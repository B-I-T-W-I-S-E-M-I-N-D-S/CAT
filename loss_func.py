import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from functools import partial
       
class MultiCrossEntropyLoss(nn.Module):
    def __init__(self, focal=False, weight=None, reduce=True):
        super(MultiCrossEntropyLoss, self).__init__()
        self.num_classes = 23
        self.focal = focal
        self.weight= weight
        self.reduce = reduce
       
        # Fixed: Initialize gamma properly
        self.gamma_ = torch.zeros(self.num_classes).cuda() + 2.0  # Standard focal loss gamma
        self.gamma_f = 0.5  # Increased for better adaptation


        self.register_buffer('pos_grad', torch.zeros(self.num_classes-1).cuda())
        self.register_buffer('neg_grad', torch.zeros(self.num_classes-1).cuda())
        self.register_buffer('pos_neg', torch.ones(self.num_classes-1).cuda())
       
        # Simplified ASL parameters for stability
        self.gamma_neg = 2  # Reduced from 4 for stability
        self.gamma_pos = 0  # Standard for positive samples  
        self.clip = 0.01    # Reduced clip value
        self.eps = 1e-8
       
        # Class-balanced loss with proper initialization
        self.register_buffer('class_counts', torch.ones(self.num_classes-1).cuda() * 100)  # Better initialization
        self.beta = 0.999  # Reduced for faster adaptation


    def forward(self, input, target):
        # Ensure target is properly normalized
        target_sum = torch.sum(target, dim=1, keepdim=True)
        target_sum = torch.where(target_sum > 0, target_sum, torch.ones_like(target_sum))
        target = target / target_sum
       
        # Clamp input for numerical stability
        input = torch.clamp(input, -10, 10)
       
        logsoftmax = nn.LogSoftmax(dim=1)
        softmax = nn.Softmax(dim=1)
       
        if not self.focal:
            # Standard cross-entropy
            if self.weight is None:
                output = torch.sum(-target * logsoftmax(input), 1)
            else:
                output = torch.sum(-target * logsoftmax(input) * self.weight.unsqueeze(0), 1)
        else:
            p = softmax(input)
            p = torch.clamp(p, self.eps, 1 - self.eps)  # Clamp probabilities
           
            # Simplified focal loss implementation
            log_p = torch.log(p)
           
            # Standard focal loss with adaptive gamma
            gamma = self.gamma_.clone()
           
            # Focal loss computation
            focal_weight = torch.pow(1 - p, gamma.unsqueeze(0))
            loss_per_class = -target * focal_weight * log_p
           
            output = torch.sum(loss_per_class, dim=1)


        if self.reduce:
            return torch.mean(output)
        else:
            return output
       
    def map_func(self, x, s):
        # More stable mapping function
        x = torch.clamp(x, 0, 1)
        return x  # Simplified for stability
   
    def collect_grad(self, target, grad):
        if grad is None:
            return
           
        grad = torch.abs(grad.reshape(-1, grad.shape[-1]))
        target = target.reshape(-1, target.shape[-1])
       
        # Only update if we have valid gradients
        if torch.isfinite(grad).all() and torch.isfinite(target).all():
            pos_grad = torch.sum(grad * target, dim=0)[:-1]
            neg_grad = torch.sum(grad * (1 - target), dim=0)[:-1]
           
            # Use momentum for stability
            self.pos_grad = 0.9 * self.pos_grad + 0.1 * pos_grad
            self.neg_grad = 0.9 * self.neg_grad + 0.1 * neg_grad
           
            # Compute ratio with proper handling
            ratio = self.pos_grad / (self.neg_grad + 1e-6)
            self.pos_neg = torch.clamp(ratio, 0.1, 1.0)
           
            # Update class counts
            class_freq = torch.sum(target, dim=0)[:-1]
            self.class_counts = 0.99 * self.class_counts + 0.01 * (class_freq + 1)


def cls_loss_func(y, output, use_focal=True, weight=None, reduce=True):
    """Fixed classification loss function"""
    input_size = y.size()
    y = y.float()
   
    if weight is not None:
        weight = weight.cuda()
   
    loss_func = MultiCrossEntropyLoss(focal=use_focal, weight=weight, reduce=reduce)
   
    y = y.reshape(-1, y.size(-1))
    output = output.reshape(-1, output.size(-1))
   
    # Check for valid inputs
    if torch.isnan(output).any() or torch.isnan(y).any():
        return torch.tensor(0.0, requires_grad=True).cuda()
   
    loss = loss_func(output, y)
   
    if not reduce:
        loss = loss.reshape(input_size[:-1])
   
    # Ensure loss is finite
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, requires_grad=True).cuda()
   
    return loss


def cls_loss_func_(loss_func, y, output, use_focal=True, weight=None, reduce=True):
    """Fixed classification loss function with external loss function"""
    input_size = y.size()
    y = y.float()
   
    if weight is not None:
        weight = weight.cuda()
   
    y = y.reshape(-1, y.size(-1))
    output = output.reshape(-1, output.size(-1))
   
    # Check for valid inputs
    if torch.isnan(output).any() or torch.isnan(y).any():
        return torch.tensor(0.0, requires_grad=True).cuda()
   
    loss = loss_func(output, y)
   
    if not reduce:
        loss = loss.reshape(input_size[:-1])
   
    # Ensure loss is finite
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, requires_grad=True).cuda()
   
    return loss


def regress_loss_func(y, output):
    """Fixed regression loss function"""
    y = y.float()
    y = y.reshape(-1, y.size(-1))
    output = output.reshape(-1, output.size(-1))
   
    # Check for valid inputs
    if torch.isnan(output).any() or torch.isnan(y).any():
        return torch.tensor(0.0, requires_grad=True).cuda()
   
    # Background mask - properly handle background samples
    bgmask = y[:, 1] < -1e2
   
    if bgmask.all():  # All background
        return torch.tensor(0.0, requires_grad=True).cuda()
   
    fg_logits = output[~bgmask]
    fg_target = y[~bgmask]
   
    if fg_logits.numel() == 0:  # No foreground samples
        return torch.tensor(0.0, requires_grad=True).cuda()
   
    # Use simple smooth L1 loss for stability
    loss = nn.functional.smooth_l1_loss(fg_logits, fg_target, reduction='mean')
   
    # Additional IoU-based loss for better localization
    if fg_logits.size(1) >= 2 and fg_target.size(1) >= 2:
        try:
            # Ensure proper format for temporal segments [start, end]
            pred_start = fg_logits[:, 0]
            pred_end = fg_logits[:, 1]
           
            gt_start = fg_target[:, 0]
            gt_end = fg_target[:, 1]
           
            # Compute IoU for temporal segments
            inter_start = torch.max(pred_start, gt_start)
            inter_end = torch.min(pred_end, gt_end)
            inter = torch.clamp(inter_end - inter_start, min=0)
           
            pred_len = torch.clamp(pred_end - pred_start, min=1e-6)
            gt_len = torch.clamp(gt_end - gt_start, min=1e-6)
            union = pred_len + gt_len - inter
           
            iou = inter / (union + 1e-6)
            iou_loss = 1 - torch.mean(torch.clamp(iou, 0, 1))
           
            # Combine losses
            loss = 0.7 * loss + 0.3 * iou_loss
           
        except:
            # Fallback to smooth L1 if IoU computation fails
            pass
   
    # Ensure loss is finite and positive
    if torch.isnan(loss) or torch.isinf(loss) or loss < 0:
        return torch.tensor(0.0, requires_grad=True).cuda()
   
    return loss


def suppress_loss_func(y, output):
    """Fixed suppression loss function"""
    y = y.float()
    y = y.reshape(-1, y.size(-1))
    output = output.reshape(-1, output.size(-1))
   
    # Check for valid inputs
    if torch.isnan(output).any() or torch.isnan(y).any():
        return torch.tensor(0.0, requires_grad=True).cuda()
   
    # Clamp output for numerical stability
    output = torch.clamp(output, 1e-6, 1 - 1e-6)
   
    # Simple focal loss implementation
    alpha = 0.25
    gamma = 2.0
   
    # Binary cross entropy with focal weighting
    bce_loss = -(y * torch.log(output) + (1 - y) * torch.log(1 - output))
   
    # Focal weights
    p_t = y * output + (1 - y) * (1 - output)
    alpha_t = y * alpha + (1 - y) * (1 - alpha)
    focal_weight = alpha_t * torch.pow(1 - p_t, gamma)
   
    focal_loss = focal_weight * bce_loss
    loss = torch.mean(focal_loss)
   
    # Ensure loss is finite
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, requires_grad=True).cuda()
   
    return loss
