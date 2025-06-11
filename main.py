import os
import json
import torch
import torchvision
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import opts_egtea as opts
import time
import h5py
from tqdm import tqdm
from iou_utils import *
from eval import evaluation_detection
from tensorboardX import SummaryWriter
from dataset import VideoDataSet
from models import MYNET, SuppressNet
from loss_func import cls_loss_func, cls_loss_func_, regress_loss_func
from loss_func import MultiCrossEntropyLoss
from functools import *


def train_one_epoch(opt, model, train_dataset, optimizer, warmup=False):
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt['batch_size'], shuffle=True,
                                               num_workers=0, pin_memory=True, drop_last=False)
    epoch_cost = 0
    epoch_cost_cls = 0
    epoch_cost_reg = 0
    epoch_cost_snip = 0
   


    total_iter = len(train_dataset) // opt['batch_size']
    cls_loss = MultiCrossEntropyLoss(focal=True)
    snip_loss = MultiCrossEntropyLoss(focal=True)
   
    model.train()  # Ensure model is in training mode


    for n_iter, (input_data, cls_label, reg_label, snip_label) in enumerate(tqdm(train_loader)):
        # Warmup learning rate schedule
        if warmup and n_iter < total_iter:
            for g in optimizer.param_groups:
                g['lr'] = (n_iter + 1) * (opt['lr']) / total_iter


        # Move data to GPU
        input_data = input_data.float().cuda()
        cls_label = cls_label.float().cuda()
        reg_label = reg_label.float().cuda()
        snip_label = snip_label.float().cuda()


        # Forward pass
        try:
            act_cls, act_reg, snip_cls, attention_mask, history_metrics = model(input_data)
        except Exception as e:
            print(f"Forward pass error: {e}")
            continue


        # Check for NaN in model outputs
        if torch.isnan(act_cls).any() or torch.isnan(act_reg).any() or torch.isnan(snip_cls).any():
            print("NaN detected in model outputs, skipping batch")
            continue


        # Register hooks for gradient collection (only if gradients exist)
        if act_cls.requires_grad:
            act_cls.register_hook(partial(cls_loss.collect_grad, cls_label))
        if snip_cls.requires_grad:
            snip_cls.register_hook(partial(snip_loss.collect_grad, snip_label))


        # Initialize costs
        cost_reg = torch.tensor(0.0).cuda()
        cost_cls = torch.tensor(0.0).cuda()
        cost_snip = torch.tensor(0.0).cuda()


        # Classification loss
        try:
            cost_cls = cls_loss_func_(cls_loss, cls_label, act_cls)
            if torch.isnan(cost_cls) or torch.isinf(cost_cls):
                cost_cls = torch.tensor(0.0).cuda()
        except Exception as e:
            print(f"Classification loss error: {e}")
            cost_cls = torch.tensor(0.0).cuda()


        # Regression loss
        try:
            cost_reg = regress_loss_func(reg_label, act_reg)
            if torch.isnan(cost_reg) or torch.isinf(cost_reg):
                cost_reg = torch.tensor(0.0).cuda()
        except Exception as e:
            print(f"Regression loss error: {e}")
            cost_reg = torch.tensor(0.0).cuda()


        # Snippet loss
        try:
            cost_snip = cls_loss_func_(snip_loss, snip_label, snip_cls)
            if torch.isnan(cost_snip) or torch.isinf(cost_snip):
                cost_snip = torch.tensor(0.0).cuda()
        except Exception as e:
            print(f"Snippet loss error: {e}")
            cost_snip = torch.tensor(0.0).cuda()


        # Total loss with proper weighting
        alpha = opt.get('alpha', 1.0)
        beta = opt.get('beta', 1.0)
        gamma = opt.get('gamma', 1.0)
       
        cost = alpha * cost_cls + beta * cost_reg + gamma * cost_snip


        # Check for valid total cost
        if torch.isnan(cost) or torch.isinf(cost) or cost <= 0:
            print(f"Invalid total cost: {cost}, skipping batch")
            continue


        # Accumulate epoch costs
        epoch_cost += cost.detach().cpu().item()
        epoch_cost_cls += cost_cls.detach().cpu().item()
        epoch_cost_reg += cost_reg.detach().cpu().item()
        epoch_cost_snip += cost_snip.detach().cpu().item()


        # Backward pass with gradient clipping
        optimizer.zero_grad()
       
        try:
            cost.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        except Exception as e:
            print(f"Backward pass error: {e}")
            optimizer.zero_grad()
            continue


        # Print progress every 100 iterations
        if n_iter % 100 == 0:
            print(f"Iter {n_iter}/{total_iter}: Total={cost.item():.4f}, "
                  f"Cls={cost_cls.item():.4f}, Reg={cost_reg.item():.4f}, "
                  f"Snip={cost_snip.item():.4f}")


    return n_iter, epoch_cost, epoch_cost_cls, epoch_cost_reg, epoch_cost_snip




def eval_one_epoch(opt, model, test_dataset):
    model.eval()  # Set model to evaluation mode
   
    with torch.no_grad():  # Disable gradient computation for evaluation
        cls_loss, reg_loss, tot_loss, output_cls, output_reg, labels_cls, labels_reg, working_time, total_frames = eval_frame(opt, model, test_dataset)
   
    # Check if outputs are valid
    if not output_cls or not output_reg:
        print("Warning: Empty outputs from eval_frame")
        return 0.0, 0.0, 0.0, 0.0
   
    try:
        result_dict = eval_map_nms(opt, test_dataset, output_cls, output_reg, labels_cls, labels_reg)
       
        # Save results
        output_dict = {"version": "VERSION 1.3", "results": result_dict, "external_data": {}}
        result_file = opt["result_file"].format(opt['exp'])
       
        with open(result_file, "w") as outfile:
            json.dump(output_dict, outfile, indent=2)
       
        # Evaluate detection performance
        IoUmAP = evaluation_detection(opt, verbose=False)
       
        if IoUmAP is not None and len(IoUmAP) > 0:
            IoUmAP_5 = sum(IoUmAP[0:]) / len(IoUmAP[0:])
        else:
            print("Warning: evaluation_detection returned empty results")
            IoUmAP_5 = 0.0
           
    except Exception as e:
        print(f"Evaluation error: {e}")
        IoUmAP_5 = 0.0


    return cls_loss, reg_loss, tot_loss, IoUmAP_5




def train(opt):
    writer = SummaryWriter()
    model = MYNET(opt).cuda()
   
    # Initialize best_map attribute
    if not hasattr(model, 'best_map'):
        model.best_map = 0.0
   
    # Separate parameters for different learning rates
    rest_of_model_params = [param for name, param in model.named_parameters() if "history_unit" not in name]
 
    # Optimizer with different learning rates
    optimizer = optim.Adam([
        {'params': model.history_unit.parameters(), 'lr': 1e-6},
        {'params': rest_of_model_params, 'lr': opt["lr"]}
    ], weight_decay=opt["weight_decay"])
   
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["lr_step"])
   
    # Datasets
    train_dataset = VideoDataSet(opt, subset="train")      
    test_dataset = VideoDataSet(opt, subset=opt['inference_subset'])
   
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
   
    warmup = True  # Enable warmup for first epoch
   
    for n_epoch in range(opt['epoch']):  
        print(f"\n=== Epoch {n_epoch + 1}/{opt['epoch']} ===")
       
        # Disable warmup after first epoch
        if n_epoch >= 1:
            warmup = False
       
        # Training
        n_iter, epoch_cost, epoch_cost_cls, epoch_cost_reg, epoch_cost_snip = train_one_epoch(
            opt, model, train_dataset, optimizer, warmup)
       
        # Check for valid training costs
        if n_iter == 0:
            print("Warning: No training iterations completed")
            continue
           
        avg_cost = epoch_cost / (n_iter + 1)
        avg_cls = epoch_cost_cls / (n_iter + 1)
        avg_reg = epoch_cost_reg / (n_iter + 1)
        avg_snip = epoch_cost_snip / (n_iter + 1)
       
        # Log training metrics
        writer.add_scalars('data/cost', {'train': avg_cost}, n_epoch)
       
        print(f"Training loss(epoch {n_epoch}): {avg_cost:.4f}, "
              f"cls - {avg_cls:.4f}, reg - {avg_reg:.4f}, snip - {avg_snip:.4f}, "
              f"lr - {optimizer.param_groups[-1]['lr']:.6f}")
       
        # Learning rate scheduling
        scheduler.step()
       
        # Evaluation
        cls_loss, reg_loss, tot_loss, IoUmAP_5 = eval_one_epoch(opt, model, test_dataset)
       
        # Log evaluation metrics
        writer.add_scalars('data/mAP', {'test': IoUmAP_5}, n_epoch)
       
        print(f"Testing loss(epoch {n_epoch}): {tot_loss:.4f}, "
              f"cls - {cls_loss:.4f}, reg - {reg_loss:.4f}, mAP Avg - {IoUmAP_5:.4f}")
                   
        # Save checkpoint
        state = {
            'epoch': n_epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_map': model.best_map
        }
       
        checkpoint_path = f"{opt['checkpoint_path']}/{opt['exp']}_checkpoint_{n_epoch+1}.pth.tar"
        torch.save(state, checkpoint_path)
       
        # Save best model
        if IoUmAP_5 > model.best_map:
            model.best_map = IoUmAP_5
            best_path = f"{opt['checkpoint_path']}/{opt['exp']}_ckp_best.pth.tar"
            torch.save(state, best_path)
            print(f"New best mAP: {IoUmAP_5:.4f}")
           
        # Set model back to training mode
        model.train()
   
    writer.close()
    print(f"\nTraining completed. Best mAP: {model.best_map:.4f}")
    return model.best_map




def eval_frame(opt, model, dataset):
    test_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt['batch_size'], shuffle=False,
                                              num_workers=0, pin_memory=True, drop_last=False)
   
    # Initialize output dictionaries
    labels_cls = {}
    labels_reg = {}
    output_cls = {}
    output_reg = {}
   
    for video_name in dataset.video_list:
        labels_cls[video_name] = []
        labels_reg[video_name] = []
        output_cls[video_name] = []
        output_reg[video_name] = []
       
    start_time = time.time()
    total_frames = 0  
    epoch_cost = 0
    epoch_cost_cls = 0
    epoch_cost_reg = 0
   
    model.eval()
   
    with torch.no_grad():
        for n_iter, (input_data, cls_label, reg_label, _) in enumerate(tqdm(test_loader)):
            try:
                # Move to GPU
                input_data = input_data.float().cuda()
                cls_label = cls_label.float().cuda()
                reg_label = reg_label.float().cuda()
               
                # Forward pass
                act_cls, act_reg, _, _, _ = model(input_data)
               
                # Check for NaN outputs
                if torch.isnan(act_cls).any() or torch.isnan(act_reg).any():
                    print(f"NaN detected in evaluation at iteration {n_iter}")
                    continue
               
                # Compute losses for monitoring
                try:
                    cost_cls = cls_loss_func(cls_label, act_cls)
                    cost_reg = regress_loss_func(reg_label, act_reg)
                   
                    if not torch.isnan(cost_cls):
                        epoch_cost_cls += cost_cls.item()
                    if not torch.isnan(cost_reg):
                        epoch_cost_reg += cost_reg.item()
                       
                    cost = opt.get('alpha', 1.0) * cost_cls + opt.get('beta', 1.0) * cost_reg
                    if not torch.isnan(cost):
                        epoch_cost += cost.item()
                       
                except Exception as e:
                    print(f"Loss computation error in evaluation: {e}")
                    continue
               
                # Apply softmax to classification outputs
                act_cls = torch.softmax(act_cls, dim=-1)
               
                # Move to CPU for processing
                act_cls_cpu = act_cls.detach().cpu().numpy()
                act_reg_cpu = act_reg.detach().cpu().numpy()
                cls_label_cpu = cls_label.detach().cpu().numpy()
                reg_label_cpu = reg_label.detach().cpu().numpy()
               
                total_frames += input_data.size(0)
               
                # Store results for each sample in batch
                for b in range(input_data.size(0)):
                    try:
                        video_name, st, ed, data_idx = dataset.inputs[n_iter * opt['batch_size'] + b]
                       
                        output_cls[video_name].append(act_cls_cpu[b, :])
                        output_reg[video_name].append(act_reg_cpu[b, :])
                        labels_cls[video_name].append(cls_label_cpu[b, :])
                        labels_reg[video_name].append(reg_label_cpu[b, :])
                       
                    except IndexError as e:
                        print(f"Index error in evaluation: {e}")
                        break
                       
            except Exception as e:
                print(f"Evaluation iteration error: {e}")
                continue
       
    end_time = time.time()
    working_time = end_time - start_time
   
    # Convert lists to numpy arrays
    for video_name in dataset.video_list:
        if len(labels_cls[video_name]) > 0:
            labels_cls[video_name] = np.stack(labels_cls[video_name], axis=0)
            labels_reg[video_name] = np.stack(labels_reg[video_name], axis=0)
            output_cls[video_name] = np.stack(output_cls[video_name], axis=0)
            output_reg[video_name] = np.stack(output_reg[video_name], axis=0)
        else:
            print(f"Warning: No valid data for video {video_name}")
   
    # Compute average losses
    if n_iter > 0:
        cls_loss = epoch_cost_cls / (n_iter + 1)
        reg_loss = epoch_cost_reg / (n_iter + 1)
        tot_loss = epoch_cost / (n_iter + 1)
    else:
        cls_loss = reg_loss = tot_loss = 0.0
     
    return cls_loss, reg_loss, tot_loss, output_cls, output_reg, labels_cls, labels_reg, working_time, total_frames








def eval_map_nms(opt, dataset, output_cls, output_reg, labels_cls, labels_reg):
    result_dict={}
    proposal_dict=[]
   
    num_class = opt["num_of_class"]
    unit_size = opt['segment_size']
    threshold=opt['threshold']
    anchors=opt['anchors']
                                             
    for video_name in dataset.video_list:
        duration = dataset.video_len[video_name]
        video_time = float(dataset.video_dict[video_name]["duration"])
        frame_to_time = 100.0*video_time / duration
         
        for idx in range(0,duration):
            cls_anc = output_cls[video_name][idx]
            reg_anc = output_reg[video_name][idx]
           
            proposal_anc_dict=[]
            for anc_idx in range(0,len(anchors)):
                cls = np.argwhere(cls_anc[anc_idx][:-1]>opt['threshold']).reshape(-1)
               
                if len(cls) == 0:
                    continue
                   
                ed= idx + anchors[anc_idx] * reg_anc[anc_idx][0]
                length = anchors[anc_idx]* np.exp(reg_anc[anc_idx][1])
                st= ed-length
               
                for cidx in range(0,len(cls)):
                    label=cls[cidx]
                    tmp_dict={}
                    tmp_dict["segment"] = [float(st*frame_to_time/100.0), float(ed*frame_to_time/100.0)]
                    tmp_dict["score"]= float(cls_anc[anc_idx][label])  # Convert to Python float
                    tmp_dict["label"]=dataset.label_name[label]
                    tmp_dict["gentime"]= float(idx*frame_to_time/100.0)
                    proposal_anc_dict.append(tmp_dict)
               
            proposal_dict+=proposal_anc_dict
       
        proposal_dict=non_max_suppression(proposal_dict, overlapThresh=opt['soft_nms'])
                   
        result_dict[video_name]=proposal_dict
        proposal_dict=[]
       
    return result_dict








def eval_map_supnet(opt, dataset, output_cls, output_reg, labels_cls, labels_reg):
    model = SuppressNet(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"]+"/ckp_best_suppress.pth.tar")
    base_dict=checkpoint['state_dict']
    model.load_state_dict(base_dict)
    model.eval()
   
    result_dict={}
    proposal_dict=[]
   
    num_class = opt["num_of_class"]
    unit_size = opt['segment_size']
    threshold=opt['threshold']
    anchors=opt['anchors']
                                             
    for video_name in dataset.video_list:
        duration = dataset.video_len[video_name]
        video_time = float(dataset.video_dict[video_name]["duration"])
        frame_to_time = 100.0*video_time / duration
        conf_queue = torch.zeros((unit_size,num_class-1))
       
        for idx in range(0,duration):
            cls_anc = output_cls[video_name][idx]
            reg_anc = output_reg[video_name][idx]
           
            proposal_anc_dict=[]
            for anc_idx in range(0,len(anchors)):
                cls = np.argwhere(cls_anc[anc_idx][:-1]>opt['threshold']).reshape(-1)
               
                if len(cls) == 0:
                    continue
                   
                ed= idx + anchors[anc_idx] * reg_anc[anc_idx][0]
                length = anchors[anc_idx]* np.exp(reg_anc[anc_idx][1])
                st= ed-length
               
                for cidx in range(0,len(cls)):
                    label=cls[cidx]
                    tmp_dict={}
                    tmp_dict["segment"] = [float(st*frame_to_time/100.0), float(ed*frame_to_time/100.0)]
                    tmp_dict["score"]= float(cls_anc[anc_idx][label])  # Convert to Python float
                    tmp_dict["label"]=dataset.label_name[label]
                    tmp_dict["gentime"]= float(idx*frame_to_time/100.0)
                    proposal_anc_dict.append(tmp_dict)
                         
            proposal_anc_dict = non_max_suppression(proposal_anc_dict, overlapThresh=opt['soft_nms'])  
               
            conf_queue[:-1,:]=conf_queue[1:,:].clone()
            conf_queue[-1,:]=0
            for proposal in proposal_anc_dict:
                cls_idx = dataset.label_name.index(proposal['label'])
                conf_queue[-1,cls_idx]=proposal["score"]
           
            minput = conf_queue.unsqueeze(0)
            suppress_conf = model(minput.cuda())
            suppress_conf=suppress_conf.squeeze(0).detach().cpu().numpy()
           
            for cls in range(0,num_class-1):
                if suppress_conf[cls] > opt['sup_threshold']:
                    for proposal in proposal_anc_dict:
                        if proposal['label'] == dataset.label_name[cls]:
                            if check_overlap_proposal(proposal_dict, proposal, overlapThresh=opt['soft_nms']) is None:
                                proposal_dict.append(proposal)
           
        result_dict[video_name]=proposal_dict
        proposal_dict=[]
       
    return result_dict




 
def test_frame(opt):
    model = MYNET(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"] + "/ckp_best.pth.tar")
    base_dict = checkpoint['state_dict']
    model.load_state_dict(base_dict)
    model.eval()
   
    dataset = VideoDataSet(opt, subset=opt['inference_subset'])    
    outfile = h5py.File(opt['frame_result_file'].format(opt['exp']), 'w')
   
    cls_loss, reg_loss, tot_loss, output_cls, output_reg, labels_cls, labels_reg, working_time, total_frames = eval_frame(opt, model, dataset)
   
    print("testing loss: %f, cls_loss: %f, reg_loss: %f" % (tot_loss, cls_loss, reg_loss))
   
    for video_name in dataset.video_list:
        o_cls = output_cls[video_name]
        o_reg = output_reg[video_name]
        l_cls = labels_cls[video_name]
        l_reg = labels_reg[video_name]
       
        dset_predcls = outfile.create_dataset(video_name + '/pred_cls', o_cls.shape, maxshape=o_cls.shape, chunks=True, dtype=np.float32)
        dset_predcls[:, :] = o_cls[:, :]  
        dset_predreg = outfile.create_dataset(video_name + '/pred_reg', o_reg.shape, maxshape=o_reg.shape, chunks=True, dtype=np.float32)
        dset_predreg[:, :] = o_reg[:, :]  
        dset_labelcls = outfile.create_dataset(video_name + '/label_cls', l_cls.shape, maxshape=l_cls.shape, chunks=True, dtype=np.float32)
        dset_labelcls[:, :] = l_cls[:, :]  
        dset_labelreg = outfile.create_dataset(video_name + '/label_reg', l_reg.shape, maxshape=l_reg.shape, chunks=True, dtype=np.float32)
        dset_labelreg[:, :] = l_reg[:, :]  
    outfile.close()
                   
    print("working time : {}s, {}fps, {} frames".format(working_time, total_frames / working_time, total_frames))
   
def patch_attention(m):
    forward_orig = m.forward




    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False




        return forward_orig(*args, **kwargs)




    m.forward = wrap








class SaveOutput:
    def __init__(self):
        self.outputs = []




    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])




    def clear(self):
        self.outputs = []




def test(opt):
    model = MYNET(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"] + "/" + opt['exp'] + "_ckp_best.pth.tar")
    base_dict = checkpoint['state_dict']
    model.load_state_dict(base_dict)
    model.eval()
   
    dataset = VideoDataSet(opt, subset=opt['inference_subset'])
   
    cls_loss, reg_loss, tot_loss, output_cls, output_reg, labels_cls, labels_reg, working_time, total_frames = eval_frame(opt, model, dataset)
   
    if opt["pptype"] == "nms":
        result_dict = eval_map_nms(opt, dataset, output_cls, output_reg, labels_cls, labels_reg)
    if opt["pptype"] == "net":
        result_dict = eval_map_supnet(opt, dataset, output_cls, output_reg, labels_cls, labels_reg)
    output_dict = {"version": "VERSION 1.3", "results": result_dict, "external_data": {}}
    outfile = open(opt["result_file"].format(opt['exp']), "w")
    json.dump(output_dict, outfile, indent=2)
    outfile.close()
   
    mAP = evaluation_detection(opt)




def test_online(opt):
    model = MYNET(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"] + "/ckp_best.pth.tar")
    base_dict = checkpoint['state_dict']
    model.load_state_dict(base_dict)
    model.eval()
   
    sup_model = SuppressNet(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"] + "/ckp_best_suppress.pth.tar")
    base_dict = checkpoint['state_dict']
    sup_model.load_state_dict(base_dict)
    sup_model.eval()
   
    dataset = VideoDataSet(opt, subset=opt['inference_subset'])
    test_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=1, shuffle=False,
                                              num_workers=0, pin_memory=True, drop_last=False)
   
    result_dict = {}
    proposal_dict = []
   
    num_class = opt["num_of_class"]
    unit_size = opt['segment_size']
    threshold = opt['threshold']
    anchors = opt['anchors']
   
    start_time = time.time()
    total_frames = 0
   
    for video_name in dataset.video_list:
        input_queue = torch.zeros((unit_size, opt['feat_dim']))
        sup_queue = torch.zeros((unit_size, num_class - 1))
   
        duration = dataset.video_len[video_name]
        video_time = float(dataset.video_dict[video_name]["duration"])
        frame_to_time = 100.0 * video_time / duration
       
        for idx in range(0, duration):
            total_frames += 1
            input_queue[:-1, :] = input_queue[1:, :].clone()
            input_queue[-1:, :] = dataset._get_base_data(video_name, idx, idx + 1)
           
            minput = input_queue.unsqueeze(0)
            act_cls, act_reg, _, _, _ = model(minput.cuda())  # Ignore snip_cls, attention_mask, and history_metrics
            act_cls = torch.softmax(act_cls, dim=-1)
           
            cls_anc = act_cls.squeeze(0).detach().cpu().numpy()
            reg_anc = act_reg.squeeze(0).detach().cpu().numpy()
           
            proposal_anc_dict = []
            for anc_idx in range(0, len(anchors)):
                cls = np.argwhere(cls_anc[anc_idx][:-1] > opt['threshold']).reshape(-1)
               
                if len(cls) == 0:
                    continue
                   
                ed = idx + anchors[anc_idx] * reg_anc[anc_idx][0]
                length = anchors[anc_idx] * np.exp(reg_anc[anc_idx][1])
                st = ed - length
               
                for cidx in range(0, len(cls)):
                    label = cls[cidx]
                    tmp_dict = {}
                    tmp_dict["segment"] = [float(st * frame_to_time / 100.0), float(ed * frame_to_time / 100.0)]
                    tmp_dict["score"] = float(cls_anc[anc_idx][label])
                    tmp_dict["label"] = dataset.label_name[label]
                    tmp_dict["gentime"] = float(idx * frame_to_time / 100.0)
                    proposal_anc_dict.append(tmp_dict)
                         
            proposal_anc_dict = non_max_suppression(proposal_anc_dict, overlapThresh=opt['soft_nms'])  
               
            sup_queue[:-1, :] = sup_queue[1:, :].clone()
            sup_queue[-1, :] = 0
            for proposal in proposal_anc_dict:
                cls_idx = dataset.label_name.index(proposal['label'])
                sup_queue[-1, cls_idx] = proposal["score"]
           
            minput = sup_queue.unsqueeze(0)
            suppress_conf = sup_model(minput.cuda())
            suppress_conf = suppress_conf.squeeze(0).detach().cpu().numpy()
           
            for cls in range(0, num_class - 1):
                if suppress_conf[cls] > opt['sup_threshold']:
                    for proposal in proposal_anc_dict:
                        if proposal['label'] == dataset.label_name[cls]:
                            if check_overlap_proposal(proposal_dict, proposal, overlapThresh=opt['soft_nms']) is None:
                                proposal_dict.append(proposal)
           
        result_dict[video_name] = proposal_dict
        proposal_dict = []
   
    end_time = time.time()
    working_time = end_time - start_time
    print("working time : {}s, {}fps, {} frames".format(working_time, total_frames / working_time, total_frames))
   
    output_dict = {"version": "VERSION 1.3", "results": result_dict, "external_data": {}}
    outfile = open(opt["result_file"].format(opt['exp']), "w")
    json.dump(output_dict, outfile, indent=2)
    outfile.close()
   
    evaluation_detection(opt)








def main(opt):
    max_perf=0
    if opt['mode'] == 'train':
        max_perf=train(opt)
    if opt['mode'] == 'test':
        test(opt)
    if opt['mode'] == 'test_frame':
        test_frame(opt)
    if opt['mode'] == 'test_online':
        test_online(opt)
    if opt['mode'] == 'eval':
        evaluation_detection(opt)
       
    return max_perf




if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])
    opt_file=open(opt["checkpoint_path"]+"/"+opt["exp"]+"_opts.json","w")
    json.dump(opt,opt_file)
    opt_file.close()
   
    if opt['seed'] >= 0:
        seed = opt['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        #random.seed(seed)
           
    opt['anchors'] = [int(item) for item in opt['anchors'].split(',')]  
           
    main(opt)
    while(opt['wterm']):
        pass
