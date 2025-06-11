import numpy as np
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from torch.nn.functional import normalize


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, maxlen: int = 750):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class AdaptiveSpanAttention(nn.Module):
    """
    Enhanced Adaptive Span Attention with dynamic head scaling and attention regularization.
    """
    def __init__(self, embedding_dim, num_heads, dropout, max_span=200, min_span=8):
        super(AdaptiveSpanAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.max_span = max_span
        self.min_span = min_span
        
        # Dynamic head scaling
        self.head_scale = nn.Parameter(torch.ones(num_heads))
        
        # Multi-head attention with regularization
        self.content_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout
        )
        
        # Span prediction network
        self.span_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 4),
            nn.GELU(),
            nn.Linear(embedding_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Relevance scorer
        self.relevance_scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Temporal decay weights
        self.temporal_decay = nn.Parameter(torch.tensor(0.95))
        
        # Adaptive threshold learning
        self.adaptive_threshold = nn.Parameter(torch.tensor(0.3))
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Attention regularization
        self.attention_reg = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, long_x, encoded_x):
        batch_size = long_x.shape[1]
        seq_len = long_x.shape[0]
        
        # Get current context representation
        current_context = torch.mean(encoded_x, dim=0, keepdim=True)
        
        # Predict adaptive span length
        span_logits = self.span_predictor(current_context).squeeze(0)
        adaptive_spans = (span_logits * (self.max_span - self.min_span) + self.min_span).int()
        
        # Initialize outputs
        attention_weights = torch.zeros(seq_len, batch_size, device=long_x.device)
        selected_features = []
        actual_spans = []
        
        for b in range(batch_size):
            current_span = min(adaptive_spans[b].item(), seq_len)
            actual_spans.append(current_span)
            
            # Select recent frames
            start_idx = max(0, seq_len - current_span)
            relevant_history = long_x[start_idx:, b:b+1, :]
            
            # Compute relevance scores
            current_expanded = current_context[:, b:b+1, :].expand(relevant_history.shape[0], -1, -1)
            combined_features = torch.cat([relevant_history, current_expanded], dim=-1)
            
            relevance_scores = self.relevance_scorer(combined_features).squeeze(-1).squeeze(-1)
            
            # Apply temporal decay
            temporal_positions = torch.arange(relevant_history.shape[0], device=long_x.device, dtype=torch.float)
            decay_weights = self.temporal_decay ** (relevant_history.shape[0] - 1 - temporal_positions)
            
            # Combine relevance and temporal decay
            final_scores = relevance_scores * decay_weights
            
            # Adaptive thresholding
            threshold_mask = final_scores > self.adaptive_threshold
            final_scores = final_scores * threshold_mask.float()
            
            # Apply head scaling
            scaled_scores = final_scores.unsqueeze(-1) * self.head_scale.unsqueeze(0)
            
            # Aggregate across heads (take mean)
            final_scores = scaled_scores.mean(dim=-1)
            
            # Normalize scores with regularization
            if final_scores.sum() > 0:
                final_scores = final_scores / (final_scores.sum() + self.attention_reg + 1e-8)
            else:
                final_scores = torch.ones_like(final_scores) / final_scores.shape[0]
            
            attention_weights[start_idx:, b] = final_scores
            
            # Compute weighted feature representation
            weighted_features = (relevant_history.squeeze(1) * final_scores.unsqueeze(-1)).sum(dim=0)
            selected_features.append(weighted_features)
        
        adaptive_features = torch.stack(selected_features, dim=0)
        
        return attention_weights, adaptive_features, actual_spans


class HierarchicalContextEncoder(nn.Module):
    """
    Hierarchical encoder with improved temporal pooling.
    """
    def __init__(self, embedding_dim, num_heads, dropout):
        super(HierarchicalContextEncoder, self).__init__()
        
        self.fine_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim, nhead=num_heads, 
                dropout=dropout, activation='gelu'
            ), num_layers=2
        )
        
        self.coarse_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim, nhead=num_heads, 
                dropout=dropout, activation='gelu'
            ), num_layers=2
        )
        
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        self.scale_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, features, attention_mask):
        fine_features = self.fine_encoder(features)
        
        seq_len, batch_size, dim = features.shape
        if seq_len >= 4:
            pooled_features = features.permute(1, 2, 0)
            pooled_features = F.avg_pool1d(pooled_features, kernel_size=4, stride=4)
            pooled_features = pooled_features.permute(2, 0, 1)
            
            coarse_features = self.coarse_encoder(pooled_features)
            
            coarse_upsampled = F.interpolate(
                coarse_features.permute(1, 2, 0), 
                size=seq_len, mode='linear', align_corners=False
            ).permute(2, 0, 1)
        else:
            coarse_upsampled = fine_features
        
        fused_features = torch.cat([fine_features, coarse_upsampled], dim=-1)
        fused_features = self.scale_fusion(fused_features)
        
        return self.layer_norm(fused_features)


class MemoryEfficientHistoryUnit(nn.Module):
    """
    Enhanced Memory-Efficient History Unit with temporal consistency loss.
    """
    def __init__(self, opt):
        super(MemoryEfficientHistoryUnit, self).__init__()
        self.n_feature = opt["feat_dim"]
        n_class = opt["num_of_class"]
        n_embedding_dim = opt["hidden_dim"]
        self.anchors = opt["anchors"]
        self.history_tokens = 16
        self.short_window_size = 16
        dropout = 0.3
        
        self.max_history_length = opt.get("max_history_length", 200)
        self.min_history_length = opt.get("min_history_length", 8)
        
        self.history_positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)
        
        self.adaptive_span_attention = AdaptiveSpanAttention(
            embedding_dim=n_embedding_dim,
            num_heads=4,
            dropout=dropout,
            max_span=self.max_history_length,
            min_span=self.min_history_length
        )
        
        self.hierarchical_encoder = HierarchicalContextEncoder(
            embedding_dim=n_embedding_dim,
            num_heads=4,
            dropout=dropout
        )
        
        self.adaptive_history_generator = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim),
            nn.GELU(),
            nn.Linear(n_embedding_dim, self.history_tokens * n_embedding_dim)
        )
        
        self.context_decoder_1 = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=n_embedding_dim, nhead=4, 
                dropout=dropout, activation='gelu'
            ), num_layers=3, norm=nn.LayerNorm(n_embedding_dim)
        )
        
        self.context_decoder_2 = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=n_embedding_dim, nhead=4, 
                dropout=dropout, activation='gelu'
            ), num_layers=2, norm=nn.LayerNorm(n_embedding_dim)
        )
        
        self.efficiency_head = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_embedding_dim // 2, n_embedding_dim // 4),
            nn.GELU()
        )
        
        self.snippet_classifier = nn.Sequential(
            nn.Linear(self.history_tokens * n_embedding_dim // 4, n_embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_embedding_dim, n_embedding_dim // 2),
            nn.GELU(),
            nn.Linear(n_embedding_dim // 2, n_class)
        )
        
        self.span_consistency_weight = nn.Parameter(torch.tensor(0.1))
        
        self.norm1 = nn.LayerNorm(n_embedding_dim)
        self.norm2 = nn.LayerNorm(n_embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Temporal consistency loss
        self.temporal_consistency_loss = nn.MSELoss()
        
    def forward(self, long_x, encoded_x):
        attention_weights, adaptive_features, actual_spans = self.adaptive_span_attention(long_x, encoded_x)
        
        hist_pe_x = self.history_positional_encoding(long_x)
        
        weighted_history = hist_pe_x * attention_weights.unsqueeze(-1)
        
        hierarchical_features = self.hierarchical_encoder(weighted_history, attention_weights)
        
        current_context = torch.mean(encoded_x, dim=0)
        adaptive_tokens = self.adaptive_history_generator(current_context)
        adaptive_tokens = adaptive_tokens.view(-1, self.history_tokens, adaptive_tokens.shape[-1] // self.history_tokens)
        adaptive_tokens = adaptive_tokens.permute(1, 0, 2)
        
        hist_encoded_1 = self.context_decoder_1(adaptive_tokens, hierarchical_features)
        hist_encoded_1 = self.norm1(hist_encoded_1 + self.dropout1(adaptive_tokens))
        
        hist_encoded_2 = self.context_decoder_2(hist_encoded_1, encoded_x)
        hist_encoded_2 = self.norm2(hist_encoded_2 + self.dropout2(hist_encoded_1))
        
        snippet_features = self.efficiency_head(hist_encoded_1)
        snippet_features = torch.flatten(snippet_features.permute(1, 0, 2), start_dim=1)
        snippet_predictions = self.snippet_classifier(snippet_features)
        
        # Compute temporal consistency loss
        temporal_loss = self.temporal_consistency_loss(
            snippet_predictions[:, :-1], snippet_predictions[:, 1:]
        )
        
        avg_span_length = sum(actual_spans) / len(actual_spans)
        span_variance = torch.var(torch.tensor(actual_spans, dtype=torch.float))
        
        return hist_encoded_2, snippet_predictions, attention_weights, {
            'adaptive_spans': actual_spans,
            'avg_span_length': avg_span_length,
            'span_variance': span_variance,
            'adaptive_features': adaptive_features,
            'temporal_loss': temporal_loss * self.span_consistency_weight
        }


class MYNET(torch.nn.Module):
    def __init__(self, opt):
        super(MYNET, self).__init__()
        self.n_feature=opt["feat_dim"]
        n_class=opt["num_of_class"]
        n_embedding_dim=opt["hidden_dim"]
        n_enc_layer=opt["enc_layer"]
        n_enc_head=opt["enc_head"]
        n_dec_layer=opt["dec_layer"]
        n_dec_head=opt["dec_head"]
        n_comb_dec_head = 4
        n_comb_dec_layer = 6  # Increased depth for better anchor refinement
        n_seglen=opt["segment_size"]
        self.anchors=opt["anchors"]
        self.history_tokens = 16
        self.short_window_size = 16
        self.anchors_stride=[]
        dropout=0.3
        self.best_loss=1000000
        self.best_map=0

        # Enhanced feature fusion
        self.feature_reduction_rgb = nn.Sequential(
            nn.Linear(self.n_feature//2, n_embedding_dim//2),
            nn.GELU(),
            nn.LayerNorm(n_embedding_dim//2)
        )
        self.feature_reduction_flow = nn.Sequential(
            nn.Linear(self.n_feature//2, n_embedding_dim//2),
            nn.GELU(),
            nn.LayerNorm(n_embedding_dim//2)
        )
       
        self.positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)      
       
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=n_embedding_dim,
                                      nhead=n_enc_head,
                                      dropout=dropout,
                                      activation='gelu'),
            n_enc_layer,
            nn.LayerNorm(n_embedding_dim))
                                           
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=n_embedding_dim,
                                      nhead=n_dec_head,
                                      dropout=dropout,
                                      activation='gelu'),
            n_dec_layer,
            nn.LayerNorm(n_embedding_dim))  

        self.history_unit = MemoryEfficientHistoryUnit(opt)

        self.history_anchor_decoder_block1 = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=n_embedding_dim,
                                      nhead=n_comb_dec_head,
                                      dropout=dropout,
                                      activation='gelu'),
            n_comb_dec_layer,
            nn.LayerNorm(n_embedding_dim))  
           
        self.classifier = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim),
            nn.GELU(),
            nn.Linear(n_embedding_dim, n_class)
        )
        self.regressor = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim),
            nn.GELU(),
            nn.Linear(n_embedding_dim, 2)
        )    
                           
        self.decoder_token = nn.Parameter(torch.zeros(len(self.anchors), 1, n_embedding_dim))

        self.norm1 = nn.LayerNorm(n_embedding_dim)
        self.dropout1 = nn.Dropout(0.1)

        self.relu = nn.ReLU(True)
        self.softmaxd1 = nn.Softmax(dim=-1)
        
        self.span_consistency_loss = nn.MSELoss()

    def forward(self, inputs):
        base_x_rgb = self.feature_reduction_rgb(inputs[:,:,:self.n_feature//2].float())
        base_x_flow = self.feature_reduction_flow(inputs[:,:,self.n_feature//2:].float())
        base_x = torch.cat([base_x_rgb, base_x_flow], dim=-1)

        base_x = base_x.permute([1,0,2])

        short_x = base_x[-self.short_window_size:]
        long_x = base_x[:-self.short_window_size]

        pe_x = self.positional_encoding(short_x)
        encoded_x = self.encoder(pe_x)
        decoder_token = self.decoder_token.expand(-1, encoded_x.shape[1], -1)
        decoded_x = self.decoder(decoder_token, encoded_x)

        hist_encoded_x, snip_cls, attention_mask, history_metrics = self.history_unit(long_x, encoded_x)

        decoded_anchor_feat = self.history_anchor_decoder_block1(decoded_x, hist_encoded_x)
        decoded_anchor_feat = decoded_anchor_feat + self.dropout1(decoded_x)
        decoded_anchor_feat = self.norm1(decoded_anchor_feat)
        decoded_anchor_feat = decoded_anchor_feat.permute([1, 0, 2])

        anc_cls = self.classifier(decoded_anchor_feat)
        anc_reg = self.regressor(decoded_anchor_feat)

        return anc_cls, anc_reg, snip_cls, attention_mask, history_metrics

 
class SuppressNet(torch.nn.Module):
    def __init__(self, opt):
        super(SuppressNet, self).__init__()
        n_class=opt["num_of_class"]-1
        n_seglen=opt["segment_size"]
        n_embedding_dim=2*n_seglen
        dropout=0.3
        self.best_loss=1000000
        self.best_map=0
       
        self.mlp1 = nn.Linear(n_seglen, n_embedding_dim)
        self.mlp2 = nn.Linear(n_embedding_dim, 1)
        self.norm = nn.InstanceNorm1d(n_class)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
       
    def forward(self, inputs):
        base_x = inputs.permute([0,2,1])
        base_x = self.norm(base_x)
        x = self.relu(self.mlp1(base_x))
        x = self.sigmoid(self.mlp2(x))
        x = x.squeeze(-1)
       
        return x
