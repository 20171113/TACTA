"""
Proposed model: PGE + TACTA for joint machinability estimation.
"""

import torch
import torch.nn as nn

from .backbone import ResNet1D
from .attention import SxTAM, SABlock, TemporalAttentionBlock
from .components import TaskProjection, AuxiliaryPredictionHead, FinalPredictionHead


class ProposedModel(nn.Module):
    """
    - Architecture
        Input -> Backbone B(·)
              -> Task-specific projections P_t(·)
              -> Auxiliary prediction heads H_aux(·)
              -> Cross-task attention
              -> Self-attention
              -> Temporal attention
              -> Final prediction heads H_final(·)
    
    - Args
        task_num:    Number of tasks, Default 6 (5 machinability factors + 1 auxiliary)
                     The phase branch is used for auxilary/physics-guided learning,
                     but is excluded from the final supervised loss.
                     
        hidden_dim:  Hidden dimension of task-specific representations
        num_heads:   Number of heads in temporal multi-head attention
        temp_dim:    Inner dimension of temporal attention FFN
    """
    
    def __init__(self, task_num=6, hidden_dim=128, num_heads=8, temp_dim=256):
        super().__init__()
        self.task_num = task_num
        
        # Backbone B(·)
        self.backbone = ResNet1D()
        
        # Task-specific projections P_t(·)
        self.task_projs = nn.ModuleList([
            TaskProjection(in_ch=1024, out_ch=hidden_dim) for _ in range(task_num)
        ])
        
        # Auxiliary prediction heads H_aux(·)
        self.pred_nets_prev = nn.ModuleList([
            AuxiliaryPredictionHead(in_ch=hidden_dim) for _ in range(task_num)
        ])
        
        # Cross-task attention
        self.cattn_nets = nn.ModuleList([
            SxTAM(in_ch=hidden_dim, s=5, use_alpha=True)
            for _ in range(task_num * (task_num - 1))
        ])
        
        # Self-attention
        self.sattn_nets = nn.ModuleList([
            SABlock(in_channels=hidden_dim, out_channels=hidden_dim)
            for _ in range(task_num * (task_num - 1))
        ])
        
        self.final_projs = nn.ModuleList([
            TaskProjection(in_ch=2 * hidden_dim * (task_num - 1), out_ch=hidden_dim)
            for _ in range(task_num)
        ])
        
        # Temporal attention
        self.tattn_nets = nn.ModuleList([
            TemporalAttentionBlock(embed_dim=hidden_dim, hidden_dim=temp_dim, num_heads=num_heads)
            for _ in range(task_num)
        ])
        
        # Final prediction heads H_final(·)
        # All task branches, including the auxiliary phase head, are instantiated 
        # to maintain architectural uniformity and to simplify per-task parameter analysis.
        # However, the final supervised loss is computed 
        # only for the primary machinability tasks (Fx, Fy, Fz, VB, Ra), excluding phase.

        self.pred_nets = nn.ModuleList([
            FinalPredictionHead(in_ch=hidden_dim) for _ in range(task_num)
        ])
    
    def forward(self, x):
        """
        - input
            x: Input vibration signal [B, 3, 400]
        
        - output
            aux_outputs:   Auxiliary predictions from PGE 
            final_outputs: Final predictions after TACTA 
        """
        feat = self.backbone.get_features(x)  # [B, 1024, 25]
        
        proj_feats = [proj(feat) for proj in self.task_projs]
        
        pairs = [(i, j) for i in range(self.task_num) for j in range(self.task_num) if i != j]
        
        cattn_feats = [attn(proj_feats[i], proj_feats[j]) 
                       for (i, j), attn in zip(pairs, self.cattn_nets)]
        
        sattn_feats = [attn(proj_feats[j]) for (i, j), attn in zip(pairs, self.sattn_nets)]
        
        attn_feats = [torch.cat((c, s), dim=1) for c, s in zip(cattn_feats, sattn_feats)]
        
        dir_num = self.task_num - 1
        dir_feats = [proj(torch.cat(attn_feats[i*dir_num:(i+1)*dir_num], dim=1))
                     for i, proj in enumerate(self.final_projs)]
        
        ct_feats = [fi + dji for fi, dji in zip(proj_feats, dir_feats)]
        
        final_feats = [attn(f) for f, attn in zip(ct_feats, self.tattn_nets)]
        
        aux_outputs = [head(f) for f, head in zip(proj_feats, self.pred_nets_prev)]
        final_outputs = [head(f) for f, head in zip(final_feats, self.pred_nets)]
        
        return aux_outputs, final_outputs