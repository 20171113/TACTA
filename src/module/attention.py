"""
Attention modules for the proposed model (Section 3.3 of the paper).
SxTAM: Cross-task attention
SABlock: Self-attention
TemporalAttentionBlock: Temporal attention

SxTAM and SABlock are adapted from xTAM (Lopes et al., 2023) for 1D temporal signals.

"""

import math
import torch
import torch.nn as nn


class SxTAM(nn.Module):
    """
    Cross-task attention block
    """

    def __init__(self, in_ch, s=5, use_alpha=True, r=1):
        super().__init__()
        self.cdim = in_ch // r
        
        self.conv_K = nn.Sequential(
            nn.Conv1d(in_ch, in_ch//r, 1, bias=False),
            nn.BatchNorm1d(in_ch//r),
            nn.ReLU(),
            nn.Flatten(2)
        )
        self.conv_Q = nn.Sequential(
            nn.Conv1d(in_ch, in_ch//r, 1, bias=False),
            nn.BatchNorm1d(in_ch//r),
            nn.ReLU(),
            nn.Flatten(2)
        )
        self.conv_V = nn.Sequential(
            nn.Conv1d(in_ch, in_ch//r, 1, bias=False),
            nn.BatchNorm1d(in_ch//r),
            nn.ReLU()
        )
        
        self.softmax = nn.Softmax(dim=-1)
        self.use_alpha = use_alpha
        if self.use_alpha:
            self.alpha = nn.Parameter(torch.zeros(1, in_ch//r, 1))
        
        self.down = nn.MaxPool1d(s)
        self.u = nn.Upsample(scale_factor=s)
    
    def forward(self, x, y):
        x_ = self.down(x)
        B, _, Ls = x_.shape
        cdim = self.cdim
        
        K = self.conv_K(x_).transpose(1, 2)
        Q = self.conv_Q(self.down(y))
        V = self.conv_V(self.down(y))
        
        coeff = math.sqrt(K.size(2))
        corr = self.softmax(K @ Q / coeff).transpose(1, 2)
        out = self.u((V.flatten(2) @ corr).view(B, cdim, Ls))
        
        if self.use_alpha:
            out *= self.alpha
        return out


class SABlock(nn.Module):
    """
    Spatial self-attention block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, in_channels//4, 3, padding=1, bias=False),
            nn.BatchNorm1d(in_channels//4),
            nn.ReLU(),
            nn.Conv1d(in_channels//4, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.Sigmoid()
        )
        self.conv = nn.Conv1d(in_channels, out_channels, 3, padding=1, bias=False)
    
    def forward(self, x):
        attention_mask = self.attention(x)
        features = self.conv(x)
        return features * attention_mask


class TemporalAttentionBlock(nn.Module):
    """
    Multi-head temporal attention (Standard transformer-style)
    """
    def __init__(self, embed_dim=128, hidden_dim=256, num_heads=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, x, return_attn=False):
        # x: [B, C, L] -> [B, L, C]
        x = x.permute(0, 2, 1)
        attn_out, attn_weights = self.attn(x, x, x)
        x = self.norm(x + attn_out)  # Same norm applied twice (V1 style)
        ffn_out = self.ffn(x)
        x = self.norm(x + ffn_out)
        x = x.permute(0, 2, 1)
        return (x, attn_weights) if return_attn else x