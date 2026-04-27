"""
Component modules for the proposed model.
"""

import torch
import torch.nn as nn


class TaskProjection(nn.Module):
    """
    Task-specific projection layer P_t(·)
    Maps shared backbone features to task-specific representations.
    """
    def __init__(self, in_ch=1024, out_ch=256, kernel_size=1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_ch, in_ch//2, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(in_ch//2),
            nn.ReLU(),
            nn.Conv1d(in_ch//2, out_ch, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.proj(x)


class AuxiliaryPredictionHead(nn.Module):
    """
    Auxiliary prediction head H_aux(·)
    Produces preliminary task estimates for computing l_aux and l_phy.
    """
    def __init__(self, in_ch=128, kernel_size=7):
        super().__init__()
        self.pred1 = nn.Sequential(
            nn.Conv1d(in_ch, in_ch//4, kernel_size=kernel_size, stride=1, padding=0),
            nn.BatchNorm1d(in_ch//4),
            nn.ReLU()
        )
        self.avgpool = nn.AvgPool1d(kernel_size=7, stride=1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_ch//4, 1)
    
    def forward(self, x):
        x = self.pred1(x)
        x = self.avgpool(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


class FinalPredictionHead(nn.Module):
    """
    Final prediction head H_final(·)
    Produces final task-specific outputs after TACTA.
    """
    def __init__(self, in_ch=128, kernel_size=7):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_ch, in_ch*4, kernel_size=1, padding=0),
            nn.BatchNorm1d(in_ch*4),
            nn.ReLU()
        )
        self.pred1 = nn.Sequential(
            nn.Conv1d(in_ch*4, in_ch*2, kernel_size),
            nn.BatchNorm1d(in_ch*2),
            nn.ReLU()
        )
        self.pred2 = nn.Sequential(
            nn.Conv1d(in_ch*2, in_ch, kernel_size),
            nn.BatchNorm1d(in_ch),
            nn.ReLU()
        )
        self.avgpool = nn.AvgPool1d(kernel_size=7, stride=1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_ch, 1)
    
    def forward(self, x):
        x = self.proj(x)
        x = self.pred1(x)
        x = self.pred2(x)
        x = self.avgpool(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)