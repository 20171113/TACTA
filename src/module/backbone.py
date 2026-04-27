"""
ResNet-based 1D-CNN backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.activation = activation
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        if self.activation:
            x = self.relu(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 kernel_size=3, padding=1, plain=False):
        super().__init__()
        self.relu = nn.ReLU()
        self.plain = plain
        
        if in_channels == 64:
            stride = 1
            self.iden = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        elif in_channels == out_channels:
            stride = 1
            self.iden = nn.Identity()
        else:
            stride = 2
            self.iden = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2)
        
        self.convseq = nn.Sequential(
            ConvBlock(in_channels, hidden_channels, kernel_size=1, padding=0, stride=stride),
            ConvBlock(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            ConvBlock(hidden_channels, out_channels, activation=False, kernel_size=1, padding=0)
        )
    
    def forward(self, x):
        x_ = self.convseq(x)
        if self.plain:
            x = x_
        else:
            x = x_ + self.iden(x)
        return self.relu(x)


class ResNet1D(nn.Module):
    """
    ResNet-based 1D-CNN backbone B(·)
    
    For the proposed method, only get_features() is used,
    which returns [B, 1024, 25] representation after conv4_x.
    """
    def __init__(self, in_channels=3, num_output=5, num_sigma=None,
                 is_plain=False, use_sigma_layer=False, tensor_init=None,
                 scaled=False):
        super().__init__()
        self.num_output = num_output
        self.scaled = scaled
        self.num_sigma = num_output if num_sigma is None else num_sigma
        
        self.conv1 = ConvBlock(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.conv2_x = nn.Sequential(*[
            ResBlock(64 if i == 0 else 256, 64, 256, 9, 4, is_plain) for i in range(3)
        ])
        self.conv3_x = nn.Sequential(*[
            ResBlock(256 if i == 0 else 512, 128, 512, 9, 4, is_plain) for i in range(4)
        ])
        self.conv4_x = nn.Sequential(*[
            ResBlock(512 if i == 0 else 1024, 256, 1024, 9, 4, is_plain) for i in range(6)
        ])
        # conv5_x retained for checkpoint compatibility (not used in get_features)
        self.conv5_x = nn.Sequential(*[
            ResBlock(1024 if i == 0 else 2048, 512, 2048, 9, 4, is_plain) for i in range(3)
        ])
        
        # FC layers retained for checkpoint compatibility
        self.avgpool = nn.AvgPool1d(kernel_size=7, stride=1)
        self.relu = nn.ReLU()
        self.fc_mu = nn.Linear(14336, num_output)
        self.use_sigma_layer = use_sigma_layer
        if self.use_sigma_layer:
            self.fc_sigma = nn.Linear(14336, self.num_sigma)
        else:
            self.log_sigma = nn.Parameter(torch.ones(self.num_sigma))
        
        if tensor_init:
            self.log_sigma = nn.Parameter(torch.tensor(tensor_init))
    
    def get_features(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        return x  # [B, 1024, 25]
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x_mu = self.fc_mu(x)
        if self.use_sigma_layer:
            x_sigma = F.softplus(self.fc_sigma(x))
        else:
            x_sigma = F.softplus(self.log_sigma)
        if self.scaled:
            x_mu = self.relu(x_mu)
        return x_mu, x_sigma