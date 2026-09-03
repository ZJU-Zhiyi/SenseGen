"""Shared CNN model components for SenseGen classification experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> Conv -> BN, with residual connection."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class SpectrogramCNN(nn.Module):
    """Residual CNN: 4-block for WiFi/mmWave, 5-block for ultrasound."""

    def __init__(self, in_channels=1, num_classes=15, dropout=0.3,
                 num_blocks=4):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        if num_blocks == 4:
            self.layer1 = ConvBlock(in_channels, 32, stride=2)
            self.layer2 = ConvBlock(32, 64, stride=2)
            self.layer3 = ConvBlock(64, 128, stride=2)
            self.layer4 = ConvBlock(128, 256, stride=2)
            self.layers = nn.ModuleList([self.layer1, self.layer2,
                                         self.layer3, self.layer4])
            fc_in = 256
        elif num_blocks == 5:
            self.layer1 = ConvBlock(in_channels, 32, stride=2)
            self.layer2 = ConvBlock(32, 64, stride=2)
            self.layer3 = ConvBlock(64, 128, stride=2)
            self.layer4 = ConvBlock(128, 256, stride=2)
            self.layer5 = ConvBlock(256, 512, stride=2)
            self.layers = nn.ModuleList([self.layer1, self.layer2,
                                         self.layer3, self.layer4, self.layer5])
            fc_in = 512
        else:
            raise ValueError(f"Unsupported num_blocks: {num_blocks}")

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_in, num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * F.log_softmax(pred, dim=-1), dim=-1))
