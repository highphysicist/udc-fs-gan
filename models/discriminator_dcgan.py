from __future__ import annotations

import torch
from torch import nn


class DCGANDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, target_size: int = 256):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 4, c * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 8, c * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c * 8, 1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.features(x)
        pooled = self.pool(feat).flatten(1)
        logits = self.classifier(pooled)
        if return_features:
            return logits, pooled
        return logits
