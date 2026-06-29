from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .discriminator_dcgan import DCGANDiscriminator


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class SRGANGenerator(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 64,
                 scale_factor: int = 4, target_size: int = 256):
        super().__init__()
        self.target_size = target_size
        self.scale_factor = scale_factor
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 9, padding=4),
            nn.PReLU(),
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(8)])
        self.mid = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
        )
        up_blocks = []
        n_up = 2 if scale_factor >= 4 else 1
        for _ in range(n_up):
            up_blocks += [
                nn.Conv2d(base_channels, base_channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU(),
            ]
        self.upsample = nn.Sequential(*up_blocks)
        self.tail = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 9, padding=4),
            nn.Tanh(),
        )

    def forward(self, x):
        if x.shape[-1] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size), mode="bicubic", align_corners=False)
        h = self.head(x)
        r = self.res_blocks(h)
        h = self.mid(r) + h
        h = self.upsample(h)
        return self.tail(h)


class SRGANWrapper(nn.Module):
    def __init__(self, target_size: int = 256, scale_factor: int = 4, base_channels: int = 64):
        super().__init__()
        self.generator = SRGANGenerator(target_size=target_size, scale_factor=scale_factor, base_channels=base_channels)
        self.discriminator = DCGANDiscriminator(target_size=target_size, base_channels=base_channels)

    def forward(self, x):
        return self.generator(x)
