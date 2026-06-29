from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .discriminator_dcgan import DCGANDiscriminator


class HaarDWT(nn.Module):
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 2.0
        lh = torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32) / 2.0
        hl = torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32) / 2.0
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 2.0
        kernels = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer('kernels', kernels)

    def forward(self, x):
        c = x.shape[1]
        weight = self.kernels.repeat(c, 1, 1, 1)
        x = F.conv2d(x, weight, stride=2, groups=c)
        return x


class WIPAGenerator(nn.Module):
    """Wavelet-inspired SR generator for comparison baselines."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 64,
                 target_size: int = 256):
        super().__init__()
        self.target_size = target_size
        self.resize = nn.Upsample(size=(target_size, target_size), mode='bicubic', align_corners=False)
        self.dwt = HaarDWT()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels * 4, base_channels, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(True),
        )
        self.body = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(True),
        )
        self.head = nn.Sequential(nn.Conv2d(base_channels, out_channels, 3, padding=1), nn.Tanh())

    def forward(self, x):
        x = self.resize(x)
        wf = self.dwt(x)
        wf = F.interpolate(wf, size=x.shape[-2:], mode='nearest')
        h = self.stem(wf)
        h = self.body(h) + h
        return self.head(h)


class WIPAWrapper(nn.Module):
    def __init__(self, target_size: int = 256, base_channels: int = 64):
        super().__init__()
        self.generator = WIPAGenerator(target_size=target_size, base_channels=base_channels)
        self.discriminator = DCGANDiscriminator(target_size=target_size, base_channels=base_channels)

    def forward(self, x):
        return self.generator(x)
