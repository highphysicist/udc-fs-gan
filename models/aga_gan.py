from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .discriminator_dcgan import DCGANDiscriminator


class AttentionGate(nn.Module):
    def __init__(self, g_ch: int, x_ch: int, inter_ch: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(g_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.W_x = nn.Sequential(nn.Conv2d(x_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi


class AGAGenerator(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 64,
                 target_size: int = 256):
        super().__init__()
        self.target_size = target_size
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, base_channels, 3, padding=1), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1), nn.BatchNorm2d(base_channels * 2), nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1), nn.BatchNorm2d(base_channels * 4), nn.ReLU(True))
        self.enc4 = nn.Sequential(nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1), nn.BatchNorm2d(base_channels * 8), nn.ReLU(True))
        self.bottleneck = nn.Sequential(nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1), nn.ReLU(True))
        self.up4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.att4 = AttentionGate(base_channels * 4, base_channels * 4, base_channels * 2)
        self.dec4 = nn.Sequential(nn.Conv2d(base_channels * 8, base_channels * 4, 3, padding=1), nn.BatchNorm2d(base_channels * 4), nn.ReLU(True))
        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.att3 = AttentionGate(base_channels * 2, base_channels * 2, base_channels)
        self.dec3 = nn.Sequential(nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1), nn.BatchNorm2d(base_channels * 2), nn.ReLU(True))
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.att2 = AttentionGate(base_channels, base_channels, base_channels // 2)
        self.dec2 = nn.Sequential(nn.Conv2d(base_channels * 2, base_channels, 3, padding=1), nn.BatchNorm2d(base_channels), nn.ReLU(True))
        self.head = nn.Sequential(nn.Conv2d(base_channels, out_channels, 3, padding=1), nn.Tanh())

    def forward(self, x):
        if x.shape[-1] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size), mode='bicubic', align_corners=False)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        d4 = self.up4(b)
        s4 = self.att4(d4, e3)
        d4 = self.dec4(torch.cat([d4, s4], dim=1))
        d3 = self.up3(d4)
        s3 = self.att3(d3, e2)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))
        d2 = self.up2(d3)
        s2 = self.att2(d2, e1)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))
        return self.head(d2)


class AGAGANWrapper(nn.Module):
    def __init__(self, target_size: int = 256, base_channels: int = 64):
        super().__init__()
        self.generator = AGAGenerator(target_size=target_size, base_channels=base_channels)
        self.discriminator = DCGANDiscriminator(target_size=target_size, base_channels=base_channels)

    def forward(self, x):
        return self.generator(x)
