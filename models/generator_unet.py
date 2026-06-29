from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_bn: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.extend([nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=not use_bn)])
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.conv(x)
        return feat, self.pool(feat)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetSRGenerator(nn.Module):
    """U-Net style generator for 4x face super-resolution."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 64,
                 target_size: int = 256, resize_mode: str = "nearest"):
        super().__init__()
        self.target_size = target_size
        self.resize_mode = resize_mode
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)
        self.down4 = DownBlock(base_channels * 8, base_channels * 8)
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)
        self.up4 = UpBlock(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up3 = UpBlock(base_channels * 8, base_channels * 8, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2, base_channels * 2, base_channels)
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.target_size and x.shape[-2] == self.target_size:
            return x
        mode = self.resize_mode
        if mode in {"nearest", "area"}:
            return F.interpolate(x, size=(self.target_size, self.target_size), mode=mode)
        return F.interpolate(x, size=(self.target_size, self.target_size), mode=mode, align_corners=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._resize(x)
        x0 = self.stem(x)
        s1, x = self.down1(x0)
        s2, x = self.down2(x)
        s3, x = self.down3(x)
        s4, x = self.down4(x)
        latent = self.bottleneck(x)
        return latent

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        x = self._resize(x)
        x0 = self.stem(x)
        s1, x = self.down1(x0)
        s2, x = self.down2(x)
        s3, x = self.down3(x)
        s4, x = self.down4(x)
        latent = self.bottleneck(x)
        x = self.up4(latent, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        out = self.head(x)
        if return_latent:
            return out, latent
        return out
