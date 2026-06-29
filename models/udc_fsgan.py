from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .generator_unet import UNetSRGenerator
from .discriminator_dcgan import DCGANDiscriminator


@dataclass
class UDCFSGANConfig:
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 64
    target_size: int = 256
    resize_mode: str = "nearest"


class UDCFSGAN(nn.Module):
    def __init__(self, config: UDCFSGANConfig | None = None):
        super().__init__()
        self.config = config or UDCFSGANConfig()
        self.generator = UNetSRGenerator(
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            base_channels=self.config.base_channels,
            target_size=self.config.target_size,
            resize_mode=self.config.resize_mode,
        )
        self.discriminator = DCGANDiscriminator(
            in_channels=self.config.out_channels,
            base_channels=self.config.base_channels,
            target_size=self.config.target_size,
        )

    def forward(self, lr: torch.Tensor, return_latent: bool = False):
        return self.generator(lr, return_latent=return_latent)

    def generate(self, lr: torch.Tensor):
        return self.forward(lr)

    def discriminate(self, image: torch.Tensor, return_features: bool = False):
        return self.discriminator(image, return_features=return_features)

    def extract_latent(self, lr: torch.Tensor) -> torch.Tensor:
        _, latent = self.forward(lr, return_latent=True)
        return latent.flatten(1)
