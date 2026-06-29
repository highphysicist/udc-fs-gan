from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn
from torchvision import models


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_ids: Sequence[int] = (3, 8, 17, 26, 35), pretrained: bool = True):
        super().__init__()
        weights = models.VGG19_Weights.DEFAULT if pretrained else None
        vgg = models.vgg19(weights=weights).features
        self.slices = nn.ModuleList()
        last = 0
        for layer_id in layer_ids:
            self.slices.append(nn.Sequential(*[vgg[i] for i in range(last, layer_id + 1)]))
            last = layer_id + 1
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.l1 = nn.L1Loss()

    def _normalize(self, x):
        return (x + 1.0) / 2.0

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        fake = self._normalize(fake)
        real = self._normalize(real)
        fake = (fake - self.mean) / self.std
        real = (real - self.mean) / self.std
        loss = 0.0
        for block in self.slices:
            fake = block(fake)
            real = block(real)
            loss = loss + self.l1(fake, real)
        return loss
