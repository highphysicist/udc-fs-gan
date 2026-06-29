from __future__ import annotations

import torch
from torch import nn


class PixelLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.loss = nn.L1Loss(reduction=reduction)

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        return self.loss(fake, real)
