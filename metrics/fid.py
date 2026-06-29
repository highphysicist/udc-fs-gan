from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except Exception as exc:  # pragma: no cover
    FrechetInceptionDistance = None


class FIDMetric:
    def __init__(self, device: str | torch.device = 'cpu'):
        if FrechetInceptionDistance is None:
            raise ImportError('torchmetrics is required for FID computation')
        self.metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        self.device = device

    def update_real(self, images: torch.Tensor):
        self.metric.update(images.to(self.device), real=True)

    def update_fake(self, images: torch.Tensor):
        self.metric.update(images.to(self.device), real=False)

    def compute(self) -> float:
        return float(self.metric.compute().item())

    def reset(self):
        self.metric.reset()
