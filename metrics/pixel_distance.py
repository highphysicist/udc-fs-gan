from __future__ import annotations

import torch


def compute_pixel_distance(fake: torch.Tensor, real: torch.Tensor, normalize: bool = False) -> float:
    dist = torch.sum(torch.abs(fake - real)).item()
    if normalize:
        dist = dist / fake.numel()
    return float(round(dist)) if not normalize else float(dist)
