from __future__ import annotations

from typing import Union

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(0, 1)
    x = x.permute(1, 2, 0).numpy()
    return x


def compute_ssim(fake: torch.Tensor, real: torch.Tensor) -> float:
    fake_np = _to_numpy(fake)
    real_np = _to_numpy(real)
    return float(ssim(real_np, fake_np, channel_axis=-1, data_range=1.0))
