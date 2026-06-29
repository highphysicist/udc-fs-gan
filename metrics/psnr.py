from __future__ import annotations

import math
import torch


def compute_psnr(fake: torch.Tensor, real: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((fake - real) ** 2).item()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val) - 10 * math.log10(mse)
