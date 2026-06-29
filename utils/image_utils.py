from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid, save_image


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    if x.ndim == 4:
        x = x[0]
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1.0) / 2.0
    x = (x * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def save_image_tensor(path: str | Path, tensor: torch.Tensor):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = tensor.detach().cpu()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    x = x.clamp(-1, 1)
    save_image((x + 1.0) / 2.0, path)


def save_image_grid(path: str | Path, tensors: Iterable[torch.Tensor], nrow: int = 4):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    batch = torch.stack([t.detach().cpu() if t.ndim == 3 else t.squeeze(0).detach().cpu() for t in tensors])
    batch = batch.clamp(-1, 1)
    grid = make_grid((batch + 1.0) / 2.0, nrow=nrow)
    save_image(grid, path)
