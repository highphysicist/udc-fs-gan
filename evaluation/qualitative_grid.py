from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch

from utils.image_utils import tensor_to_pil


def save_qualitative_grid(items: Dict[str, torch.Tensor], out_path: str):
    names = list(items.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        ax.imshow(tensor_to_pil(items[name]))
        ax.set_title(name)
        ax.axis('off')
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
