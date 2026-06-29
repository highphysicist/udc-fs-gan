from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from PIL import Image

RESIZE_MODES = {
    "nearest": Image.Resampling.NEAREST,
    "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
    "lanczos": Image.Resampling.LANCZOS,
    "gaussian": Image.Resampling.BOX,
}


def resize_image(image: Image.Image, size: Tuple[int, int], mode: str = "bicubic") -> Image.Image:
    resample = RESIZE_MODES.get(mode, Image.Resampling.BICUBIC)
    return image.resize(size, resample=resample)


def build_sr_pair(image: Image.Image, lr_size: int, hr_size: int, downsample_mode: str = "bicubic",
                  upsample_mode: str = "nearest"):
    hr = image.convert("RGB").resize((hr_size, hr_size), resample=Image.Resampling.BICUBIC)
    lr = hr.resize((lr_size, lr_size), resample=RESIZE_MODES.get(downsample_mode, Image.Resampling.BICUBIC))
    lr_up = lr.resize((hr_size, hr_size), resample=RESIZE_MODES.get(upsample_mode, Image.Resampling.NEAREST))
    return lr, lr_up, hr


def build_manifest(root: str | Path, exts: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp")) -> List[str]:
    root = Path(root)
    paths = []
    for ext in exts:
        paths.extend(str(p) for p in root.rglob(f"*{ext}"))
        paths.extend(str(p) for p in root.rglob(f"*{ext.upper()}"))
    paths = sorted(set(paths))
    return paths
