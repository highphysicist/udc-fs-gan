from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .preprocess import build_sr_pair


class PairedSuperResolutionDataset(Dataset):
    def __init__(self, lr_root: str, hr_root: str, image_size: int = 256, lr_size: int = 64,
                 downsample_mode: str = "bicubic", upsample_mode: str = "nearest"):
        self.lr_root = Path(lr_root)
        self.hr_root = Path(hr_root)
        self.image_size = image_size
        self.lr_size = lr_size
        self.downsample_mode = downsample_mode
        self.upsample_mode = upsample_mode
        self.lr_paths = sorted([p for p in self.lr_root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
        self.hr_paths = sorted([p for p in self.hr_root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
        self.to_tensor = transforms.ToTensor()
        if not self.hr_paths:
            raise FileNotFoundError(f"No HR images found in {hr_root}")

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr_path = self.hr_paths[idx]
        hr = Image.open(hr_path).convert("RGB").resize((self.image_size, self.image_size), resample=Image.Resampling.BICUBIC)
        lr = hr.resize((self.lr_size, self.lr_size), resample=Image.Resampling.BICUBIC)
        lr_up = lr.resize((self.image_size, self.image_size), resample=Image.Resampling.NEAREST)
        return {
            "lr": self.to_tensor(lr),
            "lr_up": self.to_tensor(lr_up),
            "hr": self.to_tensor(hr),
            "path": str(hr_path),
        }
