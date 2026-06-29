from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .preprocess import build_manifest, build_sr_pair
from .transforms import build_eval_transforms, build_train_transforms


class ImageFolderSRDataset(Dataset):
    def __init__(self, root: str, image_size: int = 256, lr_size: int = 64,
                 downsample_mode: str = "bicubic", upsample_mode: str = "nearest",
                 train: bool = True):
        self.root = Path(root)
        self.image_size = image_size
        self.lr_size = lr_size
        self.downsample_mode = downsample_mode
        self.upsample_mode = upsample_mode
        self.train = train
        self.paths = build_manifest(self.root)
        if not self.paths:
            raise FileNotFoundError(f"No images found in {root}")
        self.transform = build_train_transforms(image_size) if train else build_eval_transforms(image_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        hr = self.transform(image)
        lr, lr_up, hr_img = build_sr_pair(image, self.lr_size, self.image_size,
                                          downsample_mode=self.downsample_mode,
                                          upsample_mode=self.upsample_mode)
        lr = transforms.ToTensor()(lr)
        lr_up = transforms.ToTensor()(lr_up)
        hr_img = transforms.ToTensor()(hr_img)
        return {
            "lr": lr,
            "lr_up": lr_up,
            "hr": hr_img,
            "path": path,
        }


class CelebAHQSRDataset(ImageFolderSRDataset):
    pass
