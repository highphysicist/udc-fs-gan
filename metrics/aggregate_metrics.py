from __future__ import annotations

from typing import Dict

import torch

from .psnr import compute_psnr
from .ssim import compute_ssim
from .pixel_distance import compute_pixel_distance
from .face_detection_score import count_faces


def aggregate_metrics(fake: torch.Tensor, real: torch.Tensor) -> Dict[str, float]:
    return {
        'psnr': compute_psnr(fake, real),
        'ssim': compute_ssim(fake[0], real[0]) if fake.ndim == 4 else compute_ssim(fake, real),
        'pixel_distance': compute_pixel_distance(fake, real),
        'faces_detected': float(sum(count_faces(img) for img in fake)),
    }
