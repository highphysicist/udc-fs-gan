from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from PIL import Image

try:
    import cv2
    HAAR_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
except Exception:  # pragma: no cover
    cv2 = None
    HAAR_PATH = None


def _to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu().clamp(0, 1)
    x = (x * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def count_faces(image: torch.Tensor | Image.Image) -> int:
    if cv2 is None:
        return 0
    if isinstance(image, torch.Tensor):
        image = _to_pil(image)
    arr = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    detector = cv2.CascadeClassifier(HAAR_PATH)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    return int(len(faces))
