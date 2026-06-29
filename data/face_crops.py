from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def detect_faces(image: Image.Image | np.ndarray) -> list[Tuple[int, int, int, int]]:
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    detector = cv2.CascadeClassifier(HAAR_PATH)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    return [tuple(map(int, face)) for face in faces]


def crop_largest_face(image: Image.Image) -> Optional[Image.Image]:
    faces = detect_faces(image)
    if not faces:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return image.crop((x, y, x + w, y + h))
