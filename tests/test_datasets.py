from pathlib import Path
from PIL import Image
import torch

from data.preprocess import build_sr_pair
from data.datasets import ImageFolderSRDataset


def test_build_pair(tmp_path):
    img = Image.new('RGB', (512, 512), color='white')
    lr, lr_up, hr = build_sr_pair(img, 64, 256)
    assert lr.size == (64, 64)
    assert lr_up.size == (256, 256)
    assert hr.size == (256, 256)
