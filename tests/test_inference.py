import torch
from models.generator_unet import UNetSRGenerator


def test_inference_forward():
    model = UNetSRGenerator(target_size=256, base_channels=16)
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    assert y.shape == (1, 3, 256, 256)
