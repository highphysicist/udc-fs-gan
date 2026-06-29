import torch
from models.generator_unet import UNetSRGenerator
from models.discriminator_dcgan import DCGANDiscriminator


def test_generator_shape():
    model = UNetSRGenerator(target_size=256, base_channels=16)
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 3, 256, 256)


def test_discriminator_shape():
    model = DCGANDiscriminator(base_channels=16)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    assert y.shape == (2, 1)
