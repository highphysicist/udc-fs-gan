from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .adversarial_loss import GANAdversarialLoss
from .pixel_loss import PixelLoss
from .perceptual_loss import VGGPerceptualLoss


@dataclass
class LossWeights:
    pixel: float = 1.0
    perceptual: float = 1.0
    adversarial: float = 1e-3


class CompositeSuperResolutionLoss(nn.Module):
    def __init__(self, weights: LossWeights | None = None, perceptual_pretrained: bool = True):
        super().__init__()
        self.weights = weights or LossWeights()
        self.pixel = PixelLoss()
        self.perceptual = VGGPerceptualLoss(pretrained=perceptual_pretrained)
        self.adv = GANAdversarialLoss()

    def generator_loss(self, fake, real, fake_logits):
        pixel = self.pixel(fake, real)
        perceptual = self.perceptual(fake, real)
        adv = self.adv.generator_loss(fake_logits)
        total = (self.weights.pixel * pixel +
                 self.weights.perceptual * perceptual +
                 self.weights.adversarial * adv)
        return total, {'pixel': pixel.detach(), 'perceptual': perceptual.detach(), 'adv': adv.detach()}

    def discriminator_loss(self, real_logits, fake_logits):
        return self.adv.discriminator_loss(real_logits, fake_logits)
