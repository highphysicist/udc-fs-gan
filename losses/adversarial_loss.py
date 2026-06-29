from __future__ import annotations

import torch
from torch import nn


class GANAdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def discriminator_loss(self, real_logits, fake_logits):
        real_targets = torch.ones_like(real_logits)
        fake_targets = torch.zeros_like(fake_logits)
        return self.criterion(real_logits, real_targets) + self.criterion(fake_logits, fake_targets)

    def generator_loss(self, fake_logits):
        targets = torch.ones_like(fake_logits)
        return self.criterion(fake_logits, targets)
