from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses.composite_loss import CompositeSuperResolutionLoss, LossWeights
from metrics.aggregate_metrics import aggregate_metrics
from utils.image_utils import save_image_tensor
from .checkpointing import save_checkpoint


@dataclass
class TrainerConfig:
    device: str = 'cuda'
    log_every: int = 50
    save_every: int = 1
    output_dir: str = 'outputs'


class SuperResolutionTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 device: str = 'cuda', lr: float = 2e-4, beta1: float = 0.5, beta2: float = 0.999,
                 weights: LossWeights | None = None, output_dir: str = 'outputs'):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.losses = CompositeSuperResolutionLoss(weights=weights)
        self.opt_g = torch.optim.Adam(self.model.generator.parameters(), lr=lr, betas=(beta1, beta2))
        self.opt_d = torch.optim.Adam(self.model.discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    def train_batch(self, batch):
        lr_img = batch['lr'].to(self.device)
        hr_img = batch['hr'].to(self.device)
        fake = self.model(lr_img)

        # discriminator
        real_logits = self.model.discriminator(hr_img)
        fake_logits = self.model.discriminator(fake.detach())
        d_loss = self.losses.discriminator_loss(real_logits, fake_logits)
        self.opt_d.zero_grad()
        d_loss.backward()
        self.opt_d.step()

        # generator
        fake_logits_g = self.model.discriminator(fake)
        g_loss, components = self.losses.generator_loss(fake, hr_img, fake_logits_g)
        self.opt_g.zero_grad()
        g_loss.backward()
        self.opt_g.step()

        return {
            'g_loss': float(g_loss.detach().cpu().item()),
            'd_loss': float(d_loss.detach().cpu().item()),
            **{k: float(v.cpu().item()) for k, v in components.items()},
        }

    @torch.no_grad()
    def validate(self, epoch: int = 0):
        self.model.eval()
        metrics = []
        for i, batch in enumerate(self.val_loader):
            lr_img = batch['lr'].to(self.device)
            hr_img = batch['hr'].to(self.device)
            fake = self.model(lr_img)
            metrics.append(aggregate_metrics(fake, hr_img))
            if i == 0:
                save_image_tensor(self.output_dir / f'epoch_{epoch:04d}_sample.png', fake[0].detach().cpu())
        self.model.train()
        if not metrics:
            return {}
        keys = metrics[0].keys()
        return {k: float(sum(m[k] for m in metrics) / len(metrics)) for k in keys}

    def fit(self, epochs: int, checkpoint_dir: str = 'weights', start_epoch: int = 1):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        history = []
        for epoch in range(start_epoch, start_epoch + epochs):
            self.model.train()
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
            running = []
            for batch in pbar:
                logs = self.train_batch(batch)
                running.append(logs)
                pbar.set_postfix({k: f'{v:.4f}' for k, v in logs.items() if k.endswith('loss')})
            val_metrics = self.validate(epoch)
            if epoch % 1 == 0:
                save_checkpoint(checkpoint_dir / f'udc_fsgan_epoch_{epoch:04d}.pt', self.model, self.opt_g, self.opt_d, epoch=epoch, extra={'val': val_metrics})
            epoch_log = {'epoch': epoch}
            if running:
                for key in running[0].keys():
                    epoch_log[key] = float(sum(x[key] for x in running) / len(running))
            epoch_log.update({f'val_{k}': v for k, v in val_metrics.items()})
            history.append(epoch_log)
        return history
