from __future__ import annotations

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR


def build_scheduler(optimizer, kind: str = 'step', **kwargs):
    kind = kind.lower()
    if kind == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=kwargs.get('t_max', 100))
    return StepLR(optimizer, step_size=kwargs.get('step_size', 10), gamma=kwargs.get('gamma', 0.5))
