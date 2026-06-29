from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import torch


def save_checkpoint(path: str | Path, model, optimizer_g=None, optimizer_d=None, epoch: int = 0, extra: Dict[str, Any] | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'model_state': model.state_dict(),
        'epoch': epoch,
        'extra': extra or {},
    }
    if optimizer_g is not None:
        payload['optimizer_g_state'] = optimizer_g.state_dict()
    if optimizer_d is not None:
        payload['optimizer_d_state'] = optimizer_d.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model, optimizer_g=None, optimizer_d=None, map_location='cpu'):
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload['model_state'])
    if optimizer_g is not None and 'optimizer_g_state' in payload:
        optimizer_g.load_state_dict(payload['optimizer_g_state'])
    if optimizer_d is not None and 'optimizer_d_state' in payload:
        optimizer_d.load_state_dict(payload['optimizer_d_state'])
    return payload
