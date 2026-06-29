from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .udc_fsgan import UDCFSGAN, UDCFSGANConfig
from .srgan import SRGANWrapper
from .aga_gan import AGAGANWrapper
from .wipa import WIPAWrapper


def build_model(name: str, cfg: Dict[str, Any] | None = None):
    cfg = cfg or {}
    name = name.lower()
    target_size = cfg.get('target_size', 256)
    base_channels = cfg.get('base_channels', 64)
    if name in {'udc_fsgan', 'udc', 'ours'}:
        return UDCFSGAN(UDCFSGANConfig(target_size=target_size, base_channels=base_channels,
                                       resize_mode=cfg.get('resize_mode', 'nearest')))
    if name == 'srgan':
        return SRGANWrapper(target_size=target_size, base_channels=base_channels)
    if name == 'aga_gan':
        return AGAGANWrapper(target_size=target_size, base_channels=base_channels)
    if name == 'wipa':
        return WIPAWrapper(target_size=target_size, base_channels=base_channels)
    raise ValueError(f'Unknown model name: {name}')
