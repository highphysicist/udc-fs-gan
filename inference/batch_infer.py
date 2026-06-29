from __future__ import annotations

from pathlib import Path

from .upscale_image import upscale_image


def batch_infer(model, input_dir: str, output_dir: str, device='cuda'):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in input_dir.rglob('*'):
        if path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.bmp'}:
            continue
        out = output_dir / path.name
        upscale_image(model, str(path), str(out), device=device)
