from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

from data.datasets import CelebAHQSRDataset
from metrics import compute_psnr, compute_ssim, compute_pixel_distance, count_faces, FIDMetric
from models.model_factory import build_model
from training.checkpointing import load_checkpoint
from utils.image_utils import save_image_grid


@torch.no_grad()
def evaluate_udc_fsgan(model, loader, device='cuda'):
    model.eval()
    fid = FIDMetric(device=device)
    rows = []
    for batch in loader:
        lr = batch['lr'].to(device)
        hr = batch['hr'].to(device)
        fake = model(lr)
        fid.update_real(hr)
        fid.update_fake(fake)
        rows.append({
            'psnr': compute_psnr(fake, hr),
            'ssim': compute_ssim(fake[0], hr[0]),
            'pixel_distance': compute_pixel_distance(fake, hr),
            'faces_detected': float(sum(count_faces(img) for img in fake)),
        })
    out = {k: float(sum(r[k] for r in rows) / len(rows)) for k in rows[0].keys()}
    out['fid'] = fid.compute()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    dcfg = cfg['dataset']
    model = build_model('udc_fsgan', {'target_size': dcfg['image_size'], 'base_channels': cfg['model']['generator_base_channels'], 'resize_mode': dcfg.get('resize_mode', 'nearest')})
    load_checkpoint(args.checkpoint, model, map_location=cfg.get('device', 'cuda'))
    dataset = CelebAHQSRDataset(dcfg['root'], image_size=dcfg['image_size'], lr_size=dcfg['lr_size'], downsample_mode=dcfg.get('downsample_mode', 'bicubic'), upsample_mode=dcfg.get('resize_mode', 'nearest'), train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    metrics = evaluate_udc_fsgan(model.to(cfg.get('device', 'cuda')), loader, device=cfg.get('device', 'cuda'))
    Path('outputs').mkdir(exist_ok=True)
    with open('outputs/eval_udc_fsgan.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(metrics)


if __name__ == '__main__':
    main()
