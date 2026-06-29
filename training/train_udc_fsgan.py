from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from torch.utils.data import DataLoader, random_split

from data.datasets import CelebAHQSRDataset
from losses.composite_loss import LossWeights
from models.model_factory import build_model
from training.trainer import SuperResolutionTrainer
from utils.seed import set_seed



from pathlib import Path
import yaml


def load_config(path: str):
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if isinstance(data, dict) and 'include' in data:
        base_path = path.parent / data['include']
        base = load_config(str(base_path))
        data = {k: v for k, v in data.items() if k != 'include'}
        def merge(a, b):
            for k, v in b.items():
                if isinstance(v, dict) and isinstance(a.get(k), dict):
                    a[k] = merge(a[k], v)
                else:
                    a[k] = v
            return a
        return merge(base, data)
    return data



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get('seed', 42))
    dcfg = cfg['dataset']
    tcfg = cfg['train']
    model = build_model('udc_fsgan', {'target_size': dcfg['image_size'], 'base_channels': cfg['model']['generator_base_channels'], 'resize_mode': dcfg.get('resize_mode', 'nearest')})
    dataset = CelebAHQSRDataset(dcfg['root'], image_size=dcfg['image_size'], lr_size=dcfg['lr_size'],
                                downsample_mode=dcfg.get('downsample_mode', 'bicubic'), upsample_mode=dcfg.get('resize_mode', 'nearest'), train=True)
    n_train = int(len(dataset) * dcfg.get('train_split', 0.95))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=tcfg['batch_size'], shuffle=True, num_workers=dcfg.get('num_workers', 2))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=dcfg.get('num_workers', 2))
    trainer = SuperResolutionTrainer(model, train_loader, val_loader, device=cfg.get('device', 'cuda'), lr=tcfg['lr'], beta1=tcfg['beta1'], beta2=tcfg['beta2'], weights=LossWeights(tcfg['lambda_pixel'], tcfg['lambda_perceptual'], tcfg['lambda_adv']), output_dir=cfg['output']['dir'])
    trainer.fit(epochs=tcfg['epochs'], checkpoint_dir=cfg['output']['weights_dir'])


if __name__ == '__main__':
    main()
