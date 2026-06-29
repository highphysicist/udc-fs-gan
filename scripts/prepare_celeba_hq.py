from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from data.preprocess import build_sr_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--lr-size', type=int, default=64)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    hr_dir = output_dir / 'hr'
    lr_dir = output_dir / 'lr'
    lr_up_dir = output_dir / 'lr_up'
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)
    lr_up_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in input_dir.rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}]
    for path in tqdm(images):
        image = Image.open(path).convert('RGB')
        lr, lr_up, hr = build_sr_pair(image, args.lr_size, args.image_size)
        stem = path.stem
        hr.save(hr_dir / f'{stem}.png')
        lr.save(lr_dir / f'{stem}.png')
        lr_up.save(lr_up_dir / f'{stem}.png')


if __name__ == '__main__':
    main()
