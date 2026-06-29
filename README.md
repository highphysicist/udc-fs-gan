# UDC-FSGAN

PyTorch reproduction repo for **A Novel Generative Adversarial Network-Based Super-Resolution Approach for Face Recognition**.

This repository implements:

- a **U-Net style generator**,
- a **DCGAN style discriminator**,
- a perceptual + adversarial training objective,
- the paper's evaluation metrics:
  - SSIM
  - PSNR
  - FID
  - pixel-wise distance
  - face detection count
- comparison baselines:
  - nearest neighbor
  - bicubic
  - SR-GAN
  - AGA-GAN
  - WIPA

## Dataset

The paper uses **CelebA-HQ** with 4x super-resolution from **64x64 -> 256x256**. The dataset loader here also supports other scales, but the reference configuration is the 4x setup described in the paper.

## Quick start

```bash
conda env create -f environment.yml
conda activate udc-fsgan
pip install -r requirements.txt
```

Prepare data and train:

```bash
python scripts/prepare_celeba_hq.py --input-dir /path/to/CelebA-HQ --output-dir data/processed
python -m training.train_udc_fsgan --config configs/celeba_hq_4x.yaml
```

Evaluate:

```bash
python -m evaluation.eval_udc_fsgan --config configs/eval.yaml --checkpoint weights/udc_fsgan.pt
```

Compare with other methods:

```bash
python -m evaluation.eval_comparisons --config configs/compare_models.yaml
```

## Notes

- The generator uses a resizing stem and a U-Net backbone.
- The discriminator is a DCGAN-style classifier.
- Loss is VGG perceptual loss + a small adversarial term, matching the paper's setup.
- The face detection score uses OpenCV Haar cascades.
