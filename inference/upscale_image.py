from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from utils.image_utils import tensor_to_pil, save_image_tensor


@torch.no_grad()
def upscale_image(model, image_path: str, out_path: str, device='cuda'):
    image = Image.open(image_path).convert('RGB')
    x = transforms.ToTensor()(image).unsqueeze(0).to(device)
    y = model(x)[0].detach().cpu()
    save_image_tensor(out_path, y)
    return out_path
