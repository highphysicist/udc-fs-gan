import torch
from metrics.psnr import compute_psnr
from metrics.pixel_distance import compute_pixel_distance


def test_metrics_simple():
    a = torch.zeros(1, 3, 16, 16)
    b = torch.zeros(1, 3, 16, 16)
    assert compute_psnr(a, b) == float('inf')
    assert compute_pixel_distance(a, b) == 0.0
