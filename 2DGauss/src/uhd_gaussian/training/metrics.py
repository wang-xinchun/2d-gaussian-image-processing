from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def psnr(prediction: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    mse = F.mse_loss(prediction.clamp(0.0, data_range), target.clamp(0.0, data_range)).item()
    if mse <= 0.0:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def _gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    center = window_size // 2
    coords = torch.arange(window_size, device=device, dtype=dtype) - center
    gaussian_1d = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
    gaussian_1d = gaussian_1d / gaussian_1d.sum().clamp_min(1e-12)
    window = gaussian_1d[:, None] * gaussian_1d[None, :]
    return window


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    prediction = prediction.clamp(0.0, data_range)
    target = target.clamp(0.0, data_range)
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    if prediction.dim() == 3:
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)
    if prediction.dim() != 4:
        raise ValueError(f"expected 3D or 4D tensor, got {prediction.dim()}D")

    _, channels, height, width = prediction.shape
    window_size = min(window_size, height, width)
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        return 1.0

    window = _gaussian_window(window_size, sigma, prediction.device, prediction.dtype)
    window = window.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)
    padding = window_size // 2

    mu_x = F.conv2d(prediction, window, padding=padding, groups=channels)
    mu_y = F.conv2d(target, window, padding=padding, groups=channels)
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(prediction * prediction, window, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(prediction * target, window, padding=padding, groups=channels) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2) + 1e-12
    )
    return float(ssim_map.mean().item())
