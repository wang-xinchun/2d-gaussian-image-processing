from __future__ import annotations

import torch
import torch.nn.functional as F


def charbonnier_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((prediction - target).square() + eps * eps).mean()


def gradient_l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def laplacian_l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    channels = prediction.shape[1]
    kernel = prediction.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    kernel = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    pred_lap = F.conv2d(prediction, kernel, padding=1, groups=channels)
    target_lap = F.conv2d(target, kernel, padding=1, groups=channels)
    return F.l1_loss(pred_lap, target_lap)


def luminance(image: torch.Tensor) -> torch.Tensor:
    if image.shape[1] < 3:
        return image.mean(dim=1, keepdim=True)
    weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (image[:, :3] * weights).sum(dim=1, keepdim=True)


def _local_mean(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = max(int(kernel_size), 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size <= 1:
        return image
    pad = kernel_size // 2
    padded = F.pad(image, (pad, pad, pad, pad), mode="replicate")
    return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)


def _local_std(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    mean = _local_mean(image, kernel_size)
    second = _local_mean(image.square(), kernel_size)
    return torch.sqrt((second - mean.square()).clamp_min(0.0) + 1e-6)


def _sobel_magnitude(image: torch.Tensor) -> torch.Tensor:
    gray = luminance(image)
    sobel_x = image.new_tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
    ).unsqueeze(0)
    sobel_y = image.new_tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
    ).unsqueeze(0)
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    return torch.sqrt((grad_x.square() + grad_y.square()).clamp_min(1e-8))


def edge_magnitude_l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(_sobel_magnitude(prediction), _sobel_magnitude(target))


def high_frequency_l1_loss(prediction: torch.Tensor, target: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    pred_high = prediction - _local_mean(prediction, kernel_size)
    target_high = target - _local_mean(target, kernel_size)
    return F.l1_loss(pred_high, target_high)


def local_contrast_l1_loss(prediction: torch.Tensor, target: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    pred_contrast = _local_std(luminance(prediction), kernel_size)
    target_contrast = _local_std(luminance(target), kernel_size)
    return F.l1_loss(pred_contrast, target_contrast)


def luminance_l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(luminance(prediction), luminance(target))


def chroma_l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape[1] < 3 or target.shape[1] < 3:
        return prediction.new_tensor(0.0)
    pred_y = luminance(prediction)
    target_y = luminance(target)
    pred_chroma = prediction[:, :3] - pred_y
    target_chroma = target[:, :3] - target_y
    return F.l1_loss(pred_chroma, target_chroma)


def saturation_l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape[1] < 3 or target.shape[1] < 3:
        return prediction.new_tensor(0.0)
    pred_sat = prediction[:, :3].max(dim=1, keepdim=True).values - prediction[:, :3].min(dim=1, keepdim=True).values
    target_sat = target[:, :3].max(dim=1, keepdim=True).values - target[:, :3].min(dim=1, keepdim=True).values
    return F.l1_loss(pred_sat, target_sat)


def dark_channel_l1_loss(prediction: torch.Tensor, target: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
    if prediction.shape[1] < 3 or target.shape[1] < 3:
        return prediction.new_tensor(0.0)
    kernel_size = max(int(kernel_size), 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    pred_min = prediction[:, :3].min(dim=1, keepdim=True).values
    target_min = target[:, :3].min(dim=1, keepdim=True).values
    pred_dark = -F.max_pool2d(-F.pad(pred_min, (pad, pad, pad, pad), mode="replicate"), kernel_size, stride=1)
    target_dark = -F.max_pool2d(-F.pad(target_min, (pad, pad, pad, pad), mode="replicate"), kernel_size, stride=1)
    return F.l1_loss(pred_dark, target_dark)
