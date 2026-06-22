from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import GaussianRendererConfig


class DirectionalFrequencyProjector(nn.Module):
    out_channels = 6

    def __init__(self, cfg: GaussianRendererConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(self, params: torch.Tensor, target_size: tuple[int, int] | None = None) -> torch.Tensor:
        amplitude = torch.sigmoid(params[..., 0])
        theta = math.pi * torch.tanh(params[..., 5])
        texture_gain = self.cfg.max_texture_gain * torch.sigmoid(params[..., 6])
        frequency = self.cfg.max_frequency * torch.sigmoid(params[..., 7])
        beta = 0.1 + 2.0 * torch.sigmoid(params[..., 9])

        weight = amplitude * (1.0 + texture_gain)
        denom = weight.sum(dim=3).clamp_min(1e-6)

        dir_x = (weight * torch.cos(theta)).sum(dim=3) / denom
        dir_y = (weight * torch.sin(theta)).sum(dim=3) / denom
        freq_mean = (weight * frequency).sum(dim=3) / denom
        texture_mean = (weight * texture_gain).sum(dim=3) / denom
        beta_mean = (weight * beta).sum(dim=3) / denom
        coherence = torch.sqrt(dir_x.square() + dir_y.square()).clamp(0.0, 1.0)

        freq_norm = freq_mean / max(self.cfg.max_frequency, 1e-6)
        texture_norm = texture_mean / max(self.cfg.max_texture_gain, 1e-6) if self.cfg.max_texture_gain > 0.0 else texture_mean
        beta_norm = ((beta_mean - 0.1) / 2.0).clamp(0.0, 1.0)

        prior = torch.stack(
            [
                dir_x,
                dir_y,
                freq_norm.clamp(0.0, 1.0),
                texture_norm.clamp(0.0, 1.0),
                coherence,
                beta_norm,
            ],
            dim=1,
        )
        if target_size is not None and prior.shape[-2:] != target_size:
            prior = F.interpolate(prior, size=target_size, mode="bilinear", align_corners=False)
        return prior
