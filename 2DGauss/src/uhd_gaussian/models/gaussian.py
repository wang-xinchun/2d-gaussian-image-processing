from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import GaussianRendererConfig
from .blocks import ResidualDWBlock


class GaussianParameterHead(nn.Module):
    def __init__(self, in_channels: int, gaussian_cfg: GaussianRendererConfig, affine_rank: int) -> None:
        super().__init__()
        self.gaussian_cfg = gaussian_cfg
        self.affine_rank = affine_rank
        hidden_channels = in_channels
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            ResidualDWBlock(hidden_channels),
        )
        self.param_head = nn.Conv2d(
            hidden_channels,
            gaussian_cfg.atoms * gaussian_cfg.param_dim,
            kernel_size=1,
        )
        self.affine_head = nn.Conv2d(hidden_channels, affine_rank, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, _, height, width = x.shape
        shared = self.shared(x)
        params = self.param_head(shared)
        params = params.view(
            batch,
            self.gaussian_cfg.atoms,
            self.gaussian_cfg.param_dim,
            height,
            width,
        ).permute(0, 3, 4, 1, 2).contiguous()
        coeffs = self.affine_head(shared)
        return {
            "params": params,
            "coeffs": coeffs,
        }


class TextureAwareGaussianRenderer(nn.Module):
    def __init__(self, cfg: GaussianRendererConfig) -> None:
        super().__init__()
        self.cfg = cfg
        patch = cfg.patch_size
        coords = torch.linspace(0.5 / patch, 1.0 - 0.5 / patch, patch)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1)
        self.register_buffer("grid", grid, persistent=False)

    def forward(
        self,
        params: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        patch = self.cfg.patch_size
        grid = self.grid.view(1, 1, 1, 1, patch, patch, 2)

        amplitude = torch.sigmoid(params[..., 0])[..., None, None]
        mu = torch.sigmoid(params[..., 1:3])[..., None, None, :]
        sigma = 0.05 + 0.45 * torch.sigmoid(params[..., 3:5])[..., None, None, :]
        theta = math.pi * torch.tanh(params[..., 5])[..., None, None]
        texture_gain = self.cfg.max_texture_gain * torch.sigmoid(params[..., 6])[..., None, None]
        frequency = self.cfg.max_frequency * torch.sigmoid(params[..., 7])[..., None, None]
        phase = math.pi * torch.tanh(params[..., 8])[..., None, None]
        beta = 0.1 + 2.0 * torch.sigmoid(params[..., 9])[..., None, None]

        delta = grid - mu
        delta_x = delta[..., 0]
        delta_y = delta[..., 1]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        u_x = cos_t * delta_x + sin_t * delta_y
        u_y = -sin_t * delta_x + cos_t * delta_y

        gauss = torch.exp(
            -0.5
            * (
                (u_x / sigma[..., 0]).square()
                + (u_y / sigma[..., 1]).square()
            )
        )
        carrier = 1.0 + texture_gain * torch.cos(2.0 * math.pi * frequency * u_x + phase)
        spectral = torch.exp(-beta * frequency.square())
        patches = (amplitude * gauss * carrier * spectral).sum(dim=3)
        patches = torch.sigmoid(patches)

        attention = patches.permute(0, 1, 3, 2, 4).reshape(
            params.shape[0],
            1,
            params.shape[1] * patch,
            params.shape[2] * patch,
        )
        if target_size is not None and attention.shape[-2:] != target_size:
            attention = F.interpolate(attention, size=target_size, mode="bilinear", align_corners=False)
        return attention
