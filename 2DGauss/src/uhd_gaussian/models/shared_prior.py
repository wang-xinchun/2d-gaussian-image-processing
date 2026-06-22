from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualDWBlock


class GaussianSharedPriorEncoder(nn.Module):
    def __init__(
        self,
        channels: int,
        directional_channels: int = 0,
        blocks: int = 1,
    ) -> None:
        super().__init__()
        self.directional_channels = directional_channels
        prior_channels = 2 + directional_channels
        encoder_layers: list[nn.Module] = [
            nn.Conv2d(prior_channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        encoder_layers.extend(ResidualDWBlock(channels) for _ in range(blocks))
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(
        self,
        attention_map: torch.Tensor,
        directional_prior: torch.Tensor | None = None,
        target_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if target_size is not None and attention_map.shape[-2:] != target_size:
            attention_map = F.interpolate(
                attention_map,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        local_contrast = attention_map - F.avg_pool2d(attention_map, kernel_size=3, stride=1, padding=1)
        prior_inputs = [attention_map, local_contrast]
        if self.directional_channels > 0:
            if directional_prior is None:
                directional_prior = attention_map.new_zeros(
                    attention_map.shape[0],
                    self.directional_channels,
                    attention_map.shape[-2],
                    attention_map.shape[-1],
                )
            elif target_size is not None and directional_prior.shape[-2:] != target_size:
                directional_prior = F.interpolate(
                    directional_prior,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            prior_inputs.append(directional_prior)
        return self.encoder(torch.cat(prior_inputs, dim=1))
