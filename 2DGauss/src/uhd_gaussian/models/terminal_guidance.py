from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualDWBlock


class GaussianGuidedTerminalResidualHead(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        directional_channels: int = 0,
        blocks: int = 1,
        strength: float = 0.10,
    ) -> None:
        super().__init__()
        self.directional_channels = directional_channels
        self.strength = strength
        prior_channels = 2 + directional_channels
        encoder_layers: list[nn.Module] = [
            nn.Conv2d(prior_channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        encoder_layers.extend(ResidualDWBlock(channels) for _ in range(blocks))
        self.prior_encoder = nn.Sequential(*encoder_layers)
        self.feature_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        decoder_feature: torch.Tensor,
        attention_map: torch.Tensor,
        directional_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_map.shape[-2:] != decoder_feature.shape[-2:]:
            attention_map = F.interpolate(
                attention_map,
                size=decoder_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        local_contrast = attention_map - F.avg_pool2d(attention_map, kernel_size=3, stride=1, padding=1)
        prior_inputs = [attention_map, local_contrast]
        if self.directional_channels > 0:
            if directional_prior is None:
                directional_prior = decoder_feature.new_zeros(
                    decoder_feature.shape[0],
                    self.directional_channels,
                    decoder_feature.shape[-2],
                    decoder_feature.shape[-1],
                )
            elif directional_prior.shape[-2:] != decoder_feature.shape[-2:]:
                directional_prior = F.interpolate(
                    directional_prior,
                    size=decoder_feature.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            prior_inputs.append(directional_prior)
        prior_feature = self.prior_encoder(torch.cat(prior_inputs, dim=1))
        gate = self.feature_gate(torch.cat([decoder_feature, prior_feature], dim=1))
        guided_feature = decoder_feature * gate
        return self.strength * self.out(guided_feature)
