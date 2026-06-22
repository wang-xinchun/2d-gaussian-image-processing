from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualDWBlock


class GaussianGuidedDecoderModulator(nn.Module):
    def __init__(
        self,
        channels: int,
        directional_channels: int = 0,
        blocks: int = 1,
        strength: float = 0.15,
        use_shared_prior_stem: bool = False,
        use_decoder_state_gate: bool = False,
    ) -> None:
        super().__init__()
        self.directional_channels = directional_channels
        self.strength = strength
        self.use_shared_prior_stem = use_shared_prior_stem
        self.use_decoder_state_gate = use_decoder_state_gate
        if not use_shared_prior_stem:
            prior_channels = 2 + directional_channels
            encoder_layers: list[nn.Module] = [
                nn.Conv2d(prior_channels, channels, kernel_size=3, padding=1),
                nn.GELU(),
            ]
            encoder_layers.extend(ResidualDWBlock(channels) for _ in range(blocks))
            self.prior_encoder = nn.Sequential(*encoder_layers)
        else:
            self.prior_encoder = None
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.scale_head = nn.Conv2d(channels, channels, kernel_size=1)
        self.bias_head = nn.Conv2d(channels, channels, kernel_size=1)
        if use_decoder_state_gate:
            self.state_gate = nn.Sequential(
                nn.Conv2d(channels + 4, channels, kernel_size=1),
                nn.GELU(),
                ResidualDWBlock(channels),
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.state_gate = None

    def forward(
        self,
        features: torch.Tensor,
        attention_map: torch.Tensor,
        directional_prior: torch.Tensor | None = None,
        shared_prior_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_shared_prior_stem:
            if shared_prior_feature is None:
                raise ValueError("shared_prior_feature is required when use_shared_prior_stem=True")
            prior_feature = shared_prior_feature
            if prior_feature.shape[-2:] != features.shape[-2:]:
                prior_feature = F.interpolate(
                    prior_feature,
                    size=features.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
        else:
            if attention_map.shape[-2:] != features.shape[-2:]:
                attention_map = F.interpolate(
                    attention_map,
                    size=features.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            local_contrast = attention_map - F.avg_pool2d(attention_map, kernel_size=3, stride=1, padding=1)
            prior_inputs = [attention_map, local_contrast]
            if self.directional_channels > 0:
                if directional_prior is None:
                    directional_prior = features.new_zeros(
                        features.shape[0],
                        self.directional_channels,
                        features.shape[-2],
                        features.shape[-1],
                    )
                elif directional_prior.shape[-2:] != features.shape[-2:]:
                    directional_prior = F.interpolate(
                        directional_prior,
                        size=features.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                prior_inputs.append(directional_prior)
            prior_feature = self.prior_encoder(torch.cat(prior_inputs, dim=1))
        gate = self.gate(prior_feature)
        scale = torch.tanh(self.scale_head(prior_feature))
        bias = torch.tanh(self.bias_head(prior_feature))
        if self.state_gate is not None:
            decoder_energy = features.square().mean(dim=1, keepdim=True)
            local_energy = F.avg_pool2d(decoder_energy, kernel_size=3, stride=1, padding=1)
            decoder_detail = (decoder_energy - local_energy).abs() / (local_energy + 1e-6)
            prior_energy = prior_feature.square().mean(dim=1, keepdim=True)
            local_prior_energy = F.avg_pool2d(prior_energy, kernel_size=3, stride=1, padding=1)
            prior_detail = (prior_energy - local_prior_energy).abs() / (local_prior_energy + 1e-6)
            gate_inputs = torch.cat(
                [
                    features,
                    decoder_energy / (local_energy + 1e-6),
                    decoder_detail,
                    prior_energy / (local_prior_energy + 1e-6),
                    prior_detail,
                ],
                dim=1,
            )
            gate = gate * self.state_gate(gate_inputs)
        return features * (1.0 + self.strength * gate * scale) + self.strength * gate * bias
