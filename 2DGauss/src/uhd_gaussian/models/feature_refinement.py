from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualDWBlock


class GaussianGuidedFeatureRefinement(nn.Module):
    def __init__(
        self,
        channels: int,
        directional_channels: int = 0,
        blocks: int = 1,
        strength: float = 0.12,
        use_decoder_detail_hint: bool = False,
        use_edge_aware_prior: bool = False,
        use_confidence_normalization: bool = False,
        use_shared_prior_stem: bool = False,
        use_intervention_demand: bool = False,
        use_decoder_detail_gate: bool = False,
    ) -> None:
        super().__init__()
        self.directional_channels = directional_channels
        self.strength = strength
        self.use_decoder_detail_hint = use_decoder_detail_hint
        self.use_edge_aware_prior = use_edge_aware_prior
        self.use_confidence_normalization = use_confidence_normalization
        self.use_shared_prior_stem = use_shared_prior_stem
        self.use_intervention_demand = use_intervention_demand
        self.use_decoder_detail_gate = use_decoder_detail_gate
        if use_shared_prior_stem and (use_decoder_detail_hint or use_edge_aware_prior):
            raise ValueError("shared prior stem cannot be combined with direct extra cue injection")
        prior_channels = 2 + directional_channels + (2 if use_decoder_detail_hint else 0) + (5 if use_edge_aware_prior else 0)
        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "laplacian",
            torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3),
            persistent=False,
        )

        if not use_shared_prior_stem:
            prior_layers: list[nn.Module] = [
                nn.Conv2d(prior_channels, channels, kernel_size=3, padding=1),
                nn.GELU(),
            ]
            prior_layers.extend(ResidualDWBlock(channels) for _ in range(blocks))
            self.prior_encoder = nn.Sequential(*prior_layers)
        else:
            self.prior_encoder = None

        self.mix = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GELU(),
            ResidualDWBlock(channels),
        )
        self.scale_head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Tanh(),
        )
        self.bias_head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Tanh(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        if use_confidence_normalization:
            confidence_channels = 5 + (1 if directional_channels > 0 else 0) + (2 if use_shared_prior_stem else 0)
            self.confidence_encoder = nn.Sequential(
                nn.Conv2d(confidence_channels, channels, kernel_size=3, padding=1),
                nn.GELU(),
                ResidualDWBlock(channels),
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.confidence_encoder = None
        if use_intervention_demand:
            demand_channels = 5 + (1 if directional_channels > 0 else 0) + (2 if use_shared_prior_stem else 0)
            self.demand_encoder = nn.Sequential(
                nn.Conv2d(demand_channels, channels, kernel_size=3, padding=1),
                nn.GELU(),
                ResidualDWBlock(channels),
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.demand_encoder = None
        if use_decoder_detail_gate:
            self.detail_gate = nn.Sequential(
                nn.Conv2d(channels + 2, channels, kernel_size=1),
                nn.GELU(),
                ResidualDWBlock(channels),
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.detail_gate = None

    def edge_descriptor(self, rgb: torch.Tensor) -> torch.Tensor:
        gray = rgb.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(torch.clamp(grad_x.square() + grad_y.square(), min=1e-8))
        lap = F.conv2d(gray, self.laplacian, padding=1)
        local_highpass = gray - F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
        return torch.cat(
            [
                gray,
                grad_x.abs(),
                grad_y.abs(),
                grad_mag,
                lap.abs() + local_highpass.abs(),
            ],
            dim=1,
        )

    def forward(
        self,
        decoder_feature: torch.Tensor,
        attention_map: torch.Tensor,
        directional_prior: torch.Tensor | None = None,
        highres_rgb: torch.Tensor | None = None,
        shared_prior_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_map.shape[-2:] != decoder_feature.shape[-2:]:
            attention_map = F.interpolate(
                attention_map,
                size=decoder_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        local_contrast = attention_map - F.avg_pool2d(attention_map, kernel_size=3, stride=1, padding=1)
        decoder_energy = decoder_feature.square().mean(dim=1, keepdim=True)
        local_energy = F.avg_pool2d(decoder_energy, kernel_size=3, stride=1, padding=1)
        detail_hint = decoder_energy - local_energy
        edge_descriptor = None
        if highres_rgb is None:
            highres_rgb = decoder_feature.new_zeros(
                decoder_feature.shape[0],
                3,
                decoder_feature.shape[-2],
                decoder_feature.shape[-1],
            )
        elif highres_rgb.shape[-2:] != decoder_feature.shape[-2:]:
            highres_rgb = F.interpolate(
                highres_rgb,
                size=decoder_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if self.use_edge_aware_prior or self.use_confidence_normalization or self.use_intervention_demand:
            edge_descriptor = self.edge_descriptor(highres_rgb)

        if self.use_shared_prior_stem:
            if shared_prior_feature is None:
                raise ValueError("shared_prior_feature is required when use_shared_prior_stem=True")
            prior_feature = shared_prior_feature
            if prior_feature.shape[-2:] != decoder_feature.shape[-2:]:
                prior_feature = F.interpolate(
                    prior_feature,
                    size=decoder_feature.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
        else:
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
            if self.use_decoder_detail_hint:
                prior_inputs.extend([decoder_energy, detail_hint])
            if self.use_edge_aware_prior and edge_descriptor is not None:
                prior_inputs.append(edge_descriptor)
            prior_feature = self.prior_encoder(torch.cat(prior_inputs, dim=1))
        fused = self.mix(torch.cat([decoder_feature, prior_feature], dim=1))
        gate = self.gate(fused)
        scale = self.scale_head(fused)
        bias = self.bias_head(fused)
        if self.detail_gate is not None:
            detail_strength = detail_hint.abs() / (local_energy + 1e-6)
            detail_inputs = torch.cat(
                [
                    prior_feature,
                    decoder_energy / (local_energy + 1e-6),
                    detail_strength,
                ],
                dim=1,
            )
            gate = gate * self.detail_gate(detail_inputs)
        confidence_inputs = None
        if self.use_confidence_normalization or self.use_intervention_demand:
            detail_strength = detail_hint.abs() / (local_energy + 1e-6)
            edge_strength = decoder_feature.new_zeros(
                decoder_feature.shape[0],
                1,
                decoder_feature.shape[-2],
                decoder_feature.shape[-1],
            )
            if edge_descriptor is not None:
                edge_strength = 0.5 * (edge_descriptor[:, 3:4] + edge_descriptor[:, 4:5])
                edge_strength = edge_strength / (
                    F.avg_pool2d(edge_strength, kernel_size=3, stride=1, padding=1) + 1e-6
                )
            confidence_inputs = [
                attention_map,
                local_contrast.abs(),
                decoder_energy / (local_energy + 1e-6),
                detail_strength,
                edge_strength,
            ]
            if self.use_shared_prior_stem:
                shared_prior_energy = prior_feature.square().mean(dim=1, keepdim=True)
                shared_prior_local_energy = F.avg_pool2d(shared_prior_energy, kernel_size=3, stride=1, padding=1)
                confidence_inputs.extend(
                    [
                        shared_prior_energy / (shared_prior_local_energy + 1e-6),
                        (shared_prior_energy - shared_prior_local_energy).abs()
                        / (shared_prior_local_energy + 1e-6),
                    ]
                )
            if self.directional_channels > 0:
                if directional_prior is None:
                    coherence = decoder_feature.new_zeros(
                        decoder_feature.shape[0],
                        1,
                        decoder_feature.shape[-2],
                        decoder_feature.shape[-1],
                    )
                else:
                    coherence = directional_prior[:, 4:5] if directional_prior.shape[1] >= 5 else directional_prior.abs().mean(
                        dim=1,
                        keepdim=True,
                    )
                confidence_inputs.append(coherence)
        if self.use_confidence_normalization:
            confidence = self.confidence_encoder(torch.cat(confidence_inputs, dim=1))
            gate = gate * (1.0 - confidence)
        if self.demand_encoder is not None:
            demand = self.demand_encoder(torch.cat(confidence_inputs, dim=1))
            gate = gate * demand
        return decoder_feature * (1.0 + self.strength * gate * scale) + self.strength * gate * bias
