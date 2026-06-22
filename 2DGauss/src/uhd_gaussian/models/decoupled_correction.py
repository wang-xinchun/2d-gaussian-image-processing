from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualDWBlock


class DecoupledCorrectionModule(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        blocks: int = 1,
        strength: float = 0.10,
        use_gradient_detached_selectivity: bool = True,
        amplitude_channels: int = 0,
        amplitude_mode: str = "residual",
    ) -> None:
        super().__init__()
        if amplitude_mode not in {"residual", "detail"}:
            raise ValueError("amplitude_mode must be residual or detail")
        self.strength = strength
        self.use_gradient_detached_selectivity = use_gradient_detached_selectivity
        self.amplitude_mode = amplitude_mode
        hidden_channels = amplitude_channels if amplitude_channels > 0 else channels

        self.selectivity_encoder = nn.Sequential(
            nn.Conv2d(2, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            *[ResidualDWBlock(hidden_channels) for _ in range(blocks)],
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.amplitude_encoder = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            *[ResidualDWBlock(hidden_channels) for _ in range(blocks)],
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            nn.Tanh(),
        )

    def forward(
        self,
        decoder_feature: torch.Tensor,
        control_attention_map: torch.Tensor,
        control_canvas: torch.Tensor | None = None,
        detail_reference: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_size = decoder_feature.shape[-2:]
        if control_attention_map.shape[-2:] != target_size:
            control_attention_map = F.interpolate(
                control_attention_map,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        if control_canvas is None:
            control_canvas = control_attention_map
        elif control_canvas.shape[-2:] != target_size:
            control_canvas = F.interpolate(
                control_canvas,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        selectivity_input = torch.cat([control_attention_map, control_canvas], dim=1)
        selectivity_map = self.selectivity_encoder(selectivity_input)
        if self.use_gradient_detached_selectivity:
            selectivity_gate = selectivity_map.detach()
        else:
            selectivity_gate = selectivity_map

        amplitude_source = decoder_feature
        if self.amplitude_mode == "detail" and detail_reference is not None:
            if detail_reference.shape[-2:] != target_size:
                detail_reference = F.interpolate(
                    detail_reference,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            if detail_reference.shape[1] == decoder_feature.shape[1]:
                amplitude_source = decoder_feature + detail_reference
        amplitude_map = self.amplitude_encoder(amplitude_source)
        correction = self.strength * selectivity_gate * amplitude_map
        return correction, selectivity_map, amplitude_map
