from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualDWBlock


class GaussianAttentionDemandBridge(nn.Module):
    def __init__(
        self,
        channels: int,
        blocks: int = 1,
        strength: float = 0.20,
        use_input_structure: bool = True,
    ) -> None:
        super().__init__()
        self.strength = strength
        self.use_input_structure = use_input_structure
        in_channels = channels + 2 + (1 if use_input_structure else 0)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            *[ResidualDWBlock(channels) for _ in range(blocks)],
        )
        self.head = nn.Sequential(
            nn.Conv2d(channels, max(channels // 2, 8), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(max(channels // 2, 8), 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        highres_feature: torch.Tensor,
        attention_map: torch.Tensor,
        input_structure: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_size = highres_feature.shape[-2:]
        if attention_map.shape[-2:] != target_size:
            attention_map = F.interpolate(
                attention_map,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        local_contrast = attention_map - F.avg_pool2d(attention_map, kernel_size=3, stride=1, padding=1)
        bridge_inputs = [highres_feature, attention_map, local_contrast]
        if self.use_input_structure:
            if input_structure is None:
                input_structure = attention_map.new_zeros(
                    attention_map.shape[0],
                    1,
                    target_size[0],
                    target_size[1],
                )
            elif input_structure.shape[-2:] != target_size:
                input_structure = F.interpolate(
                    input_structure,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            bridge_inputs.append(input_structure)
        demand_map = self.head(self.encoder(torch.cat(bridge_inputs, dim=1)))
        bridged_attention = attention_map.lerp(demand_map, float(self.strength))
        return bridged_attention, demand_map
