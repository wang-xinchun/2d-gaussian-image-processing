from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualDWBlock


class HighResolutionEdgeGuidedRefiner(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        blocks: int = 2,
        strength: float = 0.25,
        directional_channels: int = 0,
        use_control_gate: bool = False,
    ) -> None:
        super().__init__()
        self.strength = strength
        self.use_control_gate = use_control_gate
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
        self.edge_proj = nn.Sequential(
            nn.Conv2d(5, channels, kernel_size=3, padding=1),
            nn.GELU(),
            ResidualDWBlock(channels),
        )
        self.directional_proj: nn.Module | None = None
        fuse_in_channels = channels * 2 + 1
        if directional_channels > 0:
            self.directional_proj = nn.Sequential(
                nn.Conv2d(directional_channels, channels, kernel_size=3, padding=1),
                nn.GELU(),
                ResidualDWBlock(channels),
            )
            fuse_in_channels += channels
        refinement_blocks = [ResidualDWBlock(channels) for _ in range(blocks)]
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in_channels, channels, kernel_size=1),
            *refinement_blocks,
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        if use_control_gate:
            self.control_gate = nn.Sequential(
                nn.Conv2d(channels + 3, channels, kernel_size=1),
                nn.GELU(),
                ResidualDWBlock(channels),
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.control_gate = None
        self.out = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)

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
        rgb: torch.Tensor,
        decoder_feature: torch.Tensor,
        attention_map: torch.Tensor,
        directional_prior: torch.Tensor | None = None,
        control_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        edge_feature = self.edge_proj(self.edge_descriptor(rgb))
        directional_feature = None
        gate_source = edge_feature
        if directional_prior is not None and self.directional_proj is not None:
            if directional_prior.shape[-2:] != decoder_feature.shape[-2:]:
                directional_prior = F.interpolate(
                    directional_prior,
                    size=decoder_feature.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            directional_feature = self.directional_proj(directional_prior)
            gate_source = edge_feature + directional_feature
        if attention_map.shape[-2:] != decoder_feature.shape[-2:]:
            attention_map = F.interpolate(attention_map, size=decoder_feature.shape[-2:], mode="bilinear", align_corners=False)
        if self.control_gate is not None:
            if control_feature is None:
                control_feature = decoder_feature.new_zeros(decoder_feature.shape)
            elif control_feature.shape[-2:] != decoder_feature.shape[-2:]:
                control_feature = F.interpolate(
                    control_feature,
                    size=decoder_feature.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            decoder_energy = decoder_feature.square().mean(dim=1, keepdim=True)
            local_energy = F.avg_pool2d(decoder_energy, kernel_size=3, stride=1, padding=1)
            decoder_detail = (decoder_energy - local_energy).abs() / (local_energy + 1e-6)
            control_inputs = torch.cat(
                [
                    control_feature,
                    attention_map,
                    decoder_energy / (local_energy + 1e-6),
                    decoder_detail,
                ],
                dim=1,
            )
            gate_source = gate_source * self.control_gate(control_inputs)
        fuse_inputs = [decoder_feature, edge_feature]
        if directional_feature is not None:
            fuse_inputs.append(directional_feature)
        fuse_inputs.append(attention_map)
        fused = self.fuse(torch.cat(fuse_inputs, dim=1))
        refined = self.out(fused * self.gate(gate_source))
        return self.strength * refined
