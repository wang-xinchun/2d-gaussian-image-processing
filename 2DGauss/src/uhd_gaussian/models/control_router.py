from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualDWBlock


class GaussianStructuredControlRouter(nn.Module):
    def __init__(
        self,
        channels: int,
        directional_channels: int = 0,
        blocks: int = 1,
        use_decoder_state: bool = True,
        use_edge_state: bool = True,
        role_delta_strength: float = 0.50,
        use_competitive_role_gate: bool = False,
        role_gate_temperature: float = 1.0,
        use_role_residual_centering: bool = False,
        use_demand_conditioned_competition: bool = False,
        role_competition_start: float = 0.45,
        role_competition_strength: float = 0.35,
        role_competition_floor: float = 0.30,
    ) -> None:
        super().__init__()
        self.directional_channels = directional_channels
        self.use_decoder_state = use_decoder_state
        self.use_edge_state = use_edge_state
        self.role_delta_strength = role_delta_strength
        self.use_competitive_role_gate = use_competitive_role_gate
        self.role_gate_temperature = role_gate_temperature
        self.use_role_residual_centering = use_role_residual_centering
        self.use_demand_conditioned_competition = use_demand_conditioned_competition
        self.role_competition_start = role_competition_start
        self.role_competition_strength = role_competition_strength
        self.role_competition_floor = role_competition_floor
        self.register_buffer("training_progress", torch.tensor(1.0), persistent=False)
        input_channels = channels + 2
        if directional_channels > 0:
            input_channels += 1
        if use_decoder_state:
            input_channels += 2
        if use_edge_state:
            input_channels += 5
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
        layers: list[nn.Module] = [
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        layers.extend(ResidualDWBlock(channels) for _ in range(blocks))
        self.context = nn.Sequential(*layers)
        self.role_delta = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.role_gate = nn.Conv2d(channels, 3, kernel_size=1)
        if use_demand_conditioned_competition:
            demand_channels = max(channels // 4, 8)
            self.role_demand = nn.Sequential(
                nn.Conv2d(channels, demand_channels, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(demand_channels, demand_channels, kernel_size=3, padding=1, groups=demand_channels),
                nn.GELU(),
                nn.Conv2d(demand_channels, 1, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.role_demand = None

    def set_training_progress(self, progress: float) -> None:
        progress = max(0.0, min(float(progress), 1.0))
        self.training_progress.fill_(progress)

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
        shared_prior_feature: torch.Tensor,
        decoder_feature: torch.Tensor,
        attention_map: torch.Tensor,
        directional_prior: torch.Tensor | None = None,
        highres_rgb: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        target_size = decoder_feature.shape[-2:]
        if shared_prior_feature.shape[-2:] != target_size:
            shared_prior_feature = F.interpolate(
                shared_prior_feature,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        if attention_map.shape[-2:] != target_size:
            attention_map = F.interpolate(
                attention_map,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        local_contrast = attention_map - F.avg_pool2d(attention_map, kernel_size=3, stride=1, padding=1)
        router_inputs = [shared_prior_feature, attention_map, local_contrast]
        if self.directional_channels > 0:
            if directional_prior is None:
                coherence = decoder_feature.new_zeros(
                    decoder_feature.shape[0],
                    1,
                    target_size[0],
                    target_size[1],
                )
            else:
                if directional_prior.shape[-2:] != target_size:
                    directional_prior = F.interpolate(
                        directional_prior,
                        size=target_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                coherence = (
                    directional_prior[:, 4:5]
                    if directional_prior.shape[1] >= 5
                    else directional_prior.abs().mean(dim=1, keepdim=True)
                )
            router_inputs.append(coherence)
        if self.use_decoder_state:
            decoder_energy = decoder_feature.square().mean(dim=1, keepdim=True)
            local_energy = F.avg_pool2d(decoder_energy, kernel_size=3, stride=1, padding=1)
            decoder_detail = (decoder_energy - local_energy).abs() / (local_energy + 1e-6)
            router_inputs.extend(
                [
                    decoder_energy / (local_energy + 1e-6),
                    decoder_detail,
                ]
            )
        if self.use_edge_state:
            if highres_rgb is None:
                highres_rgb = decoder_feature.new_zeros(
                    decoder_feature.shape[0],
                    3,
                    target_size[0],
                    target_size[1],
                )
            elif highres_rgb.shape[-2:] != target_size:
                highres_rgb = F.interpolate(
                    highres_rgb,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            router_inputs.append(self.edge_descriptor(highres_rgb))
        context = self.context(torch.cat(router_inputs, dim=1))
        role_deltas = self.role_delta(context).view(
            context.shape[0],
            3,
            context.shape[1],
            context.shape[2],
            context.shape[3],
        )
        if self.use_role_residual_centering:
            role_deltas = role_deltas - role_deltas.mean(dim=1, keepdim=True)
        delta_decoder = role_deltas[:, 0]
        delta_feature = role_deltas[:, 1]
        delta_edge = role_deltas[:, 2]
        role_gate_logits = self.role_gate(context)
        if self.use_competitive_role_gate:
            role_gate = torch.softmax(role_gate_logits / self.role_gate_temperature, dim=1)
        else:
            role_gate = torch.sigmoid(role_gate_logits)
        if self.role_demand is not None:
            progress = self.training_progress.to(dtype=context.dtype, device=context.device)
            ramp = torch.clamp((progress - self.role_competition_start) / max(1.0 - self.role_competition_start, 1e-6), 0.0, 1.0)
            demand = self.role_demand(context) * ramp * self.role_competition_strength
            competitive_gate = torch.softmax(role_gate_logits / self.role_gate_temperature, dim=1)
            competitive_gate = self.role_competition_floor + (1.0 - self.role_competition_floor) * competitive_gate
            role_gate = role_gate * (1.0 - demand) + role_gate * competitive_gate * demand
        strength = self.role_delta_strength
        return {
            "decoder": shared_prior_feature + strength * role_gate[:, 0:1] * torch.tanh(delta_decoder),
            "feature": shared_prior_feature + strength * role_gate[:, 1:2] * torch.tanh(delta_feature),
            "edge": shared_prior_feature + strength * role_gate[:, 2:3] * torch.tanh(delta_edge),
        }
