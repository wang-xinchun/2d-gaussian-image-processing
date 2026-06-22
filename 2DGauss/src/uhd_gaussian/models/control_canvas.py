from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualDWBlock


class GaussianControlCanvasModule(nn.Module):
    def __init__(
        self,
        channels: int,
        directional_channels: int = 0,
        blocks: int = 1,
        feedback_strength: float = 0.20,
        use_target_conditioned_feedback: bool = False,
        feedback_mode: str = "multiplicative",
        use_confidence_normalization: bool = False,
        residual_delta_strength: float = 1.0,
        feedback_selectivity_mode: str = "none",
        feedback_selectivity_temperature: float = 2.0,
        feedback_demand_power: float = 1.0,
        feedback_budget: float = 0.0,
        feedback_floor: float = 0.0,
        feedback_target_mean: float = 0.0,
        feedback_mean_boost_cap: float = 2.0,
        feedback_demand_source: str = "target",
    ) -> None:
        super().__init__()
        if feedback_mode not in {"multiplicative", "additive"}:
            raise ValueError("feedback_mode must be multiplicative or additive")
        if feedback_selectivity_mode not in {"none", "demand_budget"}:
            raise ValueError("feedback_selectivity_mode must be none or demand_budget")
        if feedback_demand_source not in {"target", "canvas", "mixed"}:
            raise ValueError("feedback_demand_source must be target, canvas, or mixed")
        self.directional_channels = directional_channels
        self.feedback_strength = feedback_strength
        self.use_target_conditioned_feedback = use_target_conditioned_feedback
        self.feedback_mode = feedback_mode
        self.use_confidence_normalization = use_confidence_normalization
        self.residual_delta_strength = residual_delta_strength
        self.feedback_selectivity_mode = feedback_selectivity_mode
        self.feedback_selectivity_temperature = feedback_selectivity_temperature
        self.feedback_demand_power = feedback_demand_power
        self.feedback_budget = feedback_budget
        self.feedback_floor = feedback_floor
        self.feedback_target_mean = feedback_target_mean
        self.feedback_mean_boost_cap = feedback_mean_boost_cap
        self.feedback_demand_source = feedback_demand_source
        in_channels = channels + 2 + directional_channels
        self.canvas_encoder = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            *[ResidualDWBlock(channels) for _ in range(blocks)],
        )
        self.canvas_head = nn.Sequential(
            nn.Conv2d(channels, max(channels // 2, 8), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(max(channels // 2, 8), 1, kernel_size=1),
            nn.Sigmoid(),
        )
        feedback_channels = channels + 2
        if use_target_conditioned_feedback:
            feedback_channels += 2
        self.feedback_gate = nn.Sequential(
            nn.Conv2d(feedback_channels, channels, kernel_size=1),
            nn.GELU(),
            ResidualDWBlock(channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.feedback_scale = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Tanh(),
        )
        self.feedback_delta: nn.Sequential | None = None
        if feedback_mode == "additive":
            self.feedback_delta = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.Tanh(),
            )

    def forward(
        self,
        decoder_feature: torch.Tensor,
        attention_map: torch.Tensor,
        directional_prior: torch.Tensor | None = None,
        structure_target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_size = decoder_feature.shape[-2:]
        if attention_map.shape[-2:] != target_size:
            attention_map = F.interpolate(
                attention_map,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        local_contrast = attention_map - F.avg_pool2d(attention_map, kernel_size=3, stride=1, padding=1)
        canvas_inputs = [decoder_feature, attention_map, local_contrast]
        if self.directional_channels > 0:
            if directional_prior is None:
                directional_prior = decoder_feature.new_zeros(
                    decoder_feature.shape[0],
                    self.directional_channels,
                    target_size[0],
                    target_size[1],
                )
            elif directional_prior.shape[-2:] != target_size:
                directional_prior = F.interpolate(
                    directional_prior,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            canvas_inputs.append(directional_prior)
        canvas_feature = self.canvas_encoder(torch.cat(canvas_inputs, dim=1))
        canvas = self.canvas_head(canvas_feature)

        feedback_inputs = [canvas_feature, canvas, attention_map]
        if self.use_target_conditioned_feedback:
            if structure_target is None:
                structure_target = canvas.detach()
            elif structure_target.shape[-2:] != target_size:
                structure_target = F.interpolate(
                    structure_target,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            feedback_inputs.extend([structure_target, (canvas - structure_target).abs()])
        gate = self.feedback_gate(torch.cat(feedback_inputs, dim=1))
        if self.use_confidence_normalization:
            normalizer = gate.detach().mean(dim=(-2, -1), keepdim=True).clamp_min(0.05)
            gate = torch.clamp(gate / normalizer, 0.0, 1.0)
        if self.feedback_selectivity_mode == "demand_budget":
            demand_reference = self._select_demand_reference(canvas, structure_target)
            gate = self._apply_demand_budget(gate, demand_reference)
        if self.feedback_target_mean > 0.0:
            gate = self._apply_feedback_target_mean(gate)
        scale = self.feedback_scale(canvas_feature)
        feedback_map = gate.mean(dim=1, keepdim=True)
        if self.feedback_mode == "additive":
            if self.feedback_delta is None:
                raise RuntimeError("additive feedback requires feedback_delta")
            delta = self.feedback_delta(canvas_feature)
            refined = decoder_feature + self.feedback_strength * self.residual_delta_strength * gate * delta
        else:
            refined = decoder_feature * (1.0 + self.feedback_strength * gate * scale)
        return refined, canvas, feedback_map

    def _select_demand_reference(
        self,
        canvas: torch.Tensor,
        structure_target: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.feedback_demand_source == "canvas" or structure_target is None:
            return canvas
        if self.feedback_demand_source == "mixed":
            return 0.5 * canvas + 0.5 * structure_target
        return structure_target

    def _apply_demand_budget(self, gate: torch.Tensor, demand_reference: torch.Tensor) -> torch.Tensor:
        if demand_reference.shape[-2:] != gate.shape[-2:]:
            demand_reference = F.interpolate(
                demand_reference,
                size=gate.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if demand_reference.shape[1] != 1:
            demand_reference = demand_reference.mean(dim=1, keepdim=True)

        temperature = max(float(self.feedback_selectivity_temperature), 1e-6)
        gate_mean = gate.mean(dim=(-2, -1), keepdim=True)
        gate_std = gate.std(dim=(-2, -1), keepdim=True, unbiased=False).detach().clamp_min(0.03)
        gate_selective = torch.sigmoid(temperature * (gate - gate_mean) / gate_std)

        demand = demand_reference.detach().clamp(0.0, 1.0)
        demand_mean = demand.mean(dim=(-2, -1), keepdim=True)
        demand_std = demand.std(dim=(-2, -1), keepdim=True, unbiased=False).detach().clamp_min(0.03)
        demand_selective = torch.sigmoid(temperature * (demand - demand_mean) / demand_std)
        demand_selective = demand_selective.clamp_min(1e-6).pow(float(self.feedback_demand_power))

        floor = float(self.feedback_floor)
        selective_gate = gate_selective * (floor + (1.0 - floor) * demand_selective)
        if self.feedback_budget > 0.0:
            current_mean = selective_gate.mean(dim=(-2, -1), keepdim=True).detach().clamp_min(1e-6)
            budget_scale = torch.clamp(float(self.feedback_budget) / current_mean, max=1.0)
            selective_gate = selective_gate * budget_scale
        return selective_gate.clamp(0.0, 1.0)

    def _apply_feedback_target_mean(self, gate: torch.Tensor) -> torch.Tensor:
        target_mean = float(self.feedback_target_mean)
        if target_mean <= 0.0:
            return gate
        current_mean = gate.mean(dim=(-2, -1), keepdim=True)
        current_mean_scalar = current_mean.detach().mean().clamp_min(1e-6)
        if float(current_mean_scalar) >= target_mean:
            return gate
        boost_cap = max(float(self.feedback_mean_boost_cap), 1.0)
        boost_value = min(max(target_mean / float(current_mean_scalar), 1.0), boost_cap)
        boost_scale = gate.new_tensor(boost_value)
        return (gate * boost_scale).clamp(0.0, 1.0)
