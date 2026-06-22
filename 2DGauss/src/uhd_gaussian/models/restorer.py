from __future__ import annotations

from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..utils import crop_to_size, pad_to_multiple
from .affine import LowRankAffineModulator
from .attention_bridge import GaussianAttentionDemandBridge
from .blocks import ResidualDWBlock, StackedResidualBlocks
from .control_router import GaussianStructuredControlRouter
from .control_canvas import GaussianControlCanvasModule
from .decoupled_correction import DecoupledCorrectionModule
from .decoder_guidance import GaussianGuidedDecoderModulator
from .directional import DirectionalFrequencyProjector
from .edge import HighResolutionEdgeGuidedRefiner
from .feature_refinement import GaussianGuidedFeatureRefinement
from .frequency import ExplicitFrequencySplitter, TextureFrequencyExtractor
from .gaussian import GaussianParameterHead, TextureAwareGaussianRenderer
from .shared_prior import GaussianSharedPriorEncoder
from .terminal_guidance import GaussianGuidedTerminalResidualHead


def _zero_init_last_conv2d(module: nn.Module) -> None:
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Conv2d):
            nn.init.zeros_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
            return


class InputStdRouteGlobalGateHead(nn.Module):
    def __init__(self, feature_channels: int) -> None:
        super().__init__()
        hidden_channels = max(feature_channels // 2, 1)
        self.local = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
        )
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
        )
        self.out = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.out(self.local(feature) + self.global_context(feature))


def _make_input_std_route_gate_head(feature_channels: int, learned_gate_mode: str) -> nn.Module:
    if learned_gate_mode == "bounded_residual_global":
        return InputStdRouteGlobalGateHead(feature_channels)
    hidden_channels = max(feature_channels // 2, 1)
    head = nn.Sequential(
        nn.Conv2d(feature_channels, hidden_channels, kernel_size=1),
        nn.GELU(),
        nn.Conv2d(hidden_channels, 1, kernel_size=1),
    )
    if learned_gate_mode in {"residual", "bounded_residual"}:
        _zero_init_last_conv2d(head)
    return head


def _blend_input_std_route_gate(
    prior_map: torch.Tensor,
    shared_feature: torch.Tensor,
    gate_head: nn.Module | None,
    *,
    learned_gate_weight: float,
    learned_gate_mode: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if gate_head is None or learned_gate_weight <= 0.0:
        return prior_map, None
    raw_gate = gate_head(shared_feature)
    if learned_gate_mode == "blend":
        learned_gate = torch.sigmoid(raw_gate)
        gate = (1.0 - learned_gate_weight) * prior_map + learned_gate_weight * learned_gate
    elif learned_gate_mode == "residual":
        prior_logit = torch.logit(prior_map.clamp(1e-5, 1.0 - 1e-5))
        gate = torch.sigmoid(prior_logit + learned_gate_weight * raw_gate)
        learned_gate = gate
    elif learned_gate_mode in {"bounded_residual", "bounded_residual_global"}:
        prior = prior_map.clamp(1e-5, 1.0 - 1e-5)
        residual = learned_gate_weight * torch.tanh(raw_gate)
        gate = torch.where(
            residual >= 0.0,
            prior + (1.0 - prior) * residual,
            prior + prior * residual,
        )
        learned_gate = gate
    else:
        raise ValueError(
            "input_std route learned_gate_mode must be blend, residual, bounded_residual, "
            "or bounded_residual_global"
        )
    return gate, learned_gate


class InputStdRouteResidualHead(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        image_channels: int,
        out_channels: int,
        *,
        blocks: int,
        strength: float,
        threshold: float,
        softness: float,
        direction: str,
        learned_gate_weight: float,
        learned_gate_mode: str,
        detach_gate: bool,
    ) -> None:
        super().__init__()
        self.strength = float(strength)
        self.threshold = float(threshold)
        self.softness = float(softness)
        self.direction = str(direction)
        self.learned_gate_weight = float(learned_gate_weight)
        self.learned_gate_mode = str(learned_gate_mode)
        self.detach_gate = bool(detach_gate)

        route_channels = feature_channels + image_channels + 1
        self.shared = nn.Sequential(
            nn.Conv2d(route_channels, feature_channels, kernel_size=1),
            nn.GELU(),
            StackedResidualBlocks(feature_channels, blocks),
        )
        self.high_blocks = StackedResidualBlocks(feature_channels, blocks)
        self.low_blocks = StackedResidualBlocks(feature_channels, blocks)
        self.high_out = nn.Conv2d(feature_channels, out_channels, kernel_size=3, padding=1)
        self.low_out = nn.Conv2d(feature_channels, out_channels, kernel_size=3, padding=1)
        if self.learned_gate_weight > 0.0:
            self.gate_head = _make_input_std_route_gate_head(feature_channels, self.learned_gate_mode)
        else:
            self.gate_head = None
        self._zero_init_route_outputs()

    def _zero_init_route_outputs(self) -> None:
        nn.init.zeros_(self.high_out.weight)
        nn.init.zeros_(self.high_out.bias)
        nn.init.zeros_(self.low_out.weight)
        nn.init.zeros_(self.low_out.bias)

    def _sample_prior(self, image: torch.Tensor) -> torch.Tensor:
        input_std = image.detach().flatten(start_dim=1).std(dim=1, unbiased=False).view(-1, 1, 1, 1)
        if self.direction in {"gt", "gte"}:
            prior = torch.sigmoid((input_std - self.threshold) / self.softness)
        elif self.direction in {"lt", "lte"}:
            prior = torch.sigmoid((self.threshold - input_std) / self.softness)
        else:
            raise ValueError("input_std_route_residual.direction must be gt, gte, lt, or lte")
        return prior

    def forward(self, decoder_feature: torch.Tensor, image: torch.Tensor) -> Dict[str, torch.Tensor | None]:
        target_size = decoder_feature.shape[-2:]
        route_image = image
        if route_image.shape[-2:] != target_size:
            route_image = F.interpolate(
                route_image,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        prior_sample = self._sample_prior(image)
        prior_map = prior_sample.expand(-1, 1, target_size[0], target_size[1])
        shared = self.shared(torch.cat([decoder_feature, route_image, prior_map], dim=1))
        high_delta = self.strength * self.high_out(self.high_blocks(shared))
        low_delta = self.strength * self.low_out(self.low_blocks(shared))
        gate, learned_gate = _blend_input_std_route_gate(
            prior_map,
            shared,
            self.gate_head,
            learned_gate_weight=self.learned_gate_weight,
            learned_gate_mode=self.learned_gate_mode,
        )
        blend_gate = gate.detach() if self.detach_gate else gate
        delta = blend_gate * high_delta + (1.0 - blend_gate) * low_delta
        return {
            "delta": delta,
            "high_delta": high_delta,
            "low_delta": low_delta,
            "gate": gate,
            "prior": prior_map,
            "learned_gate": learned_gate,
        }


class InputStdRouteTerminalHeads(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        image_channels: int,
        out_channels: int,
        *,
        blocks: int,
        strength: float,
        threshold: float,
        softness: float,
        direction: str,
        learned_gate_weight: float,
        learned_gate_mode: str,
        detach_gate: bool,
    ) -> None:
        super().__init__()
        self.strength = float(strength)
        self.threshold = float(threshold)
        self.softness = float(softness)
        self.direction = str(direction)
        self.learned_gate_weight = float(learned_gate_weight)
        self.learned_gate_mode = str(learned_gate_mode)
        self.detach_gate = bool(detach_gate)

        route_channels = feature_channels + image_channels + 1
        self.shared = nn.Sequential(
            nn.Conv2d(route_channels, feature_channels, kernel_size=1),
            nn.GELU(),
            StackedResidualBlocks(feature_channels, blocks),
        )
        self.high_expert = nn.Sequential(
            StackedResidualBlocks(feature_channels, blocks),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
            nn.GELU(),
        )
        self.low_expert = nn.Sequential(
            StackedResidualBlocks(feature_channels, blocks),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
            nn.GELU(),
        )
        self.high_out = nn.Conv2d(feature_channels, out_channels, kernel_size=3, padding=1)
        self.low_out = nn.Conv2d(feature_channels, out_channels, kernel_size=3, padding=1)
        if self.learned_gate_weight > 0.0:
            self.gate_head = _make_input_std_route_gate_head(feature_channels, self.learned_gate_mode)
        else:
            self.gate_head = None
        self._zero_init_route_outputs()

    def _zero_init_route_outputs(self) -> None:
        nn.init.zeros_(self.high_out.weight)
        nn.init.zeros_(self.high_out.bias)
        nn.init.zeros_(self.low_out.weight)
        nn.init.zeros_(self.low_out.bias)

    def _sample_prior(self, image: torch.Tensor) -> torch.Tensor:
        input_std = image.detach().flatten(start_dim=1).std(dim=1, unbiased=False).view(-1, 1, 1, 1)
        if self.direction in {"gt", "gte"}:
            prior = torch.sigmoid((input_std - self.threshold) / self.softness)
        elif self.direction in {"lt", "lte"}:
            prior = torch.sigmoid((self.threshold - input_std) / self.softness)
        else:
            raise ValueError("input_std_route_terminal_heads.direction must be gt, gte, lt, or lte")
        return prior

    def forward(
        self,
        decoder_feature: torch.Tensor,
        image: torch.Tensor,
        base_residual: torch.Tensor,
    ) -> Dict[str, torch.Tensor | None]:
        target_size = decoder_feature.shape[-2:]
        route_image = image
        if route_image.shape[-2:] != target_size:
            route_image = F.interpolate(
                route_image,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        prior_sample = self._sample_prior(image)
        prior_map = prior_sample.expand(-1, 1, target_size[0], target_size[1])
        shared = self.shared(torch.cat([decoder_feature, route_image, prior_map], dim=1))
        high_delta = self.strength * self.high_out(self.high_expert(shared))
        low_delta = self.strength * self.low_out(self.low_expert(shared))
        gate, learned_gate = _blend_input_std_route_gate(
            prior_map,
            shared,
            self.gate_head,
            learned_gate_weight=self.learned_gate_weight,
            learned_gate_mode=self.learned_gate_mode,
        )
        blend_gate = gate.detach() if self.detach_gate else gate
        high_residual = base_residual + high_delta
        low_residual = base_residual + low_delta
        residual = blend_gate * high_residual + (1.0 - blend_gate) * low_residual
        return {
            "delta": residual - base_residual,
            "residual": residual,
            "high_delta": high_delta,
            "low_delta": low_delta,
            "gate": gate,
            "prior": prior_map,
            "learned_gate": learned_gate,
        }


class InputStdRouteFeatureExpert(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        image_channels: int,
        *,
        blocks: int,
        strength: float,
        threshold: float,
        softness: float,
        direction: str,
        learned_gate_weight: float,
        learned_gate_mode: str,
        detach_gate: bool,
    ) -> None:
        super().__init__()
        self.strength = float(strength)
        self.threshold = float(threshold)
        self.softness = float(softness)
        self.direction = str(direction)
        self.learned_gate_weight = float(learned_gate_weight)
        self.learned_gate_mode = str(learned_gate_mode)
        self.detach_gate = bool(detach_gate)

        route_channels = feature_channels + image_channels + 1
        self.shared = nn.Sequential(
            nn.Conv2d(route_channels, feature_channels, kernel_size=1),
            nn.GELU(),
            StackedResidualBlocks(feature_channels, blocks),
        )
        self.high_expert = nn.Sequential(
            StackedResidualBlocks(feature_channels, blocks),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
            nn.GELU(),
        )
        self.low_expert = nn.Sequential(
            StackedResidualBlocks(feature_channels, blocks),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
            nn.GELU(),
        )
        self.high_out = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        self.low_out = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        if self.learned_gate_weight > 0.0:
            self.gate_head = _make_input_std_route_gate_head(feature_channels, self.learned_gate_mode)
        else:
            self.gate_head = None
        self._zero_init_route_outputs()

    def _zero_init_route_outputs(self) -> None:
        nn.init.zeros_(self.high_out.weight)
        nn.init.zeros_(self.high_out.bias)
        nn.init.zeros_(self.low_out.weight)
        nn.init.zeros_(self.low_out.bias)

    def _sample_prior(self, image: torch.Tensor) -> torch.Tensor:
        input_std = image.detach().flatten(start_dim=1).std(dim=1, unbiased=False).view(-1, 1, 1, 1)
        if self.direction in {"gt", "gte"}:
            prior = torch.sigmoid((input_std - self.threshold) / self.softness)
        elif self.direction in {"lt", "lte"}:
            prior = torch.sigmoid((self.threshold - input_std) / self.softness)
        else:
            raise ValueError("input_std_route_feature_expert.direction must be gt, gte, lt, or lte")
        return prior

    def forward(self, decoder_feature: torch.Tensor, image: torch.Tensor) -> Dict[str, torch.Tensor | None]:
        target_size = decoder_feature.shape[-2:]
        route_image = image
        if route_image.shape[-2:] != target_size:
            route_image = F.interpolate(
                route_image,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        prior_sample = self._sample_prior(image)
        prior_map = prior_sample.expand(-1, 1, target_size[0], target_size[1])
        shared = self.shared(torch.cat([decoder_feature, route_image, prior_map], dim=1))
        high_delta = self.strength * self.high_out(self.high_expert(shared))
        low_delta = self.strength * self.low_out(self.low_expert(shared))
        gate, learned_gate = _blend_input_std_route_gate(
            prior_map,
            shared,
            self.gate_head,
            learned_gate_weight=self.learned_gate_weight,
            learned_gate_mode=self.learned_gate_mode,
        )
        blend_gate = gate.detach() if self.detach_gate else gate
        delta = blend_gate * high_delta + (1.0 - blend_gate) * low_delta
        return {
            "feature": decoder_feature + delta,
            "delta": delta,
            "high_feature": decoder_feature + high_delta,
            "low_feature": decoder_feature + low_delta,
            "high_delta": high_delta,
            "low_delta": low_delta,
            "gate": gate,
            "prior": prior_map,
            "learned_gate": learned_gate,
        }


class InputStdRouteDecoderHeads(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        image_channels: int,
        out_channels: int,
        *,
        blocks: int,
        strength: float,
        threshold: float,
        softness: float,
        direction: str,
        learned_gate_weight: float,
        learned_gate_mode: str,
        detach_gate: bool,
        base_head: nn.Conv2d,
    ) -> None:
        super().__init__()
        self.strength = float(strength)
        self.threshold = float(threshold)
        self.softness = float(softness)
        self.direction = str(direction)
        self.learned_gate_weight = float(learned_gate_weight)
        self.learned_gate_mode = str(learned_gate_mode)
        self.detach_gate = bool(detach_gate)

        route_channels = feature_channels + image_channels + 1
        self.shared = nn.Sequential(
            nn.Conv2d(route_channels, feature_channels, kernel_size=1),
            nn.GELU(),
            StackedResidualBlocks(feature_channels, blocks),
        )
        self.high_feature = nn.Sequential(
            StackedResidualBlocks(feature_channels, blocks),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
            nn.GELU(),
        )
        self.low_feature = nn.Sequential(
            StackedResidualBlocks(feature_channels, blocks),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
            nn.GELU(),
        )
        self.high_head = nn.Conv2d(feature_channels, out_channels, kernel_size=3, padding=1)
        self.low_head = nn.Conv2d(feature_channels, out_channels, kernel_size=3, padding=1)
        if self.learned_gate_weight > 0.0:
            self.gate_head = _make_input_std_route_gate_head(feature_channels, self.learned_gate_mode)
        else:
            self.gate_head = None
        self._zero_init_feature_deltas()
        self._copy_base_head(base_head)

    def _zero_init_feature_deltas(self) -> None:
        high_out = self.high_feature[1]
        low_out = self.low_feature[1]
        if not isinstance(high_out, nn.Conv2d) or not isinstance(low_out, nn.Conv2d):
            raise TypeError("input_std route decoder head feature experts must end with Conv2d")
        nn.init.zeros_(high_out.weight)
        nn.init.zeros_(high_out.bias)
        nn.init.zeros_(low_out.weight)
        nn.init.zeros_(low_out.bias)

    def _copy_base_head(self, base_head: nn.Conv2d) -> None:
        with torch.no_grad():
            self.high_head.weight.copy_(base_head.weight)
            self.high_head.bias.copy_(base_head.bias)
            self.low_head.weight.copy_(base_head.weight)
            self.low_head.bias.copy_(base_head.bias)

    def _sample_prior(self, image: torch.Tensor) -> torch.Tensor:
        input_std = image.detach().flatten(start_dim=1).std(dim=1, unbiased=False).view(-1, 1, 1, 1)
        if self.direction in {"gt", "gte"}:
            prior = torch.sigmoid((input_std - self.threshold) / self.softness)
        elif self.direction in {"lt", "lte"}:
            prior = torch.sigmoid((self.threshold - input_std) / self.softness)
        else:
            raise ValueError("input_std_route_decoder_heads.direction must be gt, gte, lt, or lte")
        return prior

    def forward(
        self,
        decoder_feature: torch.Tensor,
        image: torch.Tensor,
        base_residual: torch.Tensor,
    ) -> Dict[str, torch.Tensor | None]:
        target_size = decoder_feature.shape[-2:]
        route_image = image
        if route_image.shape[-2:] != target_size:
            route_image = F.interpolate(
                route_image,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        prior_sample = self._sample_prior(image)
        prior_map = prior_sample.expand(-1, 1, target_size[0], target_size[1])
        shared = self.shared(torch.cat([decoder_feature, route_image, prior_map], dim=1))
        high_feature = decoder_feature + self.strength * self.high_feature(shared)
        low_feature = decoder_feature + self.strength * self.low_feature(shared)
        high_residual = self.high_head(high_feature)
        low_residual = self.low_head(low_feature)
        gate, learned_gate = _blend_input_std_route_gate(
            prior_map,
            shared,
            self.gate_head,
            learned_gate_weight=self.learned_gate_weight,
            learned_gate_mode=self.learned_gate_mode,
        )
        blend_gate = gate.detach() if self.detach_gate else gate
        residual = blend_gate * high_residual + (1.0 - blend_gate) * low_residual
        return {
            "delta": residual - base_residual,
            "residual": residual,
            "high_delta": high_residual - base_residual,
            "low_delta": low_residual - base_residual,
            "gate": gate,
            "prior": prior_map,
            "learned_gate": learned_gate,
        }


class InputStdRoutePostAdapter(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        image_channels: int,
        out_channels: int,
        *,
        blocks: int,
        strength: float,
        threshold: float,
        softness: float,
        direction: str,
        detach_gate: bool,
    ) -> None:
        super().__init__()
        self.strength = float(strength)
        self.threshold = float(threshold)
        self.softness = float(softness)
        self.direction = str(direction)
        self.detach_gate = bool(detach_gate)

        route_channels = feature_channels + image_channels + out_channels + 1
        self.shared = nn.Sequential(
            nn.Conv2d(route_channels, feature_channels, kernel_size=1),
            nn.GELU(),
            StackedResidualBlocks(feature_channels, blocks),
        )
        self.high_feature = nn.Sequential(
            StackedResidualBlocks(feature_channels, blocks),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
            nn.GELU(),
        )
        self.high_out = nn.Conv2d(feature_channels, out_channels, kernel_size=3, padding=1)
        self._zero_init_output()

    def _zero_init_output(self) -> None:
        nn.init.zeros_(self.high_out.weight)
        nn.init.zeros_(self.high_out.bias)

    def _sample_prior(self, image: torch.Tensor) -> torch.Tensor:
        input_std = image.detach().flatten(start_dim=1).std(dim=1, unbiased=False).view(-1, 1, 1, 1)
        if self.direction in {"gt", "gte"}:
            prior = torch.sigmoid((input_std - self.threshold) / self.softness)
        elif self.direction in {"lt", "lte"}:
            prior = torch.sigmoid((self.threshold - input_std) / self.softness)
        else:
            raise ValueError("input_std_route_post_adapter.direction must be gt, gte, lt, or lte")
        return prior

    def forward(
        self,
        decoder_feature: torch.Tensor,
        image: torch.Tensor,
        restored_reference: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        target_size = decoder_feature.shape[-2:]
        route_image = image
        if route_image.shape[-2:] != target_size:
            route_image = F.interpolate(
                route_image,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        route_restored = restored_reference
        if route_restored.shape[-2:] != target_size:
            route_restored = F.interpolate(
                route_restored,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        prior_sample = self._sample_prior(image)
        prior_map = prior_sample.expand(-1, 1, target_size[0], target_size[1])
        shared = self.shared(torch.cat([decoder_feature, route_image, route_restored, prior_map], dim=1))
        high_delta = self.strength * self.high_out(self.high_feature(shared))
        blend_gate = prior_map.detach() if self.detach_gate else prior_map
        return {
            "delta": blend_gate * high_delta,
            "high_delta": high_delta,
            "gate": prior_map,
            "prior": prior_map,
        }


class GaussianRouteControl(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        *,
        blocks: int,
        strength: float,
        threshold: float,
        softness: float,
        direction: str,
        learned_gate_weight: float,
        detach_gate: bool,
        adapter_count: int = 1,
    ) -> None:
        super().__init__()
        self.strength = float(strength)
        self.threshold = float(threshold)
        self.softness = float(softness)
        self.direction = str(direction)
        self.learned_gate_weight = float(learned_gate_weight)
        self.detach_gate = bool(detach_gate)
        if adapter_count not in {1, 2}:
            raise ValueError("GaussianRouteControl adapter_count must be 1 or 2")
        self.adapter_count = int(adapter_count)

        # Inputs: decoder feature plus Gaussian/control/context scalar maps.
        route_channels = feature_channels + 8
        self.prior_encoder = nn.Sequential(
            nn.Conv2d(route_channels, feature_channels, kernel_size=1),
            nn.GELU(),
            StackedResidualBlocks(feature_channels, blocks),
        )
        self.high_prior_encoder: nn.Module | None = None
        self.low_prior_encoder: nn.Module | None = None
        if self.adapter_count == 2:
            self.high_prior_encoder = nn.Sequential(
                nn.Conv2d(route_channels, feature_channels, kernel_size=1),
                nn.GELU(),
                StackedResidualBlocks(feature_channels, blocks),
            )
            self.low_prior_encoder = nn.Sequential(
                nn.Conv2d(route_channels, feature_channels, kernel_size=1),
                nn.GELU(),
                StackedResidualBlocks(feature_channels, blocks),
            )
        self.high_control = nn.Sequential(
            StackedResidualBlocks(feature_channels, blocks),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
            nn.GELU(),
        )
        self.low_control = nn.Sequential(
            StackedResidualBlocks(feature_channels, blocks),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
            nn.GELU(),
        )
        self.high_delta = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        self.low_delta = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        self.gate_head = nn.Sequential(
            nn.Conv2d(feature_channels, max(feature_channels // 2, 1), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(max(feature_channels // 2, 1), 1, kernel_size=1),
        )
        self.confidence_head = nn.Sequential(
            nn.Conv2d(feature_channels, max(feature_channels // 2, 1), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(max(feature_channels // 2, 1), 1, kernel_size=1),
        )
        self._zero_init_outputs()

    def _zero_init_outputs(self) -> None:
        nn.init.zeros_(self.high_delta.weight)
        nn.init.zeros_(self.high_delta.bias)
        nn.init.zeros_(self.low_delta.weight)
        nn.init.zeros_(self.low_delta.bias)
        _zero_init_last_conv2d(self.gate_head)
        _zero_init_last_conv2d(self.confidence_head)

    def _sample_prior(self, image: torch.Tensor) -> torch.Tensor:
        input_std = image.detach().flatten(start_dim=1).std(dim=1, unbiased=False).view(-1, 1, 1, 1)
        if self.direction in {"gt", "gte"}:
            prior = torch.sigmoid((input_std - self.threshold) / self.softness)
        elif self.direction in {"lt", "lte"}:
            prior = torch.sigmoid((self.threshold - input_std) / self.softness)
        else:
            raise ValueError("gaussian_route_control.direction must be gt, gte, lt, or lte")
        return prior

    def _resize_scalar(
        self,
        value: torch.Tensor | None,
        target_size: tuple[int, int],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if value is None:
            return reference.new_zeros((reference.shape[0], 1, target_size[0], target_size[1]))
        if value.shape[-2:] != target_size:
            value = F.interpolate(value, size=target_size, mode="bilinear", align_corners=False)
        if value.shape[1] != 1:
            value = value.mean(dim=1, keepdim=True)
        return value

    def forward(
        self,
        decoder_feature: torch.Tensor,
        image: torch.Tensor,
        attention_map: torch.Tensor,
        *,
        control_attention_map: torch.Tensor | None = None,
        control_canvas: torch.Tensor | None = None,
        feedback_map: torch.Tensor | None = None,
        directional_prior: torch.Tensor | None = None,
        shared_prior_feature: torch.Tensor | None = None,
        structure_target: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        target_size = decoder_feature.shape[-2:]
        prior_sample = self._sample_prior(image)
        prior_map = prior_sample.expand(-1, 1, target_size[0], target_size[1])
        attention = self._resize_scalar(attention_map, target_size, decoder_feature)
        control_attention = self._resize_scalar(control_attention_map, target_size, decoder_feature)
        canvas = self._resize_scalar(control_canvas, target_size, decoder_feature)
        feedback = self._resize_scalar(feedback_map, target_size, decoder_feature)
        structure = self._resize_scalar(structure_target, target_size, decoder_feature)

        if directional_prior is None:
            direction_summary = decoder_feature.new_zeros((decoder_feature.shape[0], 1, target_size[0], target_size[1]))
        else:
            direction_summary = self._resize_scalar(directional_prior, target_size, decoder_feature)
        if shared_prior_feature is None:
            shared_summary = decoder_feature.new_zeros((decoder_feature.shape[0], 1, target_size[0], target_size[1]))
        else:
            shared_summary = self._resize_scalar(shared_prior_feature, target_size, decoder_feature)

        local_contrast = attention - F.avg_pool2d(attention, kernel_size=3, stride=1, padding=1)
        route_input = torch.cat(
            [
                decoder_feature,
                attention,
                local_contrast,
                control_attention,
                canvas + feedback,
                direction_summary,
                shared_summary,
                structure,
                prior_map,
            ],
            dim=1,
        )
        route_feature = self.prior_encoder(route_input)
        raw_gate = self.gate_head(route_feature)
        residual = self.learned_gate_weight * torch.tanh(raw_gate)
        prior = prior_map.clamp(1e-5, 1.0 - 1e-5)
        gate = torch.where(
            residual >= 0.0,
            prior + (1.0 - prior) * residual,
            prior + prior * residual,
        )
        confidence = torch.sigmoid(self.confidence_head(route_feature))
        if self.high_prior_encoder is not None and self.low_prior_encoder is not None:
            high_route_feature = self.high_prior_encoder(route_input)
            low_route_feature = self.low_prior_encoder(route_input)
        else:
            high_route_feature = route_feature
            low_route_feature = route_feature
        high_delta = self.strength * self.high_delta(self.high_control(high_route_feature))
        low_delta = self.strength * self.low_delta(self.low_control(low_route_feature))
        blend_gate = gate.detach() if self.detach_gate else gate
        delta = blend_gate * high_delta + (1.0 - blend_gate) * low_delta
        return {
            "feature": decoder_feature + delta,
            "delta": delta,
            "high_delta": high_delta,
            "low_delta": low_delta,
            "gate": gate,
            "prior": prior_map,
            "confidence": confidence,
        }


class CrossResolutionGaussianRestorer(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self._training_progress = 1.0
        width = self.config.width
        lowres_in_channels = self.config.in_channels * (self.config.lowres_scale ** 2)

        self.highres_stem = nn.Sequential(
            nn.Conv2d(self.config.in_channels, width, kernel_size=3, padding=1),
            StackedResidualBlocks(width, self.config.highres_blocks),
        )
        self.lowres_proj = nn.Sequential(
            nn.Conv2d(lowres_in_channels, width, kernel_size=1),
            StackedResidualBlocks(width, self.config.lowres_blocks),
        )
        self.texture_frequency = TextureFrequencyExtractor(width, width)
        self.lowres_fusion = nn.Sequential(
            nn.Conv2d(width * 2, width, kernel_size=1),
            ResidualDWBlock(width),
        )
        self.explicit_frequency_splitter: ExplicitFrequencySplitter | None = None
        if self.config.explicit_frequency_split.enabled:
            self.explicit_frequency_splitter = ExplicitFrequencySplitter(
                width,
                kernel_size=self.config.explicit_frequency_split.lowpass_kernel_size,
                highres_high_band_strength=self.config.explicit_frequency_split.highres_high_band_strength,
                gaussian_feedback_strength=self.config.explicit_frequency_split.gaussian_feedback_strength,
                gaussian_feedback_attention_source=(
                    self.config.explicit_frequency_split.gaussian_feedback_attention_source
                ),
                detail_blocks=self.config.explicit_frequency_split.detail_blocks,
                use_structure_hint=self.config.explicit_frequency_split.use_structure_hint,
                structure_hint_strength=self.config.explicit_frequency_split.structure_hint_strength,
            )
        self.gaussian_head = GaussianParameterHead(width, self.config.gaussian, self.config.affine_rank)
        self.gaussian_renderer = TextureAwareGaussianRenderer(self.config.gaussian)
        self.directional_projector: DirectionalFrequencyProjector | None = None
        if self.config.directional_frequency_guidance.enabled:
            self.directional_projector = DirectionalFrequencyProjector(self.config.gaussian)
        self.affine_modulator = LowRankAffineModulator(width, self.config.affine_rank)
        self.decoder = nn.Sequential(
            nn.Conv2d(width * 2, width, kernel_size=1),
            StackedResidualBlocks(width, self.config.decoder_blocks),
            nn.Conv2d(width, self.config.out_channels, kernel_size=3, padding=1),
        )
        self.gaussian_decoder_guidance: GaussianGuidedDecoderModulator | None = None
        if self.config.gaussian_decoder_guidance.enabled:
            self.gaussian_decoder_guidance = GaussianGuidedDecoderModulator(
                width,
                directional_channels=(
                    DirectionalFrequencyProjector.out_channels
                    if self.config.directional_frequency_guidance.enabled
                    else 0
                ),
                blocks=self.config.gaussian_decoder_guidance.blocks,
                strength=self.config.gaussian_decoder_guidance.strength,
                use_shared_prior_stem=self.config.gaussian_shared_prior.enabled,
                use_decoder_state_gate=self.config.gaussian_decoder_guidance.use_decoder_state_gate,
            )
        self.gaussian_shared_prior: GaussianSharedPriorEncoder | None = None
        if self.config.gaussian_shared_prior.enabled:
            self.gaussian_shared_prior = GaussianSharedPriorEncoder(
                width,
                directional_channels=(
                    DirectionalFrequencyProjector.out_channels
                    if self.config.directional_frequency_guidance.enabled
                    else 0
                ),
                blocks=self.config.gaussian_shared_prior.blocks,
            )
        self.gaussian_control_router: GaussianStructuredControlRouter | None = None
        if self.config.gaussian_control_router.enabled:
            self.gaussian_control_router = GaussianStructuredControlRouter(
                width,
                directional_channels=(
                    DirectionalFrequencyProjector.out_channels
                    if self.config.directional_frequency_guidance.enabled
                    else 0
                ),
                blocks=self.config.gaussian_control_router.blocks,
                use_decoder_state=self.config.gaussian_control_router.use_decoder_state,
                use_edge_state=self.config.gaussian_control_router.use_edge_state,
                role_delta_strength=self.config.gaussian_control_router.role_delta_strength,
                use_competitive_role_gate=self.config.gaussian_control_router.use_competitive_role_gate,
                role_gate_temperature=self.config.gaussian_control_router.role_gate_temperature,
                use_role_residual_centering=self.config.gaussian_control_router.use_role_residual_centering,
                use_demand_conditioned_competition=self.config.gaussian_control_router.use_demand_conditioned_competition,
                role_competition_start=self.config.gaussian_control_router.role_competition_start,
                role_competition_strength=self.config.gaussian_control_router.role_competition_strength,
                role_competition_floor=self.config.gaussian_control_router.role_competition_floor,
            )
        self.gaussian_terminal_residual: GaussianGuidedTerminalResidualHead | None = None
        if self.config.gaussian_terminal_residual.enabled:
            self.gaussian_terminal_residual = GaussianGuidedTerminalResidualHead(
                width,
                self.config.out_channels,
                directional_channels=(
                    DirectionalFrequencyProjector.out_channels
                    if self.config.directional_frequency_guidance.enabled
                    else 0
                ),
                blocks=self.config.gaussian_terminal_residual.blocks,
                strength=self.config.gaussian_terminal_residual.strength,
            )
        self.gaussian_feature_refinement: GaussianGuidedFeatureRefinement | None = None
        if self.config.gaussian_feature_refinement.enabled:
            self.gaussian_feature_refinement = GaussianGuidedFeatureRefinement(
                width,
                directional_channels=(
                    DirectionalFrequencyProjector.out_channels
                    if self.config.directional_frequency_guidance.enabled
                    else 0
                ),
                blocks=self.config.gaussian_feature_refinement.blocks,
                strength=self.config.gaussian_feature_refinement.strength,
                use_decoder_detail_hint=self.config.gaussian_feature_refinement.use_decoder_detail_hint,
                use_edge_aware_prior=self.config.gaussian_feature_refinement.use_edge_aware_prior,
                use_confidence_normalization=self.config.gaussian_feature_refinement.use_confidence_normalization,
                use_intervention_demand=self.config.gaussian_feature_refinement.use_intervention_demand,
                use_decoder_detail_gate=self.config.gaussian_feature_refinement.use_decoder_detail_gate,
                use_shared_prior_stem=self.config.gaussian_shared_prior.enabled,
            )
        self.gaussian_attention_demand_bridge: GaussianAttentionDemandBridge | None = None
        if self.config.gaussian_attention_demand_bridge.enabled:
            self.gaussian_attention_demand_bridge = GaussianAttentionDemandBridge(
                width,
                blocks=self.config.gaussian_attention_demand_bridge.blocks,
                strength=self.config.gaussian_attention_demand_bridge.strength,
                use_input_structure=self.config.gaussian_attention_demand_bridge.use_input_structure,
            )
        self.gaussian_control_canvas: GaussianControlCanvasModule | None = None
        if self.config.gaussian_control_canvas.enabled:
            self.gaussian_control_canvas = GaussianControlCanvasModule(
                width,
                directional_channels=(
                    DirectionalFrequencyProjector.out_channels
                    if self.config.directional_frequency_guidance.enabled
                    else 0
                ),
                blocks=self.config.gaussian_control_canvas.blocks,
                feedback_strength=self.config.gaussian_control_canvas.strength,
                use_target_conditioned_feedback=self.config.gaussian_control_canvas.use_target_conditioned_feedback,
                feedback_mode=self.config.gaussian_control_canvas.feedback_mode,
                use_confidence_normalization=self.config.gaussian_control_canvas.use_confidence_normalization,
                residual_delta_strength=self.config.gaussian_control_canvas.residual_delta_strength,
                feedback_selectivity_mode=self.config.gaussian_control_canvas.feedback_selectivity_mode,
                feedback_selectivity_temperature=self.config.gaussian_control_canvas.feedback_selectivity_temperature,
                feedback_demand_power=self.config.gaussian_control_canvas.feedback_demand_power,
                feedback_budget=self.config.gaussian_control_canvas.feedback_budget,
                feedback_floor=self.config.gaussian_control_canvas.feedback_floor,
                feedback_target_mean=self.config.gaussian_control_canvas.feedback_target_mean,
                feedback_mean_boost_cap=self.config.gaussian_control_canvas.feedback_mean_boost_cap,
                feedback_demand_source=self.config.gaussian_control_canvas.feedback_demand_source,
            )
        self.edge_refiner: HighResolutionEdgeGuidedRefiner | None = None
        if self.config.edge_guidance.enabled:
            self.edge_refiner = HighResolutionEdgeGuidedRefiner(
                width,
                self.config.out_channels,
                blocks=self.config.edge_guidance.blocks,
                strength=self.config.edge_guidance.strength,
                directional_channels=(
                    DirectionalFrequencyProjector.out_channels
                    if self.config.directional_frequency_guidance.enabled
                    else 0
                ),
                use_control_gate=self.config.edge_guidance.use_control_gate,
            )
        self.decoupled_correction: DecoupledCorrectionModule | None = None
        if self.config.decoupled_correction.enabled:
            self.decoupled_correction = DecoupledCorrectionModule(
                width,
                self.config.out_channels,
                blocks=self.config.decoupled_correction.blocks,
                strength=self.config.decoupled_correction.strength,
                use_gradient_detached_selectivity=self.config.decoupled_correction.use_gradient_detached_selectivity,
                amplitude_channels=self.config.decoupled_correction.amplitude_channels,
                amplitude_mode=self.config.decoupled_correction.amplitude_mode,
            )
        self.input_std_route_residual: InputStdRouteResidualHead | None = None
        if self.config.input_std_route_residual.enabled:
            route_cfg = self.config.input_std_route_residual
            self.input_std_route_residual = InputStdRouteResidualHead(
                width,
                self.config.in_channels,
                self.config.out_channels,
                blocks=route_cfg.blocks,
                strength=route_cfg.strength,
                threshold=route_cfg.threshold,
                softness=route_cfg.softness,
                direction=route_cfg.direction,
                learned_gate_weight=route_cfg.learned_gate_weight,
                learned_gate_mode=route_cfg.learned_gate_mode,
                detach_gate=route_cfg.detach_gate,
            )
        self.input_std_route_terminal_heads: InputStdRouteTerminalHeads | None = None
        if self.config.input_std_route_terminal_heads.enabled:
            route_cfg = self.config.input_std_route_terminal_heads
            self.input_std_route_terminal_heads = InputStdRouteTerminalHeads(
                width,
                self.config.in_channels,
                self.config.out_channels,
                blocks=route_cfg.blocks,
                strength=route_cfg.strength,
                threshold=route_cfg.threshold,
                softness=route_cfg.softness,
                direction=route_cfg.direction,
                learned_gate_weight=route_cfg.learned_gate_weight,
                learned_gate_mode=route_cfg.learned_gate_mode,
                detach_gate=route_cfg.detach_gate,
            )
        self.input_std_route_feature_expert: InputStdRouteFeatureExpert | None = None
        if self.config.input_std_route_feature_expert.enabled:
            route_cfg = self.config.input_std_route_feature_expert
            self.input_std_route_feature_expert = InputStdRouteFeatureExpert(
                width,
                self.config.in_channels,
                blocks=route_cfg.blocks,
                strength=route_cfg.strength,
                threshold=route_cfg.threshold,
                softness=route_cfg.softness,
                direction=route_cfg.direction,
                learned_gate_weight=route_cfg.learned_gate_weight,
                learned_gate_mode=route_cfg.learned_gate_mode,
                detach_gate=route_cfg.detach_gate,
            )
        self.input_std_route_decoder_heads: InputStdRouteDecoderHeads | None = None
        if self.config.input_std_route_decoder_heads.enabled:
            route_cfg = self.config.input_std_route_decoder_heads
            self.input_std_route_decoder_heads = InputStdRouteDecoderHeads(
                width,
                self.config.in_channels,
                self.config.out_channels,
                blocks=route_cfg.blocks,
                strength=route_cfg.strength,
                threshold=route_cfg.threshold,
                softness=route_cfg.softness,
                direction=route_cfg.direction,
                learned_gate_weight=route_cfg.learned_gate_weight,
                learned_gate_mode=route_cfg.learned_gate_mode,
                detach_gate=route_cfg.detach_gate,
                base_head=self.decoder[2],
            )
        self.input_std_route_post_adapter: InputStdRoutePostAdapter | None = None
        if self.config.input_std_route_post_adapter.enabled:
            route_cfg = self.config.input_std_route_post_adapter
            self.input_std_route_post_adapter = InputStdRoutePostAdapter(
                width,
                self.config.in_channels,
                self.config.out_channels,
                blocks=route_cfg.blocks,
                strength=route_cfg.strength,
                threshold=route_cfg.threshold,
                softness=route_cfg.softness,
                direction=route_cfg.direction,
                detach_gate=route_cfg.detach_gate,
            )
        self.gaussian_route_control: GaussianRouteControl | None = None
        if self.config.gaussian_route_control.enabled:
            route_cfg = self.config.gaussian_route_control
            self.gaussian_route_control = GaussianRouteControl(
                width,
                blocks=route_cfg.blocks,
                strength=route_cfg.strength,
                threshold=route_cfg.threshold,
                softness=route_cfg.softness,
                direction=route_cfg.direction,
                learned_gate_weight=route_cfg.learned_gate_weight,
                detach_gate=route_cfg.detach_gate,
                adapter_count=route_cfg.adapter_count,
            )

    def initialize_input_std_route_decoder_heads_from_base(self) -> None:
        if self.input_std_route_decoder_heads is not None:
            self.input_std_route_decoder_heads._copy_base_head(self.decoder[2])

    def _luminance(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] < 3:
            return image.mean(dim=1, keepdim=True)
        weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (image[:, :3] * weights).sum(dim=1, keepdim=True)

    def _control_guidance_blend_map(self, image: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        canvas_cfg = self.config.gaussian_control_canvas
        blend = float(canvas_cfg.share_to_guidance_blend)
        batch = image.shape[0]
        base = image.new_full((batch, 1, target_hw[0], target_hw[1]), blend)
        if (
            not canvas_cfg.share_to_guidance_detail_preserve
            or canvas_cfg.share_to_guidance_detail_preserve_strength <= 0.0
            or blend <= 0.0
        ):
            return base

        rgb = image
        if rgb.shape[-2:] != target_hw:
            rgb = F.interpolate(rgb, size=target_hw, mode="bilinear", align_corners=False)
        gray = self._luminance(rgb)
        kernel_size = max(int(canvas_cfg.share_to_guidance_detail_preserve_kernel_size), 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        pad = kernel_size // 2
        if kernel_size > 1:
            local_mean = F.avg_pool2d(
                F.pad(gray, (pad, pad, pad, pad), mode="replicate"),
                kernel_size=kernel_size,
                stride=1,
            )
        else:
            local_mean = gray
        highpass = (gray - local_mean).abs()
        grad_x = gray[..., :, 1:] - gray[..., :, :-1]
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode="replicate")
        grad_y = gray[..., 1:, :] - gray[..., :-1, :]
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode="replicate")
        detail = highpass + torch.sqrt((grad_x.square() + grad_y.square()).clamp_min(1e-8))
        detail_mean = detail.flatten(start_dim=1).mean(dim=1).view(batch, 1, 1, 1).clamp_min(1e-6)
        detail_norm = detail / detail_mean
        schedule_factor = self._detail_preserve_schedule_factor()
        strength = float(canvas_cfg.share_to_guidance_detail_preserve_strength) * schedule_factor
        floor = 1.0 + (float(canvas_cfg.share_to_guidance_detail_preserve_floor) - 1.0) * schedule_factor
        detail_suppression = detail_norm / (detail_norm + 1.0)
        preserve_gate = (1.0 - strength * detail_suppression).clamp(min=floor, max=1.0)
        return base * preserve_gate

    def _apply_control_guidance_reliability_gate(
        self,
        blend_map: torch.Tensor,
        control_canvas: torch.Tensor,
        attention_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        canvas_cfg = self.config.gaussian_control_canvas
        if (
            not canvas_cfg.share_to_guidance_reliability_gate
            or canvas_cfg.share_to_guidance_reliability_strength <= 0.0
        ):
            return blend_map, None

        if attention_map.shape[-2:] != blend_map.shape[-2:]:
            attention_map = F.interpolate(
                attention_map,
                size=blend_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if control_canvas.shape[-2:] != blend_map.shape[-2:]:
            control_canvas = F.interpolate(
                control_canvas,
                size=blend_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if control_canvas.shape[1] != 1:
            control_canvas = control_canvas.mean(dim=1, keepdim=True)
        if attention_map.shape[1] != 1:
            attention_map = attention_map.mean(dim=1, keepdim=True)

        kernel_size = max(int(canvas_cfg.share_to_guidance_reliability_kernel_size), 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        mismatch = (control_canvas - attention_map).abs()
        if kernel_size > 1:
            pad = kernel_size // 2
            mismatch = F.avg_pool2d(
                F.pad(mismatch, (pad, pad, pad, pad), mode="replicate"),
                kernel_size=kernel_size,
                stride=1,
            )
        mismatch_mean = mismatch.flatten(start_dim=1).mean(dim=1).view(
            mismatch.shape[0],
            1,
            1,
            1,
        ).clamp_min(1e-6)
        mismatch_norm = mismatch / mismatch_mean
        mismatch_ratio = mismatch_norm / (mismatch_norm + 1.0)
        schedule_factor = self._control_guidance_schedule_factor(
            canvas_cfg.share_to_guidance_reliability_schedule,
            canvas_cfg.share_to_guidance_reliability_schedule_start,
            canvas_cfg.share_to_guidance_reliability_schedule_end,
        )
        strength = float(canvas_cfg.share_to_guidance_reliability_strength) * schedule_factor
        floor = 1.0 + (float(canvas_cfg.share_to_guidance_reliability_floor) - 1.0) * schedule_factor
        reliability_gate = (1.0 - strength * mismatch_ratio).clamp(min=floor, max=1.0)
        return blend_map * reliability_gate, reliability_gate

    def _detail_preserve_schedule_factor(self) -> float:
        canvas_cfg = self.config.gaussian_control_canvas
        return self._control_guidance_schedule_factor(
            canvas_cfg.share_to_guidance_detail_preserve_schedule,
            canvas_cfg.share_to_guidance_detail_preserve_schedule_start,
            canvas_cfg.share_to_guidance_detail_preserve_schedule_end,
        )

    def _control_guidance_schedule_factor(
        self,
        schedule: str,
        start_value: float,
        end_value: float,
    ) -> float:
        if schedule == "none":
            return 1.0
        start = float(start_value)
        end = float(end_value)
        progress = max(0.0, min(1.0, float(self._training_progress)))
        if progress <= start:
            return 0.0
        if progress >= end:
            return 1.0
        denom = max(end - start, 1e-6)
        return (progress - start) / denom

    def forward(self, x: torch.Tensor, return_aux: bool | str = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        original_size = x.shape[-2:]
        padded, _ = pad_to_multiple(x, self.config.lowres_scale)
        structure_target = self.structure_canvas_target(padded)

        highres_feature = self.highres_stem(padded)
        gaussian_input = padded
        explicit_low_band = None
        explicit_high_band = None
        guided_high_detail = None
        if self.explicit_frequency_splitter is not None:
            explicit_low_band, explicit_high_band = self.explicit_frequency_splitter.split(padded)
            gaussian_input = explicit_low_band

        lowres_input = F.pixel_unshuffle(gaussian_input, self.config.lowres_scale)
        lowres_feature = self.lowres_proj(lowres_input)
        lowres_rgb = F.interpolate(
            gaussian_input,
            size=lowres_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        texture_feature = self.texture_frequency(lowres_rgb, lowres_feature)
        lowres_feature = self.lowres_fusion(torch.cat([lowres_feature, texture_feature], dim=1))

        head_outputs = self.gaussian_head(lowres_feature)
        attention_map = self.gaussian_renderer(head_outputs["params"], target_size=highres_feature.shape[-2:])
        raw_attention_map = attention_map
        control_attention_map = attention_map
        attention_bridge_map = None
        if self.gaussian_attention_demand_bridge is not None:
            bridged_attention_map, attention_bridge_map = self.gaussian_attention_demand_bridge(
                highres_feature,
                attention_map,
                input_structure=structure_target,
            )
            if self.config.gaussian_attention_demand_bridge.apply_to == "main":
                attention_map = bridged_attention_map
                control_attention_map = bridged_attention_map
            elif self.config.gaussian_attention_demand_bridge.apply_to == "control":
                control_attention_map = bridged_attention_map
            else:
                raise ValueError("gaussian_attention_demand_bridge.apply_to must be main or control")
        directional_prior = None
        if self.directional_projector is not None:
            directional_prior = self.directional_projector(
                head_outputs["params"],
                target_size=highres_feature.shape[-2:],
            )
        if self.explicit_frequency_splitter is not None:
            detail_attention_map = attention_map
            if self.config.explicit_frequency_split.gaussian_feedback_attention_source == "control":
                detail_attention_map = control_attention_map
            explicit_low_band, explicit_high_band, highres_feature, guided_high_detail = self.explicit_frequency_splitter(
                padded,
                highres_feature,
                detail_attention_map,
            )
        highpass = highres_feature - F.avg_pool2d(highres_feature, kernel_size=3, stride=1, padding=1)
        modulated = highres_feature + attention_map * highpass
        modulated = self.affine_modulator(modulated, head_outputs["coeffs"])

        upsampled_lowres = F.interpolate(
            lowres_feature,
            size=modulated.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoder_input = torch.cat([modulated, upsampled_lowres], dim=1)
        decoder_feature = self.decoder[0](decoder_input)
        decoder_feature = self.decoder[1](decoder_feature)
        control_canvas = None
        feedback_map = None
        control_guidance_blend_map = None
        control_guidance_reliability_map = None
        decoupled_selectivity_map = None
        decoupled_amplitude_map = None
        decoupled_correction_map = None
        input_std_route_delta = None
        input_std_route_high_delta = None
        input_std_route_low_delta = None
        input_std_route_gate = None
        input_std_route_prior = None
        input_std_route_learned_gate = None
        input_std_route_high_restored = None
        input_std_route_low_restored = None
        input_std_route_post_adapter_delta = None
        input_std_route_post_adapter_high_delta = None
        input_std_route_post_adapter_gate = None
        input_std_route_post_adapter_prior = None
        gaussian_route_control_delta = None
        gaussian_route_control_high_delta = None
        gaussian_route_control_low_delta = None
        gaussian_route_control_gate = None
        gaussian_route_control_prior = None
        gaussian_route_control_confidence = None
        gaussian_route_control_high_restored = None
        gaussian_route_control_low_restored = None
        if (
            self.gaussian_control_canvas is not None
            and self.config.gaussian_control_canvas.stage == "pre_guidance"
        ):
            decoder_feature, control_canvas, feedback_map = self.gaussian_control_canvas(
                decoder_feature,
                control_attention_map,
                directional_prior,
                structure_target=structure_target,
            )
            if self.config.gaussian_control_canvas.share_to_guidance:
                control_guidance_blend_map = self._control_guidance_blend_map(
                    padded,
                    attention_map.shape[-2:],
                )
                control_guidance_blend_map, control_guidance_reliability_map = (
                    self._apply_control_guidance_reliability_gate(
                        control_guidance_blend_map,
                        control_canvas,
                        attention_map,
                    )
                )
                control_attention_map = (
                    control_attention_map
                    + control_guidance_blend_map * (control_canvas - control_attention_map)
                )
                attention_map = attention_map + control_guidance_blend_map * (control_canvas - attention_map)
        shared_prior_feature = None
        if self.gaussian_shared_prior is not None:
            shared_prior_feature = self.gaussian_shared_prior(
                attention_map,
                directional_prior,
                target_size=decoder_feature.shape[-2:],
            )
        decoder_control_feature = shared_prior_feature
        feature_control_feature = shared_prior_feature
        edge_control_feature = shared_prior_feature
        if self.gaussian_control_router is not None:
            if shared_prior_feature is None:
                raise ValueError("gaussian_control_router requires shared_prior_feature")
            control_features = self.gaussian_control_router(
                shared_prior_feature,
                decoder_feature,
                attention_map,
                directional_prior,
                highres_rgb=padded,
            )
            decoder_control_feature = control_features["decoder"]
            feature_control_feature = control_features["feature"]
            edge_control_feature = control_features["edge"]
        if self.gaussian_decoder_guidance is not None:
            # Keep the backbone unchanged and let Gaussian priors only modulate decoder features.
            decoder_feature = self.gaussian_decoder_guidance(
                decoder_feature,
                attention_map,
                directional_prior,
                shared_prior_feature=decoder_control_feature,
            )
        if self.gaussian_feature_refinement is not None:
            # Keep Gaussian as a structural controller, but refine in feature space rather than direct RGB.
            decoder_feature = self.gaussian_feature_refinement(
                decoder_feature,
                attention_map,
                directional_prior,
                highres_rgb=padded,
                shared_prior_feature=feature_control_feature,
            )
        if (
            self.gaussian_control_canvas is not None
            and self.config.gaussian_control_canvas.stage == "post_refinement"
        ):
            decoder_feature, control_canvas, feedback_map = self.gaussian_control_canvas(
                decoder_feature,
                control_attention_map,
                directional_prior,
                structure_target=structure_target,
            )
        if self.gaussian_route_control is not None:
            route_base_decoder_feature = decoder_feature
            route_outputs = self.gaussian_route_control(
                decoder_feature,
                padded,
                attention_map,
                control_attention_map=control_attention_map,
                control_canvas=control_canvas,
                feedback_map=feedback_map,
                directional_prior=directional_prior,
                shared_prior_feature=shared_prior_feature,
                structure_target=structure_target,
            )
            decoder_feature = route_outputs["feature"]
            gaussian_route_control_delta = route_outputs["delta"]
            gaussian_route_control_high_delta = route_outputs["high_delta"]
            gaussian_route_control_low_delta = route_outputs["low_delta"]
            gaussian_route_control_gate = route_outputs["gate"]
            gaussian_route_control_prior = route_outputs["prior"]
            gaussian_route_control_confidence = route_outputs["confidence"]
            gaussian_route_control_high_restored = padded + self.decoder[2](
                route_base_decoder_feature + gaussian_route_control_high_delta
            )
            gaussian_route_control_low_restored = padded + self.decoder[2](
                route_base_decoder_feature + gaussian_route_control_low_delta
            )
        if self.input_std_route_feature_expert is not None:
            base_decoder_feature = decoder_feature
            route_outputs = self.input_std_route_feature_expert(decoder_feature, padded)
            route_high_feature = route_outputs["high_feature"]
            route_low_feature = route_outputs["low_feature"]
            input_std_route_gate = route_outputs["gate"]
            input_std_route_prior = route_outputs["prior"]
            input_std_route_learned_gate = route_outputs["learned_gate"]
            if route_high_feature is None or route_low_feature is None:
                raise ValueError("input_std_route_feature_expert must return high_feature and low_feature")
            route_base_residual = self.decoder[2](base_decoder_feature)
            route_high_residual = self.decoder[2](route_high_feature)
            route_low_residual = self.decoder[2](route_low_feature)
            input_std_route_high_delta = route_high_residual - route_base_residual
            input_std_route_low_delta = route_low_residual - route_base_residual
            input_std_route_delta = self.decoder[2](route_outputs["feature"]) - route_base_residual
            input_std_route_high_restored = padded + route_high_residual
            input_std_route_low_restored = padded + route_low_residual
            decoder_feature = route_outputs["feature"]
        residual = self.decoder[2](decoder_feature)
        if self.decoupled_correction is not None:
            decoupled_correction_map, decoupled_selectivity_map, decoupled_amplitude_map = self.decoupled_correction(
                decoder_feature,
                control_attention_map,
                control_canvas=control_canvas,
                detail_reference=guided_high_detail,
            )
            residual = residual + decoupled_correction_map
        if self.gaussian_terminal_residual is not None:
            residual = residual + self.gaussian_terminal_residual(
                decoder_feature,
                attention_map,
                directional_prior,
            )
        if self.edge_refiner is not None:
            residual = residual + self.edge_refiner(
                padded,
                decoder_feature,
                attention_map,
                directional_prior,
                control_feature=edge_control_feature,
            )
        base_residual = residual
        if self.input_std_route_residual is not None:
            route_outputs = self.input_std_route_residual(decoder_feature, padded)
            input_std_route_delta = route_outputs["delta"]
            input_std_route_high_delta = route_outputs["high_delta"]
            input_std_route_low_delta = route_outputs["low_delta"]
            input_std_route_gate = route_outputs["gate"]
            input_std_route_prior = route_outputs["prior"]
            input_std_route_learned_gate = route_outputs["learned_gate"]
            input_std_route_high_restored = padded + base_residual + input_std_route_high_delta
            input_std_route_low_restored = padded + base_residual + input_std_route_low_delta
            residual = base_residual + input_std_route_delta
        if self.input_std_route_terminal_heads is not None:
            route_outputs = self.input_std_route_terminal_heads(decoder_feature, padded, base_residual)
            input_std_route_delta = route_outputs["delta"]
            input_std_route_high_delta = route_outputs["high_delta"]
            input_std_route_low_delta = route_outputs["low_delta"]
            input_std_route_gate = route_outputs["gate"]
            input_std_route_prior = route_outputs["prior"]
            input_std_route_learned_gate = route_outputs["learned_gate"]
            input_std_route_high_restored = padded + base_residual + input_std_route_high_delta
            input_std_route_low_restored = padded + base_residual + input_std_route_low_delta
            residual = route_outputs["residual"]
        if self.input_std_route_decoder_heads is not None:
            route_outputs = self.input_std_route_decoder_heads(decoder_feature, padded, base_residual)
            input_std_route_delta = route_outputs["delta"]
            input_std_route_high_delta = route_outputs["high_delta"]
            input_std_route_low_delta = route_outputs["low_delta"]
            input_std_route_gate = route_outputs["gate"]
            input_std_route_prior = route_outputs["prior"]
            input_std_route_learned_gate = route_outputs["learned_gate"]
            input_std_route_high_restored = padded + base_residual + input_std_route_high_delta
            input_std_route_low_restored = padded + base_residual + input_std_route_low_delta
            residual = route_outputs["residual"]
        if self.input_std_route_post_adapter is not None:
            restored_reference = padded + residual
            previous_high_restored = input_std_route_high_restored
            previous_low_restored = input_std_route_low_restored
            route_outputs = self.input_std_route_post_adapter(
                decoder_feature,
                padded,
                restored_reference,
            )
            input_std_route_post_adapter_delta = route_outputs["delta"]
            input_std_route_post_adapter_high_delta = route_outputs["high_delta"]
            input_std_route_post_adapter_gate = route_outputs["gate"]
            input_std_route_post_adapter_prior = route_outputs["prior"]
            residual = residual + input_std_route_post_adapter_delta
            input_std_route_delta = (
                input_std_route_post_adapter_delta
                if input_std_route_delta is None
                else input_std_route_delta + input_std_route_post_adapter_delta
            )
            input_std_route_high_delta = (
                input_std_route_post_adapter_high_delta
                if input_std_route_high_delta is None
                else input_std_route_high_delta + input_std_route_post_adapter_high_delta
            )
            if input_std_route_low_delta is None:
                input_std_route_low_delta = torch.zeros_like(input_std_route_post_adapter_high_delta)
            input_std_route_gate = input_std_route_post_adapter_gate
            input_std_route_prior = input_std_route_post_adapter_prior
            input_std_route_high_restored = (
                restored_reference + input_std_route_post_adapter_high_delta
                if previous_high_restored is None
                else previous_high_restored + input_std_route_post_adapter_high_delta
            )
            input_std_route_low_restored = (
                restored_reference
                if previous_low_restored is None
                else previous_low_restored
            )
        restored = padded + residual
        restored = crop_to_size(restored, original_size)
        input_std_route_high_restored = (
            crop_to_size(input_std_route_high_restored, original_size)
            if input_std_route_high_restored is not None
            else None
        )
        input_std_route_low_restored = (
            crop_to_size(input_std_route_low_restored, original_size)
            if input_std_route_low_restored is not None
            else None
        )
        control_canvas_target = None
        if self.config.gaussian_control_canvas.target_mode == "structure":
            control_canvas_target = structure_target

        if not return_aux:
            return restored
        if return_aux == "loss":
            return {
                "restored": restored,
                "attention_map": crop_to_size(attention_map, original_size),
                "raw_attention_map": crop_to_size(raw_attention_map, original_size),
                "control_attention_map": crop_to_size(control_attention_map, original_size),
                "attention_bridge_map": (
                    crop_to_size(attention_bridge_map, original_size)
                    if attention_bridge_map is not None
                    else None
                ),
                "control_canvas": crop_to_size(control_canvas, original_size) if control_canvas is not None else None,
                "control_guidance_blend_map": (
                    crop_to_size(control_guidance_blend_map, original_size)
                    if control_guidance_blend_map is not None
                    else None
                ),
                "control_guidance_reliability_map": (
                    crop_to_size(control_guidance_reliability_map, original_size)
                    if control_guidance_reliability_map is not None
                    else None
                ),
                "control_canvas_target": (
                    crop_to_size(control_canvas_target, original_size)
                    if control_canvas_target is not None
                    else None
                ),
                "control_feedback_map": crop_to_size(feedback_map, original_size) if feedback_map is not None else None,
                "decoupled_selectivity_map": (
                    crop_to_size(decoupled_selectivity_map, original_size)
                    if decoupled_selectivity_map is not None
                    else None
                ),
                "decoupled_amplitude_map": (
                    crop_to_size(decoupled_amplitude_map, original_size)
                    if decoupled_amplitude_map is not None
                    else None
                ),
                "input_std_route_delta": (
                    crop_to_size(input_std_route_delta, original_size)
                    if input_std_route_delta is not None
                    else None
                ),
                "input_std_route_high_delta": (
                    crop_to_size(input_std_route_high_delta, original_size)
                    if input_std_route_high_delta is not None
                    else None
                ),
                "input_std_route_low_delta": (
                    crop_to_size(input_std_route_low_delta, original_size)
                    if input_std_route_low_delta is not None
                    else None
                ),
                "input_std_route_gate": (
                    crop_to_size(input_std_route_gate, original_size)
                    if input_std_route_gate is not None
                    else None
                ),
                "input_std_route_prior": (
                    crop_to_size(input_std_route_prior, original_size)
                    if input_std_route_prior is not None
                    else None
                ),
                "input_std_route_learned_gate": (
                    crop_to_size(input_std_route_learned_gate, original_size)
                    if input_std_route_learned_gate is not None
                    else None
                ),
                "input_std_route_post_adapter_delta": (
                    crop_to_size(input_std_route_post_adapter_delta, original_size)
                    if input_std_route_post_adapter_delta is not None
                    else None
                ),
                "input_std_route_post_adapter_high_delta": (
                    crop_to_size(input_std_route_post_adapter_high_delta, original_size)
                    if input_std_route_post_adapter_high_delta is not None
                    else None
                ),
                "input_std_route_post_adapter_gate": (
                    crop_to_size(input_std_route_post_adapter_gate, original_size)
                    if input_std_route_post_adapter_gate is not None
                    else None
                ),
                "input_std_route_post_adapter_prior": (
                    crop_to_size(input_std_route_post_adapter_prior, original_size)
                    if input_std_route_post_adapter_prior is not None
                    else None
                ),
                "gaussian_route_control_delta": (
                    crop_to_size(gaussian_route_control_delta, original_size)
                    if gaussian_route_control_delta is not None
                    else None
                ),
                "gaussian_route_control_high_delta": (
                    crop_to_size(gaussian_route_control_high_delta, original_size)
                    if gaussian_route_control_high_delta is not None
                    else None
                ),
                "gaussian_route_control_low_delta": (
                    crop_to_size(gaussian_route_control_low_delta, original_size)
                    if gaussian_route_control_low_delta is not None
                    else None
                ),
                "gaussian_route_control_gate": (
                    crop_to_size(gaussian_route_control_gate, original_size)
                    if gaussian_route_control_gate is not None
                    else None
                ),
                "gaussian_route_control_prior": (
                    crop_to_size(gaussian_route_control_prior, original_size)
                    if gaussian_route_control_prior is not None
                    else None
                ),
                "gaussian_route_control_confidence": (
                    crop_to_size(gaussian_route_control_confidence, original_size)
                    if gaussian_route_control_confidence is not None
                    else None
                ),
                "gaussian_route_control_high_restored": (
                    crop_to_size(gaussian_route_control_high_restored, original_size)
                    if gaussian_route_control_high_restored is not None
                    else None
                ),
                "gaussian_route_control_low_restored": (
                    crop_to_size(gaussian_route_control_low_restored, original_size)
                    if gaussian_route_control_low_restored is not None
                    else None
                ),
                "input_std_route_high_restored": input_std_route_high_restored,
                "input_std_route_low_restored": input_std_route_low_restored,
            }

        return {
            "restored": restored,
            "attention_map": crop_to_size(attention_map, original_size),
            "raw_attention_map": crop_to_size(raw_attention_map, original_size),
            "control_attention_map": crop_to_size(control_attention_map, original_size),
            "attention_bridge_map": (
                crop_to_size(attention_bridge_map, original_size)
                if attention_bridge_map is not None
                else None
            ),
            "gaussian_params": head_outputs["params"],
            "affine_coeffs": head_outputs["coeffs"],
            "lowres_features": lowres_feature,
            "texture_features": texture_feature,
            "directional_prior": crop_to_size(directional_prior, original_size) if directional_prior is not None else None,
            "explicit_low_band": crop_to_size(explicit_low_band, original_size) if explicit_low_band is not None else None,
            "explicit_high_band": crop_to_size(explicit_high_band, original_size) if explicit_high_band is not None else None,
            "guided_high_detail": crop_to_size(guided_high_detail, original_size) if guided_high_detail is not None else None,
            "control_canvas": crop_to_size(control_canvas, original_size) if control_canvas is not None else None,
            "control_guidance_blend_map": (
                crop_to_size(control_guidance_blend_map, original_size)
                if control_guidance_blend_map is not None
                else None
            ),
            "control_guidance_reliability_map": (
                crop_to_size(control_guidance_reliability_map, original_size)
                if control_guidance_reliability_map is not None
                else None
            ),
            "control_canvas_target": (
                crop_to_size(control_canvas_target, original_size)
                if control_canvas_target is not None
                else None
            ),
            "control_feedback_map": crop_to_size(feedback_map, original_size) if feedback_map is not None else None,
            "decoupled_selectivity_map": (
                crop_to_size(decoupled_selectivity_map, original_size)
                if decoupled_selectivity_map is not None
                else None
            ),
            "decoupled_amplitude_map": (
                crop_to_size(decoupled_amplitude_map, original_size)
                if decoupled_amplitude_map is not None
                else None
            ),
            "decoupled_correction_map": (
                crop_to_size(decoupled_correction_map, original_size)
                if decoupled_correction_map is not None
                else None
            ),
            "input_std_route_delta": (
                crop_to_size(input_std_route_delta, original_size)
                if input_std_route_delta is not None
                else None
            ),
            "input_std_route_high_delta": (
                crop_to_size(input_std_route_high_delta, original_size)
                if input_std_route_high_delta is not None
                else None
            ),
            "input_std_route_low_delta": (
                crop_to_size(input_std_route_low_delta, original_size)
                if input_std_route_low_delta is not None
                else None
            ),
            "input_std_route_gate": (
                crop_to_size(input_std_route_gate, original_size)
                if input_std_route_gate is not None
                else None
            ),
            "input_std_route_prior": (
                crop_to_size(input_std_route_prior, original_size)
                if input_std_route_prior is not None
                else None
            ),
            "input_std_route_learned_gate": (
                crop_to_size(input_std_route_learned_gate, original_size)
                if input_std_route_learned_gate is not None
                else None
            ),
            "input_std_route_post_adapter_delta": (
                crop_to_size(input_std_route_post_adapter_delta, original_size)
                if input_std_route_post_adapter_delta is not None
                else None
            ),
            "input_std_route_post_adapter_high_delta": (
                crop_to_size(input_std_route_post_adapter_high_delta, original_size)
                if input_std_route_post_adapter_high_delta is not None
                else None
            ),
            "input_std_route_post_adapter_gate": (
                crop_to_size(input_std_route_post_adapter_gate, original_size)
                if input_std_route_post_adapter_gate is not None
                else None
            ),
            "input_std_route_post_adapter_prior": (
                crop_to_size(input_std_route_post_adapter_prior, original_size)
                if input_std_route_post_adapter_prior is not None
                else None
            ),
            "gaussian_route_control_delta": (
                crop_to_size(gaussian_route_control_delta, original_size)
                if gaussian_route_control_delta is not None
                else None
            ),
            "gaussian_route_control_high_delta": (
                crop_to_size(gaussian_route_control_high_delta, original_size)
                if gaussian_route_control_high_delta is not None
                else None
            ),
            "gaussian_route_control_low_delta": (
                crop_to_size(gaussian_route_control_low_delta, original_size)
                if gaussian_route_control_low_delta is not None
                else None
            ),
            "gaussian_route_control_gate": (
                crop_to_size(gaussian_route_control_gate, original_size)
                if gaussian_route_control_gate is not None
                else None
            ),
            "gaussian_route_control_prior": (
                crop_to_size(gaussian_route_control_prior, original_size)
                if gaussian_route_control_prior is not None
                else None
            ),
            "gaussian_route_control_confidence": (
                crop_to_size(gaussian_route_control_confidence, original_size)
                if gaussian_route_control_confidence is not None
                else None
            ),
            "gaussian_route_control_high_restored": (
                crop_to_size(gaussian_route_control_high_restored, original_size)
                if gaussian_route_control_high_restored is not None
                else None
            ),
            "gaussian_route_control_low_restored": (
                crop_to_size(gaussian_route_control_low_restored, original_size)
                if gaussian_route_control_low_restored is not None
                else None
            ),
            "input_std_route_high_restored": input_std_route_high_restored,
            "input_std_route_low_restored": input_std_route_low_restored,
        }

    def set_training_progress(self, progress: float) -> None:
        self._training_progress = max(0.0, min(1.0, float(progress)))
        if self.gaussian_control_router is not None:
            self.gaussian_control_router.set_training_progress(self._training_progress)

    @staticmethod
    def structure_canvas_target(rgb: torch.Tensor) -> torch.Tensor:
        gray = rgb.mean(dim=1, keepdim=True)
        grad_x = gray[..., :, 1:] - gray[..., :, :-1]
        grad_x = F.pad(grad_x.abs(), (0, 1, 0, 0))
        grad_y = gray[..., 1:, :] - gray[..., :-1, :]
        grad_y = F.pad(grad_y.abs(), (0, 0, 0, 1))
        local = gray - F.avg_pool2d(gray, kernel_size=5, stride=1, padding=2)
        structure = grad_x + grad_y + local.abs()
        denom = F.avg_pool2d(structure, kernel_size=15, stride=1, padding=7).detach().clamp_min(1e-4)
        return torch.clamp(structure / (structure + denom), 0.0, 1.0)
