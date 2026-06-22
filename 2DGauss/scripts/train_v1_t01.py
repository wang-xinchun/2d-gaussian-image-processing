#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
import sys
import time
import tomllib
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict

from gpu_safety import apply_cuda_visible_devices_policy

apply_cuda_visible_devices_policy("train_v1_t01.py")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uhd_gaussian import (  # noqa: E402
    CrossResolutionGaussianRestorer,
    DirectionalFrequencyGuidanceConfig,
    EdgeGuidanceConfig,
    EdgeWeightedLossConfig,
    ExplicitFrequencySplitConfig,
    GaussianAttentionDemandBridgeConfig,
    GaussianControlCanvasConfig,
    GaussianRouteControlConfig,
    DecoupledCorrectionConfig,
    GaussianDecoderGuidanceConfig,
    GaussianControlRouterConfig,
    GaussianFeatureRefinementConfig,
    InputStdRouteDecoderHeadsConfig,
    InputStdRouteFeatureExpertConfig,
    InputStdRoutePostAdapterConfig,
    InputStdRouteResidualConfig,
    InputStdRouteTerminalHeadsConfig,
    GaussianRendererConfig,
    GaussianSharedPriorConfig,
    GaussianTerminalResidualConfig,
    ModelConfig,
    TrainInferenceConsistencyConfig,
)
from uhd_gaussian.data import PairedImageDataset  # noqa: E402
from uhd_gaussian.training import (  # noqa: E402
    charbonnier_loss,
    chroma_l1_loss,
    dark_channel_l1_loss,
    edge_magnitude_l1_loss,
    gradient_l1_loss,
    high_frequency_l1_loss,
    laplacian_l1_loss,
    local_contrast_l1_loss,
    luminance_l1_loss,
    psnr,
    saturation_l1_loss,
)


def load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model_config(config: Dict[str, Any]) -> ModelConfig:
    model_cfg = config.get("model", {})
    gaussian_cfg = GaussianRendererConfig(
        atoms=int(model_cfg.get("gaussian_atoms", 4)),
        param_dim=int(model_cfg.get("gaussian_param_dim", 10)),
        patch_size=int(model_cfg.get("lowres_scale", 4)),
        max_frequency=float(model_cfg.get("max_frequency", 4.0)),
        max_texture_gain=float(model_cfg.get("max_texture_gain", 1.0)),
    )
    edge_cfg = EdgeGuidanceConfig(
        enabled=bool(model_cfg.get("use_edge_guidance", False)),
        blocks=int(model_cfg.get("edge_guidance_blocks", 2)),
        strength=float(model_cfg.get("edge_guidance_strength", 0.25)),
        use_control_gate=bool(model_cfg.get("edge_guidance_use_control_gate", False)),
    )
    directional_cfg = DirectionalFrequencyGuidanceConfig(
        enabled=bool(model_cfg.get("use_directional_frequency_guidance", False)),
    )
    explicit_split_cfg = ExplicitFrequencySplitConfig(
        enabled=bool(model_cfg.get("use_explicit_frequency_split", False)),
        lowpass_kernel_size=int(model_cfg.get("explicit_frequency_split_lowpass_kernel_size", 7)),
        highres_high_band_strength=float(
            model_cfg.get("explicit_frequency_split_highres_high_band_strength", 0.50)
        ),
        gaussian_feedback_strength=float(
            model_cfg.get("explicit_frequency_split_gaussian_feedback_strength", 0.10)
        ),
        gaussian_feedback_attention_source=str(
            model_cfg.get("explicit_frequency_split_gaussian_feedback_attention_source", "main")
        ),
        detail_blocks=int(model_cfg.get("explicit_frequency_split_detail_blocks", 1)),
        use_structure_hint=bool(model_cfg.get("explicit_frequency_split_use_structure_hint", False)),
        structure_hint_strength=float(model_cfg.get("explicit_frequency_split_structure_hint_strength", 0.25)),
    )
    consistency_cfg = TrainInferenceConsistencyConfig(
        enabled=bool(model_cfg.get("use_train_inference_consistency", False)),
        train_context_size=int(model_cfg.get("train_context_size", 0)),
        train_supervision_crop_size=int(model_cfg.get("train_supervision_crop_size", 0)),
    )
    decoder_guidance_cfg = GaussianDecoderGuidanceConfig(
        enabled=bool(model_cfg.get("use_gaussian_decoder_guidance", False)),
        blocks=int(model_cfg.get("gaussian_decoder_guidance_blocks", 1)),
        strength=float(model_cfg.get("gaussian_decoder_guidance_strength", 0.15)),
        use_decoder_state_gate=bool(model_cfg.get("gaussian_decoder_guidance_use_decoder_state_gate", False)),
    )
    shared_prior_cfg = GaussianSharedPriorConfig(
        enabled=bool(model_cfg.get("use_gaussian_shared_prior", False)),
        blocks=int(model_cfg.get("gaussian_shared_prior_blocks", 1)),
    )
    control_router_cfg = GaussianControlRouterConfig(
        enabled=bool(model_cfg.get("use_gaussian_control_router", False)),
        blocks=int(model_cfg.get("gaussian_control_router_blocks", 1)),
        use_decoder_state=bool(model_cfg.get("gaussian_control_router_use_decoder_state", True)),
        use_edge_state=bool(model_cfg.get("gaussian_control_router_use_edge_state", True)),
        role_delta_strength=float(model_cfg.get("gaussian_control_router_role_delta_strength", 0.50)),
        use_competitive_role_gate=bool(
            model_cfg.get("gaussian_control_router_use_competitive_role_gate", False)
        ),
        role_gate_temperature=float(model_cfg.get("gaussian_control_router_role_gate_temperature", 1.0)),
        use_role_residual_centering=bool(
            model_cfg.get("gaussian_control_router_use_role_residual_centering", False)
        ),
        use_demand_conditioned_competition=bool(
            model_cfg.get("gaussian_control_router_use_demand_conditioned_competition", False)
        ),
        role_competition_start=float(model_cfg.get("gaussian_control_router_role_competition_start", 0.45)),
        role_competition_strength=float(model_cfg.get("gaussian_control_router_role_competition_strength", 0.35)),
        role_competition_floor=float(model_cfg.get("gaussian_control_router_role_competition_floor", 0.30)),
    )
    terminal_residual_cfg = GaussianTerminalResidualConfig(
        enabled=bool(model_cfg.get("use_gaussian_terminal_residual", False)),
        blocks=int(model_cfg.get("gaussian_terminal_residual_blocks", 1)),
        strength=float(model_cfg.get("gaussian_terminal_residual_strength", 0.10)),
    )
    feature_refinement_cfg = GaussianFeatureRefinementConfig(
        enabled=bool(model_cfg.get("use_gaussian_feature_refinement", False)),
        blocks=int(model_cfg.get("gaussian_feature_refinement_blocks", 1)),
        strength=float(model_cfg.get("gaussian_feature_refinement_strength", 0.12)),
        use_decoder_detail_hint=bool(model_cfg.get("gaussian_feature_refinement_use_decoder_detail_hint", False)),
        use_edge_aware_prior=bool(model_cfg.get("gaussian_feature_refinement_use_edge_aware_prior", False)),
        use_confidence_normalization=bool(
            model_cfg.get("gaussian_feature_refinement_use_confidence_normalization", False)
        ),
        use_intervention_demand=bool(
            model_cfg.get("gaussian_feature_refinement_use_intervention_demand", False)
        ),
        use_decoder_detail_gate=bool(
            model_cfg.get("gaussian_feature_refinement_use_decoder_detail_gate", False)
        ),
    )
    attention_bridge_cfg = GaussianAttentionDemandBridgeConfig(
        enabled=bool(model_cfg.get("use_gaussian_attention_demand_bridge", False)),
        blocks=int(model_cfg.get("gaussian_attention_demand_bridge_blocks", 1)),
        strength=float(model_cfg.get("gaussian_attention_demand_bridge_strength", 0.20)),
        use_input_structure=bool(
            model_cfg.get("gaussian_attention_demand_bridge_use_input_structure", True)
        ),
        apply_to=str(model_cfg.get("gaussian_attention_demand_bridge_apply_to", "main")),
    )
    control_canvas_cfg = GaussianControlCanvasConfig(
        enabled=bool(model_cfg.get("use_gaussian_control_canvas", False)),
        blocks=int(model_cfg.get("gaussian_control_canvas_blocks", 1)),
        strength=float(model_cfg.get("gaussian_control_canvas_strength", 0.20)),
        stage=str(model_cfg.get("gaussian_control_canvas_stage", "post_refinement")),
        share_to_guidance=bool(model_cfg.get("gaussian_control_canvas_share_to_guidance", False)),
        share_to_guidance_blend=float(
            model_cfg.get("gaussian_control_canvas_share_to_guidance_blend", 1.0)
        ),
        share_to_guidance_detail_preserve=bool(
            model_cfg.get("gaussian_control_canvas_share_to_guidance_detail_preserve", False)
        ),
        share_to_guidance_detail_preserve_strength=float(
            model_cfg.get("gaussian_control_canvas_share_to_guidance_detail_preserve_strength", 0.0)
        ),
        share_to_guidance_detail_preserve_floor=float(
            model_cfg.get("gaussian_control_canvas_share_to_guidance_detail_preserve_floor", 0.25)
        ),
        share_to_guidance_detail_preserve_kernel_size=int(
            model_cfg.get("gaussian_control_canvas_share_to_guidance_detail_preserve_kernel_size", 5)
        ),
        share_to_guidance_detail_preserve_schedule=str(
            model_cfg.get("gaussian_control_canvas_share_to_guidance_detail_preserve_schedule", "none")
        ),
        share_to_guidance_detail_preserve_schedule_start=float(
            model_cfg.get("gaussian_control_canvas_share_to_guidance_detail_preserve_schedule_start", 0.0)
        ),
        share_to_guidance_detail_preserve_schedule_end=float(
            model_cfg.get("gaussian_control_canvas_share_to_guidance_detail_preserve_schedule_end", 1.0)
        ),
        share_to_guidance_reliability_gate=bool(
            model_cfg.get("gaussian_control_canvas_share_to_guidance_reliability_gate", False)
        ),
        share_to_guidance_reliability_strength=float(
            model_cfg.get("gaussian_control_canvas_share_to_guidance_reliability_strength", 0.0)
        ),
        share_to_guidance_reliability_floor=float(
            model_cfg.get("gaussian_control_canvas_share_to_guidance_reliability_floor", 0.50)
        ),
        share_to_guidance_reliability_kernel_size=int(
            model_cfg.get("gaussian_control_canvas_share_to_guidance_reliability_kernel_size", 7)
        ),
        share_to_guidance_reliability_schedule=str(
            model_cfg.get("gaussian_control_canvas_share_to_guidance_reliability_schedule", "none")
        ),
        share_to_guidance_reliability_schedule_start=float(
            model_cfg.get("gaussian_control_canvas_share_to_guidance_reliability_schedule_start", 0.0)
        ),
        share_to_guidance_reliability_schedule_end=float(
            model_cfg.get("gaussian_control_canvas_share_to_guidance_reliability_schedule_end", 1.0)
        ),
        use_target_conditioned_feedback=bool(
            model_cfg.get("gaussian_control_canvas_use_target_conditioned_feedback", False)
        ),
        target_mode=str(model_cfg.get("gaussian_control_canvas_target_mode", "structure")),
        feedback_mode=str(model_cfg.get("gaussian_control_canvas_feedback_mode", "multiplicative")),
        use_confidence_normalization=bool(
            model_cfg.get("gaussian_control_canvas_use_confidence_normalization", False)
        ),
        residual_delta_strength=float(model_cfg.get("gaussian_control_canvas_residual_delta_strength", 1.0)),
        feedback_selectivity_mode=str(model_cfg.get("gaussian_control_canvas_feedback_selectivity_mode", "none")),
        feedback_selectivity_temperature=float(
            model_cfg.get("gaussian_control_canvas_feedback_selectivity_temperature", 2.0)
        ),
        feedback_demand_power=float(model_cfg.get("gaussian_control_canvas_feedback_demand_power", 1.0)),
        feedback_budget=float(model_cfg.get("gaussian_control_canvas_feedback_budget", 0.0)),
        feedback_floor=float(model_cfg.get("gaussian_control_canvas_feedback_floor", 0.0)),
        feedback_target_mean=float(model_cfg.get("gaussian_control_canvas_feedback_target_mean", 0.0)),
        feedback_mean_boost_cap=float(model_cfg.get("gaussian_control_canvas_feedback_mean_boost_cap", 2.0)),
        feedback_demand_source=str(model_cfg.get("gaussian_control_canvas_feedback_demand_source", "target")),
    )
    gaussian_route_control_cfg = GaussianRouteControlConfig(
        enabled=bool(model_cfg.get("use_gaussian_route_control", False)),
        blocks=int(model_cfg.get("gaussian_route_control_blocks", 1)),
        strength=float(model_cfg.get("gaussian_route_control_strength", 0.10)),
        threshold=float(model_cfg.get("input_std_route_threshold", 0.07026463374495506)),
        softness=float(model_cfg.get("input_std_route_softness", 0.003)),
        direction=str(model_cfg.get("input_std_route_direction", "gt")),
        learned_gate_weight=float(model_cfg.get("gaussian_route_control_learned_gate_weight", 1.0)),
        detach_gate=bool(model_cfg.get("gaussian_route_control_detach_gate", True)),
        adapter_count=int(model_cfg.get("gaussian_route_control_adapter_count", 1)),
    )
    decoupled_correction_cfg = DecoupledCorrectionConfig(
        enabled=bool(model_cfg.get("use_decoupled_correction", False)),
        blocks=int(model_cfg.get("decoupled_correction_blocks", 1)),
        strength=float(model_cfg.get("decoupled_correction_strength", 0.10)),
        use_gradient_detached_selectivity=bool(
            model_cfg.get("decoupled_correction_use_gradient_detached_selectivity", True)
        ),
        amplitude_channels=int(model_cfg.get("decoupled_correction_amplitude_channels", 0)),
        amplitude_mode=str(model_cfg.get("decoupled_correction_amplitude_mode", "residual")),
    )
    input_std_route_cfg = InputStdRouteResidualConfig(
        enabled=bool(model_cfg.get("use_input_std_route_residual", False)),
        blocks=int(model_cfg.get("input_std_route_residual_blocks", 1)),
        strength=float(model_cfg.get("input_std_route_residual_strength", 0.12)),
        threshold=float(model_cfg.get("input_std_route_threshold", 0.07026463374495506)),
        softness=float(model_cfg.get("input_std_route_softness", 0.005)),
        direction=str(model_cfg.get("input_std_route_direction", "gt")),
        learned_gate_weight=float(model_cfg.get("input_std_route_learned_gate_weight", 0.0)),
        learned_gate_mode=str(model_cfg.get("input_std_route_learned_gate_mode", "blend")),
        detach_gate=bool(model_cfg.get("input_std_route_detach_gate", True)),
    )
    input_std_route_terminal_heads_cfg = InputStdRouteTerminalHeadsConfig(
        enabled=bool(model_cfg.get("use_input_std_route_terminal_heads", False)),
        blocks=int(model_cfg.get("input_std_route_terminal_heads_blocks", 2)),
        strength=float(model_cfg.get("input_std_route_terminal_heads_strength", 0.60)),
        threshold=float(model_cfg.get("input_std_route_threshold", 0.07026463374495506)),
        softness=float(model_cfg.get("input_std_route_softness", 0.005)),
        direction=str(model_cfg.get("input_std_route_direction", "gt")),
        learned_gate_weight=float(model_cfg.get("input_std_route_learned_gate_weight", 0.10)),
        learned_gate_mode=str(model_cfg.get("input_std_route_learned_gate_mode", "blend")),
        detach_gate=bool(model_cfg.get("input_std_route_detach_gate", True)),
    )
    input_std_route_feature_expert_cfg = InputStdRouteFeatureExpertConfig(
        enabled=bool(model_cfg.get("use_input_std_route_feature_expert", False)),
        blocks=int(model_cfg.get("input_std_route_feature_expert_blocks", 2)),
        strength=float(model_cfg.get("input_std_route_feature_expert_strength", 0.35)),
        threshold=float(model_cfg.get("input_std_route_threshold", 0.07026463374495506)),
        softness=float(model_cfg.get("input_std_route_softness", 0.005)),
        direction=str(model_cfg.get("input_std_route_direction", "gt")),
        learned_gate_weight=float(model_cfg.get("input_std_route_learned_gate_weight", 0.05)),
        learned_gate_mode=str(model_cfg.get("input_std_route_learned_gate_mode", "blend")),
        detach_gate=bool(model_cfg.get("input_std_route_detach_gate", True)),
    )
    input_std_route_decoder_heads_cfg = InputStdRouteDecoderHeadsConfig(
        enabled=bool(model_cfg.get("use_input_std_route_decoder_heads", False)),
        blocks=int(model_cfg.get("input_std_route_decoder_heads_blocks", 1)),
        strength=float(model_cfg.get("input_std_route_decoder_heads_strength", 1.0)),
        threshold=float(model_cfg.get("input_std_route_threshold", 0.07026463374495506)),
        softness=float(model_cfg.get("input_std_route_softness", 0.005)),
        direction=str(model_cfg.get("input_std_route_direction", "gt")),
        learned_gate_weight=float(model_cfg.get("input_std_route_learned_gate_weight", 0.02)),
        learned_gate_mode=str(model_cfg.get("input_std_route_learned_gate_mode", "blend")),
        detach_gate=bool(model_cfg.get("input_std_route_detach_gate", True)),
    )
    input_std_route_post_adapter_cfg = InputStdRoutePostAdapterConfig(
        enabled=bool(model_cfg.get("use_input_std_route_post_adapter", False)),
        blocks=int(model_cfg.get("input_std_route_post_adapter_blocks", 1)),
        strength=float(model_cfg.get("input_std_route_post_adapter_strength", 0.08)),
        threshold=float(model_cfg.get("input_std_route_post_adapter_threshold", model_cfg.get("input_std_route_threshold", 0.07026463374495506))),
        softness=float(model_cfg.get("input_std_route_post_adapter_softness", model_cfg.get("input_std_route_softness", 0.003))),
        direction=str(model_cfg.get("input_std_route_post_adapter_direction", model_cfg.get("input_std_route_direction", "gt"))),
        detach_gate=bool(model_cfg.get("input_std_route_post_adapter_detach_gate", model_cfg.get("input_std_route_detach_gate", True))),
    )
    return ModelConfig(
        in_channels=int(model_cfg.get("in_channels", 3)),
        out_channels=int(model_cfg.get("out_channels", 3)),
        width=int(model_cfg.get("width", 32)),
        lowres_scale=int(model_cfg.get("lowres_scale", 4)),
        highres_blocks=int(model_cfg.get("highres_blocks", 2)),
        lowres_blocks=int(model_cfg.get("lowres_blocks", 4)),
        decoder_blocks=int(model_cfg.get("decoder_blocks", 2)),
        affine_rank=int(model_cfg.get("affine_rank", 4)),
        gaussian=gaussian_cfg,
        edge_guidance=edge_cfg,
        directional_frequency_guidance=directional_cfg,
        explicit_frequency_split=explicit_split_cfg,
        train_inference_consistency=consistency_cfg,
        gaussian_decoder_guidance=decoder_guidance_cfg,
        gaussian_shared_prior=shared_prior_cfg,
        gaussian_control_router=control_router_cfg,
        gaussian_terminal_residual=terminal_residual_cfg,
        gaussian_feature_refinement=feature_refinement_cfg,
        gaussian_attention_demand_bridge=attention_bridge_cfg,
        gaussian_control_canvas=control_canvas_cfg,
        gaussian_route_control=gaussian_route_control_cfg,
        decoupled_correction=decoupled_correction_cfg,
        input_std_route_residual=input_std_route_cfg,
        input_std_route_terminal_heads=input_std_route_terminal_heads_cfg,
        input_std_route_feature_expert=input_std_route_feature_expert_cfg,
        input_std_route_decoder_heads=input_std_route_decoder_heads_cfg,
        input_std_route_post_adapter=input_std_route_post_adapter_cfg,
    )


def cuda_memory_fraction_from_env() -> float | None:
    raw = os.environ.get("TWODGAUSS_CUDA_MEMORY_FRACTION", "").strip()
    if not raw:
        return None
    try:
        fraction = float(raw)
    except ValueError as exc:
        raise ValueError(f"TWODGAUSS_CUDA_MEMORY_FRACTION must be numeric, got {raw!r}") from exc
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"TWODGAUSS_CUDA_MEMORY_FRACTION must be in (0, 1], got {fraction}")
    return fraction


def set_cuda_memory_fraction(device: torch.device | int) -> float | None:
    fraction = cuda_memory_fraction_from_env()
    if fraction is None:
        return None
    if isinstance(device, torch.device) and device.type == "cuda" and device.index is None:
        device = torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)
    return fraction


def init_distributed_if_needed() -> Dict[str, int | bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    state: Dict[str, int | bool] = {
        "enabled": False,
        "rank": 0,
        "local_rank": 0,
        "world_size": 1,
    }
    if world_size <= 1:
        return state
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA on this project.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    set_cuda_memory_fraction(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    state.update(
        {
            "enabled": True,
            "rank": dist.get_rank(),
            "local_rank": local_rank,
            "world_size": dist.get_world_size(),
        }
    )
    return state


def cleanup_distributed(distributed: Dict[str, int | bool]) -> None:
    if bool(distributed.get("enabled", False)) and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(distributed: Dict[str, int | bool]) -> bool:
    return int(distributed.get("rank", 0)) == 0


def distributed_barrier(distributed: Dict[str, int | bool]) -> None:
    if bool(distributed.get("enabled", False)):
        dist.barrier()


def choose_device(config: Dict[str, Any], distributed: Dict[str, int | bool] | None = None) -> torch.device:
    if distributed is not None and bool(distributed.get("enabled", False)):
        return torch.device("cuda", int(distributed.get("local_rank", 0)))
    requested = str(config.get("training", {}).get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def apply_cuda_memory_fraction(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return set_cuda_memory_fraction(device)


def cuda_memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    query_device: torch.device | int = device
    if device.index is None:
        query_device = torch.cuda.current_device()
    torch.cuda.synchronize(query_device)
    return {
        "cuda_memory_allocated_gb": torch.cuda.memory_allocated(query_device) / 1024**3,
        "cuda_memory_reserved_gb": torch.cuda.memory_reserved(query_device) / 1024**3,
        "cuda_max_memory_allocated_gb": torch.cuda.max_memory_allocated(query_device) / 1024**3,
        "cuda_max_memory_reserved_gb": torch.cuda.max_memory_reserved(query_device) / 1024**3,
    }


def make_datasets(config: Dict[str, Any], train_limit_override: int | None = None) -> tuple[PairedImageDataset, PairedImageDataset]:
    dataset_cfg = config["dataset"]
    model_cfg = config.get("model", {})
    consistency_enabled = bool(model_cfg.get("use_train_inference_consistency", False))
    train_context_size = int(model_cfg.get("train_context_size", 0)) if consistency_enabled else 0
    train_supervision_crop_size = int(model_cfg.get("train_supervision_crop_size", 0)) if consistency_enabled else 0
    training_cfg = config.get("training", {})
    route_crop_threshold = None
    route_crop_attempts = 0
    if bool(training_cfg.get("input_std_route_balanced_crop_sampling", False)):
        route_crop_threshold = float(
            training_cfg.get(
                "input_std_route_balanced_threshold",
                model_cfg.get("input_std_route_threshold", 0.07026463374495506),
            )
        )
        route_crop_attempts = int(training_cfg.get("input_std_route_balanced_crop_attempts", 8))
    train_limit = int(dataset_cfg.get("train_limit", 0))
    if train_limit_override is not None:
        train_limit = train_limit_override
    val_limit = int(dataset_cfg.get("val_limit", 0))
    train_dataset = PairedImageDataset(
        dataset_cfg["train_input"],
        dataset_cfg["train_target"],
        crop_size=int(dataset_cfg.get("train_crop_size", 256)),
        training=True,
        augment_flip=bool(dataset_cfg.get("augment_flip", True)),
        offset=int(dataset_cfg.get("train_offset", 0)),
        limit=train_limit,
        context_crop_size=train_context_size,
        supervision_crop_size=train_supervision_crop_size,
        route_crop_threshold=route_crop_threshold,
        route_crop_attempts=route_crop_attempts,
    )
    val_dataset = PairedImageDataset(
        dataset_cfg["val_input"],
        dataset_cfg["val_target"],
        crop_size=int(dataset_cfg.get("val_crop_size", 512)),
        training=False,
        augment_flip=False,
        offset=int(dataset_cfg.get("val_offset", 0)),
        limit=val_limit,
    )
    return train_dataset, val_dataset


def make_transfer_validation_dataset(config: Dict[str, Any]) -> PairedImageDataset:
    dataset_cfg = config["dataset"]
    training_cfg = config.get("training", {})
    return PairedImageDataset(
        dataset_cfg["val_input"],
        dataset_cfg["val_target"],
        crop_size=int(training_cfg.get("transfer_validation_crop_size", dataset_cfg.get("val_crop_size", 512))),
        training=False,
        augment_flip=False,
        offset=int(training_cfg.get("transfer_validation_offset", 0)),
        limit=int(training_cfg.get("transfer_validation_limit", 0)),
    )


def input_std_route_balanced_sampling_enabled(config: Dict[str, Any]) -> bool:
    training_cfg = config.get("training", {})
    return bool(training_cfg.get("input_std_route_balanced_sampling", False))


def _sample_input_std_for_sampler(path: Path, max_side: int = 128) -> float:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
        if max_side > 0 and array.ndim >= 2:
            height, width = array.shape[:2]
            step = max(1, int(np.ceil(max(height, width) / float(max_side))))
            array = array[::step, ::step]
        if array.dtype == np.uint16:
            values = array.astype(np.float32) / 65535.0
        elif array.dtype == np.uint8:
            values = array.astype(np.float32) / 255.0
        else:
            values = array.astype(np.float32)
            if values.max(initial=0.0) > 1.0:
                values = values / 65535.0
        return float(np.clip(values, 0.0, 1.0).std())
    with Image.open(path) as image:
        image = image.convert("RGB")
        if max_side > 0:
            image.thumbnail((max_side, max_side))
        values = np.asarray(image, dtype=np.float32) / 255.0
    return float(values.std())


class InputStdRouteBalancedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: PairedImageDataset,
        *,
        batch_size: int,
        threshold: float,
        drop_last: bool,
        seed: int,
        high_fraction: float = 0.5,
        stats_max_side: int = 128,
        cache_path: Path | None = None,
        single_sample_probabilistic: bool = False,
        hard_case_cache_path: Path | None = None,
        hard_case_fraction: float = 0.0,
        hard_case_route_label: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 < high_fraction < 1.0:
            raise ValueError("input_std_route_balanced_high_fraction must be in (0, 1)")
        if not 0.0 <= hard_case_fraction <= 1.0:
            raise ValueError("input_std_route_balanced_hard_case_fraction must be in [0, 1]")
        if hard_case_route_label not in (0, 1):
            raise ValueError("input_std_route_balanced_hard_case_route_label must be 0 or 1")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.threshold = float(threshold)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.high_fraction = float(high_fraction)
        self.stats_max_side = int(stats_max_side)
        self.cache_path = cache_path
        self.single_sample_probabilistic = bool(single_sample_probabilistic)
        self.hard_case_cache_path = hard_case_cache_path
        self.hard_case_fraction = float(hard_case_fraction)
        self.hard_case_route_label = int(hard_case_route_label)
        self.epoch = 0
        self.high_count = max(1, min(self.batch_size - 1, round(self.batch_size * self.high_fraction)))
        self.low_count = self.batch_size - self.high_count

        sample_stds = self._load_or_compute_sample_stds()
        high_indices: list[int] = []
        low_indices: list[int] = []
        for index, value in enumerate(sample_stds):
            if value >= self.threshold:
                high_indices.append(index)
            else:
                low_indices.append(index)
        if not high_indices or not low_indices:
            raise ValueError(
                "input_std_route_balanced_sampling requires both high and low samples; "
                f"got high={len(high_indices)} low={len(low_indices)} threshold={self.threshold}"
            )
        self.high_indices = high_indices
        self.low_indices = low_indices
        self.sample_stds = sample_stds
        self.hard_case_indices = self._load_hard_case_indices()

    def _cache_metadata(self) -> Dict[str, Any]:
        return {
            "sample_count": len(self.dataset.samples),
            "stats_max_side": self.stats_max_side,
            "input_dir": str(self.dataset.input_dir),
            "sample_names": [sample.name for sample in self.dataset.samples],
        }

    def _load_or_compute_sample_stds(self) -> list[float]:
        metadata = self._cache_metadata()
        if self.cache_path is not None and self.cache_path.is_file():
            try:
                cached = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cached = {}
            if (
                cached.get("metadata") == metadata
                and isinstance(cached.get("sample_stds"), list)
                and len(cached["sample_stds"]) == len(self.dataset.samples)
            ):
                return [float(value) for value in cached["sample_stds"]]

        sample_stds = [
            _sample_input_std_for_sampler(sample.input_path, max_side=self.stats_max_side)
            for sample in self.dataset.samples
        ]
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(
                    {
                        "metadata": metadata,
                        "sample_stds": sample_stds,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        return sample_stds

    def _load_hard_case_indices(self) -> list[int]:
        if self.hard_case_fraction <= 0.0:
            return []
        if self.hard_case_cache_path is None:
            raise ValueError(
                "input_std_route_balanced_hard_case_fraction > 0 requires "
                "input_std_route_balanced_hard_case_cache"
            )
        if not self.hard_case_cache_path.is_file():
            raise FileNotFoundError(
                f"input_std_route_balanced_hard_case_cache does not exist: {self.hard_case_cache_path}"
            )
        payload = json.loads(self.hard_case_cache_path.read_text(encoding="utf-8"))
        raw_names = payload.get("hard_case_names")
        if raw_names is None and isinstance(payload.get("hard_records"), list):
            raw_names = [record.get("name") for record in payload["hard_records"]]
        raw_indices = payload.get("hard_case_indices")

        name_to_index = {sample.name: index for index, sample in enumerate(self.dataset.samples)}
        indices: list[int] = []
        if isinstance(raw_names, list):
            missing_names: list[str] = []
            for name in raw_names:
                if not isinstance(name, str):
                    continue
                index = name_to_index.get(name)
                if index is None:
                    missing_names.append(name)
                    continue
                indices.append(index)
            if missing_names and len(missing_names) == len(raw_names):
                raise ValueError(
                    "input_std_route_balanced_hard_case_cache names do not overlap the training dataset"
                )
        elif isinstance(raw_indices, list):
            for value in raw_indices:
                index = int(value)
                if 0 <= index < len(self.dataset.samples):
                    indices.append(index)

        unique_indices = sorted(set(indices))
        if not unique_indices:
            raise ValueError("input_std_route_balanced_hard_case_cache produced zero usable hard cases")
        return unique_indices

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def _draw(self, pool: list[int], count: int, rng: random.Random) -> list[int]:
        if count <= len(pool):
            return rng.sample(pool, count)
        return [rng.choice(pool) for _ in range(count)]

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        for _ in range(len(self)):
            if self.single_sample_probabilistic and self.batch_size == 1:
                if self.hard_case_indices and rng.random() < self.hard_case_fraction:
                    batch = [(self._draw(self.hard_case_indices, 1, rng)[0], self.hard_case_route_label)]
                else:
                    use_high = rng.random() < self.high_fraction
                    pool = self.high_indices if use_high else self.low_indices
                    label = 1 if use_high else 0
                    batch = [(self._draw(pool, 1, rng)[0], label)]
            else:
                batch = [(index, 1) for index in self._draw(self.high_indices, self.high_count, rng)]
                batch.extend((index, 0) for index in self._draw(self.low_indices, self.low_count, rng))
            rng.shuffle(batch)
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def info(self) -> Dict[str, Any]:
        values = np.asarray(self.sample_stds, dtype=np.float32)
        return {
            "input_std_route_balanced_sampling": True,
            "input_std_route_balanced_threshold": self.threshold,
            "input_std_route_balanced_batch_size": self.batch_size,
            "input_std_route_balanced_high_per_batch": self.high_count,
            "input_std_route_balanced_low_per_batch": self.low_count,
            "input_std_route_balanced_single_sample_probabilistic": self.single_sample_probabilistic,
            "input_std_route_balanced_single_sample_high_probability": (
                self.high_fraction if self.single_sample_probabilistic and self.batch_size == 1 else 0.0
            ),
            "input_std_route_balanced_hard_case_cache": (
                str(self.hard_case_cache_path) if self.hard_case_cache_path is not None else ""
            ),
            "input_std_route_balanced_hard_case_fraction": self.hard_case_fraction,
            "input_std_route_balanced_hard_case_route_label": self.hard_case_route_label,
            "input_std_route_balanced_hard_case_samples": len(self.hard_case_indices),
            "input_std_route_balanced_high_samples": len(self.high_indices),
            "input_std_route_balanced_low_samples": len(self.low_indices),
            "input_std_route_balanced_input_std_min": float(values.min()),
            "input_std_route_balanced_input_std_mean": float(values.mean()),
            "input_std_route_balanced_input_std_max": float(values.max()),
        }


def make_train_batch_sampler(
    dataset: PairedImageDataset,
    config: Dict[str, Any],
    distributed: Dict[str, int | bool] | None = None,
) -> tuple[Sampler[list[int]] | None, Dict[str, Any]]:
    if not input_std_route_balanced_sampling_enabled(config):
        return None, {"input_std_route_balanced_sampling": False}
    if distributed is not None and bool(distributed.get("enabled", False)):
        raise ValueError("input_std_route_balanced_sampling is currently single-process only")
    training_cfg = config["training"]
    model_cfg = config.get("model", {})
    threshold = float(
        training_cfg.get(
            "input_std_route_balanced_threshold",
            model_cfg.get("input_std_route_threshold", 0.07026463374495506),
        )
    )
    sampler = InputStdRouteBalancedBatchSampler(
        dataset,
        batch_size=int(training_cfg.get("batch_size", 1)),
        threshold=threshold,
        drop_last=True,
        seed=int(training_cfg.get("seed", 20260424)),
        high_fraction=float(training_cfg.get("input_std_route_balanced_high_fraction", 0.5)),
        stats_max_side=int(training_cfg.get("input_std_route_balanced_stats_max_side", 128)),
        cache_path=(
            ROOT
            / str(
                training_cfg.get(
                    "input_std_route_balanced_cache",
                    "experiments/V1/runs/.cache/input_std_route_balanced_sampler.json",
                )
            )
        ),
        single_sample_probabilistic=bool(
            training_cfg.get("input_std_route_balanced_single_sample_probabilistic", False)
        ),
        hard_case_cache_path=(
            (
                Path(str(training_cfg.get("input_std_route_balanced_hard_case_cache"))).resolve()
                if Path(str(training_cfg.get("input_std_route_balanced_hard_case_cache"))).is_absolute()
                else ROOT / str(training_cfg.get("input_std_route_balanced_hard_case_cache"))
            )
            if training_cfg.get("input_std_route_balanced_hard_case_cache") is not None
            else None
        ),
        hard_case_fraction=float(training_cfg.get("input_std_route_balanced_hard_case_fraction", 0.0)),
        hard_case_route_label=int(training_cfg.get("input_std_route_balanced_hard_case_route_label", 0)),
    )
    return sampler, sampler.info()


def make_loader(
    dataset: PairedImageDataset,
    config: Dict[str, Any],
    training: bool,
    sampler: DistributedSampler | None = None,
    batch_sampler: Sampler[list[int]] | None = None,
) -> DataLoader:
    training_cfg = config["training"]
    dataset_cfg = config["dataset"]
    if batch_sampler is not None:
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=int(dataset_cfg.get("num_workers", 2)),
            pin_memory=torch.cuda.is_available(),
        )
    return DataLoader(
        dataset,
        batch_size=int(training_cfg.get("batch_size", 1)) if training else 1,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=int(dataset_cfg.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
        drop_last=training,
    )


TASK_AWARE_LOSS_WEIGHT_CAP = 0.10
TASK_AWARE_LOSS_TOTAL_CAP = 0.25
TASK_AWARE_KERNEL_CAP = 63
CONTROL_CANVAS_LOSS_WEIGHT_CAP = 0.10
GAUSSIAN_ATTENTION_LOSS_WEIGHT_CAP = 0.05
GAUSSIAN_ATTENTION_BRIDGE_LOSS_WEIGHT_CAP = 0.10


def _nonnegative_loss_value(
    loss_cfg: Dict[str, Any],
    key: str,
    default: float,
    *,
    max_value: float | None = None,
) -> float:
    value = float(loss_cfg.get(key, default))
    if value < 0.0:
        raise ValueError(f"{key} must be non-negative, got {value}")
    if max_value is not None and value > max_value and not bool(
        loss_cfg.get("allow_high_task_aware_loss_weights", False)
    ):
        raise ValueError(
            f"{key}={value} exceeds the conservative cap {max_value}; "
            "set allow_high_task_aware_loss_weights=true only for an explicit diagnostic experiment"
        )
    return value


def _positive_kernel_size(loss_cfg: Dict[str, Any], key: str, default: int) -> int:
    value = int(loss_cfg.get(key, default))
    if value <= 0:
        raise ValueError(f"{key} must be positive, got {value}")
    if value > TASK_AWARE_KERNEL_CAP and not bool(loss_cfg.get("allow_large_task_aware_kernels", False)):
        raise ValueError(
            f"{key}={value} exceeds the conservative cap {TASK_AWARE_KERNEL_CAP}; "
            "set allow_large_task_aware_kernels=true only for an explicit diagnostic experiment"
        )
    return value


def compute_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    config: Dict[str, Any],
    *,
    input_image: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    loss_cfg = config.get("loss", {})
    charbonnier_weight = _nonnegative_loss_value(loss_cfg, "charbonnier_weight", 1.0)
    mse_weight = _nonnegative_loss_value(loss_cfg, "mse_weight", 0.0)
    low_demand_identity_weight = _nonnegative_loss_value(
        loss_cfg,
        "low_demand_identity_weight",
        0.0,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    low_demand_identity_threshold = _nonnegative_loss_value(
        loss_cfg,
        "low_demand_identity_threshold",
        0.04,
    )
    low_demand_identity_softness = float(loss_cfg.get("low_demand_identity_softness", 0.02))
    if low_demand_identity_softness <= 0.0:
        raise ValueError("low_demand_identity_softness must be positive")
    gradient_weight = _nonnegative_loss_value(loss_cfg, "gradient_weight", 0.0)
    laplacian_weight = _nonnegative_loss_value(loss_cfg, "laplacian_weight", 0.0)
    edge_magnitude_weight = _nonnegative_loss_value(
        loss_cfg,
        "edge_magnitude_weight",
        0.0,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    high_frequency_weight = _nonnegative_loss_value(
        loss_cfg,
        "high_frequency_weight",
        0.0,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    high_frequency_kernel = _positive_kernel_size(loss_cfg, "high_frequency_kernel_size", 5)
    local_contrast_weight = _nonnegative_loss_value(
        loss_cfg,
        "local_contrast_weight",
        0.0,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    local_contrast_kernel = _positive_kernel_size(loss_cfg, "local_contrast_kernel_size", 7)
    luminance_weight = _nonnegative_loss_value(
        loss_cfg,
        "luminance_weight",
        0.0,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    chroma_weight = _nonnegative_loss_value(
        loss_cfg,
        "chroma_weight",
        0.0,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    saturation_weight = _nonnegative_loss_value(
        loss_cfg,
        "saturation_weight",
        0.0,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    dark_channel_weight = _nonnegative_loss_value(
        loss_cfg,
        "dark_channel_weight",
        0.0,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    dark_channel_kernel = _positive_kernel_size(loss_cfg, "dark_channel_kernel_size", 15)
    task_aware_total_weight = (
        edge_magnitude_weight
        + high_frequency_weight
        + local_contrast_weight
        + luminance_weight
        + chroma_weight
        + saturation_weight
        + dark_channel_weight
    )
    if charbonnier_weight <= 0.0 and not bool(loss_cfg.get("allow_proxy_only_loss", False)):
        raise ValueError("charbonnier_weight must stay positive unless allow_proxy_only_loss=true")
    if task_aware_total_weight > TASK_AWARE_LOSS_TOTAL_CAP and not bool(
        loss_cfg.get("allow_high_task_aware_loss_weights", False)
    ):
        raise ValueError(
            f"task-aware loss weights sum to {task_aware_total_weight}, "
            f"exceeding the conservative cap {TASK_AWARE_LOSS_TOTAL_CAP}; "
            "reduce proxy weights or set allow_high_task_aware_loss_weights=true for a diagnostic-only run"
        )
    edge_weighted_cfg = EdgeWeightedLossConfig(
        enabled=bool(loss_cfg.get("use_edge_weighted_reconstruction", False)),
        strength=_nonnegative_loss_value(loss_cfg, "edge_weighted_reconstruction_strength", 0.0),
        floor=_nonnegative_loss_value(loss_cfg, "edge_weighted_reconstruction_floor", 1.0),
    )
    reconstruction = charbonnier_loss(prediction, target)
    mse = F.mse_loss(prediction, target) if mse_weight > 0.0 else prediction.new_tensor(0.0)
    low_demand_identity = prediction.new_tensor(0.0)
    low_demand_mask_mean = 0.0
    low_demand_value_mean = 0.0
    if low_demand_identity_weight > 0.0:
        if input_image is None:
            raise ValueError("low_demand_identity_weight requires compute_loss(..., input_image=inputs)")
        effective_input = input_image
        if effective_input.shape[-2:] != prediction.shape[-2:]:
            effective_input = center_crop_like(effective_input, prediction.shape[-2:])
        demand = (effective_input - target).abs().mean(dim=1, keepdim=True).detach()
        low_demand_mask = torch.sigmoid((low_demand_identity_threshold - demand) / low_demand_identity_softness)
        identity_diff = torch.sqrt((prediction - effective_input).square() + 1e-6)
        denom = (low_demand_mask.sum() * identity_diff.shape[1]).clamp_min(1.0)
        low_demand_identity = (identity_diff * low_demand_mask).sum() / denom
        low_demand_mask_mean = float(low_demand_mask.mean().detach().cpu())
        low_demand_value_mean = float(demand.mean().detach().cpu())
    gradient = gradient_l1_loss(prediction, target) if gradient_weight > 0.0 else prediction.new_tensor(0.0)
    laplacian = laplacian_l1_loss(prediction, target) if laplacian_weight > 0.0 else prediction.new_tensor(0.0)
    edge_magnitude = (
        edge_magnitude_l1_loss(prediction, target) if edge_magnitude_weight > 0.0 else prediction.new_tensor(0.0)
    )
    high_frequency = (
        high_frequency_l1_loss(prediction, target, high_frequency_kernel)
        if high_frequency_weight > 0.0
        else prediction.new_tensor(0.0)
    )
    local_contrast = (
        local_contrast_l1_loss(prediction, target, local_contrast_kernel)
        if local_contrast_weight > 0.0
        else prediction.new_tensor(0.0)
    )
    luminance_term = luminance_l1_loss(prediction, target) if luminance_weight > 0.0 else prediction.new_tensor(0.0)
    chroma = chroma_l1_loss(prediction, target) if chroma_weight > 0.0 else prediction.new_tensor(0.0)
    saturation = saturation_l1_loss(prediction, target) if saturation_weight > 0.0 else prediction.new_tensor(0.0)
    dark_channel = (
        dark_channel_l1_loss(prediction, target, dark_channel_kernel)
        if dark_channel_weight > 0.0
        else prediction.new_tensor(0.0)
    )
    edge_weighted = prediction.new_tensor(0.0)
    if edge_weighted_cfg.enabled and edge_weighted_cfg.strength > 0.0:
        gray = target.mean(dim=1, keepdim=True)
        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            device=target.device,
            dtype=target.dtype,
        ).unsqueeze(0)
        sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
            device=target.device,
            dtype=target.dtype,
        ).unsqueeze(0)
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        edge_mag = torch.sqrt(torch.clamp(grad_x.square() + grad_y.square(), min=1e-8))
        weights = edge_weighted_cfg.floor + edge_mag / edge_mag.detach().mean().clamp_min(1e-6)
        diff = torch.sqrt((prediction - target).square() + 1e-6)
        edge_weighted = (diff * weights).mean()
    total = (
        charbonnier_weight * reconstruction
        + mse_weight * mse
        + low_demand_identity_weight * low_demand_identity
        + gradient_weight * gradient
        + laplacian_weight * laplacian
        + edge_magnitude_weight * edge_magnitude
        + high_frequency_weight * high_frequency
        + local_contrast_weight * local_contrast
        + luminance_weight * luminance_term
        + chroma_weight * chroma
        + saturation_weight * saturation
        + dark_channel_weight * dark_channel
        + edge_weighted_cfg.strength * edge_weighted
    )
    return total, {
        "loss_total": float(total.detach().cpu()),
        "loss_charbonnier": float(reconstruction.detach().cpu()),
        "loss_mse": float(mse.detach().cpu()),
        "loss_low_demand_identity": float(low_demand_identity.detach().cpu()),
        "low_demand_identity_mask_mean": low_demand_mask_mean,
        "low_demand_identity_demand_mean": low_demand_value_mean,
        "loss_gradient": float(gradient.detach().cpu()),
        "loss_laplacian": float(laplacian.detach().cpu()),
        "loss_edge_magnitude": float(edge_magnitude.detach().cpu()),
        "loss_high_frequency": float(high_frequency.detach().cpu()),
        "loss_local_contrast": float(local_contrast.detach().cpu()),
        "loss_luminance": float(luminance_term.detach().cpu()),
        "loss_chroma": float(chroma.detach().cpu()),
        "loss_saturation": float(saturation.detach().cpu()),
        "loss_dark_channel": float(dark_channel.detach().cpu()),
        "loss_edge_weighted": float(edge_weighted.detach().cpu()),
    }


def compute_auxiliary_loss(
    aux: Dict[str, torch.Tensor | None],
    target: torch.Tensor,
    config: Dict[str, Any],
    inputs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    loss_cfg = config.get("loss", {})
    canvas_weight = _nonnegative_loss_value(
        loss_cfg,
        "control_canvas_weight",
        0.0,
        max_value=CONTROL_CANVAS_LOSS_WEIGHT_CAP,
    )
    attention_weight = _nonnegative_loss_value(
        loss_cfg,
        "gaussian_attention_weight",
        0.0,
        max_value=GAUSSIAN_ATTENTION_LOSS_WEIGHT_CAP,
    )
    bridge_weight = _nonnegative_loss_value(
        loss_cfg,
        "attention_bridge_weight",
        0.0,
        max_value=GAUSSIAN_ATTENTION_BRIDGE_LOSS_WEIGHT_CAP,
    )
    if canvas_weight <= 0.0 and attention_weight <= 0.0 and bridge_weight <= 0.0:
        reference = aux.get("restored")
        if reference is None:
            raise ValueError("auxiliary loss requires aux['restored'] for zero tensor creation")
        zero = reference.new_tensor(0.0)
        return zero, {
            "loss_control_canvas": 0.0,
            "loss_gaussian_attention": 0.0,
            "loss_attention_bridge": 0.0,
        }
    if canvas_weight > 0.0 and aux.get("control_canvas") is None:
        raise ValueError("control_canvas_weight requires model use_gaussian_control_canvas=true")
    if attention_weight > 0.0 and aux.get("attention_map") is None:
        raise ValueError("gaussian_attention_weight requires aux['attention_map']")
    if bridge_weight > 0.0 and aux.get("attention_bridge_map") is None:
        raise ValueError("attention_bridge_weight requires model use_gaussian_attention_demand_bridge=true")
    supervision_size = int(config.get("model", {}).get("train_supervision_crop_size", 0))
    consistency_enabled = bool(config.get("model", {}).get("use_train_inference_consistency", False))
    target_for_canvas = target
    inputs_for_canvas = inputs
    canvas = aux.get("control_canvas")
    attention = aux.get("attention_map")
    bridge = aux.get("attention_bridge_map")
    if consistency_enabled and supervision_size > 0:
        target_for_canvas = center_crop_like(target, (supervision_size, supervision_size))
        if inputs_for_canvas is not None:
            inputs_for_canvas = center_crop_like(inputs_for_canvas, (supervision_size, supervision_size))
        if canvas is not None:
            canvas = center_crop_like(canvas, (supervision_size, supervision_size))
        if attention is not None:
            attention = center_crop_like(attention, (supervision_size, supervision_size))
        if bridge is not None:
            bridge = center_crop_like(bridge, (supervision_size, supervision_size))
    canvas_target = compute_control_canvas_target(target_for_canvas, config, inputs=inputs_for_canvas)
    total = canvas_target.new_tensor(0.0)
    parts = {
        "loss_control_canvas": 0.0,
        "loss_gaussian_attention": 0.0,
        "loss_attention_bridge": 0.0,
    }
    if canvas_weight > 0.0:
        canvas_target_for_loss = canvas_target
        if canvas is None:
            raise ValueError("control_canvas_weight requires model use_gaussian_control_canvas=true")
        if canvas.shape[-2:] != canvas_target_for_loss.shape[-2:]:
            canvas_target_for_loss = F.interpolate(
                canvas_target_for_loss,
                size=canvas.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        canvas_loss = F.l1_loss(canvas, canvas_target_for_loss.detach())
        total = total + canvas_weight * canvas_loss
        parts["loss_control_canvas"] = float(canvas_loss.detach().cpu())
    if attention_weight > 0.0:
        attention_target = canvas_target
        if attention is None:
            raise ValueError("gaussian_attention_weight requires aux['attention_map']")
        if attention.shape[-2:] != attention_target.shape[-2:]:
            attention_target = F.interpolate(
                attention_target,
                size=attention.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        attention_loss = F.l1_loss(attention, attention_target.detach())
        total = total + attention_weight * attention_loss
        parts["loss_gaussian_attention"] = float(attention_loss.detach().cpu())
    if bridge_weight > 0.0:
        bridge_target = canvas_target
        if bridge is None:
            raise ValueError("attention_bridge_weight requires aux['attention_bridge_map']")
        if bridge.shape[-2:] != bridge_target.shape[-2:]:
            bridge_target = F.interpolate(
                bridge_target,
                size=bridge.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        bridge_loss = F.l1_loss(bridge, bridge_target.detach())
        total = total + bridge_weight * bridge_loss
        parts["loss_attention_bridge"] = float(bridge_loss.detach().cpu())
    return total, parts


def compute_control_canvas_target(
    target: torch.Tensor,
    config: Dict[str, Any],
    *,
    inputs: torch.Tensor | None = None,
) -> torch.Tensor:
    model_cfg = config.get("model", {})
    target_mode = str(model_cfg.get("gaussian_control_canvas_target_mode", "structure"))
    if target_mode == "structure":
        return CrossResolutionGaussianRestorer.structure_canvas_target(target)
    if target_mode != "restoration_demand":
        raise ValueError("gaussian_control_canvas_target_mode must be structure or restoration_demand")
    if inputs is None:
        raise ValueError("restoration_demand canvas target requires input images")
    return restoration_demand_canvas_target(inputs, target)


def restoration_demand_canvas_target(inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    input_structure = CrossResolutionGaussianRestorer.structure_canvas_target(inputs)
    target_structure = CrossResolutionGaussianRestorer.structure_canvas_target(target)
    structure_gap = (target_structure - input_structure).abs()
    input_high = inputs - F.avg_pool2d(inputs, kernel_size=5, stride=1, padding=2)
    target_high = target - F.avg_pool2d(target, kernel_size=5, stride=1, padding=2)
    high_gap = (target_high - input_high).abs().mean(dim=1, keepdim=True)
    color_gap = (target - inputs).abs().mean(dim=1, keepdim=True)
    demand = 0.50 * structure_gap + 0.35 * high_gap + 0.15 * color_gap
    local_mean = F.avg_pool2d(demand, kernel_size=15, stride=1, padding=7).detach().clamp_min(1e-4)
    return torch.clamp(demand / (demand + local_mean), 0.0, 1.0)


def control_diagnostics_enabled(config: Dict[str, Any]) -> bool:
    return bool(config.get("training", {}).get("control_diagnostics", False))


def _metric_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def compute_guidance_detail_map(inputs: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    if inputs.shape[-2:] != target_hw:
        inputs = F.interpolate(inputs, size=target_hw, mode="bilinear", align_corners=False)
    if inputs.shape[1] < 3:
        gray = inputs.mean(dim=1, keepdim=True)
    else:
        weights = inputs.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        gray = (inputs[:, :3] * weights).sum(dim=1, keepdim=True)
    local_mean = F.avg_pool2d(F.pad(gray, (2, 2, 2, 2), mode="replicate"), kernel_size=5, stride=1)
    highpass = (gray - local_mean).abs()
    grad_x = gray[..., :, 1:] - gray[..., :, :-1]
    grad_x = F.pad(grad_x, (0, 1, 0, 0), mode="replicate")
    grad_y = gray[..., 1:, :] - gray[..., :-1, :]
    grad_y = F.pad(grad_y, (0, 0, 0, 1), mode="replicate")
    detail = highpass + torch.sqrt((grad_x.square() + grad_y.square()).clamp_min(1e-8))
    detail_mean = detail.flatten(start_dim=1).mean(dim=1).view(inputs.shape[0], 1, 1, 1).clamp_min(1e-6)
    return (detail / detail_mean).detach().float()


def compute_control_diagnostics(
    aux: Dict[str, torch.Tensor | None],
    target: torch.Tensor,
    config: Dict[str, Any],
    *,
    inputs: torch.Tensor | None = None,
    prefix: str = "control",
) -> Dict[str, float]:
    def name(key: str) -> str:
        return f"{prefix}_{key}" if prefix else key

    canvas = aux.get("control_canvas")
    if canvas is None:
        return {name("canvas_available"): 0.0}

    supervision_size = int(config.get("model", {}).get("train_supervision_crop_size", 0))
    consistency_enabled = bool(config.get("model", {}).get("use_train_inference_consistency", False))
    target_for_canvas = target
    inputs_for_canvas = inputs
    feedback = aux.get("control_feedback_map")
    blend_map = aux.get("control_guidance_blend_map")
    reliability_map = aux.get("control_guidance_reliability_map")
    if consistency_enabled and supervision_size > 0:
        target_for_canvas = center_crop_like(target_for_canvas, (supervision_size, supervision_size))
        canvas = center_crop_like(canvas, (supervision_size, supervision_size))
        if inputs_for_canvas is not None:
            inputs_for_canvas = center_crop_like(inputs_for_canvas, (supervision_size, supervision_size))
        if feedback is not None:
            feedback = center_crop_like(feedback, (supervision_size, supervision_size))
        if blend_map is not None:
            blend_map = center_crop_like(blend_map, (supervision_size, supervision_size))
        if reliability_map is not None:
            reliability_map = center_crop_like(reliability_map, (supervision_size, supervision_size))

    with torch.no_grad():
        canvas_target = compute_control_canvas_target(target_for_canvas, config, inputs=inputs_for_canvas)
        if canvas_target.shape[-2:] != canvas.shape[-2:]:
            canvas_target = F.interpolate(
                canvas_target,
                size=canvas.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        canvas_detached = canvas.detach().float()
        target_detached = canvas_target.detach().float()
        canvas_flat = canvas_detached.flatten(start_dim=1)
        target_flat = target_detached.flatten(start_dim=1)
        canvas_centered = canvas_flat - canvas_flat.mean(dim=1, keepdim=True)
        target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
        corr_num = (canvas_centered * target_centered).mean(dim=1)
        corr_den = (
            canvas_centered.square().mean(dim=1)
            * target_centered.square().mean(dim=1)
        ).sqrt().clamp_min(1e-6)
        diagnostics = {
            name("canvas_available"): 1.0,
            name("canvas_mean"): _metric_float(canvas_detached.mean()),
            name("canvas_std"): _metric_float(canvas_detached.std(unbiased=False)),
            name("canvas_min"): _metric_float(canvas_detached.amin()),
            name("canvas_max"): _metric_float(canvas_detached.amax()),
            name("target_mean"): _metric_float(target_detached.mean()),
            name("target_std"): _metric_float(target_detached.std(unbiased=False)),
            name("canvas_l1"): _metric_float(F.l1_loss(canvas_detached, target_detached)),
            name("canvas_corr"): _metric_float((corr_num / corr_den).mean()),
        }
        if feedback is not None:
            feedback_detached = feedback.detach().float()
            diagnostics.update(
                {
                    name("feedback_mean"): _metric_float(feedback_detached.mean()),
                    name("feedback_std"): _metric_float(feedback_detached.std(unbiased=False)),
                    name("feedback_max"): _metric_float(feedback_detached.amax()),
                    name("feedback_gt_0_05_fraction"): _metric_float((feedback_detached > 0.05).float().mean()),
                }
            )
        if blend_map is not None:
            blend_detached = blend_map.detach().float()
            diagnostics.update(
                {
                    name("guidance_blend_mean"): _metric_float(blend_detached.mean()),
                    name("guidance_blend_std"): _metric_float(blend_detached.std(unbiased=False)),
                    name("guidance_blend_min"): _metric_float(blend_detached.amin()),
                    name("guidance_blend_max"): _metric_float(blend_detached.amax()),
                }
            )
            if inputs_for_canvas is not None:
                detail_map = compute_guidance_detail_map(inputs_for_canvas, blend_detached.shape[-2:])
                detail_flat = detail_map.flatten(start_dim=1)
                detail_median = detail_flat.median(dim=1).values.view(detail_map.shape[0], 1, 1, 1)
                high_mask = (detail_map >= detail_median).float()
                low_mask = 1.0 - high_mask
                high_den = high_mask.sum().clamp_min(1.0)
                low_den = low_mask.sum().clamp_min(1.0)
                blend_flat = blend_detached.flatten(start_dim=1)
                detail_centered = detail_flat - detail_flat.mean(dim=1, keepdim=True)
                blend_centered = blend_flat - blend_flat.mean(dim=1, keepdim=True)
                blend_detail_corr_num = (detail_centered * blend_centered).mean(dim=1)
                blend_detail_corr_den = (
                    detail_centered.square().mean(dim=1)
                    * blend_centered.square().mean(dim=1)
                ).sqrt().clamp_min(1e-6)
                diagnostics.update(
                    {
                        name("guidance_blend_high_detail_mean"): _metric_float(
                            (blend_detached * high_mask).sum() / high_den
                        ),
                        name("guidance_blend_low_detail_mean"): _metric_float(
                            (blend_detached * low_mask).sum() / low_den
                        ),
                        name("guidance_blend_detail_corr"): _metric_float(
                            (blend_detail_corr_num / blend_detail_corr_den).mean()
                        ),
                    }
                )
        if reliability_map is not None:
            reliability_detached = reliability_map.detach().float()
            diagnostics.update(
                {
                    name("guidance_reliability_mean"): _metric_float(reliability_detached.mean()),
                    name("guidance_reliability_std"): _metric_float(
                        reliability_detached.std(unbiased=False)
                    ),
                    name("guidance_reliability_min"): _metric_float(reliability_detached.amin()),
                    name("guidance_reliability_max"): _metric_float(reliability_detached.amax()),
                }
            )
            if blend_map is not None:
                blend_flat = blend_detached.flatten(start_dim=1)
                reliability_flat = reliability_detached.flatten(start_dim=1)
                blend_centered = blend_flat - blend_flat.mean(dim=1, keepdim=True)
                reliability_centered = reliability_flat - reliability_flat.mean(dim=1, keepdim=True)
                reliability_corr_num = (blend_centered * reliability_centered).mean(dim=1)
                reliability_corr_den = (
                    blend_centered.square().mean(dim=1)
                    * reliability_centered.square().mean(dim=1)
                ).sqrt().clamp_min(1e-6)
                diagnostics[name("guidance_blend_reliability_corr")] = _metric_float(
                    (reliability_corr_num / reliability_corr_den).mean()
                )
    return diagnostics


def compute_input_std_route_diagnostics(
    aux: Dict[str, torch.Tensor | None],
    inputs: torch.Tensor,
    prediction: torch.Tensor,
    config: Dict[str, Any],
    *,
    prefix: str = "route",
) -> Dict[str, float]:
    def name(key: str) -> str:
        return f"{prefix}_{key}" if prefix else key

    gate = aux.get("input_std_route_gate")
    prior = aux.get("input_std_route_prior")
    high_delta = aux.get("input_std_route_high_delta")
    low_delta = aux.get("input_std_route_low_delta")
    if gate is None:
        return {name("available"): 0.0}
    if gate.shape[-2:] != prediction.shape[-2:]:
        gate = center_crop_like(gate, prediction.shape[-2:])
    if prior is not None and prior.shape[-2:] != prediction.shape[-2:]:
        prior = center_crop_like(prior, prediction.shape[-2:])
    input_std, high_soft, high_hard = _input_std_route_target(inputs, prediction, config)
    diagnostics = {
        name("available"): 1.0,
        name("gate_mean"): _metric_float(gate.detach().float().mean()),
        name("gate_std"): _metric_float(gate.detach().float().std(unbiased=False)),
        name("gate_min"): _metric_float(gate.detach().float().amin()),
        name("gate_max"): _metric_float(gate.detach().float().amax()),
        name("high_soft_mean"): _metric_float(high_soft.detach().float().mean()),
        name("high_hard_mean"): _metric_float(high_hard.detach().float().mean()),
        name("input_std_mean"): _metric_float(input_std.detach().float().mean()),
        name("input_std_min"): _metric_float(input_std.detach().float().min()),
        name("input_std_max"): _metric_float(input_std.detach().float().max()),
    }
    if prior is not None:
        diagnostics.update(
            {
                name("prior_mean"): _metric_float(prior.detach().float().mean()),
                name("prior_std"): _metric_float(prior.detach().float().std(unbiased=False)),
            }
        )
    if high_delta is not None:
        if high_delta.shape[-2:] != prediction.shape[-2:]:
            high_delta = center_crop_like(high_delta, prediction.shape[-2:])
        diagnostics[name("high_delta_abs_mean")] = _metric_float(high_delta.detach().float().abs().mean())
    if low_delta is not None:
        if low_delta.shape[-2:] != prediction.shape[-2:]:
            low_delta = center_crop_like(low_delta, prediction.shape[-2:])
        diagnostics[name("low_delta_abs_mean")] = _metric_float(low_delta.detach().float().abs().mean())
    return diagnostics


def center_crop_like(x: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    target_h, target_w = target_hw
    height, width = x.shape[-2:]
    if target_h >= height and target_w >= width:
        return x
    top = max((height - target_h) // 2, 0)
    left = max((width - target_w) // 2, 0)
    return x[..., top : top + target_h, left : left + target_w]


def select_training_supervision(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    config: Dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, float]:
    supervision_size = int(config.get("model", {}).get("train_supervision_crop_size", 0))
    consistency_enabled = bool(config.get("model", {}).get("use_train_inference_consistency", False))
    effective_predictions = predictions
    effective_targets = targets
    if consistency_enabled and supervision_size > 0:
        effective_predictions = center_crop_like(predictions, (supervision_size, supervision_size))
        effective_targets = center_crop_like(targets, (supervision_size, supervision_size))
    valid_ratio = (
        float(effective_predictions.shape[-1] * effective_predictions.shape[-2])
        / float(max(predictions.shape[-1] * predictions.shape[-2], 1))
    )
    return effective_predictions, effective_targets, valid_ratio


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def cuda_rank_mapping(distributed: Dict[str, int | bool]) -> Dict[str, Any]:
    local_rank = int(distributed.get("local_rank", 0))
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_parts = [part.strip() for part in visible_devices.split(",") if part.strip()]
    physical_gpu = visible_parts[local_rank] if local_rank < len(visible_parts) else None
    return {
        "local_rank": local_rank,
        "cuda_visible_devices": visible_devices,
        "physical_gpu": physical_gpu,
    }


def snapshot_template(run_dir: Path, config_path: Path) -> None:
    snapshot_dir = run_dir / "template_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for relative in (
        "src",
        "scripts/gpu_safety.py",
        "scripts/train_v1_t01.py",
        "scripts/smoke_calibrate_v1_t01.py",
        "experiments/V1/templates/T01",
        "docs/training_deployment_gate.md",
        "docs/versioning_policy.md",
        "docs/evaluation_split_policy.md",
    ):
        source = ROOT / relative
        target = snapshot_dir / relative
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True, ignore=ignore)
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    target_config = snapshot_dir / "launch_config.toml"
    target_config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, target_config)


def run_dir_for_config(config: Dict[str, Any]) -> Path:
    identity = config["identity"]
    archive_cfg = config.get("archive", {})
    run_root = ROOT / archive_cfg.get("output_root", "experiments/V1/runs")
    return run_root / identity["run"]


def prepare_run_dir(config: Dict[str, Any], config_path: Path) -> Path:
    run_dir = run_dir_for_config(config)
    archive_cfg = config.get("archive", {})
    for subdir in ("configs", "checkpoints", "logs", "metrics"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    target_config = run_dir / "configs" / config_path.name
    # Allow launching directly from a run-local config without failing on a
    # self-copy when the source already lives under run_dir/configs.
    if config_path.resolve() != target_config.resolve():
        shutil.copy2(config_path, target_config)
    if bool(archive_cfg.get("snapshot_on_start", True)):
        snapshot_template(run_dir, config_path)
    return run_dir


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _psnr_per_sample(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (prediction - target).square().flatten(start_dim=1).mean(dim=1).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


def validation_router_metrics_enabled(config: Dict[str, Any]) -> bool:
    training_cfg = config.get("training", {})
    return bool(training_cfg.get("validation_router_metrics", False)) or bool(
        training_cfg.get("best_selection_min_delta_vs_baseline", {})
    ) or bool(
        training_cfg.get("step_no_harm_guard_enabled", False)
    ) or bool(
        training_cfg.get("step_no_harm_guard_min_delta_vs_baseline", {})
    ) or bool(training_cfg.get("best_selection_max_delta_vs_baseline", {}))


def transfer_validation_metrics_enabled(config: Dict[str, Any]) -> bool:
    training_cfg = config.get("training", {})
    return bool(training_cfg.get("transfer_validation_enabled", False)) or bool(
        training_cfg.get("transfer_validation_records", "")
    )


def _resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def _load_transfer_validation_records(config: Dict[str, Any]) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any], str]:
    training_cfg = config.get("training", {})
    records_path_value = str(training_cfg.get("transfer_validation_records", "")).strip()
    if not records_path_value:
        raise ValueError("training.transfer_validation_records is required when transfer validation is enabled")
    records_path = _resolve_project_path(records_path_value)
    records_data = json.loads(records_path.read_text(encoding="utf-8"))
    record_split = str(training_cfg.get("transfer_validation_record_split", "val_records"))
    if record_split not in records_data:
        raise ValueError(
            f"training.transfer_validation_record_split={record_split!r} is missing from {records_path}"
        )
    selected_rule = records_data.get("selected_rule")
    if record_split == "val_records":
        selected_rule = records_data.get("selected_rule_val", {}).get("rule", selected_rule)
    elif record_split == "test_records":
        selected_rule = records_data.get("selected_rule_test", {}).get("rule", selected_rule)
    if not isinstance(selected_rule, dict):
        raise ValueError(f"transfer validation records at {records_path} do not contain a selected rule")
    raw_records = records_data[record_split]
    if not isinstance(raw_records, list):
        raise ValueError(f"{record_split} in {records_path} must be a list")
    records: Dict[str, Dict[str, Any]] = {}
    for record in raw_records:
        if not isinstance(record, dict) or "name" not in record:
            raise ValueError(f"invalid transfer validation record in {records_path}: {record!r}")
        records[str(record["name"])] = record
    return records, selected_rule, record_split


def _add_masked_validation_sum(
    sums: Dict[str, float],
    counts: Dict[str, float],
    key: str,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    mask = mask.detach().bool()
    count = float(mask.float().sum().cpu())
    counts[key] = counts.get(key, 0.0) + count
    if count <= 0.0:
        return
    sums[key] = sums.get(key, 0.0) + float(values.detach()[mask].sum().cpu())


def _add_transfer_validation_metrics(
    metrics: Dict[str, float],
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
) -> None:
    records, selected_rule, record_split = _load_transfer_validation_records(config)
    feature = str(selected_rule.get("feature", "input_std"))
    direction = str(selected_rule.get("direction", "lte"))
    threshold = float(selected_rule["threshold"])
    prefix = str(config.get("training", {}).get("transfer_validation_prefix", "transfer_val"))
    max_records = int(config.get("training", {}).get("transfer_validation_max_records", 0))
    if feature != "input_std":
        raise ValueError("transfer validation currently supports selected_rule.feature='input_std' only")
    if direction not in {"lte", "lt", "gte", "gt"}:
        raise ValueError("transfer validation selected_rule.direction must be lte, lt, gte, or gt")

    sums: Dict[str, float] = {}
    counts: Dict[str, float] = {}
    seen = 0
    with torch.no_grad():
        for batch in loader:
            names = [str(name) for name in batch["name"]]
            record_indices: list[int] = []
            batch_records: list[Dict[str, Any]] = []
            for index, name in enumerate(names):
                record = records.get(name)
                if record is None:
                    continue
                record_indices.append(index)
                batch_records.append(record)
                if max_records > 0 and seen + len(batch_records) >= max_records:
                    break
            if not record_indices:
                continue

            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            predictions = model(inputs).clamp(0.0, 1.0)
            if targets.shape[-2:] != predictions.shape[-2:]:
                targets = center_crop_like(targets, predictions.shape[-2:])
            index_tensor = torch.tensor(record_indices, device=predictions.device, dtype=torch.long)
            student_psnr = _psnr_per_sample(predictions, targets).index_select(0, index_tensor)
            teacher_a_psnr = torch.tensor(
                [float(record["psnr_a"]) for record in batch_records],
                device=predictions.device,
                dtype=student_psnr.dtype,
            )
            teacher_b_psnr = torch.tensor(
                [float(record["psnr_b"]) for record in batch_records],
                device=predictions.device,
                dtype=student_psnr.dtype,
            )
            input_std = torch.tensor(
                [float(record["features"]["input_std"]) for record in batch_records],
                device=predictions.device,
                dtype=student_psnr.dtype,
            )
            if direction in {"lte", "lt"}:
                low_mask = input_std <= threshold if direction == "lte" else input_std < threshold
            else:
                low_mask = input_std >= threshold if direction == "gte" else input_std > threshold
            high_mask = ~low_mask
            a08_oracle_mask = teacher_a_psnr >= teacher_b_psnr
            a15_oracle_mask = teacher_b_psnr > teacher_a_psnr
            router_psnr = torch.where(high_mask, teacher_a_psnr, teacher_b_psnr)
            oracle_psnr = torch.maximum(teacher_a_psnr, teacher_b_psnr)
            _add_masked_validation_sum(
                sums,
                counts,
                f"{prefix}_overall_gap_mean",
                student_psnr - router_psnr,
                torch.ones_like(high_mask, dtype=torch.bool),
            )
            _add_masked_validation_sum(
                sums,
                counts,
                f"{prefix}_oracle_gap_mean",
                student_psnr - oracle_psnr,
                torch.ones_like(high_mask, dtype=torch.bool),
            )
            _add_masked_validation_sum(
                sums,
                counts,
                f"{prefix}_high_route_gap_mean",
                student_psnr - teacher_a_psnr,
                high_mask,
            )
            _add_masked_validation_sum(
                sums,
                counts,
                f"{prefix}_low_route_gap_mean",
                student_psnr - teacher_b_psnr,
                low_mask,
            )
            _add_masked_validation_sum(
                sums,
                counts,
                f"{prefix}_a08_oracle_gap_mean",
                student_psnr - teacher_a_psnr,
                a08_oracle_mask,
            )
            _add_masked_validation_sum(
                sums,
                counts,
                f"{prefix}_a15_oracle_gap_mean",
                student_psnr - teacher_b_psnr,
                a15_oracle_mask,
            )
            _add_masked_validation_sum(
                sums,
                counts,
                f"{prefix}_low_a15_oracle_gap_mean",
                student_psnr - teacher_b_psnr,
                low_mask & a15_oracle_mask,
            )
            _add_masked_validation_sum(
                sums,
                counts,
                f"{prefix}_low_a08_oracle_gap_mean",
                student_psnr - teacher_a_psnr,
                low_mask & a08_oracle_mask,
            )
            _add_masked_validation_sum(
                sums,
                counts,
                f"{prefix}_a15_oracle_advantage_mean",
                teacher_b_psnr - teacher_a_psnr,
                a15_oracle_mask,
            )
            seen += len(batch_records)
            if max_records > 0 and seen >= max_records:
                break

    for key, count in counts.items():
        metrics[key.replace("_mean", "_count")] = count
        metrics[key] = sums.get(key, 0.0) / max(count, 1.0)
    total = max(counts.get(f"{prefix}_overall_gap_mean", 0.0), 1.0)
    metrics[f"{prefix}_high_route_fraction"] = counts.get(f"{prefix}_high_route_gap_mean", 0.0) / total
    metrics[f"{prefix}_low_route_fraction"] = counts.get(f"{prefix}_low_route_gap_mean", 0.0) / total
    metrics[f"{prefix}_a08_oracle_fraction"] = counts.get(f"{prefix}_a08_oracle_gap_mean", 0.0) / total
    metrics[f"{prefix}_a15_oracle_fraction"] = counts.get(f"{prefix}_a15_oracle_gap_mean", 0.0) / total
    metrics[f"{prefix}_matched_records"] = float(counts.get(f"{prefix}_overall_gap_mean", 0.0))
    metrics[f"{prefix}_available_records"] = float(len(records))
    metrics[f"{prefix}_record_split"] = 1.0 if record_split == "val_records" else 0.0


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    *,
    router_teacher_a: torch.nn.Module | None = None,
    router_teacher_b: torch.nn.Module | None = None,
    transfer_loader: DataLoader | None = None,
) -> Dict[str, float]:
    model.eval()
    max_batches = int(config["training"].get("max_val_batches", 10))
    losses: list[float] = []
    psnrs: list[float] = []
    diagnostics_sums: Dict[str, float] = {}
    router_metrics = validation_router_metrics_enabled(config)
    transfer_metrics = transfer_validation_metrics_enabled(config)
    if router_metrics and (router_teacher_a is None or router_teacher_b is None):
        raise ValueError(
            "training.validation_router_metrics or baseline selection gates require router teacher A/B models"
        )
    router_sums: Dict[str, float] = {}
    router_counts: Dict[str, float] = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            if control_diagnostics_enabled(config):
                aux = model(inputs, return_aux="loss")
                predictions = aux["restored"].clamp(0.0, 1.0)
                diagnostics = compute_control_diagnostics(
                    aux,
                    targets,
                    config,
                    inputs=inputs,
                    prefix="val_control",
                )
                diagnostics.update(
                    compute_input_std_route_diagnostics(
                        aux,
                        inputs,
                        predictions,
                        config,
                        prefix="val_route",
                    )
                )
                for key, value in diagnostics.items():
                    diagnostics_sums[key] = diagnostics_sums.get(key, 0.0) + value
            else:
                predictions = model(inputs).clamp(0.0, 1.0)
            loss, _ = compute_loss(predictions, targets, config, input_image=inputs)
            losses.append(float(loss.cpu()))
            psnrs.append(psnr(predictions, targets))
            if router_metrics:
                assert router_teacher_a is not None and router_teacher_b is not None
                teacher_a_predictions = router_teacher_a(inputs).clamp(0.0, 1.0)
                teacher_b_predictions = router_teacher_b(inputs).clamp(0.0, 1.0)
                if teacher_a_predictions.shape[-2:] != predictions.shape[-2:]:
                    teacher_a_predictions = center_crop_like(teacher_a_predictions, predictions.shape[-2:])
                if teacher_b_predictions.shape[-2:] != predictions.shape[-2:]:
                    teacher_b_predictions = center_crop_like(teacher_b_predictions, predictions.shape[-2:])
                if targets.shape[-2:] != predictions.shape[-2:]:
                    route_targets = center_crop_like(targets, predictions.shape[-2:])
                else:
                    route_targets = targets
                student_psnr = _psnr_per_sample(predictions, route_targets)
                teacher_a_psnr = _psnr_per_sample(teacher_a_predictions, route_targets)
                teacher_b_psnr = _psnr_per_sample(teacher_b_predictions, route_targets)
                _, _, high_hard = _input_std_route_target(inputs, predictions, config)
                high_mask = high_hard.detach() >= 0.5
                low_mask = ~high_mask
                a08_oracle_mask = teacher_a_psnr >= teacher_b_psnr
                a15_oracle_mask = teacher_b_psnr > teacher_a_psnr
                router_psnr = torch.where(high_mask, teacher_a_psnr, teacher_b_psnr)
                oracle_psnr = torch.maximum(teacher_a_psnr, teacher_b_psnr)
                _add_masked_validation_sum(
                    router_sums,
                    router_counts,
                    "val_router_overall_gap_mean",
                    student_psnr - router_psnr,
                    torch.ones_like(high_mask, dtype=torch.bool),
                )
                _add_masked_validation_sum(
                    router_sums,
                    router_counts,
                    "val_router_oracle_gap_mean",
                    student_psnr - oracle_psnr,
                    torch.ones_like(high_mask, dtype=torch.bool),
                )
                _add_masked_validation_sum(
                    router_sums,
                    router_counts,
                    "val_router_high_route_gap_mean",
                    student_psnr - teacher_a_psnr,
                    high_mask,
                )
                _add_masked_validation_sum(
                    router_sums,
                    router_counts,
                    "val_router_low_route_gap_mean",
                    student_psnr - teacher_b_psnr,
                    low_mask,
                )
                _add_masked_validation_sum(
                    router_sums,
                    router_counts,
                    "val_router_a08_oracle_gap_mean",
                    student_psnr - teacher_a_psnr,
                    a08_oracle_mask,
                )
                _add_masked_validation_sum(
                    router_sums,
                    router_counts,
                    "val_router_a15_oracle_gap_mean",
                    student_psnr - teacher_b_psnr,
                    a15_oracle_mask,
                )
                _add_masked_validation_sum(
                    router_sums,
                    router_counts,
                    "val_router_low_a15_oracle_gap_mean",
                    student_psnr - teacher_b_psnr,
                    low_mask & a15_oracle_mask,
                )
                _add_masked_validation_sum(
                    router_sums,
                    router_counts,
                    "val_router_low_a08_oracle_gap_mean",
                    student_psnr - teacher_a_psnr,
                    low_mask & a08_oracle_mask,
                )
                _add_masked_validation_sum(
                    router_sums,
                    router_counts,
                    "val_router_a15_oracle_advantage_mean",
                    teacher_b_psnr - teacher_a_psnr,
                    a15_oracle_mask,
                )
            if batch_idx >= max_batches:
                break
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metrics = {
        "val_loss": sum(losses) / max(len(losses), 1),
        "val_psnr": sum(psnrs) / max(len(psnrs), 1),
        "val_batches": len(losses),
    }
    if diagnostics_sums:
        denom = max(len(losses), 1)
        metrics.update({key: value / denom for key, value in diagnostics_sums.items()})
    if router_metrics:
        for key, count in router_counts.items():
            metrics[key.replace("_mean", "_count")] = count
            metrics[key] = router_sums.get(key, 0.0) / max(count, 1.0)
        total_samples = max(router_counts.get("val_router_overall_gap_mean", 0.0), 1.0)
        metrics["val_router_high_route_fraction"] = router_counts.get("val_router_high_route_gap_mean", 0.0) / total_samples
        metrics["val_router_low_route_fraction"] = router_counts.get("val_router_low_route_gap_mean", 0.0) / total_samples
        metrics["val_router_a08_oracle_fraction"] = router_counts.get("val_router_a08_oracle_gap_mean", 0.0) / total_samples
        metrics["val_router_a15_oracle_fraction"] = router_counts.get("val_router_a15_oracle_gap_mean", 0.0) / total_samples
    if transfer_metrics:
        _add_transfer_validation_metrics(metrics, model, transfer_loader or loader, device, config)
    model.train()
    return metrics


def guard_metric_log_value(key: str) -> bool:
    return key.startswith("val_router_") or key.startswith("transfer_val_") or key in {
        "val_loss",
        "val_psnr",
        "val_batches",
    }


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, config: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def best_metric_value(metrics: Dict[str, float], selection_metric: str) -> float:
    if selection_metric not in metrics:
        raise ValueError(
            f"training.best_selection_metric={selection_metric!r} is not available; "
            f"available validation metrics: {sorted(metrics)}"
        )
    return float(metrics[selection_metric])


def is_better_metric(candidate: float, best: float, selection_metric: str) -> bool:
    if (
        selection_metric in {"val_psnr", "psnr"}
        or selection_metric.endswith("_psnr")
        or selection_metric.endswith("_gap_mean")
        or selection_metric.endswith("_fraction")
    ):
        return candidate > best
    return candidate < best


def initial_best_metric(selection_metric: str) -> float:
    if (
        selection_metric in {"val_psnr", "psnr"}
        or selection_metric.endswith("_psnr")
        or selection_metric.endswith("_gap_mean")
        or selection_metric.endswith("_fraction")
    ):
        return float("-inf")
    return float("inf")


def _metric_gate_dict(training_cfg: Dict[str, Any], key: str) -> Dict[str, float]:
    raw = training_cfg.get(key, {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"training.{key} must be a metric-to-threshold dictionary")
    return {str(metric): float(value) for metric, value in raw.items()}


def best_selection_baseline_gates_enabled(training_cfg: Dict[str, Any]) -> bool:
    return any(
        _metric_gate_dict(training_cfg, key)
        for key in (
            "best_selection_min_delta_vs_baseline",
            "best_selection_max_delta_vs_baseline",
            "best_selection_min_metric_values",
            "best_selection_max_metric_values",
        )
    )


def _selection_gate_status_for_keys(
    metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    training_cfg: Dict[str, Any],
    *,
    min_delta_key: str,
    max_delta_key: str,
    min_metric_key: str,
    max_metric_key: str,
) -> tuple[bool, Dict[str, Any]]:
    checks: list[Dict[str, Any]] = []

    def require_metric(source: Dict[str, float], metric: str, source_name: str) -> float:
        if metric not in source:
            raise ValueError(
                f"training best-selection gate references {metric!r}, "
                f"but it is missing from {source_name}; available keys: {sorted(source)}"
            )
        return float(source[metric])

    for metric, delta in _metric_gate_dict(training_cfg, min_delta_key).items():
        actual = require_metric(metrics, metric, "validation metrics")
        baseline = require_metric(baseline_metrics, metric, "selection baseline metrics")
        threshold = baseline + delta
        checks.append(
            {
                "metric": metric,
                "mode": "min_delta_vs_baseline",
                "actual": actual,
                "baseline": baseline,
                "delta": delta,
                "threshold": threshold,
                "passed": actual >= threshold,
            }
        )
    for metric, delta in _metric_gate_dict(training_cfg, max_delta_key).items():
        actual = require_metric(metrics, metric, "validation metrics")
        baseline = require_metric(baseline_metrics, metric, "selection baseline metrics")
        threshold = baseline + delta
        checks.append(
            {
                "metric": metric,
                "mode": "max_delta_vs_baseline",
                "actual": actual,
                "baseline": baseline,
                "delta": delta,
                "threshold": threshold,
                "passed": actual <= threshold,
            }
        )
    for metric, threshold in _metric_gate_dict(training_cfg, min_metric_key).items():
        actual = require_metric(metrics, metric, "validation metrics")
        checks.append(
            {
                "metric": metric,
                "mode": "min_metric_value",
                "actual": actual,
                "threshold": threshold,
                "passed": actual >= threshold,
            }
        )
    for metric, threshold in _metric_gate_dict(training_cfg, max_metric_key).items():
        actual = require_metric(metrics, metric, "validation metrics")
        checks.append(
            {
                "metric": metric,
                "mode": "max_metric_value",
                "actual": actual,
                "threshold": threshold,
                "passed": actual <= threshold,
            }
        )
    passed = all(bool(check["passed"]) for check in checks)
    return passed, {"enabled": bool(checks), "passed": passed, "checks": checks}


def best_selection_gate_status(
    metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    training_cfg: Dict[str, Any],
) -> tuple[bool, Dict[str, Any]]:
    return _selection_gate_status_for_keys(
        metrics,
        baseline_metrics,
        training_cfg,
        min_delta_key="best_selection_min_delta_vs_baseline",
        max_delta_key="best_selection_max_delta_vs_baseline",
        min_metric_key="best_selection_min_metric_values",
        max_metric_key="best_selection_max_metric_values",
    )


def step_no_harm_guard_enabled(training_cfg: Dict[str, Any]) -> bool:
    return bool(training_cfg.get("step_no_harm_guard_enabled", False)) or any(
        _metric_gate_dict(training_cfg, key)
        for key in (
            "step_no_harm_guard_min_delta_vs_baseline",
            "step_no_harm_guard_max_delta_vs_baseline",
            "step_no_harm_guard_min_metric_values",
            "step_no_harm_guard_max_metric_values",
        )
    )


def step_no_harm_guard_status(
    metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    training_cfg: Dict[str, Any],
) -> tuple[bool, Dict[str, Any]]:
    return _selection_gate_status_for_keys(
        metrics,
        baseline_metrics,
        training_cfg,
        min_delta_key="step_no_harm_guard_min_delta_vs_baseline",
        max_delta_key="step_no_harm_guard_max_delta_vs_baseline",
        min_metric_key="step_no_harm_guard_min_metric_values",
        max_metric_key="step_no_harm_guard_max_metric_values",
    )


def step_no_harm_custom_guard_status(
    metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    training_cfg: Dict[str, Any],
    *,
    prefix: str,
) -> tuple[bool, Dict[str, Any]]:
    return _selection_gate_status_for_keys(
        metrics,
        baseline_metrics,
        training_cfg,
        min_delta_key=f"{prefix}_min_delta_vs_baseline",
        max_delta_key=f"{prefix}_max_delta_vs_baseline",
        min_metric_key=f"{prefix}_min_metric_values",
        max_metric_key=f"{prefix}_max_metric_values",
    )


def step_no_harm_projection_ratios(training_cfg: Dict[str, Any]) -> list[float]:
    raw = training_cfg.get("step_no_harm_guard_projection_ratios", [0.75, 0.5, 0.25])
    if not isinstance(raw, list):
        raise ValueError("training.step_no_harm_guard_projection_ratios must be a list")
    ratios = [float(value) for value in raw]
    if any(value <= 0.0 or value >= 1.0 for value in ratios):
        raise ValueError("training.step_no_harm_guard_projection_ratios values must be in (0, 1)")
    return ratios


def validation_config_with_max_batches(config: Dict[str, Any], max_batches: int) -> Dict[str, Any]:
    guard_config = copy.deepcopy(config)
    guard_config.setdefault("training", {})["max_val_batches"] = int(max_batches)
    return guard_config


def clone_model_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in unwrap_model(model).state_dict().items()}


def load_interpolated_model_state(
    model: torch.nn.Module,
    base_state: Dict[str, torch.Tensor],
    candidate_state: Dict[str, torch.Tensor],
    ratio: float,
) -> None:
    interpolated: Dict[str, torch.Tensor] = {}
    for key, candidate_value in candidate_state.items():
        base_value = base_state[key]
        if torch.is_floating_point(candidate_value):
            interpolated[key] = base_value + (candidate_value - base_value) * ratio
        else:
            interpolated[key] = candidate_value
    unwrap_model(model).load_state_dict(interpolated)


def lr_schedule_factor(training_cfg: Dict[str, Any], step: int, max_steps: int) -> float:
    schedule = str(training_cfg.get("lr_schedule", "none")).lower()
    if schedule in {"", "none"}:
        return 1.0
    if schedule != "linear_decay":
        raise ValueError("training.lr_schedule must be 'none' or 'linear_decay'")
    start = float(training_cfg.get("lr_schedule_start", 0.0))
    end = float(training_cfg.get("lr_schedule_end", 1.0))
    final_factor = float(training_cfg.get("lr_final_factor", 1.0))
    if not 0.0 <= start <= 1.0:
        raise ValueError("training.lr_schedule_start must be in [0, 1]")
    if not 0.0 <= end <= 1.0:
        raise ValueError("training.lr_schedule_end must be in [0, 1]")
    if end < start:
        raise ValueError("training.lr_schedule_end must be >= training.lr_schedule_start")
    if final_factor <= 0.0:
        raise ValueError("training.lr_final_factor must be positive")
    progress = step / max(max_steps, 1)
    if progress <= start:
        return 1.0
    if progress >= end:
        return final_factor
    if end == start:
        return final_factor
    alpha = (progress - start) / (end - start)
    return 1.0 + alpha * (final_factor - 1.0)


def apply_lr_schedule(optimizer: torch.optim.Optimizer, base_lrs: list[float], factor: float) -> None:
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr * factor


def ema_decay_value(training_cfg: Dict[str, Any]) -> float:
    decay = float(training_cfg.get("ema_decay", 0.0))
    if decay <= 0.0:
        return 0.0
    if not 0.0 < decay < 1.0:
        raise ValueError("training.ema_decay must be in (0, 1) when enabled")
    return decay


def init_ema_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for key, value in unwrap_model(model).state_dict().items():
        if torch.is_floating_point(value):
            state[key] = value.detach().clone()
    return state


def update_ema_state(model: torch.nn.Module, ema_state: Dict[str, torch.Tensor], decay: float) -> None:
    if not ema_state:
        return
    with torch.no_grad():
        for key, value in unwrap_model(model).state_dict().items():
            if key not in ema_state:
                continue
            ema_state[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)


@contextmanager
def ema_weights_applied(model: torch.nn.Module, ema_state: Dict[str, torch.Tensor]):
    if not ema_state:
        yield
        return
    module = unwrap_model(model)
    state = module.state_dict()
    backup: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for key, ema_value in ema_state.items():
            current = state[key]
            backup[key] = current.detach().clone()
            current.copy_(ema_value.to(device=current.device, dtype=current.dtype))
    try:
        yield
    finally:
        with torch.no_grad():
            for key, value in backup.items():
                state[key].copy_(value)


def resolve_checkpoint_path(raw_path: Any, config_path: Path) -> Path:
    checkpoint_path = Path(str(raw_path)).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (config_path.parent / checkpoint_path).resolve()
    return checkpoint_path


def load_init_checkpoint_if_requested(
    model: torch.nn.Module,
    config: Dict[str, Any],
    config_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    training_cfg = config.get("training", {})
    raw_checkpoint = training_cfg.get("init_checkpoint") or training_cfg.get("initial_checkpoint")
    if not raw_checkpoint:
        return {}
    checkpoint_path = resolve_checkpoint_path(raw_checkpoint, config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"init_checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        checkpoint_step = int(checkpoint.get("step", -1))
    else:
        state_dict = checkpoint
        checkpoint_step = -1
    if not isinstance(state_dict, dict):
        raise TypeError(f"init_checkpoint does not contain a model state dict: {checkpoint_path}")
    strict = bool(training_cfg.get("init_checkpoint_strict", True))
    module = unwrap_model(model)
    incompatible = module.load_state_dict(state_dict, strict=strict)
    route_heads_init_from_base = bool(
        training_cfg.get("init_checkpoint_initialize_input_std_route_decoder_heads_from_base", True)
    )
    route_heads_reinitialized = False
    if route_heads_init_from_base and hasattr(module, "initialize_input_std_route_decoder_heads_from_base"):
        module.initialize_input_std_route_decoder_heads_from_base()
        route_heads_reinitialized = True
    return {
        "init_checkpoint": str(checkpoint_path),
        "init_checkpoint_step": checkpoint_step,
        "init_checkpoint_strict": strict,
        "init_checkpoint_initialize_input_std_route_decoder_heads_from_base": route_heads_init_from_base,
        "init_checkpoint_route_decoder_heads_reinitialized_from_base": route_heads_reinitialized,
        "init_checkpoint_missing_keys": list(incompatible.missing_keys),
        "init_checkpoint_unexpected_keys": list(incompatible.unexpected_keys),
    }


def load_checkpoint_state_dict(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[Dict[str, torch.Tensor], int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        checkpoint_step = int(checkpoint.get("step", -1))
    else:
        state_dict = checkpoint
        checkpoint_step = -1
    if not isinstance(state_dict, dict):
        raise TypeError(f"checkpoint does not contain a model state dict: {checkpoint_path}")
    return state_dict, checkpoint_step


def _copy_conv2d_from_checkpoint_head(
    target_head: torch.nn.Conv2d,
    state_dict: Dict[str, torch.Tensor],
    *,
    source_prefix: str,
    target_label: str,
) -> Dict[str, Any]:
    weight_key = f"{source_prefix}.weight"
    bias_key = f"{source_prefix}.bias"
    if weight_key not in state_dict or bias_key not in state_dict:
        raise KeyError(
            f"{target_label} route head transplant requires {weight_key!r} and {bias_key!r} in checkpoint"
        )
    source_weight = state_dict[weight_key]
    source_bias = state_dict[bias_key]
    if tuple(source_weight.shape) != tuple(target_head.weight.shape):
        raise ValueError(
            f"{target_label} route head weight shape mismatch: "
            f"checkpoint {tuple(source_weight.shape)} vs target {tuple(target_head.weight.shape)}"
        )
    if tuple(source_bias.shape) != tuple(target_head.bias.shape):
        raise ValueError(
            f"{target_label} route head bias shape mismatch: "
            f"checkpoint {tuple(source_bias.shape)} vs target {tuple(target_head.bias.shape)}"
        )
    with torch.no_grad():
        target_head.weight.copy_(source_weight.to(device=target_head.weight.device, dtype=target_head.weight.dtype))
        target_head.bias.copy_(source_bias.to(device=target_head.bias.device, dtype=target_head.bias.dtype))
    return {
        f"input_std_route_decoder_heads_{target_label}_source_prefix": source_prefix,
        f"input_std_route_decoder_heads_{target_label}_weight_shape": list(target_head.weight.shape),
        f"input_std_route_decoder_heads_{target_label}_bias_shape": list(target_head.bias.shape),
        f"input_std_route_decoder_heads_{target_label}_weight_abs_mean": float(
            target_head.weight.detach().abs().mean().cpu()
        ),
        f"input_std_route_decoder_heads_{target_label}_bias_abs_mean": float(
            target_head.bias.detach().abs().mean().cpu()
        ),
    }


def transplant_input_std_route_decoder_heads_if_requested(
    model: torch.nn.Module,
    config: Dict[str, Any],
    config_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    training_cfg = config.get("training", {})
    high_raw = training_cfg.get("input_std_route_decoder_heads_init_high_checkpoint")
    low_raw = training_cfg.get("input_std_route_decoder_heads_init_low_checkpoint")
    if not high_raw and not low_raw:
        return {"input_std_route_decoder_heads_transplant_enabled": False}
    module = unwrap_model(model)
    route_heads = getattr(module, "input_std_route_decoder_heads", None)
    if route_heads is None:
        raise ValueError(
            "input_std_route_decoder_heads checkpoint transplant requires model.use_input_std_route_decoder_heads=true"
        )

    info: Dict[str, Any] = {"input_std_route_decoder_heads_transplant_enabled": True}
    default_prefix = str(training_cfg.get("input_std_route_decoder_heads_init_source_prefix", "decoder.2"))
    for label, raw_checkpoint, target_head in (
        ("high", high_raw, route_heads.high_head),
        ("low", low_raw, route_heads.low_head),
    ):
        if not raw_checkpoint:
            info[f"input_std_route_decoder_heads_{label}_transplant_enabled"] = False
            continue
        checkpoint_path = resolve_checkpoint_path(raw_checkpoint, config_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"input_std_route_decoder_heads_init_{label}_checkpoint does not exist: {checkpoint_path}"
            )
        state_dict, checkpoint_step = load_checkpoint_state_dict(checkpoint_path, device)
        source_prefix = str(
            training_cfg.get(f"input_std_route_decoder_heads_init_{label}_source_prefix", default_prefix)
        )
        head_info = _copy_conv2d_from_checkpoint_head(
            target_head,
            state_dict,
            source_prefix=source_prefix,
            target_label=label,
        )
        info.update(
            {
                f"input_std_route_decoder_heads_{label}_transplant_enabled": True,
                f"input_std_route_decoder_heads_{label}_checkpoint": str(checkpoint_path),
                f"input_std_route_decoder_heads_{label}_checkpoint_step": checkpoint_step,
                **head_info,
            }
        )
    return info


def apply_trainable_parameter_policy(
    model: torch.nn.Module,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    training_cfg = config.get("training", {})
    prefixes: list[str] = []
    if bool(training_cfg.get("train_only_input_std_route_decoder_heads", False)):
        prefixes.append("input_std_route_decoder_heads")
        if not bool(config.get("model", {}).get("use_input_std_route_decoder_heads", False)):
            raise ValueError(
                "training.train_only_input_std_route_decoder_heads=true requires "
                "model.use_input_std_route_decoder_heads=true"
            )
    raw_prefixes = training_cfg.get("trainable_parameter_prefixes", [])
    if isinstance(raw_prefixes, str):
        raw_prefixes = [raw_prefixes]
    if raw_prefixes:
        prefixes.extend(str(prefix) for prefix in raw_prefixes)
    prefixes = [prefix for prefix in dict.fromkeys(prefixes) if prefix]

    module = unwrap_model(model)
    total_params = sum(parameter.numel() for parameter in module.parameters())
    if not prefixes:
        trainable_params = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
        return {
            "parameter_policy_enabled": False,
            "trainable_parameter_prefixes": [],
            "total_parameter_count": total_params,
            "trainable_parameter_count": trainable_params,
            "frozen_parameter_count": total_params - trainable_params,
        }

    trainable_names: list[str] = []
    frozen_names: list[str] = []
    trainable_params = 0
    for name, parameter in module.named_parameters():
        trainable = any(name.startswith(prefix) for prefix in prefixes)
        parameter.requires_grad_(trainable)
        if trainable:
            trainable_names.append(name)
            trainable_params += parameter.numel()
        else:
            frozen_names.append(name)
    if trainable_params <= 0:
        raise ValueError(f"trainable_parameter_prefixes did not match any parameters: {prefixes}")
    return {
        "parameter_policy_enabled": True,
        "trainable_parameter_prefixes": prefixes,
        "total_parameter_count": total_params,
        "trainable_parameter_count": trainable_params,
        "frozen_parameter_count": total_params - trainable_params,
        "trainable_parameter_name_count": len(trainable_names),
        "frozen_parameter_name_count": len(frozen_names),
        "trainable_parameter_names_sample": trainable_names[:20],
        "frozen_parameter_names_sample": frozen_names[:20],
    }


def load_model_state_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    *,
    strict: bool,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        checkpoint_step = int(checkpoint.get("step", -1))
    else:
        state_dict = checkpoint
        checkpoint_step = -1
    if not isinstance(state_dict, dict):
        raise TypeError(f"checkpoint does not contain a model state dict: {checkpoint_path}")
    incompatible = unwrap_model(model).load_state_dict(state_dict, strict=strict)
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "strict": strict,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def load_teacher_model_if_requested(
    config: Dict[str, Any],
    config_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module | None, Dict[str, Any]]:
    loss_cfg = config.get("loss", {})
    weight = _nonnegative_loss_value(
        loss_cfg,
        "teacher_consistency_weight",
        0.0,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    raw_checkpoint = loss_cfg.get("teacher_consistency_checkpoint")
    if weight <= 0.0:
        return None, {"teacher_consistency_enabled": False, "teacher_consistency_weight": weight}
    if not raw_checkpoint:
        raise ValueError("loss.teacher_consistency_weight > 0 requires loss.teacher_consistency_checkpoint")
    checkpoint_path = resolve_checkpoint_path(raw_checkpoint, config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"teacher_consistency_checkpoint does not exist: {checkpoint_path}")
    teacher_model = CrossResolutionGaussianRestorer(build_model_config(config)).to(device)
    strict = bool(loss_cfg.get("teacher_consistency_strict", True))
    checkpoint_info = load_model_state_from_checkpoint(
        teacher_model,
        checkpoint_path,
        device,
        strict=strict,
    )
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)
    return teacher_model, {
        "teacher_consistency_enabled": True,
        "teacher_consistency_weight": weight,
        "teacher_consistency_checkpoint": checkpoint_info["checkpoint"],
        "teacher_consistency_checkpoint_step": checkpoint_info["checkpoint_step"],
        "teacher_consistency_strict": checkpoint_info["strict"],
        "teacher_consistency_missing_keys": checkpoint_info["missing_keys"],
        "teacher_consistency_unexpected_keys": checkpoint_info["unexpected_keys"],
    }


def _freeze_teacher_model(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _load_router_teacher(
    config: Dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
    *,
    strict: bool,
    teacher_config_path: Path | None = None,
) -> tuple[torch.nn.Module, Dict[str, Any]]:
    teacher_config = config
    if teacher_config_path is not None:
        teacher_config = load_toml(teacher_config_path)
    teacher_model = CrossResolutionGaussianRestorer(build_model_config(teacher_config)).to(device)
    checkpoint_info = load_model_state_from_checkpoint(
        teacher_model,
        checkpoint_path,
        device,
        strict=strict,
    )
    checkpoint_info["config"] = str(teacher_config_path) if teacher_config_path is not None else "student"
    return _freeze_teacher_model(teacher_model), checkpoint_info


def load_router_distillation_teachers_if_requested(
    config: Dict[str, Any],
    config_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module | None, torch.nn.Module | None, Dict[str, Any]]:
    loss_cfg = config.get("loss", {})
    weight = _nonnegative_loss_value(
        loss_cfg,
        "router_distillation_weight",
        0.0,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    shared_route_branch_weight = _nonnegative_loss_value(
        loss_cfg,
        "input_std_route_branch_weight",
        0.0,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    high_route_branch_weight = _nonnegative_loss_value(
        loss_cfg,
        "input_std_route_high_branch_weight",
        shared_route_branch_weight,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    low_route_branch_weight = _nonnegative_loss_value(
        loss_cfg,
        "input_std_route_low_branch_weight",
        shared_route_branch_weight,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    route_branch_weight = max(
        shared_route_branch_weight,
        high_route_branch_weight,
        low_route_branch_weight,
    )
    a08_oracle_anchor_weight = _nonnegative_loss_value(
        loss_cfg,
        "input_std_route_a08_oracle_anchor_weight",
        0.0,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    a15_oracle_anchor_weight = _nonnegative_loss_value(
        loss_cfg,
        "input_std_route_a15_oracle_anchor_weight",
        0.0,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    oracle_gate_weight = _nonnegative_loss_value(
        loss_cfg,
        "input_std_route_oracle_gate_weight",
        0.0,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    gaussian_route_branch_weight = _nonnegative_loss_value(
        loss_cfg,
        "gaussian_route_control_branch_anchor_weight",
        0.0,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    gaussian_route_high_branch_weight = _nonnegative_loss_value(
        loss_cfg,
        "gaussian_route_control_high_branch_anchor_weight",
        gaussian_route_branch_weight,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    gaussian_route_low_branch_weight = _nonnegative_loss_value(
        loss_cfg,
        "gaussian_route_control_low_branch_anchor_weight",
        gaussian_route_branch_weight,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    gaussian_route_branch_required_weight = max(
        gaussian_route_branch_weight,
        gaussian_route_high_branch_weight,
        gaussian_route_low_branch_weight,
    )
    teachers_required = (
        weight > 0.0
        or route_branch_weight > 0.0
        or a08_oracle_anchor_weight > 0.0
        or a15_oracle_anchor_weight > 0.0
        or oracle_gate_weight > 0.0
        or gaussian_route_branch_required_weight > 0.0
    )
    if not teachers_required:
        return None, None, {
            "router_distillation_enabled": False,
            "router_distillation_weight": weight,
            "input_std_route_teacher_enabled": False,
            "input_std_route_branch_weight": shared_route_branch_weight,
            "input_std_route_high_branch_weight": high_route_branch_weight,
            "input_std_route_low_branch_weight": low_route_branch_weight,
            "input_std_route_a08_oracle_anchor_weight": a08_oracle_anchor_weight,
            "input_std_route_a15_oracle_anchor_weight": a15_oracle_anchor_weight,
            "input_std_route_oracle_gate_weight": oracle_gate_weight,
            "gaussian_route_control_branch_anchor_weight": gaussian_route_branch_weight,
            "gaussian_route_control_high_branch_anchor_weight": gaussian_route_high_branch_weight,
            "gaussian_route_control_low_branch_anchor_weight": gaussian_route_low_branch_weight,
        }
    raw_checkpoint_a = loss_cfg.get("router_distillation_checkpoint_a")
    raw_checkpoint_b = loss_cfg.get("router_distillation_checkpoint_b")
    if not raw_checkpoint_a or not raw_checkpoint_b:
        raise ValueError(
            "router or input_std route teacher losses require "
            "loss.router_distillation_checkpoint_a and loss.router_distillation_checkpoint_b"
        )
    checkpoint_a = resolve_checkpoint_path(raw_checkpoint_a, config_path)
    checkpoint_b = resolve_checkpoint_path(raw_checkpoint_b, config_path)
    if not checkpoint_a.is_file():
        raise FileNotFoundError(f"router_distillation_checkpoint_a does not exist: {checkpoint_a}")
    if not checkpoint_b.is_file():
        raise FileNotFoundError(f"router_distillation_checkpoint_b does not exist: {checkpoint_b}")
    config_a = None
    config_b = None
    raw_config_a = loss_cfg.get("router_distillation_config_a")
    raw_config_b = loss_cfg.get("router_distillation_config_b")
    if raw_config_a:
        config_a = resolve_checkpoint_path(raw_config_a, config_path)
        if not config_a.is_file():
            raise FileNotFoundError(f"router_distillation_config_a does not exist: {config_a}")
    if raw_config_b:
        config_b = resolve_checkpoint_path(raw_config_b, config_path)
        if not config_b.is_file():
            raise FileNotFoundError(f"router_distillation_config_b does not exist: {config_b}")
    strict = bool(loss_cfg.get("router_distillation_strict", True))
    teacher_a, info_a = _load_router_teacher(
        config,
        checkpoint_a,
        device,
        strict=strict,
        teacher_config_path=config_a,
    )
    teacher_b, info_b = _load_router_teacher(
        config,
        checkpoint_b,
        device,
        strict=strict,
        teacher_config_path=config_b,
    )
    return teacher_a, teacher_b, {
        "router_distillation_enabled": weight > 0.0,
        "router_distillation_weight": weight,
        "input_std_route_teacher_enabled": (
            route_branch_weight > 0.0
            or a08_oracle_anchor_weight > 0.0
            or a15_oracle_anchor_weight > 0.0
            or oracle_gate_weight > 0.0
            or gaussian_route_branch_required_weight > 0.0
        ),
        "input_std_route_branch_weight": shared_route_branch_weight,
        "input_std_route_high_branch_weight": high_route_branch_weight,
        "input_std_route_low_branch_weight": low_route_branch_weight,
        "input_std_route_a08_oracle_anchor_weight": a08_oracle_anchor_weight,
        "input_std_route_a15_oracle_anchor_weight": a15_oracle_anchor_weight,
        "input_std_route_oracle_gate_weight": oracle_gate_weight,
        "gaussian_route_control_branch_anchor_weight": gaussian_route_branch_weight,
        "gaussian_route_control_high_branch_anchor_weight": gaussian_route_high_branch_weight,
        "gaussian_route_control_low_branch_anchor_weight": gaussian_route_low_branch_weight,
        "router_distillation_checkpoint_a_label": str(loss_cfg.get("router_distillation_label_a", "A")),
        "router_distillation_checkpoint_a": info_a["checkpoint"],
        "router_distillation_config_a": info_a["config"],
        "router_distillation_checkpoint_a_step": info_a["checkpoint_step"],
        "router_distillation_checkpoint_a_missing_keys": info_a["missing_keys"],
        "router_distillation_checkpoint_a_unexpected_keys": info_a["unexpected_keys"],
        "router_distillation_checkpoint_b_label": str(loss_cfg.get("router_distillation_label_b", "B")),
        "router_distillation_checkpoint_b": info_b["checkpoint"],
        "router_distillation_config_b": info_b["config"],
        "router_distillation_checkpoint_b_step": info_b["checkpoint_step"],
        "router_distillation_checkpoint_b_missing_keys": info_b["missing_keys"],
        "router_distillation_checkpoint_b_unexpected_keys": info_b["unexpected_keys"],
        "router_distillation_strict": strict,
        "router_distillation_feature": str(loss_cfg.get("router_distillation_feature", "input_std")),
        "router_distillation_direction": str(loss_cfg.get("router_distillation_direction", "lte")),
        "router_distillation_threshold": float(loss_cfg.get("router_distillation_threshold", 0.0)),
        "router_distillation_softness": float(loss_cfg.get("router_distillation_softness", 0.01)),
    }


def load_input_std_route_trust_teacher_if_requested(
    config: Dict[str, Any],
    config_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module | None, Dict[str, Any]]:
    loss_cfg = config.get("loss", {})
    weight = _nonnegative_loss_value(
        loss_cfg,
        "input_std_route_trust_anchor_weight",
        0.0,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    if weight <= 0.0:
        return None, {
            "input_std_route_trust_anchor_enabled": False,
            "input_std_route_trust_anchor_weight": weight,
        }

    raw_checkpoint = loss_cfg.get("input_std_route_trust_anchor_checkpoint")
    if not raw_checkpoint:
        raise ValueError(
            "loss.input_std_route_trust_anchor_weight > 0 requires "
            "loss.input_std_route_trust_anchor_checkpoint"
        )
    checkpoint_path = resolve_checkpoint_path(raw_checkpoint, config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"input_std_route_trust_anchor_checkpoint does not exist: {checkpoint_path}")

    teacher_config_path = None
    raw_config = loss_cfg.get("input_std_route_trust_anchor_config")
    if raw_config:
        teacher_config_path = resolve_checkpoint_path(raw_config, config_path)
        if not teacher_config_path.is_file():
            raise FileNotFoundError(f"input_std_route_trust_anchor_config does not exist: {teacher_config_path}")

    strict = bool(loss_cfg.get("input_std_route_trust_anchor_strict", True))
    teacher, checkpoint_info = _load_router_teacher(
        config,
        checkpoint_path,
        device,
        strict=strict,
        teacher_config_path=teacher_config_path,
    )
    return teacher, {
        "input_std_route_trust_anchor_enabled": True,
        "input_std_route_trust_anchor_weight": weight,
        "input_std_route_trust_anchor_label": str(loss_cfg.get("input_std_route_trust_anchor_label", "trust")),
        "input_std_route_trust_anchor_checkpoint": checkpoint_info["checkpoint"],
        "input_std_route_trust_anchor_config": checkpoint_info["config"],
        "input_std_route_trust_anchor_checkpoint_step": checkpoint_info["checkpoint_step"],
        "input_std_route_trust_anchor_strict": checkpoint_info["strict"],
        "input_std_route_trust_anchor_missing_keys": checkpoint_info["missing_keys"],
        "input_std_route_trust_anchor_unexpected_keys": checkpoint_info["unexpected_keys"],
        "input_std_route_trust_anchor_temperature": float(
            loss_cfg.get("input_std_route_trust_anchor_temperature", 0.006)
        ),
        "input_std_route_trust_anchor_margin": float(loss_cfg.get("input_std_route_trust_anchor_margin", 0.0)),
        "input_std_route_trust_anchor_probability_floor": float(
            loss_cfg.get("input_std_route_trust_anchor_probability_floor", 0.0)
        ),
        "input_std_route_trust_anchor_require_positive_advantage": bool(
            loss_cfg.get("input_std_route_trust_anchor_require_positive_advantage", False)
        ),
    }


def input_std_route_residual_info(config: Dict[str, Any], router_info: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = config.get("model", {})
    loss_cfg = config.get("loss", {})
    residual_enabled = bool(model_cfg.get("use_input_std_route_residual", False))
    terminal_heads_enabled = bool(model_cfg.get("use_input_std_route_terminal_heads", False))
    feature_expert_enabled = bool(model_cfg.get("use_input_std_route_feature_expert", False))
    decoder_heads_enabled = bool(model_cfg.get("use_input_std_route_decoder_heads", False))
    post_adapter_enabled = bool(model_cfg.get("use_input_std_route_post_adapter", False))
    return {
        "input_std_route_residual_enabled": residual_enabled,
        "input_std_route_terminal_heads_enabled": terminal_heads_enabled,
        "input_std_route_feature_expert_enabled": feature_expert_enabled,
        "input_std_route_decoder_heads_enabled": decoder_heads_enabled,
        "input_std_route_post_adapter_enabled": post_adapter_enabled,
        "input_std_route_residual_blocks": int(model_cfg.get("input_std_route_residual_blocks", 1)),
        "input_std_route_residual_strength": float(model_cfg.get("input_std_route_residual_strength", 0.12)),
        "input_std_route_terminal_heads_blocks": int(model_cfg.get("input_std_route_terminal_heads_blocks", 2)),
        "input_std_route_terminal_heads_strength": float(
            model_cfg.get("input_std_route_terminal_heads_strength", 0.60)
        ),
        "input_std_route_feature_expert_blocks": int(model_cfg.get("input_std_route_feature_expert_blocks", 2)),
        "input_std_route_feature_expert_strength": float(
            model_cfg.get("input_std_route_feature_expert_strength", 0.35)
        ),
        "input_std_route_decoder_heads_blocks": int(model_cfg.get("input_std_route_decoder_heads_blocks", 1)),
        "input_std_route_decoder_heads_strength": float(
            model_cfg.get("input_std_route_decoder_heads_strength", 1.0)
        ),
        "input_std_route_post_adapter_blocks": int(model_cfg.get("input_std_route_post_adapter_blocks", 1)),
        "input_std_route_post_adapter_strength": float(
            model_cfg.get("input_std_route_post_adapter_strength", 0.08)
        ),
        "input_std_route_post_adapter_threshold": float(
            model_cfg.get(
                "input_std_route_post_adapter_threshold",
                model_cfg.get("input_std_route_threshold", 0.07026463374495506),
            )
        ),
        "input_std_route_post_adapter_softness": float(
            model_cfg.get("input_std_route_post_adapter_softness", model_cfg.get("input_std_route_softness", 0.003))
        ),
        "input_std_route_post_adapter_direction": str(
            model_cfg.get("input_std_route_post_adapter_direction", model_cfg.get("input_std_route_direction", "gt"))
        ),
        "input_std_route_post_adapter_detach_gate": bool(
            model_cfg.get("input_std_route_post_adapter_detach_gate", model_cfg.get("input_std_route_detach_gate", True))
        ),
        "input_std_route_threshold": float(model_cfg.get("input_std_route_threshold", 0.07026463374495506)),
        "input_std_route_softness": float(model_cfg.get("input_std_route_softness", 0.005)),
        "input_std_route_direction": str(model_cfg.get("input_std_route_direction", "gt")),
        "input_std_route_learned_gate_weight": float(model_cfg.get("input_std_route_learned_gate_weight", 0.0)),
        "input_std_route_learned_gate_mode": str(model_cfg.get("input_std_route_learned_gate_mode", "blend")),
        "input_std_route_detach_gate": bool(model_cfg.get("input_std_route_detach_gate", True)),
        "input_std_route_gate_weight": float(loss_cfg.get("input_std_route_gate_weight", 0.0)),
        "input_std_route_oracle_gate_weight": float(loss_cfg.get("input_std_route_oracle_gate_weight", 0.0)),
        "input_std_route_oracle_gate_temperature": float(
            loss_cfg.get("input_std_route_oracle_gate_temperature", 0.006)
        ),
        "input_std_route_oracle_gate_margin": float(loss_cfg.get("input_std_route_oracle_gate_margin", 0.0)),
        "input_std_route_oracle_gate_positive_target_floor": float(
            loss_cfg.get("input_std_route_oracle_gate_positive_target_floor", 0.0)
        ),
        "input_std_route_oracle_gate_negative_target_ceil": float(
            loss_cfg.get("input_std_route_oracle_gate_negative_target_ceil", 1.0)
        ),
        "input_std_route_oracle_gate_low_route_focus": bool(
            loss_cfg.get("input_std_route_oracle_gate_low_route_focus", False)
        ),
        "input_std_route_oracle_gate_focus_band": float(
            loss_cfg.get("input_std_route_oracle_gate_focus_band", 0.0)
        ),
        "input_std_route_branch_weight": float(loss_cfg.get("input_std_route_branch_weight", 0.0)),
        "input_std_route_high_branch_weight": float(
            loss_cfg.get("input_std_route_high_branch_weight", loss_cfg.get("input_std_route_branch_weight", 0.0))
        ),
        "input_std_route_low_branch_weight": float(
            loss_cfg.get("input_std_route_low_branch_weight", loss_cfg.get("input_std_route_branch_weight", 0.0))
        ),
        "input_std_route_delta_trust_weight": float(loss_cfg.get("input_std_route_delta_trust_weight", 0.0)),
        "input_std_route_delta_trust_budget": float(loss_cfg.get("input_std_route_delta_trust_budget", 0.0)),
        "input_std_route_a08_oracle_anchor_weight": float(
            loss_cfg.get("input_std_route_a08_oracle_anchor_weight", 0.0)
        ),
        "input_std_route_a08_oracle_anchor_temperature": float(
            loss_cfg.get("input_std_route_a08_oracle_anchor_temperature", 0.006)
        ),
        "input_std_route_a08_oracle_anchor_margin": float(
            loss_cfg.get("input_std_route_a08_oracle_anchor_margin", 0.0)
        ),
        "input_std_route_a08_oracle_anchor_probability_floor": float(
            loss_cfg.get("input_std_route_a08_oracle_anchor_probability_floor", 0.0)
        ),
        "input_std_route_a08_oracle_anchor_require_positive_advantage": bool(
            loss_cfg.get("input_std_route_a08_oracle_anchor_require_positive_advantage", False)
        ),
        "input_std_route_a15_oracle_anchor_weight": float(
            loss_cfg.get("input_std_route_a15_oracle_anchor_weight", 0.0)
        ),
        "input_std_route_a15_oracle_anchor_temperature": float(
            loss_cfg.get("input_std_route_a15_oracle_anchor_temperature", 0.006)
        ),
        "input_std_route_a15_oracle_anchor_margin": float(
            loss_cfg.get("input_std_route_a15_oracle_anchor_margin", 0.0)
        ),
        "input_std_route_a15_oracle_anchor_probability_floor": float(
            loss_cfg.get("input_std_route_a15_oracle_anchor_probability_floor", 0.0)
        ),
        "input_std_route_a15_oracle_anchor_require_positive_advantage": bool(
            loss_cfg.get("input_std_route_a15_oracle_anchor_require_positive_advantage", False)
        ),
        "input_std_route_trust_anchor_weight": float(loss_cfg.get("input_std_route_trust_anchor_weight", 0.0)),
        "input_std_route_trust_anchor_label": str(loss_cfg.get("input_std_route_trust_anchor_label", "")),
        "input_std_route_trust_anchor_temperature": float(
            loss_cfg.get("input_std_route_trust_anchor_temperature", 0.006)
        ),
        "input_std_route_trust_anchor_margin": float(loss_cfg.get("input_std_route_trust_anchor_margin", 0.0)),
        "input_std_route_trust_anchor_probability_floor": float(
            loss_cfg.get("input_std_route_trust_anchor_probability_floor", 0.0)
        ),
        "input_std_route_trust_anchor_require_positive_advantage": bool(
            loss_cfg.get("input_std_route_trust_anchor_require_positive_advantage", False)
        ),
        "input_std_route_branch_warmup_steps": int(loss_cfg.get("input_std_route_branch_warmup_steps", 0)),
        "input_std_route_branch_ramp_steps": int(loss_cfg.get("input_std_route_branch_ramp_steps", 0)),
        "input_std_route_gate_warmup_steps": int(loss_cfg.get("input_std_route_gate_warmup_steps", 0)),
        "input_std_route_gate_ramp_steps": int(loss_cfg.get("input_std_route_gate_ramp_steps", 0)),
        "input_std_route_teacher_enabled": bool(router_info.get("input_std_route_teacher_enabled", False)),
    }


def compute_teacher_consistency_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    teacher_prediction: torch.Tensor | None,
    config: Dict[str, Any],
) -> tuple[torch.Tensor, Dict[str, float]]:
    loss_cfg = config.get("loss", {})
    weight = _nonnegative_loss_value(
        loss_cfg,
        "teacher_consistency_weight",
        0.0,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    zero = prediction.new_tensor(0.0)
    if weight <= 0.0:
        return zero, {
            "loss_teacher_consistency": 0.0,
            "loss_teacher_consistency_weighted": 0.0,
            "teacher_consistency_mask_mean": 0.0,
            "teacher_consistency_error_mean": 0.0,
        }
    if teacher_prediction is None:
        raise ValueError("teacher_consistency_weight requires a loaded teacher model")
    if teacher_prediction.shape[-2:] != prediction.shape[-2:]:
        teacher_prediction = center_crop_like(teacher_prediction, prediction.shape[-2:])
    if target.shape[-2:] != prediction.shape[-2:]:
        target = center_crop_like(target, prediction.shape[-2:])
    threshold = _nonnegative_loss_value(loss_cfg, "teacher_consistency_threshold", 0.08)
    softness = float(loss_cfg.get("teacher_consistency_softness", 0.02))
    if softness <= 0.0:
        raise ValueError("loss.teacher_consistency_softness must be positive")
    teacher_error = (teacher_prediction - target).abs().mean(dim=1, keepdim=True).detach()
    teacher_mask = torch.sigmoid((threshold - teacher_error) / softness)
    consistency_diff = torch.sqrt((prediction - teacher_prediction.detach()).square() + 1e-6)
    denom = (teacher_mask.sum() * consistency_diff.shape[1]).clamp_min(1.0)
    consistency = (consistency_diff * teacher_mask).sum() / denom
    weighted = weight * consistency
    return weighted, {
        "loss_teacher_consistency": float(consistency.detach().cpu()),
        "loss_teacher_consistency_weighted": float(weighted.detach().cpu()),
        "teacher_consistency_mask_mean": float(teacher_mask.mean().detach().cpu()),
        "teacher_consistency_error_mean": float(teacher_error.mean().detach().cpu()),
    }


def compute_router_distillation_loss(
    prediction: torch.Tensor,
    input_image: torch.Tensor,
    teacher_a_prediction: torch.Tensor | None,
    teacher_b_prediction: torch.Tensor | None,
    config: Dict[str, Any],
) -> tuple[torch.Tensor, Dict[str, float]]:
    loss_cfg = config.get("loss", {})
    weight = _nonnegative_loss_value(
        loss_cfg,
        "router_distillation_weight",
        0.0,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    zero = prediction.new_tensor(0.0)
    if weight <= 0.0:
        return zero, {
            "loss_router_distillation": 0.0,
            "loss_router_distillation_weighted": 0.0,
            "router_distillation_choose_b_soft_mean": 0.0,
            "router_distillation_choose_b_hard_mean": 0.0,
            "router_distillation_input_std_mean": 0.0,
        }
    if teacher_a_prediction is None or teacher_b_prediction is None:
        raise ValueError("router_distillation_weight requires both router teacher predictions")
    feature = str(loss_cfg.get("router_distillation_feature", "input_std"))
    if feature != "input_std":
        raise ValueError("loss.router_distillation_feature currently supports only input_std")
    direction = str(loss_cfg.get("router_distillation_direction", "lte"))
    if direction not in {"lte", "gte"}:
        raise ValueError("loss.router_distillation_direction must be 'lte' or 'gte'")
    threshold = float(loss_cfg.get("router_distillation_threshold", 0.0))
    softness = float(loss_cfg.get("router_distillation_softness", 0.01))
    if softness <= 0.0:
        raise ValueError("loss.router_distillation_softness must be positive")
    if teacher_a_prediction.shape[-2:] != prediction.shape[-2:]:
        teacher_a_prediction = center_crop_like(teacher_a_prediction, prediction.shape[-2:])
    if teacher_b_prediction.shape[-2:] != prediction.shape[-2:]:
        teacher_b_prediction = center_crop_like(teacher_b_prediction, prediction.shape[-2:])
    effective_input = input_image
    if effective_input.shape[-2:] != prediction.shape[-2:]:
        effective_input = center_crop_like(effective_input, prediction.shape[-2:])
    input_std = effective_input.detach().flatten(start_dim=1).std(dim=1)
    if direction == "lte":
        choose_b_soft = torch.sigmoid((threshold - input_std) / softness)
        choose_b_hard = (input_std <= threshold).float()
    else:
        choose_b_soft = torch.sigmoid((input_std - threshold) / softness)
        choose_b_hard = (input_std >= threshold).float()
    blend = choose_b_soft.view(-1, 1, 1, 1)
    distillation_target = teacher_a_prediction.detach() * (1.0 - blend) + teacher_b_prediction.detach() * blend
    consistency = torch.sqrt((prediction - distillation_target).square() + 1e-6).mean()
    weighted = weight * consistency
    return weighted, {
        "loss_router_distillation": float(consistency.detach().cpu()),
        "loss_router_distillation_weighted": float(weighted.detach().cpu()),
        "router_distillation_choose_b_soft_mean": float(choose_b_soft.mean().detach().cpu()),
        "router_distillation_choose_b_hard_mean": float(choose_b_hard.mean().detach().cpu()),
        "router_distillation_input_std_mean": float(input_std.mean().detach().cpu()),
        "router_distillation_input_std_min": float(input_std.min().detach().cpu()),
        "router_distillation_input_std_max": float(input_std.max().detach().cpu()),
        "router_distillation_input_std_threshold": threshold,
    }


def _input_std_route_target(
    input_image: torch.Tensor,
    prediction: torch.Tensor,
    config: Dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loss_cfg = config.get("loss", {})
    route_cfg = config.get("model", {})
    threshold = float(
        loss_cfg.get(
            "input_std_route_threshold",
            route_cfg.get("input_std_route_threshold", 0.07026463374495506),
        )
    )
    softness = float(
        loss_cfg.get(
            "input_std_route_softness",
            route_cfg.get("input_std_route_softness", 0.005),
        )
    )
    if softness <= 0.0:
        raise ValueError("input_std_route_softness must be positive")
    direction = str(
        loss_cfg.get(
            "input_std_route_direction",
            route_cfg.get("input_std_route_direction", "gt"),
        )
    )
    if direction not in {"gt", "gte", "lt", "lte"}:
        raise ValueError("input_std_route_direction must be gt, gte, lt, or lte")
    effective_input = input_image
    if effective_input.shape[-2:] != prediction.shape[-2:]:
        effective_input = center_crop_like(effective_input, prediction.shape[-2:])
    input_std = effective_input.detach().flatten(start_dim=1).std(dim=1, unbiased=False)
    if direction in {"gt", "gte"}:
        high_soft = torch.sigmoid((input_std - threshold) / softness)
        high_hard = (input_std >= threshold).float()
    else:
        high_soft = torch.sigmoid((threshold - input_std) / softness)
        high_hard = (input_std <= threshold).float()
    return input_std, high_soft, high_hard


def compute_input_std_route_residual_loss(
    aux: Dict[str, torch.Tensor | None],
    prediction: torch.Tensor,
    input_image: torch.Tensor,
    target: torch.Tensor | None,
    teacher_a_prediction: torch.Tensor | None,
    teacher_b_prediction: torch.Tensor | None,
    trust_anchor_prediction: torch.Tensor | None,
    config: Dict[str, Any],
    *,
    step: int | None = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    loss_cfg = config.get("loss", {})
    training_cfg = config.get("training", {})
    gate_weight = _nonnegative_loss_value(
        loss_cfg,
        "input_std_route_gate_weight",
        0.0,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    shared_branch_weight = _nonnegative_loss_value(
        loss_cfg,
        "input_std_route_branch_weight",
        0.0,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    high_branch_weight = _nonnegative_loss_value(
        loss_cfg,
        "input_std_route_high_branch_weight",
        shared_branch_weight,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    low_branch_weight = _nonnegative_loss_value(
        loss_cfg,
        "input_std_route_low_branch_weight",
        shared_branch_weight,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    route_delta_weight = _nonnegative_loss_value(
        loss_cfg,
        "input_std_route_delta_trust_weight",
        0.0,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    oracle_gate_weight = _nonnegative_loss_value(
        loss_cfg,
        "input_std_route_oracle_gate_weight",
        0.0,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    a08_oracle_anchor_weight = _nonnegative_loss_value(
        loss_cfg,
        "input_std_route_a08_oracle_anchor_weight",
        0.0,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    a15_oracle_anchor_weight = _nonnegative_loss_value(
        loss_cfg,
        "input_std_route_a15_oracle_anchor_weight",
        0.0,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    trust_anchor_weight = _nonnegative_loss_value(
        loss_cfg,
        "input_std_route_trust_anchor_weight",
        0.0,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    route_delta_budget = float(loss_cfg.get("input_std_route_delta_trust_budget", 0.0))
    if route_delta_budget < 0.0:
        raise ValueError("input_std_route_delta_trust_budget must be non-negative")
    branch_warmup_steps = max(int(loss_cfg.get("input_std_route_branch_warmup_steps", 0)), 0)
    branch_ramp_steps = max(int(loss_cfg.get("input_std_route_branch_ramp_steps", 0)), 0)
    gate_warmup_steps = max(int(loss_cfg.get("input_std_route_gate_warmup_steps", 0)), 0)
    gate_ramp_steps = max(int(loss_cfg.get("input_std_route_gate_ramp_steps", 0)), 0)
    effective_step = int(step if step is not None else training_cfg.get("current_step", 0))

    def schedule_scale(warmup_steps: int, ramp_steps: int) -> float:
        if effective_step <= warmup_steps:
            return 0.0
        if ramp_steps <= 0:
            return 1.0
        return min(max((effective_step - warmup_steps) / ramp_steps, 0.0), 1.0)

    branch_scale = schedule_scale(branch_warmup_steps, branch_ramp_steps)
    gate_scale = schedule_scale(gate_warmup_steps, gate_ramp_steps)
    high_branch_weight *= branch_scale
    low_branch_weight *= branch_scale
    gate_weight *= gate_scale
    zero = prediction.new_tensor(0.0)
    default_parts = {
        "loss_input_std_route_gate": 0.0,
        "loss_input_std_route_gate_weighted": 0.0,
        "loss_input_std_route_high_branch": 0.0,
        "loss_input_std_route_low_branch": 0.0,
        "loss_input_std_route_branch_weighted": 0.0,
        "loss_input_std_route_high_branch_weighted": 0.0,
        "loss_input_std_route_low_branch_weighted": 0.0,
        "loss_input_std_route_delta_trust": 0.0,
        "loss_input_std_route_delta_trust_weighted": 0.0,
        "loss_input_std_route_oracle_gate": 0.0,
        "loss_input_std_route_oracle_gate_weighted": 0.0,
        "loss_input_std_route_a08_oracle_anchor": 0.0,
        "loss_input_std_route_a08_oracle_anchor_weighted": 0.0,
        "loss_input_std_route_a15_oracle_anchor": 0.0,
        "loss_input_std_route_a15_oracle_anchor_weighted": 0.0,
        "loss_input_std_route_trust_anchor": 0.0,
        "loss_input_std_route_trust_anchor_weighted": 0.0,
        "input_std_route_high_branch_weight": high_branch_weight,
        "input_std_route_low_branch_weight": low_branch_weight,
        "input_std_route_a08_oracle_anchor_weight": a08_oracle_anchor_weight,
        "input_std_route_a15_oracle_anchor_weight": a15_oracle_anchor_weight,
        "input_std_route_trust_anchor_weight": trust_anchor_weight,
        "input_std_route_branch_schedule_scale": branch_scale,
        "input_std_route_gate_schedule_scale": gate_scale,
        "input_std_route_delta_trust_weight": route_delta_weight,
        "input_std_route_delta_trust_budget": route_delta_budget,
        "input_std_route_oracle_gate_weight": oracle_gate_weight,
        "input_std_route_selected_delta_abs_mean": 0.0,
        "input_std_route_high_delta_abs_mean": 0.0,
        "input_std_route_low_delta_abs_mean": 0.0,
        "input_std_route_gate_mean": 0.0,
        "input_std_route_prior_mean": 0.0,
        "input_std_route_high_soft_mean": 0.0,
        "input_std_route_high_hard_mean": 0.0,
        "input_std_route_input_std_mean": 0.0,
        "input_std_route_high_branch_oracle_weighting": 0.0,
        "input_std_route_high_branch_oracle_weight_mean": 1.0,
        "input_std_route_high_branch_oracle_weight_min": 1.0,
        "input_std_route_high_branch_oracle_weight_max": 1.0,
        "input_std_route_high_branch_oracle_a_better_fraction": 0.0,
        "input_std_route_high_branch_oracle_advantage_mean": 0.0,
        "input_std_route_high_branch_oracle_teacher_a_error_mean": 0.0,
        "input_std_route_high_branch_oracle_teacher_b_error_mean": 0.0,
        "input_std_route_oracle_gate_target_mean": 0.0,
        "input_std_route_oracle_gate_weight_mean": 0.0,
        "input_std_route_oracle_gate_positive_fraction": 0.0,
        "input_std_route_oracle_gate_advantage_mean": 0.0,
        "input_std_route_oracle_gate_teacher_a_error_mean": 0.0,
        "input_std_route_oracle_gate_teacher_b_error_mean": 0.0,
        "input_std_route_oracle_gate_low_focus_fraction": 0.0,
        "input_std_route_oracle_gate_band_focus_fraction": 0.0,
        "input_std_route_a08_oracle_anchor_probability_mean": 0.0,
        "input_std_route_a08_oracle_anchor_probability_min": 0.0,
        "input_std_route_a08_oracle_anchor_probability_max": 0.0,
        "input_std_route_a08_oracle_anchor_a_better_fraction": 0.0,
        "input_std_route_a08_oracle_anchor_advantage_mean": 0.0,
        "input_std_route_a08_oracle_anchor_teacher_a_error_mean": 0.0,
        "input_std_route_a08_oracle_anchor_teacher_b_error_mean": 0.0,
        "input_std_route_a15_oracle_anchor_probability_mean": 0.0,
        "input_std_route_a15_oracle_anchor_probability_min": 0.0,
        "input_std_route_a15_oracle_anchor_probability_max": 0.0,
        "input_std_route_a15_oracle_anchor_b_better_fraction": 0.0,
        "input_std_route_a15_oracle_anchor_advantage_mean": 0.0,
        "input_std_route_a15_oracle_anchor_teacher_a_error_mean": 0.0,
        "input_std_route_a15_oracle_anchor_teacher_b_error_mean": 0.0,
        "input_std_route_trust_anchor_probability_mean": 0.0,
        "input_std_route_trust_anchor_probability_min": 0.0,
        "input_std_route_trust_anchor_probability_max": 0.0,
        "input_std_route_trust_anchor_better_fraction": 0.0,
        "input_std_route_trust_anchor_advantage_mean": 0.0,
        "input_std_route_trust_anchor_teacher_error_mean": 0.0,
        "input_std_route_trust_anchor_prediction_error_mean": 0.0,
    }
    if (
        gate_weight <= 0.0
        and high_branch_weight <= 0.0
        and low_branch_weight <= 0.0
        and route_delta_weight <= 0.0
        and oracle_gate_weight <= 0.0
        and a08_oracle_anchor_weight <= 0.0
        and a15_oracle_anchor_weight <= 0.0
        and trust_anchor_weight <= 0.0
    ):
        return zero, default_parts

    gate = aux.get("input_std_route_gate")
    prior = aux.get("input_std_route_prior")
    high_restored = aux.get("input_std_route_high_restored")
    low_restored = aux.get("input_std_route_low_restored")
    selected_delta = aux.get("input_std_route_delta")
    high_delta = aux.get("input_std_route_high_delta")
    low_delta = aux.get("input_std_route_low_delta")
    if gate is None:
        raise ValueError(
            "input_std route losses require model use_input_std_route_residual=true "
            "or use_input_std_route_terminal_heads=true "
            "or use_input_std_route_feature_expert=true "
            "or use_input_std_route_decoder_heads=true "
            "or use_input_std_route_post_adapter=true"
        )
    if gate.shape[-2:] != prediction.shape[-2:]:
        gate = center_crop_like(gate, prediction.shape[-2:])
    if prior is not None and prior.shape[-2:] != prediction.shape[-2:]:
        prior = center_crop_like(prior, prediction.shape[-2:])

    input_std, high_soft, high_hard = _input_std_route_target(input_image, prediction, config)
    target_gate = high_soft.view(-1, 1, 1, 1).expand_as(gate)
    total = zero
    gate_loss = zero
    if gate_weight > 0.0:
        gate_loss = F.binary_cross_entropy(gate.clamp(1e-5, 1.0 - 1e-5), target_gate.detach())
        total = total + gate_weight * gate_loss

    oracle_gate_loss = zero
    oracle_gate_target_mean = 0.0
    oracle_gate_weight_mean = 0.0
    oracle_gate_positive_fraction = 0.0
    oracle_gate_advantage_mean = 0.0
    oracle_gate_a_error_mean = 0.0
    oracle_gate_b_error_mean = 0.0
    oracle_gate_low_focus_fraction = 0.0
    oracle_gate_band_focus_fraction = 0.0
    if oracle_gate_weight > 0.0:
        if target is None:
            raise ValueError("input_std_route_oracle_gate_weight requires target tensor")
        if teacher_a_prediction is None or teacher_b_prediction is None:
            raise ValueError("input_std_route_oracle_gate_weight requires both router teacher predictions")
        if target.shape[-2:] != prediction.shape[-2:]:
            target = center_crop_like(target, prediction.shape[-2:])
        if teacher_a_prediction.shape[-2:] != prediction.shape[-2:]:
            teacher_a_prediction = center_crop_like(teacher_a_prediction, prediction.shape[-2:])
        if teacher_b_prediction.shape[-2:] != prediction.shape[-2:]:
            teacher_b_prediction = center_crop_like(teacher_b_prediction, prediction.shape[-2:])
        temperature = float(loss_cfg.get("input_std_route_oracle_gate_temperature", 0.006))
        margin = float(loss_cfg.get("input_std_route_oracle_gate_margin", 0.0))
        positive_floor = float(loss_cfg.get("input_std_route_oracle_gate_positive_target_floor", 0.0))
        negative_ceil = float(loss_cfg.get("input_std_route_oracle_gate_negative_target_ceil", 1.0))
        low_route_focus = bool(loss_cfg.get("input_std_route_oracle_gate_low_route_focus", False))
        focus_band = float(loss_cfg.get("input_std_route_oracle_gate_focus_band", 0.0))
        if temperature <= 0.0:
            raise ValueError("input_std_route_oracle_gate_temperature must be positive")
        if not 0.0 <= positive_floor <= 1.0:
            raise ValueError("input_std_route_oracle_gate_positive_target_floor must be in [0, 1]")
        if not 0.0 <= negative_ceil <= 1.0:
            raise ValueError("input_std_route_oracle_gate_negative_target_ceil must be in [0, 1]")
        if focus_band < 0.0:
            raise ValueError("input_std_route_oracle_gate_focus_band must be non-negative")
        teacher_a_error = torch.sqrt((teacher_a_prediction.detach() - target).square() + 1e-6).flatten(1).mean(1)
        teacher_b_error = torch.sqrt((teacher_b_prediction.detach() - target).square() + 1e-6).flatten(1).mean(1)
        teacher_a_advantage = teacher_b_error - teacher_a_error
        oracle_target = torch.sigmoid((teacher_a_advantage - margin) / temperature)
        if positive_floor > 0.0:
            oracle_target = torch.where(
                teacher_a_advantage > margin,
                torch.maximum(oracle_target, oracle_target.new_full(oracle_target.shape, positive_floor)),
                oracle_target,
            )
        if negative_ceil < 1.0:
            oracle_target = torch.where(
                teacher_a_advantage <= margin,
                torch.minimum(oracle_target, oracle_target.new_full(oracle_target.shape, negative_ceil)),
                oracle_target,
            )
        sample_weight = torch.ones_like(oracle_target)
        low_focus_mask = high_hard <= 0.0
        if low_route_focus:
            sample_weight = sample_weight * low_focus_mask.float()
        band_focus_mask = torch.ones_like(sample_weight, dtype=torch.bool)
        if focus_band > 0.0:
            threshold = float(loss_cfg.get("input_std_route_threshold", config.get("model", {}).get("input_std_route_threshold", 0.07026463374495506)))
            band_focus_mask = (input_std - threshold).abs() <= focus_band
            sample_weight = sample_weight * band_focus_mask.float()
        target_gate_oracle = oracle_target.view(-1, 1, 1, 1).expand_as(gate).detach()
        gate_bce = F.binary_cross_entropy(gate.clamp(1e-5, 1.0 - 1e-5), target_gate_oracle, reduction="none")
        gate_weight_map = sample_weight.view(-1, 1, 1, 1).detach()
        gate_weight_denom = (
            gate_weight_map.sum() * gate_bce.shape[1] * gate_bce.shape[2] * gate_bce.shape[3]
        ).clamp_min(1.0)
        oracle_gate_loss = (gate_bce * gate_weight_map).sum() / gate_weight_denom
        total = total + oracle_gate_weight * oracle_gate_loss
        oracle_gate_target_mean = float(oracle_target.detach().mean().cpu())
        oracle_gate_weight_mean = float(sample_weight.detach().mean().cpu())
        oracle_gate_positive_fraction = float((teacher_a_advantage.detach() > margin).float().mean().cpu())
        oracle_gate_advantage_mean = float(teacher_a_advantage.detach().mean().cpu())
        oracle_gate_a_error_mean = float(teacher_a_error.detach().mean().cpu())
        oracle_gate_b_error_mean = float(teacher_b_error.detach().mean().cpu())
        oracle_gate_low_focus_fraction = float(low_focus_mask.detach().float().mean().cpu())
        oracle_gate_band_focus_fraction = float(band_focus_mask.detach().float().mean().cpu())

    high_branch_loss = zero
    low_branch_loss = zero
    branch_weighted = zero
    high_weight = prediction.new_ones((prediction.shape[0], 1, 1, 1))
    high_oracle_a_error_mean = 0.0
    high_oracle_b_error_mean = 0.0
    high_oracle_advantage_mean = 0.0
    high_oracle_a_better_fraction = 0.0
    if high_branch_weight > 0.0 or low_branch_weight > 0.0:
        if high_restored is None or low_restored is None:
            raise ValueError("input_std route branch weights require route high/low restored aux outputs")
        if teacher_a_prediction is None or teacher_b_prediction is None:
            raise ValueError("input_std route branch weights require both router teacher predictions")
        if high_restored.shape[-2:] != prediction.shape[-2:]:
            high_restored = center_crop_like(high_restored, prediction.shape[-2:])
        if low_restored.shape[-2:] != prediction.shape[-2:]:
            low_restored = center_crop_like(low_restored, prediction.shape[-2:])
        if teacher_a_prediction.shape[-2:] != prediction.shape[-2:]:
            teacher_a_prediction = center_crop_like(teacher_a_prediction, prediction.shape[-2:])
        if teacher_b_prediction.shape[-2:] != prediction.shape[-2:]:
            teacher_b_prediction = center_crop_like(teacher_b_prediction, prediction.shape[-2:])

        high_mask = high_soft.view(-1, 1, 1, 1).detach()
        low_mask = (1.0 - high_soft).view(-1, 1, 1, 1).detach()
        high_diff = torch.sqrt((high_restored - teacher_a_prediction.detach()).square() + 1e-6)
        low_diff = torch.sqrt((low_restored - teacher_b_prediction.detach()).square() + 1e-6)
        high_denom = (high_mask.sum() * high_diff.shape[1] * high_diff.shape[2] * high_diff.shape[3]).clamp_min(
            1.0
        )
        low_denom = (
            low_mask.sum() * low_diff.shape[1] * low_diff.shape[2] * low_diff.shape[3]
        ).clamp_min(1.0)
        high_oracle_weighting = bool(loss_cfg.get("input_std_route_high_branch_oracle_weighting", False))
        if high_oracle_weighting:
            if target is None:
                raise ValueError("input_std_route_high_branch_oracle_weighting requires target tensor")
            if target.shape[-2:] != prediction.shape[-2:]:
                target = center_crop_like(target, prediction.shape[-2:])
            weight_min = float(loss_cfg.get("input_std_route_high_branch_oracle_weight_min", 0.5))
            weight_max = float(loss_cfg.get("input_std_route_high_branch_oracle_weight_max", 2.0))
            temperature = float(loss_cfg.get("input_std_route_high_branch_oracle_weight_temperature", 0.01))
            margin = float(loss_cfg.get("input_std_route_high_branch_oracle_weight_margin", 0.0))
            if weight_min < 0.0:
                raise ValueError("input_std_route_high_branch_oracle_weight_min must be non-negative")
            if weight_max < weight_min:
                raise ValueError("input_std_route_high_branch_oracle_weight_max must be >= min")
            if weight_max > 8.0:
                raise ValueError("input_std_route_high_branch_oracle_weight_max must be <= 8.0")
            if temperature <= 0.0:
                raise ValueError("input_std_route_high_branch_oracle_weight_temperature must be positive")
            teacher_a_error = torch.sqrt((teacher_a_prediction.detach() - target).square() + 1e-6).flatten(1).mean(1)
            teacher_b_error = torch.sqrt((teacher_b_prediction.detach() - target).square() + 1e-6).flatten(1).mean(1)
            # Positive advantage means the high/A08 teacher is closer to GT than the low/A15 teacher
            # on this training crop; use it only as a detached routing emphasis signal.
            teacher_a_advantage = teacher_b_error - teacher_a_error
            oracle_probability = torch.sigmoid((teacher_a_advantage - margin) / temperature)
            high_weight = (weight_min + (weight_max - weight_min) * oracle_probability).view(-1, 1, 1, 1).detach()
            high_oracle_a_error_mean = float(teacher_a_error.detach().mean().cpu())
            high_oracle_b_error_mean = float(teacher_b_error.detach().mean().cpu())
            high_oracle_advantage_mean = float(teacher_a_advantage.detach().mean().cpu())
            high_oracle_a_better_fraction = float((teacher_a_advantage.detach() > 0.0).float().mean().cpu())
        high_branch_loss = (high_diff * high_mask * high_weight).sum() / high_denom
        low_branch_loss = (low_diff * low_mask).sum() / low_denom
        branch_weighted = high_branch_weight * high_branch_loss + low_branch_weight * low_branch_loss
        total = total + branch_weighted

    route_delta_loss = zero
    selected_delta_abs_mean = 0.0
    high_delta_abs_mean = 0.0
    low_delta_abs_mean = 0.0
    if route_delta_weight > 0.0:
        if selected_delta is None:
            raise ValueError("input_std_route_delta_trust_weight requires input_std_route_delta aux output")
        if selected_delta.shape[-2:] != prediction.shape[-2:]:
            selected_delta = center_crop_like(selected_delta, prediction.shape[-2:])
        selected_delta_abs = selected_delta.abs()
        selected_delta_abs_mean = float(selected_delta_abs.detach().mean().cpu())
        route_delta_loss = torch.relu(selected_delta_abs - route_delta_budget).mean()
        total = total + route_delta_weight * route_delta_loss
    if high_delta is not None:
        if high_delta.shape[-2:] != prediction.shape[-2:]:
            high_delta = center_crop_like(high_delta, prediction.shape[-2:])
        high_delta_abs_mean = float(high_delta.detach().abs().mean().cpu())
    if low_delta is not None:
        if low_delta.shape[-2:] != prediction.shape[-2:]:
            low_delta = center_crop_like(low_delta, prediction.shape[-2:])
        low_delta_abs_mean = float(low_delta.detach().abs().mean().cpu())

    a08_oracle_anchor_loss = zero
    a08_anchor_probability = prediction.new_zeros((prediction.shape[0], 1, 1, 1))
    a08_anchor_a_error_mean = 0.0
    a08_anchor_b_error_mean = 0.0
    a08_anchor_advantage_mean = 0.0
    a08_anchor_a_better_fraction = 0.0
    if a08_oracle_anchor_weight > 0.0:
        if target is None:
            raise ValueError("input_std_route_a08_oracle_anchor_weight requires target tensor")
        if teacher_a_prediction is None or teacher_b_prediction is None:
            raise ValueError("input_std_route_a08_oracle_anchor_weight requires both router teacher predictions")
        if target.shape[-2:] != prediction.shape[-2:]:
            target = center_crop_like(target, prediction.shape[-2:])
        if teacher_a_prediction.shape[-2:] != prediction.shape[-2:]:
            teacher_a_prediction = center_crop_like(teacher_a_prediction, prediction.shape[-2:])
        if teacher_b_prediction.shape[-2:] != prediction.shape[-2:]:
            teacher_b_prediction = center_crop_like(teacher_b_prediction, prediction.shape[-2:])
        temperature = float(loss_cfg.get("input_std_route_a08_oracle_anchor_temperature", 0.006))
        margin = float(loss_cfg.get("input_std_route_a08_oracle_anchor_margin", 0.0))
        probability_floor = float(loss_cfg.get("input_std_route_a08_oracle_anchor_probability_floor", 0.0))
        require_positive_advantage = bool(
            loss_cfg.get("input_std_route_a08_oracle_anchor_require_positive_advantage", False)
        )
        if temperature <= 0.0:
            raise ValueError("input_std_route_a08_oracle_anchor_temperature must be positive")
        if not 0.0 <= probability_floor <= 1.0:
            raise ValueError("input_std_route_a08_oracle_anchor_probability_floor must be in [0, 1]")
        teacher_a_error = torch.sqrt((teacher_a_prediction.detach() - target).square() + 1e-6).flatten(1).mean(1)
        teacher_b_error = torch.sqrt((teacher_b_prediction.detach() - target).square() + 1e-6).flatten(1).mean(1)
        teacher_a_advantage = teacher_b_error - teacher_a_error
        oracle_probability = torch.sigmoid((teacher_a_advantage - margin) / temperature)
        if require_positive_advantage:
            oracle_probability = torch.where(
                teacher_a_advantage > margin,
                oracle_probability,
                torch.zeros_like(oracle_probability),
            )
        if probability_floor > 0.0:
            oracle_probability = torch.where(
                oracle_probability >= probability_floor,
                oracle_probability,
                torch.zeros_like(oracle_probability),
            )
        a08_anchor_probability = oracle_probability.view(-1, 1, 1, 1).detach()
        a08_anchor_diff = torch.sqrt((prediction - teacher_a_prediction.detach()).square() + 1e-6)
        a08_anchor_denom = (
            a08_anchor_probability.sum()
            * a08_anchor_diff.shape[1]
            * a08_anchor_diff.shape[2]
            * a08_anchor_diff.shape[3]
        ).clamp_min(1.0)
        a08_oracle_anchor_loss = (a08_anchor_diff * a08_anchor_probability).sum() / a08_anchor_denom
        total = total + a08_oracle_anchor_weight * a08_oracle_anchor_loss
        a08_anchor_a_error_mean = float(teacher_a_error.detach().mean().cpu())
        a08_anchor_b_error_mean = float(teacher_b_error.detach().mean().cpu())
        a08_anchor_advantage_mean = float(teacher_a_advantage.detach().mean().cpu())
        a08_anchor_a_better_fraction = float((teacher_a_advantage.detach() > 0.0).float().mean().cpu())

    a15_oracle_anchor_loss = zero
    a15_anchor_probability = prediction.new_zeros((prediction.shape[0], 1, 1, 1))
    a15_anchor_a_error_mean = 0.0
    a15_anchor_b_error_mean = 0.0
    a15_anchor_advantage_mean = 0.0
    a15_anchor_b_better_fraction = 0.0
    if a15_oracle_anchor_weight > 0.0:
        if target is None:
            raise ValueError("input_std_route_a15_oracle_anchor_weight requires target tensor")
        if teacher_a_prediction is None or teacher_b_prediction is None:
            raise ValueError("input_std_route_a15_oracle_anchor_weight requires both router teacher predictions")
        if target.shape[-2:] != prediction.shape[-2:]:
            target = center_crop_like(target, prediction.shape[-2:])
        if teacher_a_prediction.shape[-2:] != prediction.shape[-2:]:
            teacher_a_prediction = center_crop_like(teacher_a_prediction, prediction.shape[-2:])
        if teacher_b_prediction.shape[-2:] != prediction.shape[-2:]:
            teacher_b_prediction = center_crop_like(teacher_b_prediction, prediction.shape[-2:])
        temperature = float(loss_cfg.get("input_std_route_a15_oracle_anchor_temperature", 0.006))
        margin = float(loss_cfg.get("input_std_route_a15_oracle_anchor_margin", 0.0))
        probability_floor = float(loss_cfg.get("input_std_route_a15_oracle_anchor_probability_floor", 0.0))
        require_positive_advantage = bool(
            loss_cfg.get("input_std_route_a15_oracle_anchor_require_positive_advantage", False)
        )
        if temperature <= 0.0:
            raise ValueError("input_std_route_a15_oracle_anchor_temperature must be positive")
        if not 0.0 <= probability_floor <= 1.0:
            raise ValueError("input_std_route_a15_oracle_anchor_probability_floor must be in [0, 1]")
        teacher_a_error = torch.sqrt((teacher_a_prediction.detach() - target).square() + 1e-6).flatten(1).mean(1)
        teacher_b_error = torch.sqrt((teacher_b_prediction.detach() - target).square() + 1e-6).flatten(1).mean(1)
        teacher_b_advantage = teacher_a_error - teacher_b_error
        oracle_probability = torch.sigmoid((teacher_b_advantage - margin) / temperature)
        if require_positive_advantage:
            oracle_probability = torch.where(
                teacher_b_advantage > margin,
                oracle_probability,
                torch.zeros_like(oracle_probability),
            )
        if probability_floor > 0.0:
            oracle_probability = torch.where(
                oracle_probability >= probability_floor,
                oracle_probability,
                torch.zeros_like(oracle_probability),
            )
        a15_anchor_probability = oracle_probability.view(-1, 1, 1, 1).detach()
        a15_anchor_diff = torch.sqrt((prediction - teacher_b_prediction.detach()).square() + 1e-6)
        a15_anchor_denom = (
            a15_anchor_probability.sum()
            * a15_anchor_diff.shape[1]
            * a15_anchor_diff.shape[2]
            * a15_anchor_diff.shape[3]
        ).clamp_min(1.0)
        a15_oracle_anchor_loss = (a15_anchor_diff * a15_anchor_probability).sum() / a15_anchor_denom
        total = total + a15_oracle_anchor_weight * a15_oracle_anchor_loss
        a15_anchor_a_error_mean = float(teacher_a_error.detach().mean().cpu())
        a15_anchor_b_error_mean = float(teacher_b_error.detach().mean().cpu())
        a15_anchor_advantage_mean = float(teacher_b_advantage.detach().mean().cpu())
        a15_anchor_b_better_fraction = float((teacher_b_advantage.detach() > 0.0).float().mean().cpu())

    trust_anchor_loss = zero
    trust_anchor_probability = prediction.new_zeros((prediction.shape[0], 1, 1, 1))
    trust_anchor_teacher_error_mean = 0.0
    trust_anchor_prediction_error_mean = 0.0
    trust_anchor_advantage_mean = 0.0
    trust_anchor_better_fraction = 0.0
    if trust_anchor_weight > 0.0:
        if target is None:
            raise ValueError("input_std_route_trust_anchor_weight requires target tensor")
        if trust_anchor_prediction is None:
            raise ValueError("input_std_route_trust_anchor_weight requires loaded trust anchor prediction")
        if target.shape[-2:] != prediction.shape[-2:]:
            target = center_crop_like(target, prediction.shape[-2:])
        if trust_anchor_prediction.shape[-2:] != prediction.shape[-2:]:
            trust_anchor_prediction = center_crop_like(trust_anchor_prediction, prediction.shape[-2:])
        temperature = float(loss_cfg.get("input_std_route_trust_anchor_temperature", 0.006))
        margin = float(loss_cfg.get("input_std_route_trust_anchor_margin", 0.0))
        probability_floor = float(loss_cfg.get("input_std_route_trust_anchor_probability_floor", 0.0))
        require_positive_advantage = bool(
            loss_cfg.get("input_std_route_trust_anchor_require_positive_advantage", False)
        )
        if temperature <= 0.0:
            raise ValueError("input_std_route_trust_anchor_temperature must be positive")
        if not 0.0 <= probability_floor <= 1.0:
            raise ValueError("input_std_route_trust_anchor_probability_floor must be in [0, 1]")
        trust_teacher_error = torch.sqrt(
            (trust_anchor_prediction.detach() - target).square() + 1e-6
        ).flatten(1).mean(1)
        student_error = torch.sqrt((prediction.detach() - target).square() + 1e-6).flatten(1).mean(1)
        trust_advantage = student_error - trust_teacher_error
        trust_probability = torch.sigmoid((trust_advantage - margin) / temperature)
        if require_positive_advantage:
            trust_probability = torch.where(
                trust_advantage > margin,
                trust_probability,
                torch.zeros_like(trust_probability),
            )
        if probability_floor > 0.0:
            trust_probability = torch.where(
                trust_probability >= probability_floor,
                trust_probability,
                torch.zeros_like(trust_probability),
            )
        trust_anchor_probability = trust_probability.view(-1, 1, 1, 1).detach()
        trust_diff = torch.sqrt((prediction - trust_anchor_prediction.detach()).square() + 1e-6)
        trust_denom = (
            trust_anchor_probability.sum() * trust_diff.shape[1] * trust_diff.shape[2] * trust_diff.shape[3]
        ).clamp_min(1.0)
        trust_anchor_loss = (trust_diff * trust_anchor_probability).sum() / trust_denom
        total = total + trust_anchor_weight * trust_anchor_loss
        trust_anchor_teacher_error_mean = float(trust_teacher_error.detach().mean().cpu())
        trust_anchor_prediction_error_mean = float(student_error.detach().mean().cpu())
        trust_anchor_advantage_mean = float(trust_advantage.detach().mean().cpu())
        trust_anchor_better_fraction = float((trust_advantage.detach() > 0.0).float().mean().cpu())

    parts = default_parts | {
        "loss_input_std_route_gate": float(gate_loss.detach().cpu()),
        "loss_input_std_route_gate_weighted": float((gate_weight * gate_loss).detach().cpu()),
        "loss_input_std_route_high_branch": float(high_branch_loss.detach().cpu()),
        "loss_input_std_route_low_branch": float(low_branch_loss.detach().cpu()),
        "loss_input_std_route_branch_weighted": float(branch_weighted.detach().cpu()),
        "loss_input_std_route_high_branch_weighted": float(
            (high_branch_weight * high_branch_loss).detach().cpu()
        ),
        "loss_input_std_route_low_branch_weighted": float(
            (low_branch_weight * low_branch_loss).detach().cpu()
        ),
        "loss_input_std_route_delta_trust": float(route_delta_loss.detach().cpu()),
        "loss_input_std_route_delta_trust_weighted": float((route_delta_weight * route_delta_loss).detach().cpu()),
        "loss_input_std_route_oracle_gate": float(oracle_gate_loss.detach().cpu()),
        "loss_input_std_route_oracle_gate_weighted": float((oracle_gate_weight * oracle_gate_loss).detach().cpu()),
        "loss_input_std_route_a08_oracle_anchor": float(a08_oracle_anchor_loss.detach().cpu()),
        "loss_input_std_route_a08_oracle_anchor_weighted": float(
            (a08_oracle_anchor_weight * a08_oracle_anchor_loss).detach().cpu()
        ),
        "loss_input_std_route_a15_oracle_anchor": float(a15_oracle_anchor_loss.detach().cpu()),
        "loss_input_std_route_a15_oracle_anchor_weighted": float(
            (a15_oracle_anchor_weight * a15_oracle_anchor_loss).detach().cpu()
        ),
        "loss_input_std_route_trust_anchor": float(trust_anchor_loss.detach().cpu()),
        "loss_input_std_route_trust_anchor_weighted": float(
            (trust_anchor_weight * trust_anchor_loss).detach().cpu()
        ),
        "input_std_route_high_branch_weight": high_branch_weight,
        "input_std_route_low_branch_weight": low_branch_weight,
        "input_std_route_a08_oracle_anchor_weight": a08_oracle_anchor_weight,
        "input_std_route_a15_oracle_anchor_weight": a15_oracle_anchor_weight,
        "input_std_route_trust_anchor_weight": trust_anchor_weight,
        "input_std_route_branch_schedule_scale": branch_scale,
        "input_std_route_gate_schedule_scale": gate_scale,
        "input_std_route_delta_trust_weight": route_delta_weight,
        "input_std_route_delta_trust_budget": route_delta_budget,
        "input_std_route_oracle_gate_weight": oracle_gate_weight,
        "input_std_route_selected_delta_abs_mean": selected_delta_abs_mean,
        "input_std_route_high_delta_abs_mean": high_delta_abs_mean,
        "input_std_route_low_delta_abs_mean": low_delta_abs_mean,
        "input_std_route_gate_mean": float(gate.detach().mean().cpu()),
        "input_std_route_prior_mean": float(prior.detach().mean().cpu()) if prior is not None else 0.0,
        "input_std_route_high_soft_mean": float(high_soft.detach().mean().cpu()),
        "input_std_route_high_hard_mean": float(high_hard.detach().mean().cpu()),
        "input_std_route_input_std_mean": float(input_std.detach().mean().cpu()),
        "input_std_route_input_std_min": float(input_std.detach().min().cpu()),
        "input_std_route_input_std_max": float(input_std.detach().max().cpu()),
        "input_std_route_high_branch_oracle_weighting": float(
            bool(loss_cfg.get("input_std_route_high_branch_oracle_weighting", False))
        ),
        "input_std_route_high_branch_oracle_weight_mean": float(high_weight.detach().mean().cpu()),
        "input_std_route_high_branch_oracle_weight_min": float(high_weight.detach().min().cpu()),
        "input_std_route_high_branch_oracle_weight_max": float(high_weight.detach().max().cpu()),
        "input_std_route_high_branch_oracle_a_better_fraction": high_oracle_a_better_fraction,
        "input_std_route_high_branch_oracle_advantage_mean": high_oracle_advantage_mean,
        "input_std_route_high_branch_oracle_teacher_a_error_mean": high_oracle_a_error_mean,
        "input_std_route_high_branch_oracle_teacher_b_error_mean": high_oracle_b_error_mean,
        "input_std_route_oracle_gate_target_mean": oracle_gate_target_mean,
        "input_std_route_oracle_gate_weight_mean": oracle_gate_weight_mean,
        "input_std_route_oracle_gate_positive_fraction": oracle_gate_positive_fraction,
        "input_std_route_oracle_gate_advantage_mean": oracle_gate_advantage_mean,
        "input_std_route_oracle_gate_teacher_a_error_mean": oracle_gate_a_error_mean,
        "input_std_route_oracle_gate_teacher_b_error_mean": oracle_gate_b_error_mean,
        "input_std_route_oracle_gate_low_focus_fraction": oracle_gate_low_focus_fraction,
        "input_std_route_oracle_gate_band_focus_fraction": oracle_gate_band_focus_fraction,
        "input_std_route_a08_oracle_anchor_probability_mean": float(
            a08_anchor_probability.detach().mean().cpu()
        ),
        "input_std_route_a08_oracle_anchor_probability_min": float(
            a08_anchor_probability.detach().min().cpu()
        ),
        "input_std_route_a08_oracle_anchor_probability_max": float(
            a08_anchor_probability.detach().max().cpu()
        ),
        "input_std_route_a08_oracle_anchor_a_better_fraction": a08_anchor_a_better_fraction,
        "input_std_route_a08_oracle_anchor_advantage_mean": a08_anchor_advantage_mean,
        "input_std_route_a08_oracle_anchor_teacher_a_error_mean": a08_anchor_a_error_mean,
        "input_std_route_a08_oracle_anchor_teacher_b_error_mean": a08_anchor_b_error_mean,
        "input_std_route_a15_oracle_anchor_probability_mean": float(
            a15_anchor_probability.detach().mean().cpu()
        ),
        "input_std_route_a15_oracle_anchor_probability_min": float(
            a15_anchor_probability.detach().min().cpu()
        ),
        "input_std_route_a15_oracle_anchor_probability_max": float(
            a15_anchor_probability.detach().max().cpu()
        ),
        "input_std_route_a15_oracle_anchor_b_better_fraction": a15_anchor_b_better_fraction,
        "input_std_route_a15_oracle_anchor_advantage_mean": a15_anchor_advantage_mean,
        "input_std_route_a15_oracle_anchor_teacher_a_error_mean": a15_anchor_a_error_mean,
        "input_std_route_a15_oracle_anchor_teacher_b_error_mean": a15_anchor_b_error_mean,
        "input_std_route_trust_anchor_probability_mean": float(
            trust_anchor_probability.detach().mean().cpu()
        ),
        "input_std_route_trust_anchor_probability_min": float(
            trust_anchor_probability.detach().min().cpu()
        ),
        "input_std_route_trust_anchor_probability_max": float(
            trust_anchor_probability.detach().max().cpu()
        ),
        "input_std_route_trust_anchor_better_fraction": trust_anchor_better_fraction,
        "input_std_route_trust_anchor_advantage_mean": trust_anchor_advantage_mean,
        "input_std_route_trust_anchor_teacher_error_mean": trust_anchor_teacher_error_mean,
        "input_std_route_trust_anchor_prediction_error_mean": trust_anchor_prediction_error_mean,
    }
    return total, parts


def compute_gaussian_route_control_branch_loss(
    aux: Dict[str, torch.Tensor | None],
    prediction: torch.Tensor,
    target: torch.Tensor | None,
    teacher_a_prediction: torch.Tensor | None,
    teacher_b_prediction: torch.Tensor | None,
    config: Dict[str, Any],
) -> tuple[torch.Tensor, Dict[str, float]]:
    loss_cfg = config.get("loss", {})
    shared_weight = _nonnegative_loss_value(
        loss_cfg,
        "gaussian_route_control_branch_anchor_weight",
        0.0,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    high_weight = _nonnegative_loss_value(
        loss_cfg,
        "gaussian_route_control_high_branch_anchor_weight",
        shared_weight,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    low_weight = _nonnegative_loss_value(
        loss_cfg,
        "gaussian_route_control_low_branch_anchor_weight",
        shared_weight,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    non_target_delta_weight = _nonnegative_loss_value(
        loss_cfg,
        "gaussian_route_control_non_target_delta_weight",
        0.0,
        max_value=TASK_AWARE_LOSS_WEIGHT_CAP,
    )
    delta_budget = float(loss_cfg.get("gaussian_route_control_non_target_delta_budget", 0.0))
    if delta_budget < 0.0:
        raise ValueError("gaussian_route_control_non_target_delta_budget must be non-negative")
    zero = prediction.new_tensor(0.0)
    default_parts = {
        "loss_gaussian_route_control_high_branch_anchor": 0.0,
        "loss_gaussian_route_control_high_branch_anchor_weighted": 0.0,
        "loss_gaussian_route_control_low_branch_anchor": 0.0,
        "loss_gaussian_route_control_low_branch_anchor_weighted": 0.0,
        "loss_gaussian_route_control_branch_anchor_weighted": 0.0,
        "loss_gaussian_route_control_non_target_delta": 0.0,
        "loss_gaussian_route_control_non_target_delta_weighted": 0.0,
        "gaussian_route_control_branch_anchor_weight": shared_weight,
        "gaussian_route_control_high_branch_anchor_weight": high_weight,
        "gaussian_route_control_low_branch_anchor_weight": low_weight,
        "gaussian_route_control_non_target_delta_weight": non_target_delta_weight,
        "gaussian_route_control_non_target_delta_budget": delta_budget,
        "gaussian_route_control_high_branch_anchor_probability_mean": 0.0,
        "gaussian_route_control_high_branch_anchor_probability_min": 0.0,
        "gaussian_route_control_high_branch_anchor_probability_max": 0.0,
        "gaussian_route_control_high_branch_anchor_a_better_fraction": 0.0,
        "gaussian_route_control_high_branch_anchor_advantage_mean": 0.0,
        "gaussian_route_control_low_branch_anchor_probability_mean": 0.0,
        "gaussian_route_control_low_branch_anchor_probability_min": 0.0,
        "gaussian_route_control_low_branch_anchor_probability_max": 0.0,
        "gaussian_route_control_low_branch_anchor_b_better_fraction": 0.0,
        "gaussian_route_control_low_branch_anchor_advantage_mean": 0.0,
        "gaussian_route_control_high_branch_teacher_a_error_mean": 0.0,
        "gaussian_route_control_high_branch_teacher_b_error_mean": 0.0,
        "gaussian_route_control_low_branch_teacher_a_error_mean": 0.0,
        "gaussian_route_control_low_branch_teacher_b_error_mean": 0.0,
        "gaussian_route_control_high_restored_l1_vs_teacher_a": 0.0,
        "gaussian_route_control_low_restored_l1_vs_teacher_b": 0.0,
        "gaussian_route_control_high_delta_abs_mean": 0.0,
        "gaussian_route_control_low_delta_abs_mean": 0.0,
        "gaussian_route_control_a08_low_delta_over_budget_mean": 0.0,
        "gaussian_route_control_a15_high_delta_over_budget_mean": 0.0,
    }
    if high_weight <= 0.0 and low_weight <= 0.0 and non_target_delta_weight <= 0.0:
        return zero, default_parts
    if target is None:
        raise ValueError("gaussian_route_control branch anchors require target tensor")
    if teacher_a_prediction is None or teacher_b_prediction is None:
        raise ValueError("gaussian_route_control branch anchors require both router teacher predictions")

    high_restored = aux.get("gaussian_route_control_high_restored")
    low_restored = aux.get("gaussian_route_control_low_restored")
    high_delta = aux.get("gaussian_route_control_high_delta")
    low_delta = aux.get("gaussian_route_control_low_delta")
    if high_weight > 0.0 and high_restored is None:
        raise ValueError("gaussian_route_control_high_branch_anchor_weight requires high_restored aux output")
    if low_weight > 0.0 and low_restored is None:
        raise ValueError("gaussian_route_control_low_branch_anchor_weight requires low_restored aux output")
    if non_target_delta_weight > 0.0 and (high_delta is None or low_delta is None):
        raise ValueError("gaussian_route_control_non_target_delta_weight requires high/low delta aux outputs")

    if target.shape[-2:] != prediction.shape[-2:]:
        target = center_crop_like(target, prediction.shape[-2:])
    if teacher_a_prediction.shape[-2:] != prediction.shape[-2:]:
        teacher_a_prediction = center_crop_like(teacher_a_prediction, prediction.shape[-2:])
    if teacher_b_prediction.shape[-2:] != prediction.shape[-2:]:
        teacher_b_prediction = center_crop_like(teacher_b_prediction, prediction.shape[-2:])
    if high_restored is not None and high_restored.shape[-2:] != prediction.shape[-2:]:
        high_restored = center_crop_like(high_restored, prediction.shape[-2:])
    if low_restored is not None and low_restored.shape[-2:] != prediction.shape[-2:]:
        low_restored = center_crop_like(low_restored, prediction.shape[-2:])

    temperature = float(loss_cfg.get("gaussian_route_control_branch_anchor_temperature", 0.004))
    margin = float(loss_cfg.get("gaussian_route_control_branch_anchor_margin", 0.0))
    probability_floor = float(loss_cfg.get("gaussian_route_control_branch_anchor_probability_floor", 0.50))
    require_positive_advantage = bool(
        loss_cfg.get("gaussian_route_control_branch_anchor_require_positive_advantage", True)
    )
    if temperature <= 0.0:
        raise ValueError("gaussian_route_control_branch_anchor_temperature must be positive")
    if not 0.0 <= probability_floor <= 1.0:
        raise ValueError("gaussian_route_control_branch_anchor_probability_floor must be in [0, 1]")

    teacher_a_error = torch.sqrt((teacher_a_prediction.detach() - target).square() + 1e-6).flatten(1).mean(1)
    teacher_b_error = torch.sqrt((teacher_b_prediction.detach() - target).square() + 1e-6).flatten(1).mean(1)
    high_advantage = teacher_b_error - teacher_a_error
    low_advantage = teacher_a_error - teacher_b_error

    def branch_probability(advantage: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid((advantage - margin) / temperature)
        if require_positive_advantage:
            probability = torch.where(advantage > margin, probability, torch.zeros_like(probability))
        if probability_floor > 0.0:
            probability = torch.where(probability >= probability_floor, probability, torch.zeros_like(probability))
        return probability

    high_probability = branch_probability(high_advantage).view(-1, 1, 1, 1).detach()
    low_probability = branch_probability(low_advantage).view(-1, 1, 1, 1).detach()
    total = zero
    high_loss = zero
    low_loss = zero
    high_l1_mean = 0.0
    low_l1_mean = 0.0
    if high_weight > 0.0:
        high_diff = torch.sqrt((high_restored - teacher_a_prediction.detach()).square() + 1e-6)
        high_denom = (
            high_probability.sum() * high_diff.shape[1] * high_diff.shape[2] * high_diff.shape[3]
        ).clamp_min(1.0)
        high_loss = (high_diff * high_probability).sum() / high_denom
        total = total + high_weight * high_loss
        high_l1_mean = float(high_diff.detach().mean().cpu())
    if low_weight > 0.0:
        low_diff = torch.sqrt((low_restored - teacher_b_prediction.detach()).square() + 1e-6)
        low_denom = (
            low_probability.sum() * low_diff.shape[1] * low_diff.shape[2] * low_diff.shape[3]
        ).clamp_min(1.0)
        low_loss = (low_diff * low_probability).sum() / low_denom
        total = total + low_weight * low_loss
        low_l1_mean = float(low_diff.detach().mean().cpu())

    non_target_delta_loss = zero
    a08_low_delta_over_budget_mean = 0.0
    a15_high_delta_over_budget_mean = 0.0
    high_delta_abs_mean = 0.0
    low_delta_abs_mean = 0.0
    if high_delta is not None:
        if high_delta.shape[-2:] != prediction.shape[-2:]:
            high_delta = center_crop_like(high_delta, prediction.shape[-2:])
        high_delta_abs = high_delta.abs()
        high_delta_abs_mean = float(high_delta_abs.detach().mean().cpu())
    if low_delta is not None:
        if low_delta.shape[-2:] != prediction.shape[-2:]:
            low_delta = center_crop_like(low_delta, prediction.shape[-2:])
        low_delta_abs = low_delta.abs()
        low_delta_abs_mean = float(low_delta_abs.detach().mean().cpu())
    if non_target_delta_weight > 0.0:
        a08_low_over = torch.relu(low_delta_abs - delta_budget)
        a15_high_over = torch.relu(high_delta_abs - delta_budget)
        low_denom = (
            high_probability.sum() * a08_low_over.shape[1] * a08_low_over.shape[2] * a08_low_over.shape[3]
        ).clamp_min(1.0)
        high_denom = (
            low_probability.sum() * a15_high_over.shape[1] * a15_high_over.shape[2] * a15_high_over.shape[3]
        ).clamp_min(1.0)
        a08_low_delta_loss = (a08_low_over * high_probability).sum() / low_denom
        a15_high_delta_loss = (a15_high_over * low_probability).sum() / high_denom
        non_target_delta_loss = a08_low_delta_loss + a15_high_delta_loss
        total = total + non_target_delta_weight * non_target_delta_loss
        a08_low_delta_over_budget_mean = float(a08_low_over.detach().mean().cpu())
        a15_high_delta_over_budget_mean = float(a15_high_over.detach().mean().cpu())

    branch_weighted = high_weight * high_loss + low_weight * low_loss
    parts = default_parts | {
        "loss_gaussian_route_control_high_branch_anchor": float(high_loss.detach().cpu()),
        "loss_gaussian_route_control_high_branch_anchor_weighted": float(
            (high_weight * high_loss).detach().cpu()
        ),
        "loss_gaussian_route_control_low_branch_anchor": float(low_loss.detach().cpu()),
        "loss_gaussian_route_control_low_branch_anchor_weighted": float((low_weight * low_loss).detach().cpu()),
        "loss_gaussian_route_control_branch_anchor_weighted": float(branch_weighted.detach().cpu()),
        "loss_gaussian_route_control_non_target_delta": float(non_target_delta_loss.detach().cpu()),
        "loss_gaussian_route_control_non_target_delta_weighted": float(
            (non_target_delta_weight * non_target_delta_loss).detach().cpu()
        ),
        "gaussian_route_control_branch_anchor_weight": shared_weight,
        "gaussian_route_control_high_branch_anchor_weight": high_weight,
        "gaussian_route_control_low_branch_anchor_weight": low_weight,
        "gaussian_route_control_non_target_delta_weight": non_target_delta_weight,
        "gaussian_route_control_non_target_delta_budget": delta_budget,
        "gaussian_route_control_high_branch_anchor_probability_mean": float(
            high_probability.detach().mean().cpu()
        ),
        "gaussian_route_control_high_branch_anchor_probability_min": float(high_probability.detach().min().cpu()),
        "gaussian_route_control_high_branch_anchor_probability_max": float(high_probability.detach().max().cpu()),
        "gaussian_route_control_high_branch_anchor_a_better_fraction": float(
            (high_advantage.detach() > 0.0).float().mean().cpu()
        ),
        "gaussian_route_control_high_branch_anchor_advantage_mean": float(high_advantage.detach().mean().cpu()),
        "gaussian_route_control_low_branch_anchor_probability_mean": float(
            low_probability.detach().mean().cpu()
        ),
        "gaussian_route_control_low_branch_anchor_probability_min": float(low_probability.detach().min().cpu()),
        "gaussian_route_control_low_branch_anchor_probability_max": float(low_probability.detach().max().cpu()),
        "gaussian_route_control_low_branch_anchor_b_better_fraction": float(
            (low_advantage.detach() > 0.0).float().mean().cpu()
        ),
        "gaussian_route_control_low_branch_anchor_advantage_mean": float(low_advantage.detach().mean().cpu()),
        "gaussian_route_control_high_branch_teacher_a_error_mean": float(teacher_a_error.detach().mean().cpu()),
        "gaussian_route_control_high_branch_teacher_b_error_mean": float(teacher_b_error.detach().mean().cpu()),
        "gaussian_route_control_low_branch_teacher_a_error_mean": float(teacher_a_error.detach().mean().cpu()),
        "gaussian_route_control_low_branch_teacher_b_error_mean": float(teacher_b_error.detach().mean().cpu()),
        "gaussian_route_control_high_restored_l1_vs_teacher_a": high_l1_mean,
        "gaussian_route_control_low_restored_l1_vs_teacher_b": low_l1_mean,
        "gaussian_route_control_high_delta_abs_mean": high_delta_abs_mean,
        "gaussian_route_control_low_delta_abs_mean": low_delta_abs_mean,
        "gaussian_route_control_a08_low_delta_over_budget_mean": a08_low_delta_over_budget_mean,
        "gaussian_route_control_a15_high_delta_over_budget_mean": a15_high_delta_over_budget_mean,
    }
    return total, parts


def reduce_float_dict(
    values: Dict[str, float],
    device: torch.device,
    distributed: Dict[str, int | bool],
) -> Dict[str, float]:
    if not bool(distributed.get("enabled", False)):
        return values
    keys = sorted(values)
    tensor = torch.tensor([float(values[key]) for key in keys], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= int(distributed.get("world_size", 1))
    return {key: float(tensor[idx].detach().cpu()) for idx, key in enumerate(keys)}


def run_preflight(config: Dict[str, Any], args: argparse.Namespace) -> None:
    preflight_config = copy.deepcopy(config)
    dataset_cfg = preflight_config.setdefault("dataset", {})
    preflight_val_limit = int(dataset_cfg.get("preflight_val_limit", 0))
    if preflight_val_limit > 0:
        dataset_cfg["val_limit"] = preflight_val_limit
    dataset_cfg["num_workers"] = int(dataset_cfg.get("preflight_num_workers", 0))

    seed_everything(int(preflight_config["training"].get("seed", 20260424)))
    batch_size = int(preflight_config.get("training", {}).get("batch_size", 1))
    preflight_train_samples = max(int(args.preflight_train_samples), batch_size)
    train_dataset, val_dataset = make_datasets(preflight_config, train_limit_override=preflight_train_samples)
    train_batch_sampler, sampler_info = make_train_batch_sampler(train_dataset, preflight_config)
    train_loader = make_loader(train_dataset, preflight_config, training=True, batch_sampler=train_batch_sampler)
    val_loader = make_loader(val_dataset, preflight_config, training=False)
    transfer_loader = (
        make_loader(make_transfer_validation_dataset(preflight_config), preflight_config, training=False)
        if transfer_validation_metrics_enabled(preflight_config)
        else None
    )
    device = choose_device(preflight_config)
    cuda_memory_fraction = apply_cuda_memory_fraction(device)
    model = CrossResolutionGaussianRestorer(build_model_config(preflight_config)).to(device)
    init_checkpoint_info = load_init_checkpoint_if_requested(model, preflight_config, args.config, device)
    route_head_transplant_info = transplant_input_std_route_decoder_heads_if_requested(
        model,
        preflight_config,
        args.config,
        device,
    )
    parameter_policy_info = apply_trainable_parameter_policy(model, preflight_config)
    teacher_model, teacher_info = load_teacher_model_if_requested(preflight_config, args.config, device)
    router_teacher_a, router_teacher_b, router_info = load_router_distillation_teachers_if_requested(
        preflight_config,
        args.config,
        device,
    )
    trust_anchor_teacher, trust_anchor_info = load_input_std_route_trust_teacher_if_requested(
        preflight_config,
        args.config,
        device,
    )
    model.set_training_progress(1.0)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("no trainable parameters remain after applying training parameter policy")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config["optimizer"].get("lr", 2e-4)),
        weight_decay=float(config["optimizer"].get("weight_decay", 1e-4)),
    )
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    batch = next(iter(train_loader))
    inputs = batch["input"].to(device, non_blocking=True)
    targets = batch["target"].to(device, non_blocking=True)
    aux = model(inputs, return_aux="loss")
    predictions = aux["restored"]
    effective_predictions, effective_targets, _ = select_training_supervision(predictions, targets, config)
    teacher_predictions = None
    if teacher_model is not None:
        with torch.no_grad():
            teacher_predictions = teacher_model(inputs).clamp(0.0, 1.0)
    router_teacher_a_predictions = None
    router_teacher_b_predictions = None
    if router_teacher_a is not None and router_teacher_b is not None:
        with torch.no_grad():
            router_teacher_a_predictions = router_teacher_a(inputs).clamp(0.0, 1.0)
            router_teacher_b_predictions = router_teacher_b(inputs).clamp(0.0, 1.0)
    trust_anchor_predictions = None
    if trust_anchor_teacher is not None:
        with torch.no_grad():
            trust_anchor_predictions = trust_anchor_teacher(inputs).clamp(0.0, 1.0)
    loss, loss_parts = compute_loss(effective_predictions, effective_targets, preflight_config, input_image=inputs)
    aux_loss, aux_parts = compute_auxiliary_loss(aux, targets, preflight_config, inputs=inputs)
    teacher_loss, teacher_parts = compute_teacher_consistency_loss(
        effective_predictions,
        effective_targets,
        teacher_predictions,
        preflight_config,
    )
    router_loss, router_parts = compute_router_distillation_loss(
        effective_predictions,
        inputs,
        router_teacher_a_predictions,
        router_teacher_b_predictions,
        preflight_config,
    )
    route_loss, route_parts = compute_input_std_route_residual_loss(
        aux,
        effective_predictions,
        inputs,
        effective_targets,
        router_teacher_a_predictions,
        router_teacher_b_predictions,
        trust_anchor_predictions,
        preflight_config,
        step=1,
    )
    gaussian_route_loss, gaussian_route_parts = compute_gaussian_route_control_branch_loss(
        aux,
        effective_predictions,
        effective_targets,
        router_teacher_a_predictions,
        router_teacher_b_predictions,
        preflight_config,
    )
    loss = loss + aux_loss + teacher_loss + router_loss + route_loss + gaussian_route_loss
    loss_parts.update(aux_parts)
    loss_parts.update(teacher_parts)
    loss_parts.update(router_parts)
    loss_parts.update(route_parts)
    loss_parts.update(gaussian_route_parts)
    loss_parts["cuda_memory_fraction"] = cuda_memory_fraction
    if control_diagnostics_enabled(config):
        loss_parts.update(
            compute_control_diagnostics(
                aux,
                targets,
                preflight_config,
                inputs=inputs,
                prefix="preflight_control",
            )
        )
        loss_parts.update(
            compute_input_std_route_diagnostics(
                aux,
                inputs,
                effective_predictions,
                preflight_config,
                prefix="preflight_route",
            )
        )
    loss_parts["loss_total"] = float(loss.detach().cpu())
    loss.backward()
    optimizer.step()
    val_metrics = validate(
        model,
        val_loader,
        device,
        preflight_config,
        router_teacher_a=router_teacher_a,
        router_teacher_b=router_teacher_b,
        transfer_loader=transfer_loader,
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())
    print("preflight: ok")
    print(f"device: {device}")
    print(f"train_samples: {len(train_dataset)}")
    print(f"requested_preflight_train_samples: {args.preflight_train_samples}")
    print(f"effective_preflight_train_samples: {preflight_train_samples}")
    print(f"val_samples: {len(val_dataset)}")
    print(f"batch_input_shape: {tuple(inputs.shape)}")
    print(f"batch_target_shape: {tuple(targets.shape)}")
    print(f"total_params: {total_params}")
    print(
        json.dumps(
            {
                **init_checkpoint_info,
                **route_head_transplant_info,
                **parameter_policy_info,
                **teacher_info,
                **router_info,
                **trust_anchor_info,
                **sampler_info,
                **loss_parts,
                **val_metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )


def run_training(config: Dict[str, Any], config_path: Path, distributed: Dict[str, int | bool]) -> None:
    rank = int(distributed.get("rank", 0))
    seed_everything(int(config["training"].get("seed", 20260424)) + rank)
    train_dataset, val_dataset = make_datasets(config)
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=int(distributed.get("world_size", 1)),
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        if bool(distributed.get("enabled", False))
        else None
    )
    train_batch_sampler, sampler_info = make_train_batch_sampler(train_dataset, config, distributed)
    train_loader = make_loader(
        train_dataset,
        config,
        training=True,
        sampler=train_sampler,
        batch_sampler=train_batch_sampler,
    )
    val_loader = make_loader(val_dataset, config, training=False) if is_main_process(distributed) else None
    transfer_loader = (
        make_loader(make_transfer_validation_dataset(config), config, training=False)
        if is_main_process(distributed) and transfer_validation_metrics_enabled(config)
        else None
    )
    device = choose_device(config, distributed)
    cuda_memory_fraction = apply_cuda_memory_fraction(device)
    model = CrossResolutionGaussianRestorer(build_model_config(config)).to(device)
    init_checkpoint_info = load_init_checkpoint_if_requested(model, config, config_path, device)
    route_head_transplant_info = transplant_input_std_route_decoder_heads_if_requested(
        model,
        config,
        config_path,
        device,
    )
    parameter_policy_info = apply_trainable_parameter_policy(model, config)
    teacher_model, teacher_info = load_teacher_model_if_requested(config, config_path, device)
    router_teacher_a, router_teacher_b, router_info = load_router_distillation_teachers_if_requested(
        config,
        config_path,
        device,
    )
    trust_anchor_teacher, trust_anchor_info = load_input_std_route_trust_teacher_if_requested(
        config,
        config_path,
        device,
    )
    route_info = input_std_route_residual_info(config, router_info)
    if bool(distributed.get("enabled", False)):
        model = DistributedDataParallel(
            model,
            device_ids=[int(distributed.get("local_rank", 0))],
            output_device=int(distributed.get("local_rank", 0)),
            broadcast_buffers=False,
            find_unused_parameters=bool(config.get("training", {}).get("ddp_find_unused_parameters", True)),
        )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("no trainable parameters remain after applying training parameter policy")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config["optimizer"].get("lr", 2e-4)),
        weight_decay=float(config["optimizer"].get("weight_decay", 1e-4)),
    )
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if is_main_process(distributed):
        run_dir = prepare_run_dir(config, config_path)
        if init_checkpoint_info:
            (run_dir / "metrics" / "init_checkpoint.json").write_text(
                json.dumps(init_checkpoint_info, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps({"event": "init_checkpoint_loaded", **init_checkpoint_info}, sort_keys=True))
        if bool(route_head_transplant_info.get("input_std_route_decoder_heads_transplant_enabled", False)):
            (run_dir / "metrics" / "input_std_route_decoder_heads_transplant.json").write_text(
                json.dumps(route_head_transplant_info, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {"event": "input_std_route_decoder_heads_transplanted", **route_head_transplant_info},
                    sort_keys=True,
                )
            )
        if bool(parameter_policy_info.get("parameter_policy_enabled", False)):
            (run_dir / "metrics" / "parameter_policy.json").write_text(
                json.dumps(parameter_policy_info, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps({"event": "parameter_policy_applied", **parameter_policy_info}, sort_keys=True))
        if bool(teacher_info.get("teacher_consistency_enabled", False)):
            (run_dir / "metrics" / "teacher_consistency.json").write_text(
                json.dumps(teacher_info, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps({"event": "teacher_consistency_loaded", **teacher_info}, sort_keys=True))
        if bool(router_info.get("router_distillation_enabled", False)) or bool(
            router_info.get("input_std_route_teacher_enabled", False)
        ):
            (run_dir / "metrics" / "router_distillation.json").write_text(
                json.dumps(router_info, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps({"event": "router_teachers_loaded", **router_info}, sort_keys=True))
        if bool(trust_anchor_info.get("input_std_route_trust_anchor_enabled", False)):
            (run_dir / "metrics" / "input_std_route_trust_anchor.json").write_text(
                json.dumps(trust_anchor_info, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps({"event": "input_std_route_trust_anchor_loaded", **trust_anchor_info}, sort_keys=True))
        if bool(route_info.get("input_std_route_residual_enabled", False)) or bool(
            route_info.get("input_std_route_terminal_heads_enabled", False)
        ) or bool(
            route_info.get("input_std_route_feature_expert_enabled", False)
        ) or bool(
            route_info.get("input_std_route_decoder_heads_enabled", False)
        ) or bool(
            route_info.get("input_std_route_post_adapter_enabled", False)
        ):
            (run_dir / "metrics" / "input_std_route_residual.json").write_text(
                json.dumps(route_info, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps({"event": "input_std_route_enabled", **route_info}, sort_keys=True))
        if bool(sampler_info.get("input_std_route_balanced_sampling", False)):
            (run_dir / "metrics" / "input_std_route_balanced_sampler.json").write_text(
                json.dumps(sampler_info, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps({"event": "input_std_route_balanced_sampler_enabled", **sampler_info}, sort_keys=True))
    else:
        run_dir = run_dir_for_config(config)
    distributed_barrier(distributed)
    metrics_path = run_dir / "metrics" / "train_metrics.jsonl"
    rank_metrics_path = run_dir / "metrics" / "rank_metrics" / f"rank_{rank:02d}.jsonl"

    training_cfg = config["training"]
    max_steps = int(training_cfg.get("max_steps", 1000))
    grad_accum = int(training_cfg.get("grad_accum", 1))
    log_interval = int(training_cfg.get("log_interval", 10))
    val_interval = int(training_cfg.get("val_interval", 100))
    ckpt_interval = int(training_cfg.get("checkpoint_interval", 100))
    wall_clock_minutes = float(training_cfg.get("wall_clock_minutes", 55.0))
    grad_clip_norm = float(training_cfg.get("grad_clip_norm", 0.0))
    ema_decay = ema_decay_value(training_cfg)
    ema_eval = bool(training_cfg.get("ema_eval", False)) and ema_decay > 0.0
    ema_state = init_ema_state(model) if ema_decay > 0.0 else {}
    amp_enabled = bool(training_cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    if device.type == "cuda":
        peak_device: torch.device | int = device
        if device.index is None:
            peak_device = torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(peak_device)

    step = 0
    epoch = 0
    start_time = time.time()
    best_val_loss = float("inf")
    best_val_step = 0
    best_val_psnr = float("-inf")
    selection_metric = str(training_cfg.get("best_selection_metric", "val_loss"))
    best_selection_value = initial_best_metric(selection_metric)
    best_selection_requires_gate = bool(training_cfg.get("best_selection_require_baseline_gate", False))
    best_selection_gate_baseline_metrics: Dict[str, float] = {}
    best_selection_gate_info: Dict[str, Any] = {"enabled": False, "passed": True, "checks": []}
    step_guard_enabled = step_no_harm_guard_enabled(training_cfg)
    if step_guard_enabled and bool(distributed.get("enabled", False)):
        raise ValueError("step_no_harm_guard is single-process only; G-RSCN launchers must not use DDP")
    step_guard_interval = max(int(training_cfg.get("step_no_harm_guard_interval", grad_accum)), 1)
    step_guard_max_batches = max(int(training_cfg.get("step_no_harm_guard_max_batches", 8)), 1)
    step_guard_restore_optimizer = bool(training_cfg.get("step_no_harm_guard_restore_optimizer", True))
    step_guard_two_phase = bool(training_cfg.get("step_no_harm_guard_two_phase_enabled", False))
    step_guard_accept_protect_only = bool(
        training_cfg.get("step_no_harm_guard_two_phase_accept_protect_only", False)
    )
    step_guard_projection_enabled = bool(training_cfg.get("step_no_harm_guard_projection_enabled", False))
    step_guard_projection_restore_optimizer = bool(
        training_cfg.get("step_no_harm_guard_projection_restore_optimizer", True)
    )
    step_guard_projection_ratios = (
        step_no_harm_projection_ratios(training_cfg) if step_guard_projection_enabled else []
    )
    step_guard_config = validation_config_with_max_batches(config, step_guard_max_batches)
    step_guard_baseline_metrics: Dict[str, float] = {}
    step_guard_last_info: Dict[str, Any] = {
        "enabled": step_guard_enabled,
        "checked": False,
        "passed": True,
        "rolled_back": False,
        "checks": [],
    }
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if is_main_process(distributed) and (
        best_selection_baseline_gates_enabled(training_cfg) or step_guard_enabled
    ):
        assert val_loader is not None
        validation_context = ema_weights_applied(model, ema_state) if ema_eval else nullcontext()
        with validation_context:
            best_selection_gate_baseline_metrics = validate(
                unwrap_model(model),
                val_loader,
                device,
                config,
                router_teacher_a=router_teacher_a,
                router_teacher_b=router_teacher_b,
                transfer_loader=transfer_loader,
            )
        baseline_selection_value = best_metric_value(best_selection_gate_baseline_metrics, selection_metric)
        best_selection_value = baseline_selection_value
        best_val_loss = best_selection_gate_baseline_metrics["val_loss"]
        best_val_step = 0
        best_val_psnr = best_selection_gate_baseline_metrics["val_psnr"]
        save_checkpoint(run_dir / "checkpoints" / "best_val.pt", model, optimizer, 0, config)
        best_summary = {
            "best_val_step": best_val_step,
            "best_val_loss": best_val_loss,
            "best_val_psnr": best_val_psnr,
            "selection_metric": selection_metric,
            "best_selection_value": best_selection_value,
            "checkpoint": str(run_dir / "checkpoints" / "best_val.pt"),
            "ema_decay": ema_decay,
            "ema_eval": ema_eval,
            "baseline_gate_enabled": True,
            "baseline_initialized_best": True,
            "baseline_metrics": best_selection_gate_baseline_metrics,
        }
        (run_dir / "metrics" / "best_val_summary.json").write_text(
            json.dumps(best_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (run_dir / "metrics" / "best_selection_baseline_metrics.json").write_text(
            json.dumps(best_selection_gate_baseline_metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "event": "best_selection_baseline_initialized",
                    "selection_metric": selection_metric,
                    "best_selection_value": best_selection_value,
                    **best_selection_gate_baseline_metrics,
                },
                sort_keys=True,
            )
        )
        if step_guard_enabled:
            step_guard_baseline_metrics = validate(
                unwrap_model(model),
                val_loader,
                device,
                step_guard_config,
                router_teacher_a=router_teacher_a,
                router_teacher_b=router_teacher_b,
                transfer_loader=transfer_loader,
            )
            (run_dir / "metrics" / "step_no_harm_guard_baseline_metrics.json").write_text(
                json.dumps(step_guard_baseline_metrics, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "event": "step_no_harm_guard_baseline_initialized",
                        "max_batches": step_guard_max_batches,
                        **step_guard_baseline_metrics,
                    },
                    sort_keys=True,
                )
            )
    distributed_barrier(distributed)

    while step < max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if hasattr(train_batch_sampler, "set_epoch"):
            train_batch_sampler.set_epoch(epoch)  # type: ignore[union-attr]
        for batch in train_loader:
            step += 1
            lr_factor = lr_schedule_factor(training_cfg, step, max_steps)
            apply_lr_schedule(optimizer, base_lrs, lr_factor)
            unwrap_model(model).set_training_progress(step / max(max_steps, 1))
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            with autocast_context(device, amp_enabled):
                aux = model(inputs, return_aux="loss")
                predictions = aux["restored"].clamp(0.0, 1.0)
                rollback_model_state = (
                    clone_model_state(model)
                    if step_guard_enabled
                    and is_main_process(distributed)
                    and step % step_guard_interval == 0
                    else None
                )
                rollback_optimizer_state = (
                    copy.deepcopy(optimizer.state_dict())
                    if rollback_model_state is not None and step_guard_restore_optimizer
                    else None
                )
                teacher_predictions = None
                if teacher_model is not None:
                    with torch.no_grad():
                        teacher_predictions = teacher_model(inputs).clamp(0.0, 1.0)
                router_teacher_a_predictions = None
                router_teacher_b_predictions = None
                if router_teacher_a is not None and router_teacher_b is not None:
                    with torch.no_grad():
                        router_teacher_a_predictions = router_teacher_a(inputs).clamp(0.0, 1.0)
                        router_teacher_b_predictions = router_teacher_b(inputs).clamp(0.0, 1.0)
                trust_anchor_predictions = None
                if trust_anchor_teacher is not None:
                    with torch.no_grad():
                        trust_anchor_predictions = trust_anchor_teacher(inputs).clamp(0.0, 1.0)
                effective_predictions, effective_targets, valid_ratio = select_training_supervision(
                    predictions,
                    targets,
                    config,
                )
                loss, loss_parts = compute_loss(effective_predictions, effective_targets, config, input_image=inputs)
                aux_loss, aux_parts = compute_auxiliary_loss(aux, targets, config, inputs=inputs)
                teacher_loss, teacher_parts = compute_teacher_consistency_loss(
                    effective_predictions,
                    effective_targets,
                    teacher_predictions,
                    config,
                )
                router_loss, router_parts = compute_router_distillation_loss(
                    effective_predictions,
                    inputs,
                    router_teacher_a_predictions,
                    router_teacher_b_predictions,
                    config,
                )
                route_loss, route_parts = compute_input_std_route_residual_loss(
                    aux,
                    effective_predictions,
                    inputs,
                    effective_targets,
                    router_teacher_a_predictions,
                    router_teacher_b_predictions,
                    trust_anchor_predictions,
                    config,
                    step=step,
                )
                gaussian_route_loss, gaussian_route_parts = compute_gaussian_route_control_branch_loss(
                    aux,
                    effective_predictions,
                    effective_targets,
                    router_teacher_a_predictions,
                    router_teacher_b_predictions,
                    config,
                )
                loss = loss + aux_loss + teacher_loss + router_loss + route_loss + gaussian_route_loss
                loss_parts.update(aux_parts)
                loss_parts.update(teacher_parts)
                loss_parts.update(router_parts)
                loss_parts.update(route_parts)
                loss_parts.update(gaussian_route_parts)
                loss_parts["loss_total"] = float(loss.detach().cpu())
                scaled_loss = loss / grad_accum
            scaler.scale(scaled_loss).backward()
            if step % grad_accum == 0:
                if grad_clip_norm > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema_decay > 0.0:
                    update_ema_state(model, ema_state, ema_decay)
                if rollback_model_state is not None:
                    assert val_loader is not None
                    guard_metrics = validate(
                        unwrap_model(model),
                        val_loader,
                        device,
                        step_guard_config,
                        router_teacher_a=router_teacher_a,
                        router_teacher_b=router_teacher_b,
                        transfer_loader=transfer_loader,
                    )
                    guard_passed, guard_info = step_no_harm_guard_status(
                        guard_metrics,
                        step_guard_baseline_metrics,
                        training_cfg,
                    )
                    protect_passed, protect_info = step_no_harm_custom_guard_status(
                        guard_metrics,
                        step_guard_baseline_metrics,
                        training_cfg,
                        prefix="step_no_harm_guard_protect",
                    )
                    target_passed, target_info = step_no_harm_custom_guard_status(
                        guard_metrics,
                        step_guard_baseline_metrics,
                        training_cfg,
                        prefix="step_no_harm_guard_target",
                    )
                    accepted_mode = "full" if guard_passed else "rollback"
                    projection_attempts: list[Dict[str, Any]] = []
                    projection_accepted_ratio: float | None = None
                    projection_accepted_full_gate = False
                    projection_accepted_protect_gate = False
                    projection_accepted_target_gate = False

                    if (
                        not guard_passed
                        and step_guard_two_phase
                        and step_guard_accept_protect_only
                        and bool(protect_info.get("enabled", False))
                        and protect_passed
                    ):
                        guard_passed = True
                        accepted_mode = "protect_only"

                    if (
                        not guard_passed
                        and step_guard_projection_enabled
                        and bool(target_info.get("enabled", False))
                        and target_passed
                    ):
                        candidate_model_state = clone_model_state(model)
                        for ratio in step_guard_projection_ratios:
                            load_interpolated_model_state(model, rollback_model_state, candidate_model_state, ratio)
                            projected_metrics = validate(
                                unwrap_model(model),
                                val_loader,
                                device,
                                step_guard_config,
                                router_teacher_a=router_teacher_a,
                                router_teacher_b=router_teacher_b,
                                transfer_loader=transfer_loader,
                            )
                            projected_full_passed, projected_full_info = step_no_harm_guard_status(
                                projected_metrics,
                                step_guard_baseline_metrics,
                                training_cfg,
                            )
                            projected_protect_passed, projected_protect_info = step_no_harm_custom_guard_status(
                                projected_metrics,
                                step_guard_baseline_metrics,
                                training_cfg,
                                prefix="step_no_harm_guard_protect",
                            )
                            projected_target_passed, projected_target_info = step_no_harm_custom_guard_status(
                                projected_metrics,
                                step_guard_baseline_metrics,
                                training_cfg,
                                prefix="step_no_harm_guard_target",
                            )
                            projection_attempt = {
                                "ratio": ratio,
                                "full_gate_passed": projected_full_passed,
                                "protect_gate_passed": projected_protect_passed,
                                "target_gate_passed": projected_target_passed,
                                "full_gate_checks": projected_full_info.get("checks", []),
                                "protect_gate_checks": projected_protect_info.get("checks", []),
                                "target_gate_checks": projected_target_info.get("checks", []),
                            }
                            projection_attempt.update(
                                {
                                    f"metrics_{key}": value
                                    for key, value in projected_metrics.items()
                                    if guard_metric_log_value(key)
                                }
                            )
                            projection_attempts.append(projection_attempt)
                            projected_accepted = projected_full_passed or (
                                bool(projected_protect_info.get("enabled", False))
                                and bool(projected_target_info.get("enabled", False))
                                and projected_protect_passed
                                and projected_target_passed
                            )
                            if projected_accepted:
                                guard_passed = True
                                guard_metrics = projected_metrics
                                guard_info = projected_full_info
                                protect_info = projected_protect_info
                                target_info = projected_target_info
                                protect_passed = projected_protect_passed
                                target_passed = projected_target_passed
                                projection_accepted_ratio = ratio
                                projection_accepted_full_gate = projected_full_passed
                                projection_accepted_protect_gate = projected_protect_passed
                                projection_accepted_target_gate = projected_target_passed
                                accepted_mode = "projection_full" if projected_full_passed else "projection"
                                if rollback_optimizer_state is not None and step_guard_projection_restore_optimizer:
                                    optimizer.load_state_dict(rollback_optimizer_state)
                                if ema_decay > 0.0:
                                    ema_state = init_ema_state(model)
                                break

                    rolled_back = bool(guard_info.get("enabled", False)) and not guard_passed
                    if rolled_back:
                        unwrap_model(model).load_state_dict(rollback_model_state)
                        if rollback_optimizer_state is not None:
                            optimizer.load_state_dict(rollback_optimizer_state)
                        if ema_decay > 0.0:
                            ema_state = init_ema_state(model)
                    step_guard_last_info = {
                        "enabled": True,
                        "checked": True,
                        "passed": guard_passed,
                        "rolled_back": rolled_back,
                        "max_batches": step_guard_max_batches,
                        "interval": step_guard_interval,
                        "restore_optimizer": step_guard_restore_optimizer,
                        "two_phase_enabled": step_guard_two_phase,
                        "two_phase_accept_protect_only": step_guard_accept_protect_only,
                        "projection_enabled": step_guard_projection_enabled,
                        "projection_ratios": step_guard_projection_ratios,
                        "projection_restore_optimizer": step_guard_projection_restore_optimizer,
                        "accepted_mode": accepted_mode,
                        "full_gate_passed": bool(guard_info.get("passed", guard_passed)),
                        "full_gate_checks": guard_info.get("checks", []),
                        "protect_gate_enabled": bool(protect_info.get("enabled", False)),
                        "protect_gate_passed": protect_passed,
                        "protect_gate_checks": protect_info.get("checks", []),
                        "target_gate_enabled": bool(target_info.get("enabled", False)),
                        "target_gate_passed": target_passed,
                        "target_gate_checks": target_info.get("checks", []),
                        "projection_accepted_ratio": projection_accepted_ratio,
                        "projection_accepted_full_gate": projection_accepted_full_gate,
                        "projection_accepted_protect_gate": projection_accepted_protect_gate,
                        "projection_accepted_target_gate": projection_accepted_target_gate,
                        "projection_attempts": projection_attempts,
                        "checks": guard_info.get("checks", []),
                        **{
                            f"step_guard_{key}": value
                            for key, value in guard_metrics.items()
                            if guard_metric_log_value(key)
                        },
                    }
                    append_jsonl(
                        run_dir / "metrics" / "step_no_harm_guard_metrics.jsonl",
                        {
                            "step": step,
                            "elapsed_minutes": (time.time() - start_time) / 60.0,
                            "rank": rank,
                            "world_size": int(distributed.get("world_size", 1)),
                            **step_guard_last_info,
                        },
                    )
                    print(
                        json.dumps(
                            {"event": "step_no_harm_guard_checked", "step": step, **step_guard_last_info},
                            sort_keys=True,
                        )
                    )
                    model.train()
                else:
                    step_guard_last_info = {
                        "enabled": step_guard_enabled,
                        "checked": False,
                        "passed": True,
                        "rolled_back": False,
                        "checks": [],
                    }

            elapsed_minutes = (time.time() - start_time) / 60.0
            if step % log_interval == 0 or step == 1:
                if control_diagnostics_enabled(config):
                    loss_parts.update(
                        compute_control_diagnostics(
                            aux,
                            targets,
                            config,
                            inputs=inputs,
                            prefix="train_control",
                        )
                    )
                    loss_parts.update(
                        compute_input_std_route_diagnostics(
                            aux,
                            inputs,
                            effective_predictions,
                            config,
                            prefix="train_route",
                        )
                    )
                reduced_loss_parts = reduce_float_dict(loss_parts, device, distributed)
                reduced_valid_ratio = reduce_float_dict(
                    {"supervision_valid_ratio": valid_ratio},
                    device,
                    distributed,
                )["supervision_valid_ratio"]
                record = {
                    "step": step,
                    "elapsed_minutes": elapsed_minutes,
                    "lr": optimizer.param_groups[0]["lr"],
                    "lr_schedule_factor": lr_factor,
                    "grad_clip_norm": grad_clip_norm,
                    "ema_decay": ema_decay,
                    "ema_eval": ema_eval,
                    "rank": rank,
                    "world_size": int(distributed.get("world_size", 1)),
                    "cuda_memory_fraction": cuda_memory_fraction,
                    "per_rank_batch_size": int(training_cfg.get("batch_size", 1)),
                    "effective_global_batch_size": int(training_cfg.get("batch_size", 1))
                    * int(distributed.get("world_size", 1)),
                    "supervision_valid_ratio": reduced_valid_ratio,
                    **cuda_memory_stats(device),
                    **reduced_loss_parts,
                    "step_no_harm_guard_enabled": bool(step_guard_last_info.get("enabled", False)),
                    "step_no_harm_guard_checked": bool(step_guard_last_info.get("checked", False)),
                    "step_no_harm_guard_passed": bool(step_guard_last_info.get("passed", True)),
                    "step_no_harm_guard_rolled_back": bool(step_guard_last_info.get("rolled_back", False)),
                }
                rank_record = {
                    "step": step,
                    "elapsed_minutes": elapsed_minutes,
                    "lr": optimizer.param_groups[0]["lr"],
                    "lr_schedule_factor": lr_factor,
                    "grad_clip_norm": grad_clip_norm,
                    "ema_decay": ema_decay,
                    "ema_eval": ema_eval,
                    "rank": rank,
                    "world_size": int(distributed.get("world_size", 1)),
                    **cuda_rank_mapping(distributed),
                    "device": str(device),
                    "cuda_memory_fraction": cuda_memory_fraction,
                    "per_rank_batch_size": int(training_cfg.get("batch_size", 1)),
                    "effective_global_batch_size": int(training_cfg.get("batch_size", 1))
                    * int(distributed.get("world_size", 1)),
                    "supervision_valid_ratio": valid_ratio,
                    **cuda_memory_stats(device),
                    **loss_parts,
                }
                append_jsonl(rank_metrics_path, rank_record)
                if is_main_process(distributed):
                    append_jsonl(metrics_path, record)
                    print(json.dumps(record, sort_keys=True))

            if step % val_interval == 0:
                distributed_barrier(distributed)
                if is_main_process(distributed):
                    assert val_loader is not None
                    validation_context = ema_weights_applied(model, ema_state) if ema_eval else nullcontext()
                    with validation_context:
                        val_metrics = validate(
                            unwrap_model(model),
                            val_loader,
                            device,
                            config,
                            router_teacher_a=router_teacher_a,
                            router_teacher_b=router_teacher_b,
                            transfer_loader=transfer_loader,
                        )
                        if best_selection_baseline_gates_enabled(training_cfg):
                            _, best_selection_gate_info = best_selection_gate_status(
                                val_metrics,
                                best_selection_gate_baseline_metrics,
                                training_cfg,
                            )
                        else:
                            best_selection_gate_info = {"enabled": False, "passed": True, "checks": []}
                        selection_value = best_metric_value(val_metrics, selection_metric)
                        is_candidate_better = is_better_metric(selection_value, best_selection_value, selection_metric)
                        is_best_selection = is_candidate_better and (
                            bool(best_selection_gate_info.get("passed", True)) or not best_selection_requires_gate
                        )
                        if is_best_selection:
                            best_val_loss = val_metrics["val_loss"]
                            best_val_step = step
                            best_val_psnr = val_metrics["val_psnr"]
                            best_selection_value = selection_value
                            save_checkpoint(
                                run_dir / "checkpoints" / "best_val.pt",
                                model,
                                optimizer,
                                step,
                                config,
                            )
                            best_summary = {
                                "best_val_step": best_val_step,
                                "best_val_loss": best_val_loss,
                                "best_val_psnr": best_val_psnr,
                                "selection_metric": selection_metric,
                                "best_selection_value": best_selection_value,
                                "checkpoint": str(run_dir / "checkpoints" / "best_val.pt"),
                                "ema_decay": ema_decay,
                                "ema_eval": ema_eval,
                                "baseline_gate_enabled": bool(best_selection_gate_info.get("enabled", False)),
                                "baseline_gate_passed": bool(best_selection_gate_info.get("passed", True)),
                                "baseline_gate_checks": best_selection_gate_info.get("checks", []),
                            }
                            (run_dir / "metrics" / "best_val_summary.json").write_text(
                                json.dumps(best_summary, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8",
                            )
                    val_record = {
                        "step": step,
                        "elapsed_minutes": elapsed_minutes,
                        "rank": rank,
                        "world_size": int(distributed.get("world_size", 1)),
                        **val_metrics,
                        "ema_decay": ema_decay,
                        "ema_eval": ema_eval,
                        "is_best_val_loss": val_metrics["val_loss"] <= best_val_loss,
                        "is_candidate_better": is_candidate_better,
                        "is_best_selection": is_best_selection,
                        "best_selection_requires_gate": best_selection_requires_gate,
                        "best_selection_gate_enabled": bool(best_selection_gate_info.get("enabled", False)),
                        "best_selection_gate_passed": bool(best_selection_gate_info.get("passed", True)),
                        "best_selection_gate_checks": best_selection_gate_info.get("checks", []),
                        "selection_metric": selection_metric,
                        "selection_value": selection_value,
                        "best_selection_value_so_far": best_selection_value,
                        "best_val_step_so_far": best_val_step,
                        "best_val_loss_so_far": best_val_loss,
                        "best_val_psnr_so_far": best_val_psnr,
                    }
                    append_jsonl(run_dir / "metrics" / "val_metrics.jsonl", val_record)
                    print(json.dumps(val_record, sort_keys=True))
                distributed_barrier(distributed)
                model.train()

            if step % ckpt_interval == 0:
                if is_main_process(distributed):
                    save_checkpoint(run_dir / "checkpoints" / f"step_{step:06d}.pt", model, optimizer, step, config)
                    save_checkpoint(run_dir / "checkpoints" / "latest.pt", model, optimizer, step, config)
                distributed_barrier(distributed)

            if step >= max_steps or elapsed_minutes >= wall_clock_minutes:
                break
        epoch += 1
        if step >= max_steps or (time.time() - start_time) / 60.0 >= wall_clock_minutes:
            break

    if is_main_process(distributed):
        save_checkpoint(run_dir / "checkpoints" / "final.pt", model, optimizer, step, config)
        print(f"training: finished step={step} run_dir={run_dir}")
    distributed_barrier(distributed)


def main() -> None:
    parser = argparse.ArgumentParser(description="V1/T01 training entrypoint")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true", help="Run one forward/backward and one validation pass without archiving checkpoints.")
    parser.add_argument("--preflight-train-samples", type=int, default=4)
    args = parser.parse_args()

    config = load_toml(args.config)
    distributed = init_distributed_if_needed()
    try:
        if args.preflight:
            if bool(distributed.get("enabled", False)):
                raise RuntimeError("Use single-process preflight or a short distributed run; --preflight is not DDP-aware.")
            run_preflight(config, args)
        else:
            run_training(config, args.config, distributed)
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
