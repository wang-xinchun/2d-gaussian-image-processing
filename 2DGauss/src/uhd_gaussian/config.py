from __future__ import annotations

from dataclasses import dataclass, field


INPUT_STD_ROUTE_LEARNED_GATE_MODES = {"blend", "residual", "bounded_residual", "bounded_residual_global"}


@dataclass
class GaussianRendererConfig:
    atoms: int = 4
    param_dim: int = 10
    patch_size: int = 4
    max_frequency: float = 4.0
    max_texture_gain: float = 1.0


@dataclass
class EdgeGuidanceConfig:
    enabled: bool = False
    blocks: int = 2
    strength: float = 0.25
    use_control_gate: bool = False


@dataclass
class DirectionalFrequencyGuidanceConfig:
    enabled: bool = False


@dataclass
class ExplicitFrequencySplitConfig:
    enabled: bool = False
    lowpass_kernel_size: int = 7
    highres_high_band_strength: float = 0.50
    gaussian_feedback_strength: float = 0.10
    gaussian_feedback_attention_source: str = "main"
    detail_blocks: int = 1
    use_structure_hint: bool = False
    structure_hint_strength: float = 0.25


@dataclass
class TrainInferenceConsistencyConfig:
    enabled: bool = False
    train_context_size: int = 0
    train_supervision_crop_size: int = 0


@dataclass
class GaussianDecoderGuidanceConfig:
    enabled: bool = False
    blocks: int = 1
    strength: float = 0.15
    use_decoder_state_gate: bool = False


@dataclass
class GaussianSharedPriorConfig:
    enabled: bool = False
    blocks: int = 1


@dataclass
class GaussianControlRouterConfig:
    enabled: bool = False
    blocks: int = 1
    use_decoder_state: bool = True
    use_edge_state: bool = True
    role_delta_strength: float = 0.50
    use_competitive_role_gate: bool = False
    role_gate_temperature: float = 1.0
    use_role_residual_centering: bool = False
    use_demand_conditioned_competition: bool = False
    role_competition_start: float = 0.45
    role_competition_strength: float = 0.35
    role_competition_floor: float = 0.30


@dataclass
class GaussianTerminalResidualConfig:
    enabled: bool = False
    blocks: int = 1
    strength: float = 0.10


@dataclass
class GaussianFeatureRefinementConfig:
    enabled: bool = False
    blocks: int = 1
    strength: float = 0.12
    use_decoder_detail_hint: bool = False
    use_edge_aware_prior: bool = False
    use_confidence_normalization: bool = False
    use_intervention_demand: bool = False
    use_decoder_detail_gate: bool = False


@dataclass
class GaussianAttentionDemandBridgeConfig:
    enabled: bool = False
    blocks: int = 1
    strength: float = 0.20
    use_input_structure: bool = True
    apply_to: str = "main"


@dataclass
class GaussianControlCanvasConfig:
    enabled: bool = False
    blocks: int = 1
    strength: float = 0.20
    stage: str = "post_refinement"
    share_to_guidance: bool = False
    share_to_guidance_blend: float = 1.0
    share_to_guidance_detail_preserve: bool = False
    share_to_guidance_detail_preserve_strength: float = 0.0
    share_to_guidance_detail_preserve_floor: float = 0.25
    share_to_guidance_detail_preserve_kernel_size: int = 5
    share_to_guidance_detail_preserve_schedule: str = "none"
    share_to_guidance_detail_preserve_schedule_start: float = 0.0
    share_to_guidance_detail_preserve_schedule_end: float = 1.0
    share_to_guidance_reliability_gate: bool = False
    share_to_guidance_reliability_strength: float = 0.0
    share_to_guidance_reliability_floor: float = 0.50
    share_to_guidance_reliability_kernel_size: int = 7
    share_to_guidance_reliability_schedule: str = "none"
    share_to_guidance_reliability_schedule_start: float = 0.0
    share_to_guidance_reliability_schedule_end: float = 1.0
    use_target_conditioned_feedback: bool = False
    target_mode: str = "structure"
    feedback_mode: str = "multiplicative"
    use_confidence_normalization: bool = False
    residual_delta_strength: float = 1.0
    feedback_selectivity_mode: str = "none"
    feedback_selectivity_temperature: float = 2.0
    feedback_demand_power: float = 1.0
    feedback_budget: float = 0.0
    feedback_floor: float = 0.0
    feedback_target_mean: float = 0.0
    feedback_mean_boost_cap: float = 2.0
    feedback_demand_source: str = "target"


@dataclass
class GaussianRouteControlConfig:
    enabled: bool = False
    blocks: int = 1
    strength: float = 0.10
    threshold: float = 0.07026463374495506
    softness: float = 0.003
    direction: str = "gt"
    learned_gate_weight: float = 1.0
    detach_gate: bool = True
    adapter_count: int = 1


@dataclass
class DecoupledCorrectionConfig:
    enabled: bool = False
    blocks: int = 1
    strength: float = 0.10
    use_gradient_detached_selectivity: bool = True
    amplitude_channels: int = 0
    amplitude_mode: str = "residual"


@dataclass
class InputStdRouteResidualConfig:
    enabled: bool = False
    blocks: int = 1
    strength: float = 0.12
    threshold: float = 0.07026463374495506
    softness: float = 0.005
    direction: str = "gt"
    learned_gate_weight: float = 0.0
    learned_gate_mode: str = "blend"
    detach_gate: bool = True


@dataclass
class InputStdRouteTerminalHeadsConfig:
    enabled: bool = False
    blocks: int = 2
    strength: float = 0.60
    threshold: float = 0.07026463374495506
    softness: float = 0.005
    direction: str = "gt"
    learned_gate_weight: float = 0.10
    learned_gate_mode: str = "blend"
    detach_gate: bool = True


@dataclass
class InputStdRouteFeatureExpertConfig:
    enabled: bool = False
    blocks: int = 2
    strength: float = 0.35
    threshold: float = 0.07026463374495506
    softness: float = 0.005
    direction: str = "gt"
    learned_gate_weight: float = 0.05
    learned_gate_mode: str = "blend"
    detach_gate: bool = True


@dataclass
class InputStdRouteDecoderHeadsConfig:
    enabled: bool = False
    blocks: int = 1
    strength: float = 1.0
    threshold: float = 0.07026463374495506
    softness: float = 0.005
    direction: str = "gt"
    learned_gate_weight: float = 0.02
    learned_gate_mode: str = "blend"
    detach_gate: bool = True


@dataclass
class InputStdRoutePostAdapterConfig:
    enabled: bool = False
    blocks: int = 1
    strength: float = 0.08
    threshold: float = 0.07026463374495506
    softness: float = 0.003
    direction: str = "gt"
    detach_gate: bool = True


@dataclass
class EdgeWeightedLossConfig:
    enabled: bool = False
    strength: float = 0.0
    floor: float = 1.0


@dataclass
class ModelConfig:
    in_channels: int = 3
    out_channels: int = 3
    width: int = 32
    lowres_scale: int = 4
    highres_blocks: int = 2
    lowres_blocks: int = 4
    decoder_blocks: int = 2
    affine_rank: int = 4
    gaussian: GaussianRendererConfig = field(default_factory=GaussianRendererConfig)
    edge_guidance: EdgeGuidanceConfig = field(default_factory=EdgeGuidanceConfig)
    directional_frequency_guidance: DirectionalFrequencyGuidanceConfig = field(
        default_factory=DirectionalFrequencyGuidanceConfig
    )
    explicit_frequency_split: ExplicitFrequencySplitConfig = field(
        default_factory=ExplicitFrequencySplitConfig
    )
    train_inference_consistency: TrainInferenceConsistencyConfig = field(
        default_factory=TrainInferenceConsistencyConfig
    )
    gaussian_decoder_guidance: GaussianDecoderGuidanceConfig = field(
        default_factory=GaussianDecoderGuidanceConfig
    )
    gaussian_shared_prior: GaussianSharedPriorConfig = field(
        default_factory=GaussianSharedPriorConfig
    )
    gaussian_control_router: GaussianControlRouterConfig = field(
        default_factory=GaussianControlRouterConfig
    )
    gaussian_terminal_residual: GaussianTerminalResidualConfig = field(
        default_factory=GaussianTerminalResidualConfig
    )
    gaussian_feature_refinement: GaussianFeatureRefinementConfig = field(
        default_factory=GaussianFeatureRefinementConfig
    )
    gaussian_attention_demand_bridge: GaussianAttentionDemandBridgeConfig = field(
        default_factory=GaussianAttentionDemandBridgeConfig
    )
    gaussian_control_canvas: GaussianControlCanvasConfig = field(
        default_factory=GaussianControlCanvasConfig
    )
    gaussian_route_control: GaussianRouteControlConfig = field(
        default_factory=GaussianRouteControlConfig
    )
    decoupled_correction: DecoupledCorrectionConfig = field(
        default_factory=DecoupledCorrectionConfig
    )
    input_std_route_residual: InputStdRouteResidualConfig = field(
        default_factory=InputStdRouteResidualConfig
    )
    input_std_route_terminal_heads: InputStdRouteTerminalHeadsConfig = field(
        default_factory=InputStdRouteTerminalHeadsConfig
    )
    input_std_route_feature_expert: InputStdRouteFeatureExpertConfig = field(
        default_factory=InputStdRouteFeatureExpertConfig
    )
    input_std_route_decoder_heads: InputStdRouteDecoderHeadsConfig = field(
        default_factory=InputStdRouteDecoderHeadsConfig
    )
    input_std_route_post_adapter: InputStdRoutePostAdapterConfig = field(
        default_factory=InputStdRoutePostAdapterConfig
    )

    def __post_init__(self) -> None:
        if self.lowres_scale <= 1:
            raise ValueError("lowres_scale must be greater than 1")
        if self.gaussian.patch_size != self.lowres_scale:
            self.gaussian.patch_size = self.lowres_scale
        if self.edge_guidance.blocks < 0:
            raise ValueError("edge_guidance.blocks must be non-negative")
        if self.edge_guidance.strength < 0.0:
            raise ValueError("edge_guidance.strength must be non-negative")
        if self.directional_frequency_guidance.enabled and not self.edge_guidance.enabled:
            raise ValueError("directional_frequency_guidance requires edge_guidance to be enabled")
        if self.explicit_frequency_split.lowpass_kernel_size <= 0:
            raise ValueError("explicit_frequency_split.lowpass_kernel_size must be positive")
        if self.explicit_frequency_split.lowpass_kernel_size % 2 == 0:
            raise ValueError("explicit_frequency_split.lowpass_kernel_size must be odd")
        if self.explicit_frequency_split.highres_high_band_strength < 0.0:
            raise ValueError("explicit_frequency_split.highres_high_band_strength must be non-negative")
        if self.explicit_frequency_split.gaussian_feedback_strength < 0.0:
            raise ValueError("explicit_frequency_split.gaussian_feedback_strength must be non-negative")
        if self.explicit_frequency_split.gaussian_feedback_attention_source not in {"main", "control"}:
            raise ValueError("explicit_frequency_split.gaussian_feedback_attention_source must be main or control")
        if self.explicit_frequency_split.detail_blocks < 0:
            raise ValueError("explicit_frequency_split.detail_blocks must be non-negative")
        if self.explicit_frequency_split.structure_hint_strength < 0.0:
            raise ValueError("explicit_frequency_split.structure_hint_strength must be non-negative")
        if self.train_inference_consistency.train_context_size < 0:
            raise ValueError("train_inference_consistency.train_context_size must be non-negative")
        if self.train_inference_consistency.train_supervision_crop_size < 0:
            raise ValueError("train_inference_consistency.train_supervision_crop_size must be non-negative")
        if self.train_inference_consistency.enabled:
            context = self.train_inference_consistency.train_context_size
            supervision = self.train_inference_consistency.train_supervision_crop_size
            if context <= 0:
                raise ValueError("train_inference_consistency requires train_context_size > 0")
            if supervision <= 0:
                raise ValueError("train_inference_consistency requires train_supervision_crop_size > 0")
            if context < supervision:
                raise ValueError("train_context_size must be >= train_supervision_crop_size")
        if self.gaussian_decoder_guidance.blocks < 0:
            raise ValueError("gaussian_decoder_guidance.blocks must be non-negative")
        if self.gaussian_decoder_guidance.strength < 0.0:
            raise ValueError("gaussian_decoder_guidance.strength must be non-negative")
        if self.gaussian_shared_prior.blocks < 0:
            raise ValueError("gaussian_shared_prior.blocks must be non-negative")
        if self.gaussian_control_router.enabled and not self.gaussian_shared_prior.enabled:
            raise ValueError("gaussian_control_router requires gaussian_shared_prior to be enabled")
        if self.gaussian_control_router.blocks < 0:
            raise ValueError("gaussian_control_router.blocks must be non-negative")
        if self.gaussian_control_router.role_delta_strength < 0.0:
            raise ValueError("gaussian_control_router.role_delta_strength must be non-negative")
        if self.gaussian_control_router.role_gate_temperature <= 0.0:
            raise ValueError("gaussian_control_router.role_gate_temperature must be positive")
        if not 0.0 <= self.gaussian_control_router.role_competition_start <= 1.0:
            raise ValueError("gaussian_control_router.role_competition_start must be in [0, 1]")
        if self.gaussian_control_router.role_competition_strength < 0.0:
            raise ValueError("gaussian_control_router.role_competition_strength must be non-negative")
        if not 0.0 <= self.gaussian_control_router.role_competition_floor <= 1.0:
            raise ValueError("gaussian_control_router.role_competition_floor must be in [0, 1]")
        if self.gaussian_terminal_residual.blocks < 0:
            raise ValueError("gaussian_terminal_residual.blocks must be non-negative")
        if self.gaussian_terminal_residual.strength < 0.0:
            raise ValueError("gaussian_terminal_residual.strength must be non-negative")
        if self.gaussian_feature_refinement.blocks < 0:
            raise ValueError("gaussian_feature_refinement.blocks must be non-negative")
        if self.gaussian_feature_refinement.strength < 0.0:
            raise ValueError("gaussian_feature_refinement.strength must be non-negative")
        if self.gaussian_attention_demand_bridge.blocks < 0:
            raise ValueError("gaussian_attention_demand_bridge.blocks must be non-negative")
        if not 0.0 <= self.gaussian_attention_demand_bridge.strength <= 1.0:
            raise ValueError("gaussian_attention_demand_bridge.strength must be in [0, 1]")
        if self.gaussian_attention_demand_bridge.apply_to not in {"main", "control"}:
            raise ValueError("gaussian_attention_demand_bridge.apply_to must be main or control")
        if self.gaussian_control_canvas.blocks < 0:
            raise ValueError("gaussian_control_canvas.blocks must be non-negative")
        if self.gaussian_control_canvas.strength < 0.0:
            raise ValueError("gaussian_control_canvas.strength must be non-negative")
        if self.gaussian_control_canvas.stage not in {"post_refinement", "pre_guidance"}:
            raise ValueError("gaussian_control_canvas.stage must be post_refinement or pre_guidance")
        if self.gaussian_control_canvas.share_to_guidance and not self.gaussian_control_canvas.enabled:
            raise ValueError("gaussian_control_canvas.share_to_guidance requires gaussian_control_canvas.enabled")
        if (
            self.gaussian_control_canvas.share_to_guidance
            and self.gaussian_control_canvas.stage != "pre_guidance"
        ):
            raise ValueError("gaussian_control_canvas.share_to_guidance requires stage=pre_guidance")
        if not 0.0 <= self.gaussian_control_canvas.share_to_guidance_blend <= 1.0:
            raise ValueError("gaussian_control_canvas.share_to_guidance_blend must be in [0, 1]")
        if not 0.0 <= self.gaussian_control_canvas.share_to_guidance_detail_preserve_strength <= 1.0:
            raise ValueError(
                "gaussian_control_canvas.share_to_guidance_detail_preserve_strength must be in [0, 1]"
            )
        if not 0.0 <= self.gaussian_control_canvas.share_to_guidance_detail_preserve_floor <= 1.0:
            raise ValueError(
                "gaussian_control_canvas.share_to_guidance_detail_preserve_floor must be in [0, 1]"
            )
        if self.gaussian_control_canvas.share_to_guidance_detail_preserve_kernel_size <= 0:
            raise ValueError(
                "gaussian_control_canvas.share_to_guidance_detail_preserve_kernel_size must be positive"
            )
        if self.gaussian_control_canvas.share_to_guidance_detail_preserve_schedule not in {"none", "linear"}:
            raise ValueError(
                "gaussian_control_canvas.share_to_guidance_detail_preserve_schedule must be none or linear"
            )
        if not 0.0 <= self.gaussian_control_canvas.share_to_guidance_detail_preserve_schedule_start <= 1.0:
            raise ValueError(
                "gaussian_control_canvas.share_to_guidance_detail_preserve_schedule_start must be in [0, 1]"
            )
        if not 0.0 <= self.gaussian_control_canvas.share_to_guidance_detail_preserve_schedule_end <= 1.0:
            raise ValueError(
                "gaussian_control_canvas.share_to_guidance_detail_preserve_schedule_end must be in [0, 1]"
            )
        if (
            self.gaussian_control_canvas.share_to_guidance_detail_preserve_schedule_end
            < self.gaussian_control_canvas.share_to_guidance_detail_preserve_schedule_start
        ):
            raise ValueError(
                "gaussian_control_canvas.share_to_guidance_detail_preserve_schedule_end must be >= start"
            )
        if not 0.0 <= self.gaussian_control_canvas.share_to_guidance_reliability_strength <= 1.0:
            raise ValueError(
                "gaussian_control_canvas.share_to_guidance_reliability_strength must be in [0, 1]"
            )
        if not 0.0 <= self.gaussian_control_canvas.share_to_guidance_reliability_floor <= 1.0:
            raise ValueError(
                "gaussian_control_canvas.share_to_guidance_reliability_floor must be in [0, 1]"
            )
        if self.gaussian_control_canvas.share_to_guidance_reliability_kernel_size <= 0:
            raise ValueError(
                "gaussian_control_canvas.share_to_guidance_reliability_kernel_size must be positive"
            )
        if self.gaussian_control_canvas.share_to_guidance_reliability_schedule not in {"none", "linear"}:
            raise ValueError(
                "gaussian_control_canvas.share_to_guidance_reliability_schedule must be none or linear"
            )
        if not 0.0 <= self.gaussian_control_canvas.share_to_guidance_reliability_schedule_start <= 1.0:
            raise ValueError(
                "gaussian_control_canvas.share_to_guidance_reliability_schedule_start must be in [0, 1]"
            )
        if not 0.0 <= self.gaussian_control_canvas.share_to_guidance_reliability_schedule_end <= 1.0:
            raise ValueError(
                "gaussian_control_canvas.share_to_guidance_reliability_schedule_end must be in [0, 1]"
            )
        if (
            self.gaussian_control_canvas.share_to_guidance_reliability_schedule_end
            < self.gaussian_control_canvas.share_to_guidance_reliability_schedule_start
        ):
            raise ValueError(
                "gaussian_control_canvas.share_to_guidance_reliability_schedule_end must be >= start"
            )
        if self.gaussian_control_canvas.target_mode not in {"structure", "restoration_demand"}:
            raise ValueError("gaussian_control_canvas.target_mode must be structure or restoration_demand")
        if self.gaussian_control_canvas.feedback_mode not in {"multiplicative", "additive"}:
            raise ValueError("gaussian_control_canvas.feedback_mode must be multiplicative or additive")
        if self.gaussian_control_canvas.residual_delta_strength < 0.0:
            raise ValueError("gaussian_control_canvas.residual_delta_strength must be non-negative")
        if self.gaussian_control_canvas.feedback_selectivity_mode not in {"none", "demand_budget"}:
            raise ValueError("gaussian_control_canvas.feedback_selectivity_mode must be none or demand_budget")
        if self.gaussian_control_canvas.feedback_demand_source not in {"target", "canvas", "mixed"}:
            raise ValueError("gaussian_control_canvas.feedback_demand_source must be target, canvas, or mixed")
        if self.gaussian_control_canvas.feedback_selectivity_temperature <= 0.0:
            raise ValueError("gaussian_control_canvas.feedback_selectivity_temperature must be positive")
        if self.gaussian_control_canvas.feedback_demand_power <= 0.0:
            raise ValueError("gaussian_control_canvas.feedback_demand_power must be positive")
        if self.gaussian_control_canvas.feedback_budget < 0.0:
            raise ValueError("gaussian_control_canvas.feedback_budget must be non-negative")
        if not 0.0 <= self.gaussian_control_canvas.feedback_floor <= 1.0:
            raise ValueError("gaussian_control_canvas.feedback_floor must be in [0, 1]")
        if not 0.0 <= self.gaussian_control_canvas.feedback_target_mean <= 1.0:
            raise ValueError("gaussian_control_canvas.feedback_target_mean must be in [0, 1]")
        if self.gaussian_control_canvas.feedback_mean_boost_cap < 1.0:
            raise ValueError("gaussian_control_canvas.feedback_mean_boost_cap must be at least 1.0")
        if (
            self.gaussian_control_canvas.feedback_budget > 0.0
            and self.gaussian_control_canvas.feedback_target_mean > self.gaussian_control_canvas.feedback_budget
        ):
            raise ValueError(
                "gaussian_control_canvas.feedback_target_mean must not exceed feedback_budget when feedback_budget is set"
            )
        if self.gaussian_route_control.blocks < 0:
            raise ValueError("gaussian_route_control.blocks must be non-negative")
        if self.gaussian_route_control.strength < 0.0:
            raise ValueError("gaussian_route_control.strength must be non-negative")
        if self.gaussian_route_control.softness <= 0.0:
            raise ValueError("gaussian_route_control.softness must be positive")
        if self.gaussian_route_control.direction not in {"gt", "gte", "lt", "lte"}:
            raise ValueError("gaussian_route_control.direction must be gt, gte, lt, or lte")
        if not 0.0 <= self.gaussian_route_control.learned_gate_weight <= 1.0:
            raise ValueError("gaussian_route_control.learned_gate_weight must be in [0, 1]")
        if self.gaussian_route_control.adapter_count not in {1, 2}:
            raise ValueError("gaussian_route_control.adapter_count must be 1 or 2")
        if self.decoupled_correction.blocks < 0:
            raise ValueError("decoupled_correction.blocks must be non-negative")
        if self.decoupled_correction.strength < 0.0:
            raise ValueError("decoupled_correction.strength must be non-negative")
        if self.decoupled_correction.amplitude_channels < 0:
            raise ValueError("decoupled_correction.amplitude_channels must be non-negative")
        if self.decoupled_correction.amplitude_mode not in {"residual", "detail"}:
            raise ValueError("decoupled_correction.amplitude_mode must be residual or detail")
        if self.input_std_route_residual.blocks < 0:
            raise ValueError("input_std_route_residual.blocks must be non-negative")
        if self.input_std_route_residual.strength < 0.0:
            raise ValueError("input_std_route_residual.strength must be non-negative")
        if self.input_std_route_residual.softness <= 0.0:
            raise ValueError("input_std_route_residual.softness must be positive")
        if self.input_std_route_residual.direction not in {"gt", "gte", "lt", "lte"}:
            raise ValueError("input_std_route_residual.direction must be gt, gte, lt, or lte")
        if not 0.0 <= self.input_std_route_residual.learned_gate_weight <= 1.0:
            raise ValueError("input_std_route_residual.learned_gate_weight must be in [0, 1]")
        if self.input_std_route_residual.learned_gate_mode not in INPUT_STD_ROUTE_LEARNED_GATE_MODES:
            raise ValueError(
                "input_std_route_residual.learned_gate_mode must be blend, residual, bounded_residual, "
                "or bounded_residual_global"
            )
        if self.input_std_route_terminal_heads.blocks < 0:
            raise ValueError("input_std_route_terminal_heads.blocks must be non-negative")
        if self.input_std_route_terminal_heads.strength < 0.0:
            raise ValueError("input_std_route_terminal_heads.strength must be non-negative")
        if self.input_std_route_terminal_heads.softness <= 0.0:
            raise ValueError("input_std_route_terminal_heads.softness must be positive")
        if self.input_std_route_terminal_heads.direction not in {"gt", "gte", "lt", "lte"}:
            raise ValueError("input_std_route_terminal_heads.direction must be gt, gte, lt, or lte")
        if not 0.0 <= self.input_std_route_terminal_heads.learned_gate_weight <= 1.0:
            raise ValueError("input_std_route_terminal_heads.learned_gate_weight must be in [0, 1]")
        if self.input_std_route_terminal_heads.learned_gate_mode not in INPUT_STD_ROUTE_LEARNED_GATE_MODES:
            raise ValueError(
                "input_std_route_terminal_heads.learned_gate_mode must be blend, residual, bounded_residual, "
                "or bounded_residual_global"
            )
        if self.input_std_route_feature_expert.blocks < 0:
            raise ValueError("input_std_route_feature_expert.blocks must be non-negative")
        if self.input_std_route_feature_expert.strength < 0.0:
            raise ValueError("input_std_route_feature_expert.strength must be non-negative")
        if self.input_std_route_feature_expert.softness <= 0.0:
            raise ValueError("input_std_route_feature_expert.softness must be positive")
        if self.input_std_route_feature_expert.direction not in {"gt", "gte", "lt", "lte"}:
            raise ValueError("input_std_route_feature_expert.direction must be gt, gte, lt, or lte")
        if not 0.0 <= self.input_std_route_feature_expert.learned_gate_weight <= 1.0:
            raise ValueError("input_std_route_feature_expert.learned_gate_weight must be in [0, 1]")
        if self.input_std_route_feature_expert.learned_gate_mode not in INPUT_STD_ROUTE_LEARNED_GATE_MODES:
            raise ValueError(
                "input_std_route_feature_expert.learned_gate_mode must be blend, residual, bounded_residual, "
                "or bounded_residual_global"
            )
        if self.input_std_route_decoder_heads.blocks < 0:
            raise ValueError("input_std_route_decoder_heads.blocks must be non-negative")
        if self.input_std_route_decoder_heads.strength < 0.0:
            raise ValueError("input_std_route_decoder_heads.strength must be non-negative")
        if self.input_std_route_decoder_heads.softness <= 0.0:
            raise ValueError("input_std_route_decoder_heads.softness must be positive")
        if self.input_std_route_decoder_heads.direction not in {"gt", "gte", "lt", "lte"}:
            raise ValueError("input_std_route_decoder_heads.direction must be gt, gte, lt, or lte")
        if not 0.0 <= self.input_std_route_decoder_heads.learned_gate_weight <= 1.0:
            raise ValueError("input_std_route_decoder_heads.learned_gate_weight must be in [0, 1]")
        if self.input_std_route_decoder_heads.learned_gate_mode not in INPUT_STD_ROUTE_LEARNED_GATE_MODES:
            raise ValueError(
                "input_std_route_decoder_heads.learned_gate_mode must be blend, residual, bounded_residual, "
                "or bounded_residual_global"
            )
        if self.input_std_route_post_adapter.blocks < 0:
            raise ValueError("input_std_route_post_adapter.blocks must be non-negative")
        if self.input_std_route_post_adapter.strength < 0.0:
            raise ValueError("input_std_route_post_adapter.strength must be non-negative")
        if self.input_std_route_post_adapter.softness <= 0.0:
            raise ValueError("input_std_route_post_adapter.softness must be positive")
        if self.input_std_route_post_adapter.direction not in {"gt", "gte", "lt", "lte"}:
            raise ValueError("input_std_route_post_adapter.direction must be gt, gte, lt, or lte")
        terminal_route_mechanisms = sum(
            int(cfg.enabled)
            for cfg in (
                self.input_std_route_residual,
                self.input_std_route_terminal_heads,
                self.input_std_route_decoder_heads,
            )
        )
        if terminal_route_mechanisms > 1:
            raise ValueError(
                "only one terminal input_std route mechanism can be enabled at a time: "
                "residual, terminal_heads, or decoder_heads"
            )
        if (
            self.input_std_route_feature_expert.enabled
            and terminal_route_mechanisms == 1
            and not self.input_std_route_decoder_heads.enabled
        ):
            raise ValueError(
                "input_std_route_feature_expert can be enabled alone or with decoder_heads only"
            )
