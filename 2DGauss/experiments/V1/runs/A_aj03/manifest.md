# Run Manifest: V1/T01/A_aj03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_aj03`
- Full identifier: `V1/T01/A_aj03`

## Current Status

Prepared for gate screening. Not launched.

## Purpose

`A_aj03` is the first strong-mechanism probe after `A_ai03`. It tests whether a structured role router can coordinate decoder detail, feature refinement, and edge correction through one shared control surface.

## Mechanism

This branch enables the new default-off structured router:

```text
use_gaussian_control_router = true
gaussian_control_router_use_decoder_state = true
gaussian_control_router_use_edge_state = true
edge_guidance_use_control_gate = true
gaussian_feature_refinement_use_intervention_demand = false
```

The router emits three role-specific control priors: decoder, feature, and edge.

## Baselines

- Frozen medium baseline: `M04/A_y03`, official test `26.272695829299355`
- Current short leader: `A_z03`, official test `25.385710820365173`
- Best shared-control result: `A_ah03`, official test `25.345361164304613`
- Negative single-gate diagnostic: `A_ai03`, official test `25.267052413067724`

## Protocol

- Dataset: `UHD_deblur`
- Train split: after first `40` pairs
- Internal validation: first `20` train pairs, full image
- Official test: post-hoc, `300` full-image pairs
- Geometry: `batch_size=6`, `train_context_size=696`, supervision crop `512`
- Budget: `max_steps=2500`, `wall_clock_minutes=58`
- Reserved VRAM target: `85.0-93.0G`

## Gate Status

- Static py_compile: pending
- Config parser/small forward: passed, `total_params=347979`
- Formal preflight: pending
- True `6x720` smoke: pending
- Launch: pending

## Decision Rule

Launch first if it passes smoke/preflight. It is the cleanest direct test of the current bottleneck.
