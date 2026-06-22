# Run Manifest: V1/T01/A_ao03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_ao03`
- Full identifier: `V1/T01/A_ao03`

## Current Status

Prepared for gate screening. Not launched.

## Purpose

`A_ao03` is the demand-branch probe for the competitive router family. It tests whether, after making the role router competitive, feature refinement still benefits from an explicit intervention-demand map.

## Mechanism

```text
use_gaussian_control_router = true
gaussian_control_router_use_decoder_state = true
gaussian_control_router_use_edge_state = true
edge_guidance_use_control_gate = true
gaussian_control_router_use_competitive_role_gate = true
gaussian_control_router_role_gate_temperature = 0.75
gaussian_feature_refinement_use_intervention_demand = true
```

## Diagnostic Role

If this improves over `A_am03`, the remaining bottleneck is positive demand formation. If not, the router should stay the main control surface and demand should remain separate.

## Gate Status

- Static py_compile: pending
- Config parser/small forward: pending
- Formal preflight: pending
- True `6x688` smoke: pending
- Launch: pending
