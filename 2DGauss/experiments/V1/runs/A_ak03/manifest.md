# Run Manifest: V1/T01/A_ak03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_ak03`
- Full identifier: `V1/T01/A_ak03`

## Current Status

Prepared for gate screening. Not launched.

## Purpose

`A_ak03` is a structured-router attribution probe. It removes edge-state input from the router while still allowing the routed edge control prior to drive the edge module.

## Mechanism

```text
use_gaussian_control_router = true
gaussian_control_router_use_decoder_state = true
gaussian_control_router_use_edge_state = false
edge_guidance_use_control_gate = true
gaussian_feature_refinement_use_intervention_demand = false
```

## Diagnostic Role

If this tracks `A_aj03`, decoder-state routing is the main active ingredient. If it falls toward `A_ai03`, edge-state participation is necessary for the closed loop.

## Gate Status

- Static py_compile: pending
- Config parser/small forward: passed, `total_params=345819`
- Formal preflight: pending
- True `6x720` smoke: pending
- Launch: pending
