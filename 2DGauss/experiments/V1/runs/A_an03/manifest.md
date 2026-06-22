# Run Manifest: V1/T01/A_an03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_an03`
- Full identifier: `V1/T01/A_an03`

## Current Status

Prepared for gate screening. Not launched.

## Purpose

`A_an03` is the edge-state attribution probe for the competitive router family. It removes edge-state input from the router while preserving the competitive gate, so we can tell whether edge-state participation is still essential once the router itself becomes more selective.

## Mechanism

```text
use_gaussian_control_router = true
gaussian_control_router_use_decoder_state = true
gaussian_control_router_use_edge_state = false
edge_guidance_use_control_gate = true
gaussian_control_router_use_competitive_role_gate = true
gaussian_control_router_role_gate_temperature = 0.75
gaussian_feature_refinement_use_intervention_demand = false
```

## Diagnostic Role

If this tracks `A_am03`, edge-state participation is not the key bottleneck once competition is in place. If it drops hard, the closed loop still needs explicit edge-state participation.

## Gate Status

- Static py_compile: pending
- Config parser/small forward: pending
- Formal preflight: pending
- True `6x688` smoke: pending
- Launch: pending
