# Run Manifest: V1/T01/A_al03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_al03`
- Full identifier: `V1/T01/A_al03`

## Current Status

Prepared for gate screening. Not launched.

## Purpose

`A_al03` tests whether the structured router still needs explicit positive demand formation inside feature refinement.

## Mechanism

```text
use_gaussian_control_router = true
gaussian_control_router_use_decoder_state = true
gaussian_control_router_use_edge_state = true
edge_guidance_use_control_gate = true
gaussian_feature_refinement_use_intervention_demand = true
```

## Diagnostic Role

If `A_al03` improves over `A_aj03`, the missing factor after routing is active intervention-demand estimation. If it underperforms, keep the router clean and leave demand gating separate.

## Gate Status

- Static py_compile: pending
- Config parser/small forward: passed, `total_params=364251`
- Formal preflight: pending
- True `6x720` smoke: pending
- Launch: pending
