# Strong Structured Control Short Probes - 2026-05-19

## Why This Family Exists

`A_ai03` falsified the simple single-gate hypothesis: decoder-detail feedback is useful, but attaching it as one multiplicative feature-refinement gate does not preserve the `A_ah03` edge-role gain and does not recover the `A_z03` short-leader advantage.

The next mechanism must coordinate roles instead of stacking cues. The new code path adds a default-off `GaussianStructuredControlRouter` after the shared prior. It consumes the shared prior plus optional decoder-state and edge-state descriptors, then emits three role-specific control priors:

- decoder control prior for `GaussianGuidedDecoderModulator`;
- feature control prior for `GaussianGuidedFeatureRefinement`;
- edge control prior for `HighResolutionEdgeGuidedRefiner`.

This keeps one shared control language, but lets each module receive a role-specialized view.

## Probe Set

### `A_aj03` - full structured role router

Question: can role-specific control priors jointly preserve shared-prior coordination, decoder-detail feedback, and edge-role separation?

Mechanism:

```text
use_gaussian_control_router = true
gaussian_control_router_use_decoder_state = true
gaussian_control_router_use_edge_state = true
edge_guidance_use_control_gate = true
gaussian_feature_refinement_use_intervention_demand = false
```

Expected diagnostic value:

- if it beats `A_ah03`, role routing is doing useful coordination;
- if it approaches or beats `A_z03`, the router is recovering detail without raw cue stacking.

### `A_ak03` - router without edge-state input

Question: is the full router's gain driven by true cross-module role routing, or just by injecting edge descriptors into the router?

Mechanism:

```text
use_gaussian_control_router = true
gaussian_control_router_use_decoder_state = true
gaussian_control_router_use_edge_state = false
edge_guidance_use_control_gate = true
gaussian_feature_refinement_use_intervention_demand = false
```

Expected diagnostic value:

- if `A_ak03` stays close to `A_aj03`, decoder-state role routing is the key;
- if it drops toward `A_ai03`, edge-state participation is necessary for the closed loop.

### `A_al03` - router plus intervention-demand feature refinement

Question: after role routing, does feature refinement also need an explicit positive demand map rather than only a routed feature prior?

Mechanism:

```text
use_gaussian_control_router = true
gaussian_control_router_use_decoder_state = true
gaussian_control_router_use_edge_state = true
edge_guidance_use_control_gate = true
gaussian_feature_refinement_use_intervention_demand = true
```

Expected diagnostic value:

- if it improves over `A_aj03`, the missing factor is demand formation after routing;
- if it is worse, the router itself should stay cleaner and demand gating should remain separate.

## Common Protocol

All three probes start from the short-tier envelope, but the full router exceeded the preflight memory ceiling at `6x720`, and `6x704` ran but exceeded the reserved-memory window at `93.8086G`. The accepted strong-mechanism scout geometry is therefore `6x696`, preserving batch size and supervision crop while reducing full-resolution activation pressure:

- dataset: `UHD_deblur`;
- effective training split: train pairs after the first `40`;
- internal validation: first `20` train pairs, full image;
- official test: post-hoc only, full `300` pairs;
- training: `max_steps=2500`, `wall_clock_minutes=58`;
- train context: `696`, supervision crop: `512`;
- reserved VRAM target: `85.0-93.0G`;
- frozen baseline isolation: do not modify `M04/A_y03`.

## Decision Rule

Priority order for launch after gates:

1. `A_aj03`: first, because it is the full mechanism and the cleanest answer to the current bottleneck.
2. `A_ak03`: second, only if `A_aj03` is positive or ambiguous and needs edge-state attribution.
3. `A_al03`: second or third, if `A_aj03` is stable but not selective enough.

Promotion threshold remains strict: no medium promotion unless official short post-hoc at least beats `A_z03 = 25.385710820365173 dB` and shows a coherent validation trajectory.
