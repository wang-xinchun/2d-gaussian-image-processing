# Architecture Lessons For G-RSCN - 2026-06-17

Time: `2026-06-17 21:32:35 CST`

Trigger: `MSG-000698` - 用户要求先维护日志，然后继续研究架构，并强调新架构必须吸取上一个架构的教训。

## Executive Verdict

The next architecture must not merely be "A37 plus more modules." It must directly correct the old architecture's failure mode:

```text
old failure = shared Gaussian/control foundation + late input_std route patches
              cannot hold high-preserve/A08 and low-dehaze/A15 policies simultaneously.
```

The Gaussian core remains immutable. The correction is to move route-specific policy capacity into Gaussian-conditioned control and feature paths before the RGB residual is produced.

Recommended direction remains:

```text
G-RSCN: Gaussian-Preserved Route-Specific Control Network
```

But now with stricter lessons:

1. no route mechanism may be only a scalar gate;
2. no route mechanism may act only after RGB residual generation;
3. no route mechanism may be detached from Gaussian attention/prior/control;
4. shared Gaussian control may be kept as a base, but route-specific Gaussian control must be explicit;
5. promotion must require Pareto movement, not aggregate PSNR.

## What The Old Architecture Actually Does

Current `CrossResolutionGaussianRestorer` dataflow:

```text
input
-> highres_stem
-> lowres pixel_unshuffle branch
-> GaussianParameterHead
-> TextureAwareGaussianRenderer
-> attention_map
-> highpass Gaussian modulation + LowRankAffineModulator
-> shared decoder feature
-> optional shared GaussianControlCanvas
-> optional GaussianSharedPrior / GaussianStructuredControlRouter
-> GaussianGuidedDecoderModulator / GaussianGuidedFeatureRefinement
-> shared RGB residual head
-> optional input_std route residual / terminal heads / decoder heads / post adapter
```

Key source locations:

- Gaussian core construction: `/home/ldj/wxc/2DGauss/src/uhd_gaussian/models/restorer.py` lines around `645-650`.
- Decoder and Gaussian guidance/control construction: lines around `651-764`.
- Forward Gaussian prior creation and decoder path: lines around `1034-1175`.
- Late input-std route modules: lines around `1176-1289`.

This confirms the architectural issue:

- `gaussian_control_canvas` is shared.
- `GaussianStructuredControlRouter` routes Gaussian prior into decoder/feature/edge roles, not high-preserve/low-dehaze route policies.
- `input_std_route_feature_expert` acts after the shared decoder feature is already formed.
- `input_std_route_decoder_heads`, terminal heads, residual heads, and post adapter act at or after RGB residual formation.
- A46/A47 trained only the route gate head, so they changed selection but not the internal policy representation.

## Run Evidence Mapped To Lessons

| Run | Mechanism | Result | Lesson For New Architecture |
| --- | --- | --- | --- |
| A36 | Full route-specific capacity | Nearly closes route gaps vs A18 | Route-specific capacity is real; compression must preserve enough of it. |
| A37 | Shared Gaussian foundation plus high route compression, low route locked | Aggregate milestone, but high/A08 remains negative | A37 is the base, not the final. It proves the shared Gaussian foundation is valuable. |
| A39 | Adds trainable shared `gaussian_control_canvas` | Aggregate/low improves; high/A08 worsens | Shared Gaussian control is useful but too blunt; make control route-specific. |
| A40 | High-route balanced sampling | Aggregate/low preserved; high/A08 still worse than A37 | Exposure cannot substitute for route-specific policy capacity. |
| A46 | Hard-case cache plus movable bounded gate | high/A08 improves, low/A15 collapses | Gate-only lifting overgeneralizes; the model needs selective internal policy, not only boundary movement. |
| A47 | Global-context bounded gate | low/A15 protected, high/A08 suppressed | More route context in the gate still lacks route-specific Gaussian-conditioned capacity. |

## Old Failure Modes That Must Not Reappear

### 1. Late Patch Failure

Old behavior:

- route residual, terminal heads, decoder heads, and post adapter operate too late;
- they modify residuals or final decoder outputs after most representation sharing is already fixed.

New constraint:

```text
route-specific modules must operate before RGB residual generation,
inside Gaussian-guided decoder/refinement/control feature paths.
```

### 2. Gate-Capacity Confusion

Old behavior:

- A44-A47 mainly changed `input_std_route_decoder_heads.gate_head`;
- A46 proved gate movement can improve high/A08 but globally harms low/A15;
- A47 proved global gate context can protect low/A15 but suppresses high/A08.

New constraint:

```text
the route gate may choose or blend policies,
but it must not be the only trainable route mechanism.
```

### 3. Shared-Control Dilution

Old behavior:

- A39 made `gaussian_control_canvas` trainable;
- it improved aggregate/low but diluted high/A08 correction.

New constraint:

```text
Gaussian control must have route-specific high-preserve and low-dehaze branches
derived from the same Gaussian prior.
```

### 4. Input-Std Boundary Overtrust

Old behavior:

- `input_std` is a good prior but fails nominal-low/A08-oracle samples;
- A46/A47 show boundary correction alone becomes too broad or too conservative.

New constraint:

```text
input_std initializes the router, but boundary correction must use Gaussian prior statistics,
control canvas statistics, decoder detail energy, and image/content cues.
```

### 5. Teacher L1 / Aggregate PSNR Misleading Promotion

Old behavior:

- A39/A40/A41 improved high-branch L1 or aggregate PSNR without improving high/A08 alignment.

New constraint:

```text
promotion must require high-route and A08-oracle movement while preserving low-route and A15-oracle surplus.
```

## G-RSCN Required Components

### Component 1: Frozen-Compatible Gaussian Foundation

Must preserve:

- highres stem;
- lowres branch and fusion;
- `GaussianParameterHead`;
- `TextureAwareGaussianRenderer`;
- `LowRankAffineModulator`;
- existing Gaussian decoder/refinement machinery.

Initialization rule:

```text
with all new route-specific residuals zero-initialized, G-RSCN must reproduce A37 outputs.
```

This is the first guard against accidental architecture drift.

### Component 2: Gaussian Prior Feature Bank

Build an explicit feature bank before route-specific control:

- attention map;
- raw attention map;
- local contrast of attention;
- directional prior;
- shared Gaussian prior feature;
- control canvas and feedback map;
- decoder feature energy/detail;
- optional image edge/content descriptors.

This bank is the only legal input source for learned route correction beyond `input_std`.

### Component 3: Gaussian-Aware Boundary Router

The router should output:

- route probability or blend gate;
- confidence;
- optional hard-boundary diagnostic logits.

Rules:

- initialize from A18 `input_std` prior;
- add zero-initialized residual correction;
- train/evaluate on A08-vs-A15 oracle records;
- include nominal-low/A08-oracle hard samples and low/A15 hard negatives.

This router is a selector, not the capacity itself.

### Component 4: Route-Specific Gaussian Control

Replace "one shared canvas does everything" with:

```text
shared Gaussian prior
-> high-preserve Gaussian control path
-> low-dehaze Gaussian control path
-> route/confidence blend
-> decoder/refinement modulation
```

High-preserve path target:

- protect detail;
- reduce over-dehazing;
- mimic useful A08/A37 high behavior.

Low-dehaze path target:

- preserve A15-like restoration strength;
- prevent A46-style low/A15 collapse;
- keep A37/A40 low-route surplus.

### Component 5: Gaussian-Conditioned Route Adapters

Adapters are allowed only if they consume Gaussian prior/control features:

```text
adapter(feature, gaussian_control_feature, route_gate) -> zero-init residual feature delta
```

Suggested insertion points:

1. after Gaussian highpass/affine modulation, before decoder input fusion;
2. after decoder block `decoder[1]`, before RGB head;
3. inside Gaussian-guided refinement/guidance as route-specific scale/bias heads;
4. optionally around `gaussian_control_canvas`, but only route-specific.

Do not start with adapters in highres stem unless later evidence demands it; that raises risk of disturbing the Gaussian foundation too early.

### Component 6: Shared RGB Head With Pre-RGB Route Capacity

Keep a shared RGB head for compactness and paper clarity, but route-specific policy must already be present in the feature/control tensor before the head.

This avoids:

- full A36-style dual-backbone cost;
- old terminal-head-only weakness;
- output-space patching.

## Proposed First Architecture: G/B_4kd01

Purpose:

```text
test whether route-specific Gaussian control before RGB can move high/A08 without losing low/A15.
```

Minimal structure:

1. Start from A37 checkpoint.
2. Add default-off `GaussianRouteControl` module:
   - consumes shared Gaussian prior feature, attention/local contrast, decoder feature, and `input_std` prior;
   - emits `high_control_feature`, `low_control_feature`, `route_gate`, `confidence`;
   - zero-init route deltas so the blended control initially equals the old shared path.
3. Add one Gaussian-conditioned route adapter after `decoder[1]` and before `decoder[2]`.
4. Keep `input_std_route_decoder_heads` disabled or frozen for the first architecture probe, except diagnostics.
5. Keep RGB head shared.

Why only one adapter first:

- It tests the core claim with minimal blast radius.
- It avoids recreating the old "many knobs, unclear mechanism" problem.
- If one pre-RGB Gaussian-conditioned route adapter cannot beat A37 high/A08 without low/A15 collapse, deeper adapters should not be blindly added.

## Experiment Gate For G/B_4kd01

Before training, the experiment card must include:

- exact module diagram;
- exact trainable prefixes;
- A37 equivalence test with new modules enabled but zero-initialized;
- parameter count increase vs A37 and vs A36;
- planned losses and teacher checkpoints;
- hard-positive and hard-negative validation lists;
- stop condition.

Minimum preflight checks:

1. strict load from A37 has only expected new keys missing;
2. output max/mean difference vs A37 is near zero before training;
3. route gate initially equals A18 input-std prior within clamp tolerance;
4. low route selected delta remains zero/equivalent at init if low path is meant to be protected;
5. trainable parameter list contains no accidental Gaussian core replacement.

Promotion criteria:

- high-route gap better than A37 `-0.11159004428502844`;
- A08-oracle gap better than A37 `-0.19984098726138502`;
- low-route gap remains strongly positive and does not repeat A46 collapse;
- A15-oracle gap remains strongly positive and does not repeat A46 collapse;
- nominal-low/A08-oracle cases improve;
- aggregate PSNR is secondary until high/A08 moves.

## Decision

Continue architecture research under G-RSCN, but do not train yet.

The next concrete artifact should be a `G/B_4kd01` experiment card, not code or training. The card must turn the above lessons into a small, falsifiable implementation plan.
