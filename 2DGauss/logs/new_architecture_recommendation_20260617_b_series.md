# New Architecture Recommendation: B-Series Route-Specific Gaussian Adapter Network

- Time: `2026-06-17 18:51:58 CST`
- Trigger: user asked to preserve the core idea and decide what new architecture should replace the current route-patch line.
- Status: architecture recommendation, not a training launch.

## Core Idea To Preserve

Do not abandon the project core. Preserve these ideas:

1. **Cross-resolution Gaussian structural prior**
   - The model is not a pure CNN and not a pure renderer.
   - Gaussian attention/control should remain the structural prior that tells the restoration backbone where texture, haze, detail, and intervention demand live.

2. **Conditional restoration**
   - A08/A15 complementarity is real.
   - Different images need different restoration policies: high-preserve / A08-like versus low-dehaze / A15-like.

3. **Single-checkpoint compactness**
   - A36 proves full dual-backbone route capacity works, but it is too expensive as the final story.
   - The paper-worthy direction is compact route-specific capacity, not two independent full models.

4. **`input_std` as prior, not truth**
   - `input_std` is a strong prior from A18, but A42-A47 show boundary samples violate it.
   - The new router should start from `input_std` and learn content-aware correction.

5. **High-tail-first evaluation**
   - Aggregate PSNR alone is forbidden as a promotion rule.
   - Future evaluation must track high-route, low-route, A08-oracle, A15-oracle, and worst-gap cases.

## Recommended Main Architecture

Adopt a B-series architecture:

```text
Route-Specific Gaussian Adapter Network
(RSGA-Net)
```

This architecture keeps the shared cross-resolution Gaussian backbone, but moves route-specific capacity inside the model before the final RGB residual.

## Architecture Sketch

```text
input RGB
  -> shared highres stem
  -> shared pixel-unshuffle lowres branch
  -> shared GaussianParameterHead + TextureAwareGaussianRenderer
  -> shared early fusion feature

  -> content-aware router:
       prior = input_std sigmoid rule
       correction = router(image/content/Gaussian features)
       route_gate = bounded(prior + correction)

  -> route-specific internal adapters:
       high-preserve adapters: A08-like, conservative detail preservation
       low-dehaze adapters: A15-like, stronger restoration/demand correction

  -> slim dual-expert decoder blocks:
       shared decoder stem
       high decoder adapter/block
       low decoder adapter/block
       gated blend in feature space, not only RGB output

  -> route-specific Gaussian control:
       high-preserve control channel
       low-dehaze control channel
       route-gated control injected before final decoder head

  -> shared RGB projection
  -> restored = input + residual
```

The central change is this:

```text
old A-line:
shared decoder feature -> late route heads/gates -> RGB residual

new B-line:
shared early feature -> route-specific adapters/control/decoder features -> RGB residual
```

## Why This Architecture Fits The Evidence

### It keeps what worked

- A18/A36: route prior and complementarity work.
- A37: compact compression can beat the router aggregate line.
- A46/A45: bounded residual gating and hard-case infrastructure are usable.
- A18/A36/A37 diagnostics: high/low/A08/A15 decomposition catches trade-offs.

### It avoids what failed

- A35/A38: final-output adapters and terminal-only heads were too late.
- A39: shared control capacity helped aggregate but redirected pressure into low/A15.
- A40/A41: sampling/weighting without capacity did not solve high/A08.
- A46/A47: gate-only correction caused zero-sum trade-offs.

## Concrete B1 Design

### B1.1 Shared Foundation

Keep from `CrossResolutionGaussianRestorer`:

- `highres_stem`
- `lowres_proj`
- `texture_frequency`
- `lowres_fusion`
- `GaussianParameterHead`
- `TextureAwareGaussianRenderer`
- `LowRankAffineModulator`
- existing Gaussian decoder guidance/refinement as shared base, but do not make it the only trainable route capacity.

### B1.2 Content-Aware Boundary Router

Input features:

- scalar `input_std` prior;
- pooled highres stem feature;
- pooled lowres fusion feature;
- pooled Gaussian attention/control statistics;
- optional restoration-demand/control-canvas statistics.

Output:

```text
route_gate in [0, 1]
```

Initialization:

- initial route gate should equal fixed A18 `input_std` prior.
- learned correction is zero-initialized.

Training:

- supervised by A08-vs-A15 oracle records;
- use hard positives and hard negatives;
- include nominal-low/A08-oracle boundary samples.

### B1.3 Multi-Depth Route Adapters

Add small high/low adapters at multiple points:

1. after highres stem;
2. after lowres fusion;
3. before/inside Gaussian-guided decoder refinement;
4. inside decoder blocks.

Adapter form should be small and stable:

```text
adapter(x) = x + scale * route_gate * A_high(x) + scale * (1-route_gate) * A_low(x)
```

or lower-cost:

```text
x + scale * (route_gate * LoRA_high(x) + (1-route_gate) * LoRA_low(x))
```

Key rule: route-specific capacity must exist before the final RGB head.

### B1.4 Slim Dual-Expert Decoder

Replace the current single shared decoder block plus late route heads with:

```text
decoder_shared_stem
decoder_high_adapter/block
decoder_low_adapter/block
feature_blend = gate * high_feature + (1-gate) * low_feature
shared_rgb_head(feature_blend)
```

This is not A36 full dual backbone. It is a compact, partially shared decoder.

Parameter budget target:

- substantially less than A36 full dual-backbone checkpoint;
- larger than ~1k gate-only patches;
- enough to represent two internal policies.

### B1.5 Route-Specific Gaussian Control

Instead of one shared `gaussian_control_canvas`, use two route roles:

```text
control_high_preserve
control_low_dehaze
control = gate * control_high_preserve + (1-gate) * control_low_dehaze
```

The high branch should avoid excessive dehaze/detail suppression; the low branch may apply stronger restoration-demand correction.

This directly addresses A39:

- A39 showed shared control capacity improves aggregate but hurts high/A08.
- B1 should stop shared control from absorbing both policies into one averaged control map.

## Training Strategy

Do not train from scratch first. Use staged initialization.

### Stage 0: No Training Sanity

- Implement B1 default-off.
- Initialize route gate to A18 prior.
- Initialize shared modules from A37 best-val.
- Initialize high route adapters conservatively from A37/A08-like high behavior if possible.
- Initialize low route adapters from A37/A15-like low behavior if possible.
- Verify zero-init equivalence or near-equivalence to A37 before training.

### Stage 1: Router-Only Fit

- Freeze restoration body.
- Train only content-aware boundary router on A08/A15 oracle labels.
- Objective: improve boundary classification without changing image outputs.
- Validate on nominal-low/A08-oracle cases.

### Stage 2: Adapter Distillation

- Freeze most shared backbone.
- Train route adapters and slim decoder experts.
- Distill:
  - high route toward A08/A37 high-preserve target;
  - low route toward A15/A37 low-dehaze target;
  - hard boundary samples with A08/A15 oracle labels.

### Stage 3: Joint Fine-Tune With Guardrails

- Allow limited shared refinement only after adapters show non-zero benefit.
- Keep A37 trust anchor on hard cases.
- Use high-tail-first checkpoint selection.

## Promotion Criteria

A B-series run is not promoted unless:

- high-route gap beats A37 `-0.11159004428502844`;
- A08-oracle gap beats A37 `-0.19984098726138502`;
- low-route gap remains strongly positive and does not repeat A46 collapse;
- A15-oracle gap remains strongly positive and does not repeat A46 collapse;
- worst-gap boundary cases improve, especially:
  - `94_9400006.jpg`
  - `94_94000065.jpg`
  - `94_94000081.jpg`
- aggregate PSNR is secondary until high/A08 improves.

## Alternatives Considered

### Only learned router

Rejected as main architecture. A36/A18 show routing matters, but A46/A47 show routing/gating without enough internal route capacity becomes zero-sum.

### Only bigger shared backbone

Risky. A39 shows shared capacity can improve aggregate while hurting high/A08. Bigger shared capacity may repeat this unless route-specific channels are explicit.

### Full dual backbone

Mechanically valid but not the desired paper contribution. A36 already proves this upper bound. Use it as teacher/upper bound, not final architecture.

### Continue late route heads/gates

Rejected as mainline by A35/A38/A44-A47.

## Recommended First B-Series Experiment

Start with:

```text
B_4kd01: A37-initialized Route-Specific Gaussian Adapter Decoder
```

Minimal first implementation:

- shared A37 backbone;
- content-aware route correction head initialized to zero;
- route-specific decoder adapter blocks before RGB head;
- optional route-specific Gaussian control split if the decoder adapter works;
- no open-ended training until an experiment card is written.

The first B-series goal is not to beat A40 aggregate PSNR. The goal is to move the Pareto front:

```text
improve high/A08 over A37
without giving back low/A15 like A46.
```

## Paper-Level Position

The new architecture can support a cleaner paper story:

```text
2D Gaussian priors provide structural restoration control.
Conditional image restoration contains route-specific policy conflict.
Full route capacity solves the conflict but is expensive.
Naive compact sharing causes Pareto trade-offs.
Route-Specific Gaussian Adapters recover compact conditional capacity.
```

This preserves the original core idea while avoiding the current patch trap.

## 2026-06-17 Clarification: Gaussian Core Must Not Change

Trigger: `MSG-000696`.

The `Route-Specific Gaussian Adapter Network` wording must not be interpreted as replacing the Gaussian core with ordinary adapters or a generic dual decoder. The stricter interpretation is:

```text
Gaussian stays central; route-specific mechanisms only change how each route consumes and applies the Gaussian prior.
```

The clarified architecture name is:

```text
Gaussian-Preserved Route-Specific Control Network (G-RSCN)
```

Hard invariant:

- keep `GaussianParameterHead` and `TextureAwareGaussianRenderer`;
- keep Gaussian attention/control in decoder/refinement/control paths;
- make the boundary router Gaussian-aware, not `input_std`-only;
- make adapters Gaussian-conditioned, not generic feature patches;
- turn shared Gaussian control into route-specific Gaussian control before RGB output;
- do not use gate-only, post-head-only, bigger-shared-backbone, or non-Gaussian replacement as the mainline.

The companion review is `/home/ldj/wxc/2DGauss/logs/gaussian_core_preservation_architecture_review_20260617.md`.
