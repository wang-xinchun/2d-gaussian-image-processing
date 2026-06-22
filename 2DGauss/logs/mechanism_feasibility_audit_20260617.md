# Mechanism Feasibility Audit: Stop Blind Training

- Time: `2026-06-17 18:45 CST`
- Trigger: user requested log maintenance and a hard decision to stop blind training, then decompose prior experiments into viable and non-viable mechanisms.
- Scope: V1/T01 4KDehaze A18/A24/A33/A35/A36/A37-A47 and architecture stop-loss review.

## Executive Decision

Stop open-ended training on the current A37/A47 route-patch line.

Do not launch new experiments unless the proposal explicitly passes a mechanism gate:

1. it changes route-specific internal capacity or boundary representation, not only a scalar loss, sampler, gate head, or late adapter;
2. it defines which previous failure it is meant to falsify;
3. it has promotion criteria on high-route gap, A08-oracle gap, low-route gap, and A15-oracle gap, not aggregate PSNR alone;
4. it has a stop condition before training starts.

`A48 selective positive-up + hard negatives` is no longer an automatic next run. It can only be run as a bounded falsification probe, not as the mainline.

## Current Mechanism Map

### Proven Viable: Keep As Foundation

| Mechanism | Evidence | Decision |
| --- | --- | --- |
| A08/A15 complementarity | A18 fixed `input_std` router reaches `17.84693437581442`, above A08 `17.789133503142946` and A15 `17.781017099305146`; binary oracle is `17.883555795086536`. | Real signal. Do not discard. |
| `input_std` as a strong prior | A18 selected rule chooses A15 when `input_std <= 0.07026463374495506`; correct against binary oracle on `71/100` test samples. | Keep as prior, but not as hard truth. |
| Full route-specific capacity | A36 single-checkpoint dual-backbone router reaches `17.8470899392002`, gap vs A18 router `+0.00015556338577912498`; high gap `-0.00019952460737376242`, low gap `+0.00042343678412253124`. | Capacity answer is yes. Compression/sharing is the real problem. |
| Partial compressed/shared conversion | A33 reaches `17.823062311698262`; A37 reaches `17.889658453438816`, overall gap `+0.042724077624394126`. | Viable milestone, but not final architecture. |
| Native-config teacher loading and low-route lock | A37 beats A18/A36 aggregate while low selected delta remains `0.0`. | Keep as training infrastructure for future B-series. |

### Useful But Not Sufficient: Keep As Tools, Not Mainline

| Mechanism | Evidence | Decision |
| --- | --- | --- |
| Late route residual | A24 is first positive single-checkpoint conversion, `17.790202882742463`, but still `-0.056731493670697 dB` behind A18 router. | Useful diagnostic; too late/small for final. |
| Checkpoint surgery / high-low component recombination | A33 combines A31 high path and A24/A32 low protection; reaches `17.823062311698262`. | Useful for locating capacity, not a scalable method by itself. |
| Hard-case cache and sampler | A46 cache exposes `20` hard positives; sampler smoke hits hard cases `45/64`. | Exposure infrastructure works. Not a breakthrough alone. |
| Bounded residual gate | A45 proves low-prior gate can move; A46 proves it can lift hard positives. | Valid parameterization, but gate-only is not enough. |
| A37/A18/A36 alignment diagnostics | A37-A47 reveal high/low/A08/A15 trade-offs that aggregate PSNR hides. | Mandatory for future promotion. |
| Trust-anchor plumbing | A43 confirms strict-positive and A37 trust-anchor logic can be implemented cleanly. | Useful guardrail for future architectures, insufficient without capacity. |

### Non-Viable As Mainline: Stop Repeating

| Mechanism | Evidence | Decision |
| --- | --- | --- |
| Output/post-route adapters | A35 improves A33 by only `+0.00006982179538539413 dB`; high gap changes by only `+0.00014431539454373699`. | Stop. Too late and too small. |
| Terminal/head-only tuning | A38 improves A37 by only `+0.00016080611639779363 dB`; best at step10 then decays. | Stop as mainline. |
| More shared control capacity | A39 improves aggregate to `17.900369585886775`, but high gap worsens to `-0.11858325626995542`, A08 gap worsens to `-0.23903555195244477`. | Aggregate helper, not high-tail solution. |
| High-input-std balanced sampling alone | A40 is aggregate best `17.900562951956445`, but high gap `-0.1180115677202543`, A08 gap `-0.23833660113789967`, worse than A37. | Stop as high-tail solution. |
| Scalar oracle weighting / branch L1 optimization | A41 reduces high-branch L1 slightly but high gap remains `-0.11813195131188482`. | Stop treating teacher L1 as promotion proxy. |
| Selected-output A08 anchors without capacity | A42/A43 improve or control mechanics but high/A08 worsen; A43 high gap `-0.12314298527548243`, A08 gap `-0.24640859619400587`. | Not sufficient. |
| Residual-logit learned gate | A44 gate remains effectively prior, overall gate `0.42893156833639295` vs prior `0.42893063667252734`. | Reject parameterization. |
| Gate-only hard-case correction | A46 final improves high/A08 to `-0.10929431364232904` / `-0.1659231102631667`, but PSNR collapses to `17.87790077389511`, low/A15 collapse to `+0.1367771104331444` / `+0.1986870903736065`. | Not mainline; proves zero-sum. |
| Global gate-only correction | A47 final protects low/A15 but high/A08 returns to `-0.11217793301246365` / `-0.2010047959040689`. | Not mainline. |
| Aggregate PSNR-only promotion | A40 looks best overall but worsens high/A08. | Forbidden as promotion rule. |

## The Zero-Sum Pattern

The decisive evidence is not one failed run. It is the repeated trade-off:

| run | test PSNR | high gap | low gap | A08 gap | A15 gap | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A37 best | `17.889658453438816` | `-0.11159004428502844` | `+0.1591364853806252` | `-0.19984098726138502` | `+0.2493535773419097` | current compact/shared anchor |
| A40 best | `17.900562951956445` | `-0.1180115677202543` | `+0.18311149168725271` | `-0.23833660113789967` | `+0.3023396530841825` | low/A15 and aggregate improve, high/A08 worsens |
| A46 final | `17.87790077389511` | `-0.10929431364232904` | `+0.1367771104331444` | `-0.1659231102631667` | `+0.1986870903736065` | high/A08 improves, low/A15 and aggregate collapse |
| A47 final | `17.889420757227708` | `-0.11217793301246365` | `+0.1591629694888495` | `-0.2010047959040689` | `+0.24990479172066027` | low/A15 protected, high/A08 returns to A37-like failure |

This is the "拆东墙补西墙" pattern. The current compact shared backbone plus late route heads/tiny gates does not hold both policies at once.

## Mechanisms To Build Next

The next mainline should be a B-series architecture, not another A-series patch.

### B1: Slim Dual-Expert Decoder

- Share highres stem, lowres branch, Gaussian parameter head, and early fusion.
- Split only mid/late decoder blocks or route adapters before RGB residual.
- Distill high expert from A08/A37 high behavior and low expert from A15/A37 low behavior.
- Measure parameter count against A36 full dual-backbone upper bound.

### B2: Multi-Depth Conditional Adapters

- Add small route-specific adapters in highres stem, lowres fusion, Gaussian guidance/refinement, and decoder.
- Do not rely only on terminal RGB heads.
- Gate adapters with `input_std` prior plus learned content confidence.

### B3: Content-Aware Boundary Router

- Use `input_std` as prior, not hard boundary.
- Train router on A08-vs-A15 oracle records, especially nominal-low/A08-oracle samples:
  - `94_9400006.jpg`
  - `94_94000065.jpg`
  - `94_94000081.jpg`
- Promotion requires boundary-case improvement without A15 collapse.

### B4: Route-Specific Gaussian Control

- Current Gaussian control is mostly shared.
- Split control into high-preserve and low-dehaze channels before decoder feature refinement.
- Avoid A39 failure by preventing shared control from redirecting high pressure into low/A15 aggregate gain.

## New Experiment Gate

Before any new training, write a one-page experiment card with:

- mechanism category;
- which prior failure it addresses;
- expected metric movement;
- allowed regression budget;
- checkpoint selection metric;
- stop condition.

Minimum promotion criteria for the next serious run:

- high-route gap better than A37 `-0.11159004428502844`;
- A08-oracle gap better than A37 `-0.19984098726138502`;
- low-route gap remains strongly positive and must not repeat A46-final collapse;
- A15-oracle gap remains strongly positive and must not repeat A46-final collapse;
- aggregate PSNR is secondary until the high/A08 bottleneck moves.

## Final Decision

Mechanisms we should trust:

- A08/A15 complementarity;
- `input_std` prior;
- full route-specific capacity;
- A37-style native teacher infrastructure;
- A18/A36/A37 alignment diagnostics;
- hard-case cache as a diagnostic tool.

Mechanisms we should stop using as mainline:

- late output adapters;
- gate-only correction;
- sampler-only correction;
- scalar weighting-only correction;
- high-branch L1-only promotion;
- aggregate PSNR-only promotion;
- shared-control-only tuning.

The next breakthrough should be architectural: route-specific internal capacity with a learned boundary, not another blind training run.
