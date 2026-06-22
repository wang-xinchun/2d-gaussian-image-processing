# Gaussian Core Preservation Architecture Review - 2026-06-17

Time: `2026-06-17 20:59:44 CST`

Trigger: `MSG-000696` - 用户强调“核心 idea 不能变，也就是高斯不能变”，要求重新研究并认真分析其他机制。

## Verdict

The Gaussian core is immutable. The next architecture must not replace the project with a generic adapter, generic dual-decoder, Transformer, diffusion prior, or CNN-only restoration model.

The previous B-series recommendation should be read with this stricter constraint:

```text
route-specific capacity is allowed only when it is conditioned by, routed by, or injected through the Gaussian structural prior.
```

The clearer name for the next direction is:

```text
Gaussian-Preserved Route-Specific Control Network (G-RSCN)
```

The experiment name can still remain `B_4kd01` for continuity, but its mechanism must be interpreted as `Gaussian-conditioned route-specific control`, not ordinary route adapters.

## Immutable Gaussian Core

The following parts are the project identity and must stay in the model:

1. `GaussianParameterHead`
   - Reads fused low-resolution/high-resolution features.
   - Predicts per-atom Gaussian parameters plus affine coefficients.
   - Current code: `/home/ldj/wxc/2DGauss/src/uhd_gaussian/models/gaussian.py`.

2. `TextureAwareGaussianRenderer`
   - Converts Gaussian parameters into a structural attention map.
   - Uses amplitude, center, sigma, angle, texture gain, frequency, phase, and beta.
   - This is not a decorative attention block; it is the explicit 2D Gaussian structural prior.

3. Gaussian-guided decoder/refinement/control path
   - `GaussianGuidedDecoderModulator`
   - `GaussianGuidedFeatureRefinement`
   - `GaussianControlCanvasModule`
   - `GaussianSharedPriorEncoder`
   - `GaussianStructuredControlRouter`
   - These modules propagate `attention_map`, local contrast, directional prior, shared prior feature, and canvas feedback into the restoration path.

4. Cross-resolution Gaussian restoration logic
   - The model must keep the cross-resolution feature foundation and Gaussian control before final RGB output.
   - A route-specific mechanism that only edits RGB output after the fact is not enough.

5. Conditional high/low restoration evaluation
   - A08/A15 complementarity, high-route/A08-oracle, low-route/A15-oracle, and hard boundary samples remain the central evidence chain.

## What the Experiments Say

The evidence does not say that Gaussian structure failed. It says the current non-Gaussian routing and late patch mechanisms cannot interpret the Gaussian prior differently for conflicting restoration policies.

| Evidence | Reading |
| --- | --- |
| A36 full route-specific capacity nearly closes route gaps | Route-specific policy capacity works. |
| A37 compressed/shared model beats A18/A36 aggregate | Shared Gaussian foundation is strong and should be preserved. |
| A39 trainable shared `gaussian_control_canvas` improves aggregate but hurts high/A08 | Gaussian control capacity is useful, but shared control is too blunt. |
| A40 high sampling improves aggregate/low but not high/A08 | Exposure is not the missing mechanism. |
| A46 hard-case gate improves high/A08 but collapses low/A15 | Gate-only correction can move the boundary but causes zero-sum trade-off. |
| A47 global gate protects low/A15 but gives back high/A08 | More gate context without internal Gaussian-conditioned capacity is still insufficient. |

Therefore the bottleneck is:

```text
shared/late route mechanisms cannot produce two route-specific interpretations of the same Gaussian structural prior.
```

The remedy is not to weaken Gaussian. The remedy is to make the non-Gaussian mechanisms serve route-specific Gaussian control.

## Mechanism Audit Under the Gaussian-Immutable Constraint

| Mechanism | Decision | Reason |
| --- | --- | --- |
| Gaussian parameter head and texture-aware renderer | Keep unchanged as core | This is the central paper idea and the explicit 2D Gaussian prior. |
| Gaussian shared prior encoder | Keep and promote as route-control source | It converts attention/local contrast/directional cues into reusable control features. |
| Gaussian decoder guidance and feature refinement | Keep as primary injection points | These are where route-specific Gaussian conditioning should enter before RGB output. |
| Gaussian control canvas | Rebuild, not discard | A39 shows shared canvas has useful capacity but wrong selectivity. Next version should be route-specific: high-preserve canvas and low-dehaze canvas from the same Gaussian prior. |
| Gaussian structured control router | Reuse idea, change role semantics | Current roles are decoder/feature/edge. New roles should include route policy roles or route-conditioned role gates. |
| `input_std` route prior | Keep only as prior/diagnostic | It is useful but insufficient. Some nominal-low samples are A08-oracle winners. |
| Learned route boundary | Allowed only if Gaussian-aware | Router input must include Gaussian statistics, canvas statistics, shared prior statistics, and image/content statistics; not `input_std` alone. |
| Multi-depth adapters | Allowed only as Gaussian-conditioned route adapters | Generic adapters risk becoming a non-Gaussian patch. Adapter modulation must consume Gaussian prior features. |
| Slim dual-expert decoder | Allowed only as compact route policy capacity | It must share the Gaussian foundation and receive Gaussian-conditioned control; it must not become two independent non-Gaussian restorers. |
| Hard-case cache/sampler | Tool only | Useful for exposure and validation, but A46 proves exposure/gate mobility alone is not a breakthrough. |
| Oracle gate loss/trust anchors | Tool only | Useful as guardrails, not main architecture. |
| Post/output adapters and terminal heads | Stop as mainline | A35/A38/A42/A43 show late output edits are too weak or misaligned. |
| Gate-only route correction | Stop as mainline | A46/A47 show clear zero-sum behavior. |
| Bigger shared backbone | Reject as main explanation | A39/A40 suggest shared capacity tends to improve aggregate/low while worsening high/A08. |
| Non-Gaussian Transformer/diffusion/CNN replacement | Reject | Violates the project core and does not answer the current evidence. |

## Revised Architecture

Recommended direction:

```text
G-RSCN: Gaussian-Preserved Route-Specific Control Network
```

Dataflow:

1. Shared cross-resolution Gaussian foundation
   - Keep A37-style high-resolution stem, low-resolution branch, fusion, Gaussian parameter head, Gaussian renderer, and low-rank affine modulation.

2. Gaussian prior feature bank
   - Build a bank from:
     - Gaussian attention map;
     - local contrast of attention;
     - directional prior;
     - shared prior feature;
     - control canvas and feedback statistics;
     - decoder feature energy/detail cues.

3. Gaussian-aware boundary router
   - Initialize from A18 `input_std` prior.
   - Add zero-initialized residual correction from Gaussian prior statistics and image/content statistics.
   - Supervise using A08-vs-A15 oracle records, especially nominal-low/A08-oracle hard samples.

4. Route-specific Gaussian control
   - Produce two small control paths from the same Gaussian prior:
     - high-preserve control for A08-like fidelity/detail retention;
     - low-dehaze control for A15-like restoration.
   - Blend by the Gaussian-aware route gate.
   - This is the direct successor to A39, but route-specific rather than shared.

5. Gaussian-conditioned route adapters
   - Add small feature-space adapters at multiple decoder/refinement depths.
   - Each adapter receives the decoder feature plus Gaussian prior/control feature.
   - Zero-init the route-specific residual so the initial model equals A37.

6. Shared RGB head
   - Keep final image formation compact.
   - Route-specific capacity should act before RGB residual generation, not as a late patch only.

## What Changes And What Does Not

Does not change:

- 2D Gaussian parameterization.
- Texture-aware Gaussian rendering.
- Gaussian attention/control as the restoration structure prior.
- Cross-resolution foundation.
- A08/A15 conditional-restoration evidence chain.

Changes:

- Shared Gaussian control becomes route-specific Gaussian control.
- `input_std` route boundary becomes a Gaussian-aware boundary correction.
- Generic adapters become Gaussian-conditioned route adapters.
- Late gate/post-head patching stops being the mainline.

## First Experiment Rule

Before any training:

1. Write an experiment card for `G/B_4kd01`.
2. Implement default-off modules.
3. Verify A37 equivalence with all new residuals zero-initialized.
4. Verify new trainable parameter groups are only the intended Gaussian-conditioned route-control/adapters.
5. Define promotion criteria on:
   - high-route gap vs A37;
   - A08-oracle gap vs A37;
   - low-route and A15-oracle non-collapse;
   - nominal-low/A08-oracle hard samples;
   - aggregate PSNR only as a secondary check.

No training should be launched from this analysis alone.

## Paper-Level Interpretation

The paper story should be:

```text
2D Gaussian structure is the model's central restoration prior.
Conditional restoration contains conflicting route policies.
Full route-specific capacity works but is expensive.
Compact sharing without route-specific Gaussian control causes Pareto trade-offs.
Gaussian-preserved route-specific control lets one compact model interpret the same Gaussian prior differently for high-preserve and low-dehaze regimes.
```

This preserves the core idea while making a real architectural decision about the non-Gaussian mechanisms.

## 2026-06-17 Follow-Up: Lessons From The Old Architecture

Companion document: `/home/ldj/wxc/2DGauss/logs/architecture_lessons_for_g_rscn_20260617.md`.

The stricter lesson is:

```text
Gaussian core unchanged,
but route-specific policy capacity must move into Gaussian-conditioned pre-RGB control/features.
```

The old architecture's problem is not that it lacks modules. It has many modules, but the route-specific ones mostly arrive after the shared decoder representation has already collapsed the policy distinction. Therefore future G-RSCN work must obey:

- route-specific Gaussian control before RGB residual;
- Gaussian-aware boundary router as selector only, not sole capacity;
- route-specific high-preserve and low-dehaze control paths;
- Gaussian-conditioned adapters, not generic adapters;
- A37 zero-init equivalence before any training.
