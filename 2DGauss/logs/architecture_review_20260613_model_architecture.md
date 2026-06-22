# Architecture Review: Current V1/T01 Model After A24-A26

- Date: `2026-06-13 CST`
- Scope: Review the current `CrossResolutionGaussianRestorer` architecture, the A24 route-residual breakthrough, and why the remaining A18 router gap is still not closed.
- Primary code paths:
  - `/home/ldj/wxc/2DGauss/src/uhd_gaussian/models/restorer.py`
  - `/home/ldj/wxc/2DGauss/src/uhd_gaussian/config.py`
  - `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/configs/a_4kd24_fast_4kdehaze_a15_input_std_dual_route_residual.toml`
  - `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/result_analysis.md`
  - `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd26/result_analysis.md`

## Executive Conclusion

The current model is not a pure 2D Gaussian renderer. It is a compact CNN
restoration backbone whose low-resolution branch predicts Gaussian-like
attention/control priors. Those priors modulate the high-resolution CNN feature
path, and the final image is still produced as an RGB residual added to the
input image.

A24 was a real architectural move because it stopped treating the A08/A15
router as only a loss target and gave the model a structural high/low
`input_std` branch. That is why it finally exceeded A08/A15/A23 as a single
checkpoint. But A24 inserts the branch very late, after almost all shared
feature computation has already happened. Its route head is a small terminal
correction, not a full internal expert. Therefore it can polish the A15-like
backbone, but it cannot fully reproduce the two-checkpoint A18 behavior where
high-`input_std` samples get an entire A08 model path.

The next architectural step should be route-specific terminal decoder heads or
a larger route-conditioned expert path, not more sampler balancing. A25/A26
already showed that forcing 50/50 high/low route exposure hurts the natural
training trajectory.

## Current Model Flow

The forward path in `CrossResolutionGaussianRestorer` can be summarized as:

1. High-resolution stem:
   - Input RGB is padded and processed by `highres_stem`.
   - This keeps full-resolution spatial detail in a CNN feature tensor.

2. Low-resolution branch:
   - The input is pixel-unshuffled by `lowres_scale=4`.
   - `lowres_proj`, `TextureFrequencyExtractor`, and `lowres_fusion` produce a compact low-resolution feature.

3. Gaussian prior generation:
   - `GaussianParameterHead` predicts per-cell Gaussian parameters and affine coefficients.
   - `TextureAwareGaussianRenderer` renders those parameters into a high-resolution single-channel `attention_map`.
   - This attention map is the key "Gaussian" component, but it acts as a structural prior rather than directly rendering RGB.

4. High-resolution modulation:
   - The attention map modulates the high-pass part of the high-resolution feature.
   - `LowRankAffineModulator` further modulates the feature using low-resolution affine coefficients.

5. Decoder feature path:
   - The modulated high-resolution feature is concatenated with upsampled low-resolution features.
   - The decoder produces `decoder_feature`, then a final RGB residual.

6. Optional structural modules:
   - `gaussian_control_canvas` can refine decoder features and share a control canvas back into the Gaussian attention path.
   - `gaussian_decoder_guidance` and `gaussian_feature_refinement` use Gaussian priors to modulate features.
   - `decoupled_correction`, `gaussian_terminal_residual`, and `edge_refiner` can add terminal corrections.

7. A24 route residual:
   - `InputStdRouteResidualHead` receives `decoder_feature`, the input image, and an `input_std` prior.
   - It predicts `high_delta` and `low_delta`, blends them by an input-std gate, and adds the result to the base residual.
   - The final image is still `restored = input + residual`.

## What A24 Actually Changed

A24 added a default-off `InputStdRouteResidualHead` with:

- high route: intended to imitate A08 on high-`input_std` samples;
- low route: intended to imitate A15 on low-`input_std` samples;
- fixed prior threshold: `0.07026463374495506`;
- soft gate: `softness=0.005`;
- route strength: `0.20`;
- learned gate contribution: `0.20`;
- branch teacher weight: `0.08`;
- gate supervision weight: `0.04`;
- initialization from A15 best-val with non-strict load.

The important design win is that branch teacher supervision targets the branch
outputs, not only the final blended output. This avoids the main failure mode
of A23, where whole-output router distillation mostly averaged back into an
A15-like model.

The important limitation is placement and capacity. The A24 route head sits
after shared Gaussian attention, shared decoder guidance, shared feature
refinement, shared control canvas feedback, and the shared terminal RGB
projection. It does not choose between two internal feature-processing paths.
It only applies a final correction on top of a mostly A15-like backbone.

## Why A24 Is Positive But Still Partial

A24 result:

- best-val step400 `val_psnr=18.349088096641417`;
- fast-test best-val `psnr_mean=17.790202882742463`;
- A08 single `17.789133503142946`;
- A15 single `17.781017099305146`;
- A23 single `17.78064128762362`;
- A18 two-checkpoint router `17.84693437641316`.

That means A24 is the first single-checkpoint structural conversion that truly
crosses the old single-checkpoint ceiling. It is not noise-level in the wrong
direction; it is a real positive sign.

But the remaining A18 gap is about `0.05673 dB`. The reason is architectural:

1. The high route does not own the full A08 computation.
   A18 selects an entire A08 checkpoint for high-`input_std` samples. A24 gives
   those samples only a small high-route terminal delta on an A15-derived model.

2. The shared decoder is still the bottleneck.
   Both routes consume the same `decoder_feature`. If that feature has already
   lost an A08-specific preservation behavior, the terminal route head must
   reconstruct it from a compressed shared state.

3. The Gaussian/control priors are shared before routing.
   The current route does not condition the Gaussian attention map, control
   canvas, decoder guidance, or feature refinement. If A08/A15 differ in how
   they use those priors, A24 cannot express that difference strongly.

4. The route head is deliberately conservative.
   Zero-init makes loading safe and avoids an initial regression. But with
   `blocks=1`, `width=32`, `strength=0.20`, and branch loss `0.08`, the learned
   deltas remained small. At step400, A24 logged
   `train_route_high_delta_abs_mean=0.0016153835458680987` and
   `train_route_low_delta_abs_mean=0.0034981542266905308`.

5. The gate is mostly a per-sample scalar prior.
   The prior is expanded spatially from image-level `input_std`. The learned
   gate contributes only 20%. This is stable and correctly constrained, but it
   does not create a strong spatially adaptive expert allocation.

6. Starting from A15 biases the shared path.
   This was a sensible conservative choice because A15 had better loss/proxy
   behavior. But it makes the high route responsible for recovering A08-like
   behavior from a late, small branch.

## What A25/A26 Taught Architecturally

A25 and A26 tested whether the missing piece was route exposure.

- A25 used file-level high/low balancing, but random crops did not reliably
  preserve the route label.
- A26 made sampling crop-aware and technically achieved about 50/50 high/low
  route batches.

The result was negative: A26's best early validation stayed around
`18.236443073291603`, far below A24's `18.349088096641417`.

This matters because it separates two hypotheses:

- wrong: the main bottleneck is simply too few high-route samples;
- stronger: the main bottleneck is insufficient route capacity and late route
  placement under the natural distribution.

Forced balancing changes the training distribution too much. The next attempt
should preserve natural sample frequency and give the route more representational
power.

## Architecture-Level Recommendations

### 1. Route-Specific Terminal Decoder Heads

This is the cleanest next step.

Current terminal path:

```text
decoder_feature -> shared RGB residual -> small route delta -> restored image
```

Recommended A27-style path:

```text
decoder_feature -> shared trunk
                -> high terminal head -> high_restored
                -> low terminal head  -> low_restored
input_std gate blends high/low restored or high/low residual
```

Why this is better:

- high/low routes get independent RGB projection capacity;
- the route can change the main residual, not only add a tiny correction;
- zero-init can still be used safely by initializing heads from the existing
  shared terminal projection plus small learnable offsets;
- branch supervision can directly train the two heads against A08/A15 teachers.

### 2. Route-Conditioned Feature Expert Before RGB Projection

If terminal heads are not enough, add route-specific feature experts before
`decoder[2]`:

```text
decoder_feature -> high_feature_expert -> shared/high RGB head
decoder_feature -> low_feature_expert  -> shared/low RGB head
```

This gives the model a place to recover A08-like high-`input_std` feature
behavior before the image residual is formed.

### 3. Keep Input-Std As A Strong Prior, Not A Free Learned Router

A22 showed richer learned routers overfit cross-block. The gate should remain
anchored to the fixed A18 input-std threshold. A learned component can exist,
but it should be constrained, small, and audited against A18/A19/A21 records.

### 4. Do Not Continue 50/50 Sampler Variants As Mainline

A26 is enough evidence that hard 50/50 crop balancing is not the next mainline.
If route imbalance needs correction, use loss reweighting or branch-specific
teacher weights under the natural loader before changing the sampler again.

### 5. Evaluate Per Regime, Not Only Aggregate

Every future route experiment should report:

- low-`input_std` PSNR/SSIM/loss;
- high-`input_std` PSNR/SSIM/loss;
- branch high output versus A08 teacher;
- branch low output versus A15 teacher;
- final blended output versus A08, A15, and A18 router;
- cross-block checks against A18/A19/A21 saved records when possible.

## Concrete Next Run Proposal

The next serious architecture run should be `A_4kd27`:

- base: A24 config and A15 initialization;
- replace small `input_std_route_residual` with route-specific terminal decoder
  heads;
- keep natural sampling;
- keep fixed input-std prior and hard branch teacher targets;
- increase route branch capacity before increasing route loss weight;
- keep old behavior default-off in config;
- preserve A24 as fallback candidate.

Promotion bar:

1. beat A24 single-checkpoint PSNR `17.790202882742463`;
2. close at least a meaningful part of the A18 router gap;
3. not lose A24's SSIM/loss improvement;
4. show high-route branch actually moves toward A08 on high-`input_std` samples.

## Bottom Line

The architecture is close enough to the right mechanism that A24 produced a
real positive result. The remaining issue is not "more tuning" and not "more
balanced sampling." The model still lacks a sufficiently strong internal place
to store two route-specialized behaviors. Put the route earlier and give it
real terminal/expert capacity.
