# Model Architecture and Breakthrough Experiment Report

- Time: `2026-06-16 15:10:54 CST`
- Project root: `/home/ldj/wxc/2DGauss`
- Scope: current V1/T01 model architecture, A08-A41 breakthrough experiments, bottleneck analysis, and next breakthrough directions.
- Primary task line: paired 4K dehazing / ultra-high-resolution restoration on `4KDehaze_local_90_5_5_fast_probe_512`.

## Executive Summary

The active project is not a pure 2D Gaussian renderer. The current model is a compact CNN image-restoration backbone with a low-resolution Gaussian attention/control prior. The final RGB image is still produced as:

```text
restored = input + RGB residual
```

The Gaussian part predicts structural attention and control maps that modulate the CNN feature path. It is a controller, not the final image renderer.

The main scientific signal is now clear: A08 and A15 are complementary checkpoints. A15 is better on many low-`input_std`/hard haze samples, while A08 better preserves a high-fidelity A08-tail. A18/A19/A21 proved that a simple fixed `input_std` router can exploit this complementarity. A36 proved that if one checkpoint carries two full backbones internally, the route behavior can be reproduced almost exactly.

The current bottleneck is not route discovery. It is compression: how to put A08/A15 conditional specialization into one compact/shared model without losing the A08-tail.

Current best roles:

- `A_4kd37`: most important compressed/shared milestone; it beats the A18/A36 aggregate router line and remains the better high-route anchor among compressed models.
- `A_4kd40`: current aggregate numeric best, but it does not solve the high-route/A08-oracle bottleneck.
- `A_4kd36`: upper-bound proof; full dual-backbone capacity inside one checkpoint can preserve both routes.
- `A_4kd41`: useful negative result; crop-level oracle weighting improves teacher L1 slightly but does not improve high-route PSNR.

The next breakthrough should be judged high-tail-first: high-route gap, A08-oracle gap, and worst-case recovery must improve before aggregate PSNR is considered a real bottleneck breakthrough.

## 1. Current Model Architecture

### 1.1 Code and Configuration Entry Points

Primary code paths:

- `/home/ldj/wxc/2DGauss/src/uhd_gaussian/models/restorer.py`
- `/home/ldj/wxc/2DGauss/src/uhd_gaussian/models/gaussian.py`
- `/home/ldj/wxc/2DGauss/src/uhd_gaussian/config.py`
- Current representative configs:
  - `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/configs/a_4kd37_fast_4kdehaze_a33_high_route_a36_teacher_compress.toml`
  - `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd40/configs/a_4kd40_fast_4kdehaze_a37_high_route_balanced_teacher.toml`
  - `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd41/configs/a_4kd41_fast_4kdehaze_a37_high_route_oracle_weighted.toml`

Main module:

```text
CrossResolutionGaussianRestorer
```

Key boundary: this is a restoration network with Gaussian-guided feature control. It does not render the RGB output from Gaussian splats. The Gaussian renderer outputs a single-channel attention/control prior.

### 1.2 High-Level Forward Path

The model flow is:

```text
input RGB
  -> pad_to_multiple(lowres_scale)
  -> highres_stem full-resolution CNN feature
  -> optional explicit frequency split
  -> pixel_unshuffle low-resolution branch
  -> lowres_proj + texture_frequency + lowres_fusion
  -> GaussianParameterHead
  -> TextureAwareGaussianRenderer -> attention_map
  -> attention_map modulates high-pass highres feature
  -> LowRankAffineModulator
  -> concat(modulated highres, upsampled lowres)
  -> decoder feature
  -> optional Gaussian control/guidance/refinement modules
  -> optional input_std route decoder heads
  -> RGB residual
  -> restored = input + residual
```

### 1.3 Base CNN Restoration Backbone

The base architecture is cross-resolution:

- Full-resolution branch:
  - `highres_stem = Conv2d(in_channels, width, 3x3) + StackedResidualBlocks`.
  - Purpose: preserve spatial detail and high-frequency structure at original resolution.

- Low-resolution branch:
  - `pixel_unshuffle` with `lowres_scale=4`.
  - `lowres_proj = Conv2d(in_channels * 16, width, 1x1) + StackedResidualBlocks`.
  - `TextureFrequencyExtractor` reads downsampled RGB and low-res feature.
  - `lowres_fusion` merges low-res semantic/texture information.

- Fusion and decoder:
  - Gaussian-modulated high-res feature is concatenated with upsampled low-res feature.
  - Decoder:

```text
Conv2d(width * 2 -> width, 1x1)
StackedResidualBlocks(width, decoder_blocks)
Conv2d(width -> out_channels, 3x3)
```

### 1.4 Gaussian Prior Module

`GaussianParameterHead` predicts:

- `params`: per low-res cell, shape equivalent to `atoms * param_dim`.
- `coeffs`: low-rank affine coefficients for `LowRankAffineModulator`.

Current representative Gaussian settings:

- `gaussian_atoms = 4`
- `gaussian_param_dim = 10`
- `max_frequency = 4.0`
- `max_texture_gain = 1.0`
- `affine_rank = 4`

`TextureAwareGaussianRenderer` interprets the 10 parameters as amplitude, mean, sigma, rotation, texture gain, frequency, phase, and beta-like spectral damping. It renders a per-cell patch and upsamples/reshapes it to a high-resolution single-channel `attention_map`.

This `attention_map` is used as:

```text
highpass = highres_feature - avg_pool(highres_feature)
modulated = highres_feature + attention_map * highpass
modulated = LowRankAffineModulator(modulated, coeffs)
```

Interpretation: Gaussian priors decide where and how strongly to preserve/refine structure; the CNN still produces RGB restoration.

### 1.5 Optional Structural Modules

The model contains multiple default-off or config-enabled modules:

- `gaussian_decoder_guidance`: Gaussian-guided decoder feature modulation.
- `gaussian_feature_refinement`: Gaussian-guided feature refinement.
- `gaussian_control_canvas`: produces a control canvas and can share it back into guidance/attention.
- `gaussian_attention_demand_bridge`: bridges restoration demand into attention.
- `gaussian_shared_prior` and `gaussian_control_router`: shared prior and structured role routing.
- `gaussian_terminal_residual`: terminal Gaussian-guided RGB residual.
- `decoupled_correction`: correction map with selectivity/amplitude.
- `edge_refiner`: high-resolution edge-guided residual.
- `directional_frequency_guidance`: optional directional priors.
- `explicit_frequency_split`: optional low/high band split and feedback.

Current A37/A40/A41-style active structural set:

- `use_gaussian_decoder_guidance = true`
- `gaussian_decoder_guidance_strength = 0.08`
- `use_gaussian_feature_refinement = true`
- `gaussian_feature_refinement_strength = 0.08`
- `use_gaussian_control_canvas = true`
- `gaussian_control_canvas_strength = 0.06`
- `gaussian_control_canvas_stage = "pre_guidance"`
- `gaussian_control_canvas_share_to_guidance = true`
- `gaussian_control_canvas_share_to_guidance_blend = 0.25`
- `gaussian_control_canvas_target_mode = "restoration_demand"`
- `gaussian_control_canvas_feedback_mode = "additive"`
- `gaussian_control_canvas_feedback_budget = 0.20`

### 1.6 Input-Std Route Modules

The route family was added because A18/A19/A21 showed that `input_std` separates two restoration regimes.

Shared route settings:

- threshold: `0.07026463374495506`
- direction for current route modules: `gt`
- softness in A37/A40/A41: `0.003`
- meaning:

```text
input_std > threshold  -> high route, A08-like target
input_std <= threshold -> low route, A15-like target
```

Route module history:

- `InputStdRouteResidualHead`:
  - A24.
  - Late residual delta on top of shared base residual.
  - First positive single-checkpoint structural conversion, but capacity was too late/small.

- `InputStdRouteTerminalHeads`:
  - A27.
  - Independent high/low terminal RGB heads.
  - Did not solve high-route; terminal placement still insufficient.

- `InputStdRouteFeatureExpert`:
  - A28.
  - Route-conditioned decoder feature delta.
  - Did not solve high-route.

- `InputStdRouteDecoderHeads`:
  - A29 onward; important for A31/A33/A37-A41.
  - Takes `decoder_feature`, input image, and input-std prior.
  - Builds a route shared feature:

```text
shared = Conv1x1([decoder_feature, image, prior_map]) + residual blocks
high_feature = decoder_feature + strength * high_feature_expert(shared)
low_feature  = decoder_feature + strength * low_feature_expert(shared)
high_residual = high_head(high_feature)
low_residual  = low_head(low_feature)
residual = gate * high_residual + (1 - gate) * low_residual
```

  - `high_head` and `low_head` are initialized from the base decoder head.
  - A37/A40/A41 lock the low route and train high/shared late modules.

- `InputStdRoutePostAdapter`:
  - A35.
  - Post-route high adapter.
  - Marginal result; too late to matter.

### 1.7 Current Representative Model Size and Trainable Policy

A37/A40/A41 representative backbone settings:

- `width = 32`
- `lowres_scale = 4`
- `highres_blocks = 2`
- `lowres_blocks = 4`
- `decoder_blocks = 2`
- `affine_rank = 4`
- `gaussian_atoms = 4`
- `gaussian_param_dim = 10`

A37/A40/A41 trainable prefixes:

```text
input_std_route_decoder_heads.shared
input_std_route_decoder_heads.high_feature
input_std_route_decoder_heads.high_head
gaussian_feature_refinement
gaussian_decoder_guidance
decoder.1
```

A41 preflight confirmed trainable parameters:

```text
76643 / 168406
```

Interpretation: current compressed experiments are not training a full second backbone. They are attempting to recover high-route behavior through a limited set of shared/high late modules.

## 2. Data and Evaluation Context

Current active fast-probe view:

- Dataset: `4KDehaze_local_90_5_5_fast_probe_512`
- Train view: `512`
- Validation view: `100`
- Test view: `100`
- Crop size: `512`

Broader local 4KDehaze split:

- train: `14045`
- validation: `780`
- test: `781`

Important caveat: the fast-probe protocol is small and scene-concentrated, so it is suitable for fast mechanism discovery, not final generalization claims. However, A18/A19/A21/A22 used multiple blocks/offsets to verify that the router signal is not a one-block accident.

Primary metrics used in this phase:

- Standard test PSNR/SSIM/loss on 100-sample fast test.
- A18 record alignment:
  - overall gap vs fixed A18 router.
  - high-route gap.
  - low-route gap.
  - A08-oracle group gap.
  - A15-oracle group gap.
- Route-regime diagnostics:
  - selected delta magnitude.
  - high selected delta.
  - low selected delta.
  - high branch L1 vs A08.
  - low branch L1 vs A15.

## 3. Key Route Evidence Before Single-Checkpoint Compression

### 3.1 A08 and A15 Are Complementary

Base checkpoints:

- A08 fast-test PSNR: `17.789133503142946`
- A15 fast-test PSNR: `17.781017099305146`

A15 wins many samples but damages the A08 high-fidelity tail. This is why a global single checkpoint or simple averaging does not solve the problem.

### 3.2 A18 Fixed Router

A18 fixed route:

```text
input_std <= 0.07026463374495506 -> A15
input_std  > 0.07026463374495506 -> A08
```

Results:

- selected router PSNR: `17.84693437581442`
- choose A15 count: `57 / 100`
- binary oracle PSNR: `17.883555795086536`

A18 proved that the signal is real on the fast-test block.

### 3.3 A19 and A21 Cross-Block Checks

A19 offset100:

- fixed A18 rule PSNR: `15.913737915082573`
- A08: `15.87689249278197`
- A15: `15.885454203061169`
- binary oracle: `15.928149885150932`

A21 offset200:

- fixed A18 rule PSNR: `14.64236451612018`
- A08: `14.61942131606054`
- A15: `14.570231632564433`
- binary oracle: `14.662330739170043`

This shows the fixed `input_std` rule generalizes better than a one-block coincidence.

### 3.4 A22 Router Analysis

A22 showed richer learned routers overfit:

- fixed input-std best in leave-one-block-style analysis: `15.761350330784339`
- oracle agreement: `0.77`
- choose A15: `204 / 400`
- richer logistic/all-common variants were lower.

Conclusion: the route should stay anchored to the stable fixed `input_std` prior. The problem is not to discover a more complex free router; it is to make one compact model preserve both behaviors.

## 4. Experiment Results

### 4.1 Standard Fast-Test PSNR Table

For A37/A39/A40/A41, some standard eval files have `loss=0.0` because those configurations intentionally disabled ordinary reconstruction/proxy loss for teacher-compression diagnostics. A37 was separately re-evaluated with standard loss and got `loss_mean=0.11911950781941413`, `total_loss_mean=0.12235063299536705`.

| Run | Main mechanism | Test PSNR | Key result |
|---|---|---:|---|
| A08 | earlier high-fidelity/detail-protected baseline | 17.789133503142946 | strong A08-tail behavior, unstable globally |
| A15 | A08 finetune / better loss-proxy baseline | 17.781017099305146 | better on many low-std samples, damages A08 tail |
| A18 | two-checkpoint fixed input-std router | 17.84693437581442 | route signal proof |
| A23 | whole-output router distillation | 17.78064128762362 | failed; collapses to A15-like behavior |
| A24 | late input-std route residual | 17.790202882742463 | first positive single-checkpoint structural conversion |
| A27 | route terminal heads | 17.7513865188403 | failed; high route still weak |
| A28 | feature expert | 17.73849582669033 | failed; high route still weak |
| A29 | hard decoder heads | 17.76127769536133 | capacity opens but drifts harmfully |
| A30 | trust-region decoder heads | 17.75180602939797 | suppresses harmful drift, but no useful high route |
| A31 | transplanted route heads | 17.770243709681385 | high-route improves, low-route regresses |
| A32 | high A08 transplant + low lock | 17.786780968730323 | low safe, high insufficient |
| A33 | checkpoint surgery: A31 high + A24/A32 low | 17.823062311698262 | clears 17.82; best pre-A36 compact candidate |
| A34 | formal A33 refinement | 17.820851508384713 | does not improve A33 |
| A35 | post-route adapter | 17.823132133493647 | marginal only |
| A36 | single checkpoint, full dual backbone | 17.8470899392002 | upper-bound proof; route capacity sufficient |
| A37 | A33/A36 teacher compression with native teachers | 17.889658453438816 | compressed/shared aggregate milestone |
| A38 | standard-loss high-head-only micro tune | 17.889819259555214 | tiny gain; not meaningful high-route breakthrough |
| A39 | A37 + trainable control canvas | 17.900369585886775 | aggregate improves, high route worsens |
| A40 | A37 + high-balanced sampling | 17.900562951956445 | current aggregate numeric best, high route not solved |
| A41 | A40 + crop-level oracle weighting | 17.900201657598004 | negative high-route probe |

### 4.2 A18 Alignment Table

The most important table is not aggregate PSNR. It is alignment against the A18 fixed router and its oracle groups.

| Run | Overall gap | High-route gap | Low-route gap | A08-oracle gap | A15-oracle gap |
|---|---:|---:|---:|---:|---:|
| A27 | -0.09554785697412059 | -0.3054244593410088 | +0.06278010621493545 | -0.45424864545861654 | +0.2100120739571167 |
| A28 | -0.1084385491240903 | -0.3390800359165398 | +0.06555415143863476 | -0.5188876236246397 | +0.24120325508008136 |
| A29 | -0.08565668045309131 | -0.236088142902285 | +0.027826703499809195 | -0.27720066420366934 | +0.07751041681591958 |
| A30 | -0.0951283464164498 | -0.23814779446593295 | +0.012763517901581372 | -0.27065612700046393 | +0.05439531852548818 |
| A31 | -0.07669066613303403 | -0.14121316850824578 | -0.02801579592015501 | -0.14761475713935954 | -0.016273847868386378 |
| A32 | -0.060153407084098394 | -0.211220889864822 | +0.05380978168030712 | -0.1930536941607739 | +0.053057948573810354 |
| A33 | -0.023872064116158373 | -0.12677254958688497 | +0.053754617905617835 | -0.08633674234915452 | +0.029338587711949447 |
| A34 | -0.026082867429706925 | -0.1309565862555424 | +0.053032394140660195 | -0.08089136622024616 | +0.02060585376223391 |
| A35 | -0.02380224232077156 | -0.12662823419234123 | +0.05376824277532487 | -0.08644686819587206 | +0.029561698239499248 |
| A37 | +0.042724077624394126 | -0.11159004428502844 | +0.1591364853806252 | -0.19984098726138502 | +0.2493535773419097 |
| A38 | +0.04288488374079478 | -0.1111094511726412 | +0.15905604867549208 | -0.19848443524653703 | +0.2484957851003737 |
| A39 | +0.053435210072354966 | -0.11858325626995542 | +0.18320352678672946 | -0.23903555195244477 | +0.3025769703157029 |
| A40 | +0.0536285761420247 | -0.1180115677202543 | +0.18311149168725271 | -0.23833660113789967 | +0.3023396530841825 |
| A41 | +0.05326728178358334 | -0.11813195131188482 | +0.182568457627533 | -0.23811165869770295 | +0.3014789718231976 |

Reading:

- A37 is the compressed/shared aggregate milestone.
- A40 is the aggregate numeric best.
- A37/A38 remain better than A39/A40/A41 on the high-route gap.
- A39/A40/A41 gain aggregate PSNR mainly through low/A15-side improvement, while A08-oracle gap worsens.

### 4.3 Route-Regime Diagnostics

| Run | Selected delta | High selected delta | Low selected delta | High branch L1 vs A08 | Low branch L1 vs A15 |
|---|---:|---:|---:|---:|---:|
| A27 | 0.0009762731264345348 | 0.0007441960665029149 | 0.001151348803225055 | 0.01429054965156802 | 0.01471208993410855 |
| A28 | 0.0011144706892082467 | 0.0008386419774682889 | 0.0013225519980647061 | 0.015561815286271794 | 0.01525487631494016 |
| A29 | 0.00025362832675455136 | 0.0002827254742611373 | 0.0002316778470566006 | 0.262734996371491 | 0.34177368774748684 |
| A30 | 0.0000578112014773069 | 0.0000618680088908065 | 0.0000547508029022107 | 0.29880278470904326 | 0.3636353517833509 |
| A31 | 0.012584832096472383 | 0.019272921736850294 | 0.007539431139696063 | 0.12602153940256253 | 0.39231843226834345 |
| A32 | 0.006058236313983798 | 0.014088921660427437 | 0.0 | 0.23157927844413492 | 0.4520495779681624 |
| A33 | 0.008283325210213662 | 0.019263547000496886 | 0.0 | - | - |
| A34 | 0.008751856992021204 | 0.020353155795398148 | 0.0 | 0.32904680800992386 | 0.2585565087042357 |
| A35 | 0.008263196442276239 | 0.019216735912270324 | 0.0 | 0.3066226146941961 | 0.3575197092273779 |
| A37 | 0.0062688171397894624 | 0.014578644511138284 | 0.0 | 0.014976846487369648 | 0.012233833717018888 |
| A39 | 0.005716269994154572 | 0.01329365114919668 | 0.0 | 0.013812246537485789 | 0.01426901130757311 |
| A40 | 0.0056926918867975474 | 0.013238818341389645 | 0.0 | 0.013522875278668349 | 0.014295937117646661 |
| A41 | 0.005682019237428904 | 0.013213998226578845 | 0.0 | 0.01348984414754912 | 0.014252835606927411 |

Important reading: A39/A40/A41 improve high branch L1 vs A08 relative to A37, but high-route PSNR gap becomes worse. Therefore branch L1 to A08 is not sufficient as the promotion metric.

## 5. Phase-by-Phase Result Analysis

### 5.1 A23: Whole-Output Distillation Failed

A23 tried to distill the A18 two-checkpoint router into one A15-initialized student through final-output distillation.

Result:

- best-val test PSNR: `17.78064128762362`
- final checkpoint PSNR: `17.035365402315897`

Conclusion: whole-output distillation averages or biases the model back to A15-like behavior. It does not preserve two conditional optima.

### 5.2 A24: First Positive Structural Route

A24 added `InputStdRouteResidualHead`.

Result:

- PSNR: `17.790202882742463`
- SSIM: `0.7000000587105751`
- loss: `0.12000417917966842`
- total loss: `0.1232352689653635`

Conclusion: A24 was the first positive single-checkpoint structural conversion. It beat A08/A15/A23, but it remained far below A18 because the route was only a late residual correction after shared Gaussian/control/decoder features.

### 5.3 A25/A26: Balanced Sampling Was Not the Missing Piece

A25 file-level balancing failed because file-level `input_std` did not ensure crop-level route labels.

A26 crop-aware balancing technically achieved 50/50 high/low route batches, but validation stayed much lower than A24:

- A26 early val around `18.236443073291603`
- A24 val `18.349088096641417`

Conclusion: the bottleneck is not simply too few high-route samples. Forced route balancing can damage the natural training distribution.

### 5.4 A27/A28: Terminal Heads and Feature Experts Failed

A27 terminal heads:

- PSNR: `17.7513865188403`
- high-route gap: `-0.3054244593410088`
- low-route gap: `+0.06278010621493545`

A28 feature expert:

- PSNR: `17.73849582669033`
- high-route gap: `-0.3390800359165398`
- low-route gap: `+0.06555415143863476`

Conclusion: simple added route capacity was not enough. The high branch did not recover A08-like internal behavior.

### 5.5 A29/A30: Hard Capacity and Trust Region Did Not Solve It

A29 opened hard decoder-head capacity:

- PSNR: `17.76127769536133`
- high-route gap: `-0.236088142902285`

A29 showed route deltas can grow, but larger deltas hurt validation.

A30 added trust-region/staged branch pressure:

- PSNR: `17.75180602939797`
- high-route gap: `-0.23814779446593295`
- selected delta nearly zero: `0.0000578112014773069`

Conclusion: trust region suppresses harmful drift but does not create useful high-route behavior.

### 5.6 A31/A32/A33: Transplant and Surgery Created a Compact Candidate

A31 transplanted route heads:

- PSNR: `17.770243709681385`
- high-route gap improved to `-0.14121316850824578`
- low-route gap regressed to `-0.02801579592015501`

A32 protected low route:

- PSNR: `17.786780968730323`
- low-route gap: `+0.05380978168030712`
- high-route gap: `-0.211220889864822`

A33 surgery combined A31 high/shared modules with A24/A32 low protection:

- PSNR: `17.823062311698262`
- high-route gap: `-0.12677254958688497`
- low-route gap: `+0.053754617905617835`

Conclusion: A33 proved that high-route transplant plus low-route lock is productive. It became the best pre-A36 compact candidate, but still did not reach A18.

### 5.7 A34/A35: Micro-Tuning and Post-Adapter Were Not Enough

A34 formal A33 refinement:

- PSNR: `17.820851508384713`
- high-route gap: `-0.1309565862555424`

A35 post-route adapter:

- PSNR: `17.823132133493647`
- high-route gap: `-0.12662823419234123`

Conclusion: output-space micro-corrections are too late and too small.

### 5.8 A36: Full Dual-Backbone Upper Bound

A36 packed A08 and A15 into one checkpoint with an internal fixed input-std router.

Result:

- PSNR: `17.8470899392002`
- gap vs A18 router: `+0.00015556338577912498`
- high-route gap: approximately `-0.0001995`
- low-route gap: approximately `+0.0004234`

Conclusion: route-specific capacity is sufficient. The remaining problem is not route logic; it is compressing two experts into one shared/compact architecture.

### 5.9 A37: Compressed/Shared Aggregate Breakthrough

A37 starts from A33, loads A08/A15 teachers using their native configs, locks low route, and trains high/shared late modules with high-route teacher pressure and delta trust.

Result:

- test PSNR: `17.889658453438816`
- standard-loss re-eval loss: `0.11911950781941413`
- total loss: `0.12235063299536705`
- SSIM: `0.7026848953962326`
- overall gap vs A18: `+0.042724077624394126`
- high-route gap: `-0.11159004428502844`
- low-route gap: `+0.1591364853806252`

Conclusion: A37 is a real project breakthrough. It is the first compressed/shared candidate to beat the A18/A36 aggregate router line. Boundary: it still misses the desired high-route line `>-0.10 dB`.

### 5.10 A38: Standard-Loss High-Head-Only Tuning Was Too Small

A38 reintroduced standard loss and trained high head/feature only from A37.

Result:

- PSNR: `17.889819259555214`
- high-route gap: `-0.1111094511726412`

Conclusion: tiny improvement, not a real high-route breakthrough. Training peaked early and decayed.

### 5.11 A39: Trainable Shared Control Canvas Improved Aggregate but Hurt High Route

A39 made `gaussian_control_canvas` trainable from A37.

Result:

- PSNR: `17.900369585886775`
- high-route gap: `-0.11858325626995542`
- low-route gap: `+0.18320352678672946`
- A08-oracle gap: `-0.23903555195244477`

Conclusion: shared control capacity helps aggregate/low-side but dilutes high-route fidelity.

### 5.12 A40: High-Balanced Sampling Improved Aggregate but Not High Route

A40 froze `gaussian_control_canvas` again and used high-`input_std` balanced sampling/crop targeting.

Result:

- PSNR: `17.900562951956445`
- current aggregate numeric best
- high-route gap: `-0.1180115677202543`
- low-route gap: `+0.18311149168725271`
- A08-oracle gap: `-0.23833660113789967`

Conclusion: more high-route exposure alone does not recover A08-tail fidelity.

### 5.13 A41: Oracle-Weighted High-Branch Loss Was Negative for High Route

A41 added crop-level A08-vs-A15 oracle weighting to high-branch teacher loss.

Result:

- PSNR: `17.900201657598004`
- high-route gap: `-0.11813195131188482`
- low-route gap: `+0.182568457627533`
- A08-oracle gap: `-0.23811165869770295`
- high branch L1 vs A08: `0.01348984414754912`

Compared to A40:

- A40 high branch L1 vs A08: `0.013522875278668349`
- A41 improves L1 by about `0.000033031131119229395`
- but high-route PSNR gap does not improve.

Conclusion: scalar oracle reweighting is not the missing mechanism.

## 6. Bottleneck Diagnosis

### 6.1 The Signal Is Proven

A18/A19/A21 show the fixed input-std route beats both A08 and A15 across blocks. A36 shows that a single checkpoint with full dual-backbone internal capacity can reproduce A18.

Therefore the project is no longer blocked by whether routing is real.

### 6.2 The Bottleneck Is Compression and Shared Representation

A37 proves that some compression is possible. But A39/A40/A41 show a stable failure mode:

- aggregate PSNR improves;
- low/A15 side improves;
- high-route gap worsens relative to A37;
- A08-oracle gap worsens sharply relative to A37.

This means the shared model tends to convert new capacity and training pressure into low-side/aggregate gain, while still failing to protect A08-tail fidelity.

### 6.3 Late Corrections Cannot Fully Recover A08-Like Behavior

A24, A27, A35, and A38 all point to the same architecture issue: after Gaussian attention, control canvas, decoder guidance, feature refinement, and shared decoder feature have already been computed, a late RGB delta has limited ability to reconstruct a lost A08-like representation.

The high route needs either:

- stronger route-conditioned internal capacity, or
- explicit hard-case anchors that prevent shared paths from erasing A08-tail behavior.

### 6.4 L1-to-Teacher Is Not the Right Promotion Signal

A39/A40/A41 reduce high branch L1 vs A08, but high-route PSNR gets worse than A37. This means L1-to-teacher can improve by matching average/low-frequency content while missing PSNR-sensitive hard cases or final selected-route behavior.

Promotion should use:

- high-route gap;
- A08-oracle gap;
- worst-gap list recovery;
- per-sample PSNR margin vs A37/A18/A36.

### 6.5 Fixed High-Input-Std Sampling Misses Some A08-Oracle Hard Cases

Worst-case review found nominal low-route samples where A08 is still better than A15. Examples:

- `94_9400006.jpg`: route `low_b`, input_std `0.058188`, `a15_minus_a08 = -0.302716`
- `94_94000065.jpg`: route `low_b`, input_std `0.064002`, `a15_minus_a08 = -0.310023`
- `94_94000081.jpg`: route `low_b`, input_std `0.058379`, `a15_minus_a08 = -0.276238`

This explains why A40/A41 high-`input_std` sampling did not solve A08-oracle gap. The hard A08-oracle set is not identical to threshold-high samples.

## 7. Current Bottleneck Statement

The precise bottleneck is:

```text
The model has learned a compressed/shared aggregate solution, but the shared route-conditioned representation still cannot preserve A08-oracle/high-fidelity hard cases while also keeping the A15/low-side gains.
```

In plainer terms: we can now win the overall score, but the model is still paying for that by losing part of the A08-tail. The next breakthrough must stop that tradeoff.

## 8. Recommended Next Experiments

### A42: A08-Oracle Hard-Case Target Cache

Goal: train against the true A08-oracle hard set, not only fixed high-`input_std` samples.

Design:

- Build an offline cache/list from A18 saved records.
- Include samples where `a15_minus_a08 <= 0`, including nominal low-route cases.
- Include worst student-vs-router or student-vs-A08 gaps.
- Store per-sample metadata:
  - filename;
  - input_std;
  - A08 PSNR;
  - A15 PSNR;
  - A18 selected expert;
  - A08/A15 oracle label;
  - current A37/A40/A41 gap.
- Train with explicit A08 teacher anchor or GT+teacher mixed target on this set.

Promotion criteria:

- A08-oracle gap improves over A37 and does not regress to A40/A41 levels.
- high-route gap improves over A37 `-0.11159004428502844`.
- worst-gap list visibly recovers.

### A43: A37 Per-Sample Trust Anchor

Goal: use A37 as the high-route-safe compressed anchor and prevent later training from sacrificing its hard cases.

Design:

- Save A37 outputs or per-sample PSNR margins for hard cases.
- Add hinge/trust loss:

```text
new_model should not be worse than A37 on A08-tail/hard-case set by more than a small margin
```

- Allow low/A15-side improvements only if hard-case margins do not collapse.

Promotion criteria:

- high-route gap better than A37;
- A08-oracle gap not worse than A37;
- aggregate PSNR can be below A40 initially if bottleneck metrics improve.

### A44: Route-Conditioned Internal Capacity

Goal: move route-specific capacity earlier than late RGB deltas.

Candidate structures:

- route-conditioned decoder/refinement blocks before terminal RGB projection;
- partial dual-expert blocks for the last decoder/refinement stages;
- route-specific Gaussian/control branch, but with anchors to avoid A39-style shared-control redirection;
- compact A36-inspired partial dual path rather than full duplicated backbone.

Promotion criteria:

- high-route gap must beat A37 and ideally cross `>-0.10 dB`;
- A08-oracle gap must not remain around A40/A41 `~ -0.238 dB`;
- low-route selected delta remains locked or fully explained;
- aggregate PSNR is secondary until high-tail moves.

## 9. Promotion Policy Going Forward

Do not promote a run as a bottleneck breakthrough only because aggregate PSNR improves.

Required reporting for every next experiment:

- standard PSNR/SSIM/loss;
- A18 alignment:
  - overall gap;
  - high-route gap;
  - low-route gap;
  - A08-oracle gap;
  - A15-oracle gap;
- route-regime diagnostics:
  - selected delta;
  - high selected delta;
  - low selected delta;
  - high branch L1 vs A08;
  - low branch L1 vs A15;
- worst-gap list vs A18/A36/A37;
- explicit conclusion: aggregate breakthrough, high-route breakthrough, both, or neither.

High-tail-first promotion thresholds:

- must improve over A37 high-route gap `-0.11159004428502844`;
- target line: high-route gap `>-0.10 dB`;
- must improve A08-oracle gap over A37 `-0.19984098726138502`, or at minimum avoid A40/A41-level regression around `-0.238 dB`;
- low-route surplus should remain positive and explainable.

## 10. Bottom Line

The project has already achieved a real compressed aggregate breakthrough with A37 and an aggregate numeric best with A40. The remaining bottleneck is narrower and harder: preserving A08-oracle/high-fidelity hard cases inside a compact/shared model.

The next breakthrough will likely not come from another scalar loss weight, another high-sampling variant, or another late output adapter. It should come from one of these:

- A08-oracle hard-case target cache;
- A37 per-sample trust anchoring;
- stronger route-conditioned internal capacity inspired by A36 but much more compact.

That is the cleanest current bottleneck map.
