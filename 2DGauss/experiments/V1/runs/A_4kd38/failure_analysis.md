# A_4kd38 Failure Analysis

## Question

Why did A38 fail to break the remaining A37 high-route/A08-tail bottleneck, even though it slightly improved the aggregate PSNR?

## Short Answer

A38 did not fail because it lacked steps. It failed because its intervention was pointed at the wrong mechanism.

A37's gain came from route-aware teacher compression: native A08/A15 teachers, high-route branch supervision, low-route protection, and selected late shared/high modules. A38 removed that teacher signal and replaced it with ordinary GT reconstruction/proxy loss while training only a small terminal high decoder-head path. That objective can make tiny pixel-level corrections, but it cannot reconstruct the missing A08-like high-route representation.

## Key Evidence

### 1. The improvement is real but too small to be a mechanism breakthrough

| Metric | A37 | A38 best | Delta |
|---|---:|---:|---:|
| test PSNR | 17.889658453438816 | 17.889819259555214 | +0.00016080611639779363 |
| A18 overall gap | +0.042724077624394126 | +0.04288488374079478 | +0.00016080611640065245 |
| high-route gap | -0.11159004428502844 | -0.1111094511726412 | +0.0004805931123872398 |
| low-route gap | +0.1591364853806252 | +0.15905604867549208 | -0.00008043670513310786 |
| A08-oracle gap | -0.19984098726138502 | -0.19848443524653703 | +0.0013565520148479948 |
| A15-oracle gap | +0.2493535773419097 | +0.2484957851003737 | -0.0008577922415359818 |

A38 moves the right group a little: A08-oracle improves by about `+0.00136 dB`. But the required high-route gap target is `>-0.10 dB`; A38 remains at `-0.1111094511726412 dB`. The missing gap is still about `0.0111 dB`, more than twenty times the high-route improvement A38 produced.

### 2. Training longer made the mechanism worse

A38 validation:

| Step | Val PSNR | High-route delta |
|---:|---:|---:|
| 10 | 18.562811833496248 | 0.017442694883793592 |
| 60 | 18.561646188950657 | 0.017667582090944052 |
| 120 | 18.560370419887278 | 0.01791006026789546 |

The high-route delta keeps growing, but validation PSNR keeps falling. Final A18 alignment confirms the same behavior:

- A38 best high-route gap: `-0.1111094511726412`
- A38 final high-route gap: `-0.11144634980848278`

So A38 was not undertrained. It was being optimized along a direction that quickly becomes harmful.

### 3. Per-sample behavior shows cancellation, not broad improvement

Per-sample A38 minus A37 PSNR:

- Overall: `+0.00016080611640065357 dB`
- Positive samples: `33`
- Negative samples: `36`
- Unchanged samples: `31`
- High route: `+0.0004805931123872398 dB`
- Low route: `-0.00008043670513308692 dB`

The group split is sharper:

| Group | Count | Mean delta |
|---|---:|---:|
| high route + A08 oracle | 30 | +0.002047657982529927 |
| high route + A15 oracle | 13 | -0.003135710434095884 |
| low route + A15 oracle | 41 | -0.00013552549755354933 |
| low route + A08 oracle | 16 | +0.00006072832544434803 |

A38 is partially doing what we wanted on high+A08 samples, but it also hurts high+A15 samples and some hard high+A08 samples. The net aggregate gain is therefore tiny.

## Mechanism-Level Explanation

### 1. A38 removed the signal that made A37 work

A37 used native A08/A15 teacher configurations and high-route branch supervision. Its result analysis records that the successful recipe disabled ordinary reconstruction/proxy losses and kept high/A08 route branch supervision plus delta trust. That directly taught the student to imitate the A08-like high route.

A38 did the opposite:

- `router_distillation_weight = 0.0`
- `input_std_route_high_branch_weight = 0.0`
- `input_std_route_low_branch_weight = 0.0`
- `charbonnier_weight = 1.0`
- proxy losses restored

This changes the target from "be like the A08 expert on high route" to "slightly reduce GT reconstruction loss through a tiny terminal head." Those are not equivalent. The original bottleneck is not just pixel error; it is preserving the high-route expert's representation and inductive bias inside one compact checkpoint.

### 2. The trainable capacity was too late and too small

A37 trained `76643` parameters, including shared route decoder-head modules, Gaussian feature refinement, Gaussian decoder guidance, and `decoder.1`.

A38 trained only `6819` parameters:

- `input_std_route_decoder_heads.high_feature`
- `input_std_route_decoder_heads.high_head`

This protects the low route, but it also freezes the upstream representation that the high head receives. The A08-like advantage likely depends on earlier/larger feature formation, Gaussian/control interactions, and decoder state, not only the terminal RGB projection. A small terminal head can reshape residual color/detail a little; it cannot rebuild missing expert capacity.

### 3. The fixed soft gate dilutes and leaks the gradient

`InputStdRouteDecoderHeads` computes:

```text
residual = gate * high_residual + (1 - gate) * low_residual
```

The gate is a detached sigmoid prior from input_std. It is not hard except far from the threshold.

Observed test route gate means:

- high-route gate mean: `0.966229241947795`
- low-route gate mean: `0.023600109885921854`

Consequences:

- High-route samples are not pure high-head supervision; about `3.4%` of the output is still low route on average.
- Low-route samples are not totally isolated from high-head changes; about `2.36%` high-head leakage remains on average.
- Near-boundary samples receive mixed gradients that are neither pure A08 nor pure A15.

This is useful for smoothness, but it is bad for a tiny high-head-only correction when the goal is to close a precise expert gap.

### 4. Natural sampling gives sparse useful high-route updates

The test split has `43` high-route and `57` low-route samples. Validation reports `val_route_high_hard_mean = 0.37`. Under natural sampling, many batches do not provide hard high-route supervision.

Because A38 trains only the high route, low-route batches produce little useful signal for the trainable path unless the soft gate leaks a small gradient. The result is a weak, noisy, and mixed objective: many steps are almost irrelevant to the desired high/A08-tail correction.

### 5. Standard GT loss conflicts with oracle heterogeneity

The high route is not identical to the A08-oracle set. There are high-input-std samples where A15 is still oracle. A38's high-head update improved high+A08 samples on average, but hurt high+A15 samples strongly:

- high+A08: `+0.00205 dB`
- high+A15: `-0.00314 dB`

This means `input_std` is a strong prior but not a perfect oracle label. A fixed high-head update under GT loss cannot distinguish "high route but A08-like needed" from "high route but A15-like behavior is safer." A37 survived this better because its teacher pressure and delta constraints held a more useful compromise.

## Why This Is Not Just Hyperparameter Failure

Lower LR or fewer steps would preserve A37, but preserving A37 is not a breakthrough. The best A38 checkpoint is already at step10, and it only improves high-route gap by `+0.00048 dB`. More steps degrade. That tells us the local optimum around A37 has a very narrow useful direction under standard loss.

The missing `0.0111 dB` high-route gap needs a different mechanism, not more of the same objective.

## Practical Conclusion

A38 proves three useful things:

1. Low-route lock works.
2. Standard-loss high-head-only tuning can slightly help A08-oracle samples.
3. The remaining bottleneck is not terminal output correction; it is route-specific representation/capacity.

The next attempt should not be A38 medium training. It should either restore route-aware teacher signal or move capacity earlier:

- A39 option 1: A37 init + offline A08 high-route target cache + high-route distillation without online teacher memory cost.
- A39 option 2: A37 init + route-conditioned feature expert before the terminal RGB head, low route locked.
- A39 option 3: A37 init + route-conditioned Gaussian/control modulation, because A08-tail behavior likely depends on structure/detail control before final residual projection.

The scientific explanation is therefore: A38 was optimizing a small terminal correction with a generic reconstruction objective, while the remaining A37 deficit is an expert-representation deficit.
