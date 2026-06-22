# Result Analysis: V1/T01/A_ac03

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_ac03`
- Dataset: `UHD_deblur`
- Training split: `1924` pairs from `train`
- Internal validation split: `20` pairs from `train`
- Official test split: `300` pairs from `test`

## Completion Status

- Training finished successfully.
- Final step: `2500`
- Final logged elapsed time: `37.2581` minutes
- Final checkpoint: `experiments/V1/runs/A_ac03/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A_ac03/checkpoints/step_002500.pt`
- Training log: `experiments/V1/runs/A_ac03/logs/train_20260518_012147.log`
- Train metrics: `experiments/V1/runs/A_ac03/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A_ac03/metrics/val_metrics.jsonl`
- Official test metrics: `experiments/V1/runs/A_ac03/metrics/official_test_final.json`

The run completed by reaching `max_steps=2500`, not by the `58`-minute
wall-clock cap. No NaN, checkpoint failure, dataloader failure, or runtime
OOM occurred during training.

Technical note:

- `step_002500.pt`, `latest.pt`, and `final.pt` have identical `model`
  weights.
- Their serialized files may differ because of non-model payload, but the
  effective trained endpoint is the same.
- Therefore the accepted official result of `final.pt` is also the true
  result of the actual `step=2500` endpoint.

## Training Behavior

The branch showed an early internal-validation lift followed by a clear late
rollback:

- `step=500`: `val_psnr=28.5750`
- `step=1000`: `val_psnr=28.8683`
- `step=1500`: `val_psnr=28.6933`
- `step=2000`: `val_psnr=28.6603`
- `step=2500`: `val_psnr=28.3405`

Interpretation:

- The confidence-normalized refinement does not collapse training.
- But its best validation point arrives early and is not sustained.
- The late-stage downward drift is the opposite of what a promotion-ready
  short probe should show.

## Training-Time Full-Image Validation

Training-time validation used the first `20` train pairs with full-image
`crop_size=0`.

Validation trajectory:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 500 | 0.024397 | 28.5750 |
| 1000 | 0.023151 | 28.8683 |
| 1500 | 0.023523 | 28.6933 |
| 2000 | 0.023941 | 28.6603 |
| 2500 | 0.025035 | 28.3405 |

The best training-time validation PSNR is:

```text
step 1000
val_loss = 0.023150817211717366
val_psnr = 28.868337213826116
```

The lowest training-time validation loss is also:

```text
step 1000
val_loss = 0.023150817211717366
val_psnr = 28.868337213826116
```

Interpretation:

- `A_ac03` briefly looks promising on internal validation.
- But the best point is not the final point, and the final endpoint degrades
  by about `0.5279 dB` relative to the peak.
- This makes the branch much less trustworthy than a stable short winner.

## Post-Hoc Official Evaluation

Post-hoc evaluation was run on:

- official test set: `300` samples, `crop_size=0`

Evaluated checkpoint:

- `experiments/V1/runs/A_ac03/checkpoints/final.pt`

Result:

| Checkpoint | Split | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|
| `final.pt` | official test | 300 | 0.037152 | 25.1474 |

Interpretation:

- The early internal-validation high point does **not** convert into a better
  official endpoint.
- This branch underperforms not only the short leader `A_z03`, but also the
  already negative `A_ab03`.
- Therefore the current confidence-normalized formulation is not a viable
  medium-promotion candidate.

## Comparison To Existing UHD Deblur Anchors

Relevant same-task same-protocol references:

```text
A_ad03 final.pt official full-image test PSNR = 25.200779442877334
A_ab03 final.pt official full-image test PSNR = 25.172311894428464
A_z03 final.pt official full-image test PSNR = 25.385710820365173
A_u03 final.pt official full-image test PSNR = 25.310064081965912
A_t03 final.pt official full-image test PSNR = 25.294446219726133
A_n03 final.pt official full-image test PSNR = 25.291929097577473
A_r03 final.pt official full-image test PSNR = 25.286687395113344
A_v03 final.pt official full-image test PSNR = 25.271115184810476
A_s03 final.pt official full-image test PSNR = 25.25673342402521
A_aa03 final.pt official full-image test PSNR = 26.084983286081467
A_y03 final.pt official full-image test PSNR = 26.272695829299355
```

Same-task same-protocol comparison:

| Run | Checkpoint | Official Test PSNR | Delta vs A_ac03 Final |
|---|---|---:|---:|
| `A_ac03` | `final.pt` | 25.1474 | 0.0000 |
| `A_ad03` | `final.pt` | 25.2008 | +0.0534 |
| `A_ab03` | `final.pt` | 25.1723 | +0.0249 |
| `A_z03` | `final.pt` | 25.3857 | +0.2383 |
| `A_u03` | `final.pt` | 25.3101 | +0.1627 |
| `A_t03` | `final.pt` | 25.2944 | +0.1471 |
| `A_n03` | `final.pt` | 25.2919 | +0.1445 |
| `A_r03` | `final.pt` | 25.2867 | +0.1393 |
| `A_v03` | `final.pt` | 25.2711 | +0.1237 |
| `A_s03` | `final.pt` | 25.2567 | +0.1093 |
| `A_aa03` | `final.pt` | 26.0850 | +0.9376 |
| `A_y03` | `final.pt` | 26.2727 | +1.1253 |

Additional deltas:

```text
A_ac03 final.pt vs A_ad03 final.pt = -0.05338741381268575 dB
A_ac03 final.pt vs A_ab03 final.pt = -0.024919865363815108 dB
A_ac03 final.pt vs A_z03 final.pt = -0.23831879130052442 dB
A_ac03 final.pt vs A_u03 final.pt = -0.1626720529012637 dB
A_ac03 final.pt vs A_t03 final.pt = -0.1470541906614848 dB
A_ac03 final.pt vs A_y03 final.pt = -1.1253038002347066 dB
```

Interpretation:

- `A_ac03` is the weakest result among the current post-`A_aa03` scout trio
  `A_ab03/A_ac03/A_ad03`.
- It fails both the local head-to-head and the broader short-family
  comparison.
- The present confidence-normalized implementation should therefore be
  treated as a negative short structural probe.

## Main Conclusion

`V1/T01/A_ac03` is accepted as a successfully completed short probe with a
negative official result.

It proves:

- the calibrated `6x716` geometry is runtime-safe under the strict
  `85-93G` window;
- confidence-normalized refinement can be implemented without destabilizing
  training;
- this specific formulation does not improve official short-tier quality.

It does not prove:

- that confidence or uncertainty is globally useless for Gaussian control;
- that selectivity is the wrong problem to target;
- that all future normalization-style routing ideas should be abandoned.

What it does prove is narrower and more useful:

- this **current** confidence-normalized feature-refinement design should
  **not** be promoted to medium training.

## Acceptance Decision

Acceptance status:

```text
runtime/checkpoint acceptance: passed
short-probe completion: passed
official full-image test acceptance: passed
best completed post-M04 short probe so far: no
best result inside the A_ac03/A_ad03 serial pair: no
medium-promotion candidate: no
recommended next action: retire the current A_ac03 formulation and continue redesign around stronger Gaussian-decoder control mechanisms
```
