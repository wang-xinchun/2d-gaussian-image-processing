# Result Analysis: V1/T01/A_ad03

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_ad03`
- Dataset: `UHD_deblur`
- Training split: `1924` pairs from `train`
- Internal validation split: `20` pairs from `train`
- Official test split: `300` pairs from `test`

## Completion Status

- Training finished successfully.
- Final step: `2500`
- Final logged elapsed time: `33.6803` minutes
- Final checkpoint: `experiments/V1/runs/A_ad03/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A_ad03/checkpoints/step_002500.pt`
- Training log: `experiments/V1/runs/A_ad03/logs/train_20260518_012147_after_ac03.log`
- Train metrics: `experiments/V1/runs/A_ad03/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A_ad03/metrics/val_metrics.jsonl`
- Official test metrics: `experiments/V1/runs/A_ad03/metrics/official_test_final.json`

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

The branch showed a notably flatter and more stable short-run validation
trajectory than `A_ac03`:

- `step=500`: `val_psnr=28.6524`
- `step=1000`: `val_psnr=28.6875`
- `step=1500`: `val_psnr=28.6774`
- `step=2000`: `val_psnr=28.5878`
- `step=2500`: `val_psnr=28.6896`

Interpretation:

- Shared-prior coupling does not produce a dramatic peak, but it avoids the
  late rollback seen in `A_ac03`.
- Internal validation remains in a narrow stable band throughout the run.
- This is the stronger behavior inside the `A_ac03/A_ad03` serial scout pair.

## Training-Time Full-Image Validation

Training-time validation used the first `20` train pairs with full-image
`crop_size=0`.

Validation trajectory:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 500 | 0.024079 | 28.6524 |
| 1000 | 0.024158 | 28.6875 |
| 1500 | 0.024046 | 28.6774 |
| 2000 | 0.024547 | 28.5878 |
| 2500 | 0.024486 | 28.6896 |

The best training-time validation PSNR is:

```text
step 2500
val_loss = 0.02448628218844533
val_psnr = 28.68962112923873
```

The lowest training-time validation loss is:

```text
step 1500
val_loss = 0.02404614621773362
val_psnr = 28.677351757664116
```

Interpretation:

- `A_ad03` ends very close to its best internal-validation point.
- The run is better calibrated than `A_ac03` from a training-dynamics
  perspective.
- But stable internal validation alone is still not enough to imply a winning
  official result.

## Post-Hoc Official Evaluation

Post-hoc evaluation was run on:

- official test set: `300` samples, `crop_size=0`

Evaluated checkpoint:

- `experiments/V1/runs/A_ad03/checkpoints/final.pt`

Result:

| Checkpoint | Split | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|
| `final.pt` | official test | 300 | 0.036967 | 25.2008 |

Interpretation:

- `A_ad03` is the better result inside the new serial scout pair.
- However, it still fails to beat the short leader `A_z03`, and it also stays
  below the older short anchors `A_u03/A_t03/A_n03`.
- Therefore it is a local winner inside this pair, but not a true promotion
  winner for the mainline.

## Comparison To Existing UHD Deblur Anchors

Relevant same-task same-protocol references:

```text
A_ac03 final.pt official full-image test PSNR = 25.147392029064648
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

| Run | Checkpoint | Official Test PSNR | Delta vs A_ad03 Final |
|---|---|---:|---:|
| `A_ad03` | `final.pt` | 25.2008 | 0.0000 |
| `A_ab03` | `final.pt` | 25.1723 | -0.0285 |
| `A_ac03` | `final.pt` | 25.1474 | -0.0534 |
| `A_z03` | `final.pt` | 25.3857 | +0.1849 |
| `A_u03` | `final.pt` | 25.3101 | +0.1093 |
| `A_t03` | `final.pt` | 25.2944 | +0.0937 |
| `A_n03` | `final.pt` | 25.2919 | +0.0911 |
| `A_r03` | `final.pt` | 25.2867 | +0.0859 |
| `A_v03` | `final.pt` | 25.2711 | +0.0703 |
| `A_s03` | `final.pt` | 25.2567 | +0.0560 |
| `A_aa03` | `final.pt` | 26.0850 | +0.8842 |
| `A_y03` | `final.pt` | 26.2727 | +1.0719 |

Additional deltas:

```text
A_ad03 final.pt vs A_ac03 final.pt = +0.05338741381268575 dB
A_ad03 final.pt vs A_ab03 final.pt = +0.028467548448869642 dB
A_ad03 final.pt vs A_z03 final.pt = -0.18493137748783937 dB
A_ad03 final.pt vs A_u03 final.pt = -0.10928463908857866 dB
A_ad03 final.pt vs A_t03 final.pt = -0.09366677684879975 dB
A_ad03 final.pt vs A_y03 final.pt = -1.0719163864220215 dB
```

Interpretation:

- `A_ad03` is the best completed result among the three post-`A_aa03`
  redesign scouts `A_ab03/A_ac03/A_ad03`.
- But the gain is only local. It does not restore the project to the older
  short-leader level, and it remains far below the medium anchors.
- The shared-prior idea is therefore more promising than `A_ac03`, but still
  not ready for medium promotion in its current form.

## Main Conclusion

`V1/T01/A_ad03` is accepted as a successfully completed short probe with a
locally positive but globally insufficient official result.

It proves:

- shared-prior coupling is runtime-safe under the short-tier `6x728`
  envelope;
- this branch is more stable than `A_ac03` in both validation shape and
  official endpoint;
- the shared-prior direction is a better lightweight coordination candidate
  than the current confidence-normalized formulation.

It does not prove:

- that the current shared-prior stem is already strong enough for medium
  promotion;
- that the absolute bottleneck is solved;
- that the project should replace `A_z03` or `M04/A_y03` with this branch.

What it does prove is narrower and more useful:

- if the project continues along this literature-guided family, `A_ad03`
  is the branch worth refining, but it still needs a stronger control design
  before any medium-budget promotion.

## Acceptance Decision

Acceptance status:

```text
runtime/checkpoint acceptance: passed
short-probe completion: passed
official full-image test acceptance: passed
best completed post-M04 short probe so far: no
best result inside the A_ac03/A_ad03 serial pair: yes
medium-promotion candidate: no
recommended next action: keep the shared-prior coordination idea as the better local direction, but redesign before any medium training
```
