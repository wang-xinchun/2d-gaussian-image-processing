# Result Analysis: V1/T01/A_ah03

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_ah03`
- Dataset: `UHD_deblur`
- Training split: `1924` pairs from `train`
- Internal validation split: `20` pairs from `train`
- Official test split: `300` pairs from `test`

## Completion Status

- Training finished successfully.
- Final step: `2500`
- Final logged elapsed time: `36.1397` minutes
- Final checkpoint: `experiments/V1/runs/A_ah03/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A_ah03/checkpoints/step_002500.pt`
- Training log: `experiments/V1/runs/A_ah03/logs/train_20260518_234142.log`
- Train metrics: `experiments/V1/runs/A_ah03/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A_ah03/metrics/val_metrics.jsonl`
- Official test metrics: `experiments/V1/runs/A_ah03/metrics/official_test_final.json`

The run completed by reaching `max_steps=2500`, not by the `58`-minute wall-clock cap. No NaN, checkpoint failure, dataloader failure, or runtime OOM occurred during training.

Technical note:

- `step_002500.pt`, `latest.pt`, and `final.pt` have identical `model` weights.
- Their serialized files may differ because of non-model payload, but the effective trained endpoint is the same.
- Therefore the accepted official result of `final.pt` is also the true result of the actual `step=2500` endpoint.

## Training Behavior

The branch showed a healthy but not dramatic short-run validation profile:

- `step=500`: `val_psnr=28.527268073041363`
- `step=1000`: `val_psnr=28.61423288220579`
- `step=1500`: `val_psnr=28.366151293611722`
- `step=2000`: `val_psnr=28.7055744215311`
- `step=2500`: `val_psnr=29.039882709180723`

Interpretation:

- The control-gated edge branch did not collapse or become erratic.
- It recovered after the mid-run dip and ended at its best training-time validation PSNR.
- The branch is therefore a clean short probe, but not automatically a promotion winner.

## Training-Time Full-Image Validation

Training-time validation used the first `20` train pairs with full-image `crop_size=0`.

Validation trajectory:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 500 | 0.024734 | 28.5273 |
| 1000 | 0.024984 | 28.6142 |
| 1500 | 0.024842 | 28.3662 |
| 2000 | 0.024162 | 28.7056 |
| 2500 | 0.022766 | 29.0399 |

The best training-time validation PSNR is:

```text
step 2500
val_loss = 0.022765929251909255
val_psnr = 29.039882709180723
```

The lowest training-time validation loss is:

```text
step 2500
val_loss = 0.022765929251909255
val_psnr = 29.039882709180723
```

Interpretation:

- The branch ends at its best validation point, and unlike the earlier short probe family there is no endpoint rollback on the internal validation metric.
- That is a good short-probe sign, but the official test still decides whether the branch is truly competitive.

## Post-Hoc Official Evaluation

Post-hoc evaluation was run on:

- official test set: `300` samples, `crop_size=0`

Evaluated checkpoint:

- `experiments/V1/runs/A_ah03/checkpoints/final.pt`

Result:

| Checkpoint | Split | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|
| `final.pt` | official test | 300 | 0.036350 | 25.3454 |

Interpretation:

- `A_ah03` is stronger than the earlier local short probes `A_ac03`, `A_ad03`, and `A_ae03` on the official test.
- It still does not beat `A_z03`, and it remains below the frozen medium milestone `M04/A_y03`.
- So it is a meaningful diagnostic gain, but not a promotion candidate.

## Comparison To Existing UHD Deblur Anchors

Relevant same-task same-protocol references:

```text
A_ac03 final.pt official full-image test PSNR = 25.147392029064648
A_ad03 final.pt official full-image test PSNR = 25.200779442877334
A_ae03 final.pt official full-image test PSNR = 25.253274175974838
A_z03 final.pt official full-image test PSNR = 25.385710820365173
A_y03 final.pt official full-image test PSNR = 26.272695829299355
```

Same-task same-protocol comparison:

| Run | Checkpoint | Official Test PSNR | Delta vs A_ah03 Final |
|---|---|---:|---:|
| `A_ah03` | `final.pt` | 25.3454 | 0.0000 |
| `A_ae03` | `final.pt` | 25.2533 | -0.0921 |
| `A_ad03` | `final.pt` | 25.2008 | -0.1446 |
| `A_ac03` | `final.pt` | 25.1474 | -0.1980 |
| `A_z03` | `final.pt` | 25.3857 | +0.0403 |
| `A_y03` | `final.pt` | 26.2727 | +0.9273 |

Additional deltas:

```text
A_ah03 final.pt vs A_ae03 final.pt = +0.09208698832977547 dB
A_ah03 final.pt vs A_ad03 final.pt = +0.14458172142727932 dB
A_ah03 final.pt vs A_ac03 final.pt = +0.1979691352399655 dB
A_ah03 final.pt vs A_z03 final.pt = -0.040349656060559624 dB
A_ah03 final.pt vs A_y03 final.pt = -0.927334664994742 dB
```

## Main Conclusion

`V1/T01/A_ah03` is accepted as a successfully completed short probe with a real official-test gain over the earlier three short probes in the local family, but it still does not overtake the existing short leader `A_z03`.

It proves:

- control-gated edge refinement is runtime-safe under the short-tier `6x720` envelope;
- it is a better official-test result than `A_ac03`, `A_ad03`, and `A_ae03`;
- the shared-control-surface direction still has room to improve.

It does not prove:

- that this branch is strong enough to beat `A_z03`;
- that the gap to `M04/A_y03` is closed;
- that this exact control gate should be promoted to medium training as-is.

## Acceptance Decision

Acceptance status:

```text
runtime/checkpoint acceptance: passed
short-probe completion: passed
official full-image test acceptance: passed
best completed post-M04 short probe so far: no
best result inside the A_ac03/A_ad03/A_ae03/A_ah03 family: yes
medium-promotion candidate: no
recommended next action: keep refining the shared control surface with another lightweight redesign rather than replaying this exact gate
```
