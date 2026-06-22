# Result Analysis: V1/T01/A_ai03

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_ai03`
- Dataset: `UHD_deblur`
- Training split: `1924` pairs from `train`
- Internal validation split: `20` pairs from `train`
- Official test split: `300` pairs from `test`

## Completion Status

- Training finished successfully.
- Final step: `2500`
- Final logged elapsed time: `39.6542` minutes
- Final checkpoint: `experiments/V1/runs/A_ai03/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A_ai03/checkpoints/step_002500.pt`
- Training log: `experiments/V1/runs/A_ai03/logs/train_20260519_102725.log`
- Train metrics: `experiments/V1/runs/A_ai03/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A_ai03/metrics/val_metrics.jsonl`
- Official test metrics: `experiments/V1/runs/A_ai03/metrics/official_test_final.json`

The run completed by reaching `max_steps=2500`, not by the `58`-minute wall-clock cap. No NaN, checkpoint failure, dataloader failure, or runtime OOM occurred during training.

Technical note:

- `step_002500.pt`, `latest.pt`, and `final.pt` have identical `model` weights.
- Their serialized files may differ because of non-model payload, but the effective trained endpoint is the same.
- Therefore the accepted official result of `final.pt` is also the true result of the actual `step=2500` endpoint.

## Training Behavior

The branch trained safely but did not produce a strong validation trajectory:

- `step=500`: `val_psnr=28.512547438826395`
- `step=1000`: `val_psnr=28.240129841572543`
- `step=1500`: `val_psnr=28.53359034424408`
- `step=2000`: `val_psnr=28.677247346979414`
- `step=2500`: `val_psnr=28.695192038708456`

Interpretation:

- The decoder-detail gate did not collapse the run.
- The branch recovered from the `step=1000` dip and ended at its best validation PSNR.
- However, the final validation PSNR remains below `A_ah03` and below the stronger local short trajectories, so the live signal is safe but not strong.

## Training-Time Full-Image Validation

Training-time validation used the first `20` train pairs with full-image `crop_size=0`.

Validation trajectory:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 500 | 0.024728 | 28.5125 |
| 1000 | 0.025346 | 28.2401 |
| 1500 | 0.024809 | 28.5336 |
| 2000 | 0.023768 | 28.6772 |
| 2500 | 0.024005 | 28.6952 |

The best training-time validation PSNR is:

```text
step 2500
val_loss = 0.02400543577969074
val_psnr = 28.695192038708456
```

The lowest training-time validation loss is:

```text
step 2000
val_loss = 0.023768198490142823
val_psnr = 28.677247346979414
```

Interpretation:

- The branch does not show a late collapse.
- The endpoint is acceptable as the official-test checkpoint because the final model is also the true final periodic checkpoint.
- The validation profile is not competitive with `A_ah03`, whose endpoint validation reached `29.039882709180723`.

## Post-Hoc Official Evaluation

Post-hoc evaluation was run on:

- official test set: `300` samples, `crop_size=0`

Evaluated checkpoint:

- `experiments/V1/runs/A_ai03/checkpoints/final.pt`

Result:

| Checkpoint | Split | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|
| `final.pt` | official test | 300 | 0.036690 | 25.2671 |

Raw accepted official metrics:

```text
checkpoint_step = 2500
device = cuda
crop_size = 0
samples = 300
loss_mean = 0.03668969906090448
psnr_mean = 25.267052413067724
elapsed_seconds = 456.83330845832825
```

## Comparison To Existing UHD Deblur Anchors

Relevant same-task same-protocol references:

```text
A_ac03 final.pt official full-image test PSNR = 25.147392029064648
A_ad03 final.pt official full-image test PSNR = 25.200779442877334
A_ae03 final.pt official full-image test PSNR = 25.253274175974838
A_ah03 final.pt official full-image test PSNR = 25.345361164304613
A_z03 final.pt official full-image test PSNR = 25.385710820365173
A_y03 final.pt official full-image test PSNR = 26.272695829299355
```

Same-task same-protocol comparison:

| Run | Checkpoint | Official Test PSNR | Delta vs A_ai03 Final |
|---|---|---:|---:|
| `A_ai03` | `final.pt` | 25.2671 | 0.0000 |
| `A_ae03` | `final.pt` | 25.2533 | -0.0138 |
| `A_ad03` | `final.pt` | 25.2008 | -0.0663 |
| `A_ac03` | `final.pt` | 25.1474 | -0.1197 |
| `A_ah03` | `final.pt` | 25.3454 | +0.0783 |
| `A_z03` | `final.pt` | 25.3857 | +0.1187 |
| `A_y03` | `final.pt` | 26.2727 | +1.0056 |

Additional deltas:

```text
A_ai03 final.pt vs A_ae03 final.pt = +0.013778237092886059 dB
A_ai03 final.pt vs A_ad03 final.pt = +0.06627297019038991 dB
A_ai03 final.pt vs A_ac03 final.pt = +0.1196603840030761 dB
A_ai03 final.pt vs A_ah03 final.pt = -0.0783087512368894 dB
A_ai03 final.pt vs A_z03 final.pt = -0.11865840729744903 dB
A_ai03 final.pt vs A_y03 final.pt = -1.0056434162316314 dB
```

## Main Conclusion

`V1/T01/A_ai03` is accepted as a cleanly completed short probe, but it is not a breakthrough and should not be promoted.

It proves:

- the shared-prior decoder-detail gate is runtime-safe under the short-tier `6x720` envelope;
- the branch can complete the full `2500` steps cleanly;
- the mechanism is slightly better than `A_ae03`, so it is not a total failure.

It does not prove:

- that decoder-detail feedback inside the shared-control path is the missing root mechanism;
- that this formulation improves over `A_ah03`;
- that it can overtake the current short leader `A_z03`;
- that it should be promoted to medium training.

## Scientific Interpretation

The sharpened hypothesis after `A_ah03` was that shared-control might have lost the decoder-detail feedback that made `A_z03` strong. `A_ai03` tested that idea by injecting decoder-detail statistics as a multiplicative gate on the shared-prior feature-refinement path rather than as direct cue stacking.

The result is informative but negative for this exact mechanism:

- adding decoder-detail feedback in this gated form recovers only a tiny gain over `A_ae03`;
- it remains clearly below `A_ah03`, which means it does not preserve the role-separation gain from control-gated edge refinement;
- it also remains further below `A_z03`, so it does not recover the original decoder-detail short-leader signal.

Therefore the remaining bottleneck is not simply “shared-control lacks decoder detail.” The root issue is more likely that decoder detail, edge-role separation, and shared prior coordination need a stronger multi-module closed loop rather than a single extra detail gate inside feature refinement.

## Acceptance Decision

Acceptance status:

```text
runtime/checkpoint acceptance: passed
short-probe completion: passed
official full-image test acceptance: passed
best completed post-M04 short probe so far: no
best result inside the shared-control family: no
medium-promotion candidate: no
recommended next action: do not promote A_ai03; use it as evidence that decoder-detail gating alone is insufficient and return to stronger cross-module closed-loop control, with A_af03/A_ag03 still available as diagnostic probes
```
