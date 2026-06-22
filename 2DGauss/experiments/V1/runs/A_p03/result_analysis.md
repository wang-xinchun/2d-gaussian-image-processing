# Result Analysis: V1/T01/A_p03

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_p03`
- Dataset: `UHD_deblur`
- Training split: `1924` pairs from `train`
- Internal validation split: `20` pairs from `train`
- Official test split: `300` pairs from `test`

## Completion Status

- Training finished successfully.
- Final step: `21820`
- Final logged elapsed time: `300.0092` minutes
- Final checkpoint: `experiments/V1/runs/A_p03/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A_p03/checkpoints/step_021500.pt`
- Training log: `experiments/V1/runs/A_p03/logs/train_20260515_032320.log`
- Train metrics: `experiments/V1/runs/A_p03/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A_p03/metrics/val_metrics.jsonl`
- Official test metrics: `experiments/V1/runs/A_p03/metrics/official_test_final.json`

The run completed cleanly at the planned `300` minute cap. No NaN, data-loader
failure, checkpoint failure, or post-validation OOM occurred during training.

Technical note:

- `final.pt` is the accepted endpoint used for official evaluation.
- `latest.pt` and `step_021500.pt` reflect the last periodic save state.

## Training Behavior

The first logged training loss was:

```text
step 1 loss_total = 0.12189736217260361
```

The final logged training loss before clean finish was:

```text
step 21820 loss_total = 0.02943393960595131
```

Interpretation:

- The medium-budget decoder-guidance branch remained stable for the full
  five-hour window.
- The branch did not show any runtime regression under the `6x800` launch
  geometry.
- The loss curve is noisier than the short probes, which is expected for a
  longer optimization window with the same lightweight backbone.

## Training-Time Full-Image Validation

Training-time validation used the first `20` train pairs with full-image
`crop_size=0`.

Validation trajectory:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 1000 | 0.024302 | 28.5563 |
| 2000 | 0.023387 | 28.5173 |
| 3000 | 0.021157 | 29.0776 |
| 4000 | 0.021652 | 29.1126 |
| 5000 | 0.020995 | 29.1650 |
| 6000 | 0.020761 | 29.2257 |
| 7000 | 0.021882 | 28.9034 |
| 8000 | 0.021425 | 29.0885 |
| 9000 | 0.021324 | 29.1158 |
| 10000 | 0.021059 | 29.1707 |
| 11000 | 0.020361 | 29.2806 |
| 12000 | 0.020347 | 29.2897 |
| 13000 | 0.020503 | 29.2412 |
| 14000 | 0.020143 | 29.3127 |
| 15000 | 0.020516 | 29.2895 |
| 16000 | 0.020301 | 29.2065 |
| 17000 | 0.020942 | 29.1829 |
| 18000 | 0.020314 | 29.2957 |
| 19000 | 0.020067 | 29.3053 |
| 20000 | 0.020165 | 29.2935 |
| 21000 | 0.020245 | 29.3409 |

The best training-time validation PSNR is:

```text
step 21000
val_loss = 0.02024461105465889
val_psnr = 29.340949526687346
```

Interpretation:

- The internal validation curve is healthy but does not fully predict the
  final official-test ordering.
- The best internal point appears before the run end, yet the final endpoint
  is still the relevant official-test winner inside this branch.

## Post-Hoc Official Evaluation

Post-hoc evaluation was run on:

- official test set: `300` samples, `crop_size=0`

Evaluated checkpoint:

- `experiments/V1/runs/A_p03/checkpoints/final.pt`

Result:

| Checkpoint | Split | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|
| `final.pt` | official test | 300 | 0.031886 | 26.0087 |

Interpretation:

- The run is a clear improvement over the short-scout family.
- It does **not** yet beat the frozen `M03/A_k03` official baseline.
- The official-test gap to the current baseline is about `0.1913 dB`, so the
  branch is positive but still not a milestone replacement.

## Comparison To Existing UHD Deblur Baselines

Frozen baseline before this run:

```text
A_k03 final.pt official full-image test PSNR = 26.199959289578075
```

Same-task same-protocol comparison:

| Run | Checkpoint | Official Test PSNR | Delta vs A_k03 Final |
|---|---|---:|---:|
| `A_k03` | `final.pt` | 26.2000 | 0.0000 |
| `A_p03` | `final.pt` | 26.0087 | -0.1913 |

Interpretation:

- `A_p03` is a successful completed medium-budget run, but not yet a new
  official baseline.
- Compared with the current frozen milestone, it remains behind by about
  `0.1913 dB`.

## Relation To The Lightweight 90% Goal

Current working `UHD-Blur` 90% PSNR threshold:

```text
26.5743 dB
```

Comparison:

```text
A_p03 final.pt = 26.008706971730295 dB
remaining gap to 90% threshold = 0.5655930282697064 dB
```

Interpretation:

- `A_p03` closes some distance to the lightweight goal, but does not cross it.
- The run confirms that the current decoder-guidance line still has room for
  improvement, but the next step should not be a blind replay of the same
  medium-budget recipe.

## Main Conclusion

`V1/T01/A_p03` is accepted as a successful completed medium-budget
`UHD_deblur` experiment under the `A_n03` structural anchor.

It proves:

- the promoted short-scout structure can scale into a full five-hour budget
  without runtime instability;
- the branch can achieve a stronger official-test result than the short
  probes;
- the current `M03/A_k03` baseline is still ahead, so `A_p03` is a positive
  step rather than a milestone replacement.
