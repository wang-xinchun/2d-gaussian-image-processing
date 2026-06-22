# Result Analysis: V1/T01/A_e03

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_e03`
- Dataset: `UHD_deblur`
- Training split: `1864` pairs from `train`
- Internal validation split: `20` pairs from `train`
- Official test split: `300` pairs from `test`

## Completion Status

- Training finished successfully.
- Final step: `1500`
- Final logged elapsed time: `27.6250` minutes
- Final checkpoint: `experiments/V1/runs/A_e03/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A_e03/checkpoints/step_001500.pt`
- Training log: `experiments/V1/runs/A_e03/logs/train_20260509_012535.log`
- Train metrics: `experiments/V1/runs/A_e03/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A_e03/metrics/val_metrics.jsonl`

The run completed by reaching `max_steps=1500`, not by the `wall_clock_minutes=58`
cap. No NaN, data-loader failure, checkpoint failure, or post-validation OOM
occurred in `A_e03`.

Important continuity note:

- `A_e03` is the cache-safe redeployment of `A_d03`.
- `A_d03` validated the edge-guided branch's initial training behavior but
  failed immediately after step-500 full-image validation with CUDA OOM.
- `A_e03` keeps the same mechanism and high-VRAM target, and changes only
  runtime hygiene: template-side `torch.cuda.empty_cache()` after validation
  plus launch env `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Training Behavior

The first logged training loss was:

```text
step 1 loss_total = 0.15594424307346344
```

The final logged training loss was:

```text
step 1500 loss_total = 0.03862513601779938
```

Segment means:

| Step range | Mean train loss | Median | Min | Max |
|---|---:|---:|---:|---:|
| 1-250 | 0.039243 | 0.035379 | 0.024101 | 0.155944 |
| 251-500 | 0.036147 | 0.035800 | 0.025952 | 0.049874 |
| 501-1000 | 0.035156 | 0.034972 | 0.024101 | 0.050894 |
| 1001-1500 | 0.035419 | 0.035037 | 0.026753 | 0.047710 |

Interpretation:

- The loss dropped sharply after startup and then stabilized in the `0.03x`
  regime.
- The added edge-guided detail branch and Laplacian loss did not create
  unstable optimization.
- Most importantly, `A_e03` crossed the exact post-validation failure point of
  `A_d03` and finished cleanly, so the cache-safe runtime fix is validated.

## Training-Time Full-Image Validation

Training-time validation used the first `20` train pairs with full-image
`crop_size=0`.

Validation trajectory:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 500 | 0.023656 | 28.5723 |
| 1000 | 0.023560 | 28.3272 |
| 1500 | 0.022555 | 28.7618 |

The best internal-validation point is the last checkpoint:

```text
step 1500
val_loss = 0.02255534678697586
val_psnr = 28.761768145576514
```

Interpretation:

- Unlike `A03` and `A_c03`, this run does not show an early-validation best
  point overtaking the final step.
- Internal validation improves again by the end of training, suggesting the
  edge-guided branch is not simply overfitting immediately after the first
  validation peak.

## Post-Hoc Full Evaluation

Post-hoc evaluation was run on:

- full internal validation set: `20` samples, `crop_size=0`
- official test set: `300` samples, `crop_size=0`

Evaluated checkpoints:

- `experiments/V1/runs/A_e03/checkpoints/step_001500.pt`
- `experiments/V1/runs/A_e03/checkpoints/final.pt`

Results:

| Checkpoint | Split | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|
| `step_001500.pt` | full internal val | 20 | 0.022555 | 28.7618 |
| `step_001500.pt` | official test | 300 | 0.034855 | 25.2141 |
| `final.pt` | full internal val | 20 | 0.022555 | 28.7618 |
| `final.pt` | official test | 300 | 0.034855 | 25.2141 |

Technical note:

- `step_001500.pt` and `final.pt` have identical `model` weights and therefore
  identical evaluation results.
- Their file hashes differ because the checkpoint package contains non-model
  content in addition to the weights.

## Comparison To Existing UHD Deblur Baselines

Current pre-`A_e03` official baseline:

```text
A03 final.pt official full-image test PSNR = 25.1402293880564
```

Recent same-task same-protocol comparison:

| Run | Checkpoint | Official Test PSNR | Delta vs A03 Final |
|---|---|---:|---:|
| `A03` | `final.pt` | 25.1402293880564 | 0.0000 |
| `A_c03` | `final.pt` | 25.124958465158436 | -0.0153 |
| `A_e03` | `step_001500.pt` | 25.21410079036176 | +0.0739 |
| `A_e03` | `final.pt` | 25.21410079036176 | +0.0739 |

Interpretation:

- `A_e03` is the first edge-guided high-VRAM branch that both finishes cleanly
  and exceeds the accepted `A03` official baseline under the same full-image
  official test protocol.
- The gain over `A03 final.pt` is modest at about `+0.0739 dB`, so it does not
  reach the staged `+0.3 dB` threshold for a clearly meaningful jump.
- Even so, the result is important because it converts the new mechanism from a
  pure engineering/runtime fix into a real, protocol-consistent positive result.

## Main Conclusion

`V1/T01/A_e03` is accepted as a successful completed high-VRAM `UHD_deblur`
experiment and as the first accepted official-baseline improvement after `A03`.

It proves:

- the `A_d03` post-validation OOM was a runtime/cache issue rather than a
  mechanism failure;
- the edge-guided detail refiner plus Laplacian consistency loss can train
  stably under the `>=80G` deployment constraint when the runtime is fixed;
- the new mechanism produces a same-protocol official full-image improvement
  over the previously accepted `A03` baseline.

The conservative project record is therefore:

> `A_e03` completed cleanly at `step=1500`, validated the cache-safe runtime
> fix for the edge-guided branch, and improved the official `UHD_deblur`
> full-image test PSNR from `25.1402 dB` (`A03 final.pt`) to `25.2141 dB`
> (`A_e03 final.pt` / `step_001500.pt`).

## Acceptance Decision

Acceptance status:

```text
runtime/checkpoint acceptance: passed
high-VRAM gate acceptance: passed
official full-image test acceptance: passed
official baseline replacement: yes
new official UHD_deblur baseline: V1/T01/A_e03 final.pt, PSNR=25.2141 dB
```

## Recommended Next Action

The highest-value next step on the existing run line is now:

1. finish `A_a03` official full-image `crop_size=0` evaluation under the same
   protocol;
2. compare `A_a03` directly against the updated `A_e03` baseline instead of the
   old `A03` baseline;
3. if the project stays on the edge-guided route, target a larger same-protocol
   gain, ideally approaching the staged meaningful-improvement threshold of
   about `+0.3 dB`.
