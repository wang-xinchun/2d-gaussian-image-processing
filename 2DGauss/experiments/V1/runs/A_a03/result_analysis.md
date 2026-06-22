# Result Analysis: V1/T01/A_a03

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_a03`
- Dataset: `UHD_deblur`
- Training split: `1864` pairs from `train`
- Internal validation split: `100` pairs from `train`
- Official test split: `300` pairs from `test`

## Completion Status

- Training finished successfully.
- Final step: `2000`
- Final logged elapsed time: `20.2250` minutes
- Final checkpoint: `experiments/V1/runs/A_a03/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A_a03/checkpoints/step_002000.pt`
- Training log: `experiments/V1/runs/A_a03/logs/train_20260507_010106.log`
- Train metrics: `experiments/V1/runs/A_a03/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A_a03/metrics/val_metrics.jsonl`

The run completed by reaching `max_steps=2000`, not by the `58` minute
wall-clock cap. No NaN, OOM, data-loader failure, or checkpoint failure was
observed during training.

Important audit note:

- `experiments/V1/runs/A_a03/logs/train_20260507_010003.log` is a failed launch
  attempt before training. It exited with `SameFileError` because the launch
  script originally passed the run-local config path to the training entrypoint.
- The effective training run is
  `experiments/V1/runs/A_a03/logs/train_20260507_010106.log`.

## Training Behavior

The first logged training loss was:

```text
step 1 loss_total = 0.19248083233833313
```

The final logged training loss was:

```text
step 2000 loss_total = 0.023953847587108612
```

Segment means:

| Step range | Mean train loss | Median | Min | Max |
|---|---:|---:|---:|---:|
| 1-250 | 0.046043 | 0.040366 | 0.030086 | 0.192481 |
| 251-500 | 0.034864 | 0.035614 | 0.021918 | 0.047472 |
| 501-1000 | 0.036614 | 0.035567 | 0.022974 | 0.056599 |
| 1001-1500 | 0.034908 | 0.035479 | 0.023071 | 0.047906 |
| 1501-2000 | 0.034249 | 0.033921 | 0.023485 | 0.045960 |

Interpretation:

- The model learned quickly in the first few hundred steps.
- After the early drop, train loss stayed stable in the `0.03x` range.
- The larger `A_a03` capacity setting did not create runtime instability.

## Training-Time Validation

Training-time validation used `100` batches from the fixed internal validation
split.

Validation trajectory:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 250 | 0.027229 | 28.5685 |
| 500 | 0.025282 | 29.0861 |
| 750 | 0.025955 | 28.8716 |
| 1000 | 0.025984 | 28.5659 |
| 1250 | 0.025331 | 28.5676 |
| 1500 | 0.025430 | 28.4943 |
| 1750 | 0.025600 | 28.5330 |
| 2000 | 0.024599 | 28.5866 |

Internal validation selection is split:

- Best internal-validation PSNR: `step_000500.pt`, `PSNR=29.0861`.
- Best internal-validation loss: `final.pt` / step `2000`, `loss=0.024599`.

This means the branch still has the same selection-policy tension as `A03`: the
lowest loss checkpoint and highest PSNR checkpoint are not identical.

## Post-Hoc Evaluation Completed

Post-hoc evaluation completed on:

- full internal validation set with `crop_size=512`;
- diagnostic official-test center-crop setting with `crop_size=512`.

Evaluated checkpoints:

- `experiments/V1/runs/A_a03/checkpoints/step_000500.pt`
- `experiments/V1/runs/A_a03/checkpoints/final.pt`

Results:

| Checkpoint | Split | Crop | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|---:|
| `step_000500.pt` | full internal val | 512 | 100 | 0.025282 | 29.0861 |
| `final.pt` | full internal val | 512 | 100 | 0.024599 | 28.5866 |
| `step_000500.pt` | test diagnostic | 512 | 300 | 0.043987 | 23.9399 |
| `final.pt` | test diagnostic | 512 | 300 | 0.043345 | 24.0163 |

For reference, the matching `A03` diagnostic `crop_size=512` test numbers are:

| Run | Checkpoint | Crop | Test PSNR |
|---|---|---:|---:|
| `A03` | `step_000500.pt` | 512 | 23.9051 |
| `A03` | `final.pt` | 512 | 23.9060 |
| `A_a03` | `step_000500.pt` | 512 | 23.9399 |
| `A_a03` | `final.pt` | 512 | 24.0163 |

Under this diagnostic crop protocol, `A_a03 final.pt` is ahead of `A03 final.pt`
by about `+0.1103 dB`. This is useful diagnostic evidence, but it is not the
official baseline protocol.

## Official Full-Image Test Completed

Official full-image `crop_size=0` evaluation is now completed for both the
internal-val-selected checkpoint and `final.pt`.

Results:

| Checkpoint | Split | Crop | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|---:|
| `step_000500.pt` | official test | 0 | 300 | 0.035078 | 25.1216 |
| `final.pt` | official test | 0 | 300 | 0.034607 | 25.1585 |

Compared with the current accepted `UHD_deblur` baselines:

```text
A03 final.pt official full-image test PSNR = 25.1402293880564
A_e03 final.pt official full-image test PSNR = 25.21410079036176
```

Delta summary:

```text
A_a03 step_000500.pt vs A03 final.pt = -0.0186 dB
A_a03 final.pt vs A03 final.pt      = +0.0183 dB
A_a03 step_000500.pt vs A_e03 final.pt = -0.0925 dB
A_a03 final.pt vs A_e03 final.pt       = -0.0556 dB
```

Interpretation:

- `A_a03` now has a complete same-protocol official result record.
- The branch does not beat the current accepted `A_e03` official baseline.
- The internal-val-selected checkpoint is not the official-test winner; `final.pt`
  is slightly stronger on the official full-image test, mirroring the broader
  checkpoint-selection tension already seen in `A03`.

## Comparison With A03 And A_e03

What can be concluded now:

- Runtime/training acceptance: `A_a03` passes.
- Internal validation: `A_a03 step_000500.pt` slightly exceeds `A03
  step_000500.pt` on internal-val PSNR (`29.0861` vs `29.0056`).
- Diagnostic `512`-crop test: `A_a03 final.pt` slightly exceeds `A03 final.pt`
  (`24.0163` vs `23.9060`).
- Same-protocol official full-image test: `A_a03 final.pt` is slightly above
  `A03 final.pt` (`25.1585` vs `25.1402`) but still below `A_e03 final.pt`
  (`25.2141`).

What cannot be concluded:

- `A_a03` cannot be recorded as the new official `UHD_deblur` baseline because
  it does not exceed `A_e03`.

## Acceptance Conclusion

`V1/T01/A_a03` is accepted as a successful completed training run and useful
branch probe.

It proves:

- the larger capacity/Gaussian-expression branch is trainable under `T01`;
- the run reaches all planned validation/checkpoint milestones;
- the branch gives positive internal-validation and diagnostic crop-test signals.

It does not prove:

- that the capacity-only branch should replace the current `A_e03` official
  baseline;
- that larger capacity without the edge-guided mechanism is the strongest next
  direction for `UHD_deblur`.

Acceptance status:

```text
training/runtime acceptance: passed
diagnostic evaluation: positive but modest
official full-image evaluation: completed
official full-image baseline replacement: no
current official UHD_deblur baseline remains: A_e03 final.pt, PSNR=25.2141 dB
```

## Recommended Next Action

Use `A_a03` as completed evidence that capacity-only scaling is not enough to
beat the accepted edge-guided baseline, and prioritize post-`A_e03` branches
that keep the new mechanism while targeting a larger same-protocol gain.
