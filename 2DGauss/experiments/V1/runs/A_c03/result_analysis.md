# Result Analysis: V1/T01/A_c03

Date: 2026-05-08 CST

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Run: `A_c03`
- Full identifier: `V1/T01/A_c03`
- Dataset: `UHD_deblur`
- Training log: `experiments/V1/runs/A_c03/logs/train_20260508_013703.log`

## Purpose

`A_c03` is the high-VRAM replacement for `A_b03`.

The user added a hard deployment constraint: training memory usage must not be
below `80G`. `A_b03` completed successfully as a full-image-validation protocol
probe, but its observed training memory was about `62.8G`, so it does not meet
the new gate. `A_c03` keeps the same `UHD_deblur` full-image internal-validation
idea and the same capacity settings, but raises the batch size to `25`.

## Gate Summary

Rejected calibration:

| Batch | Status | Max Reserved VRAM | Decision |
|---:|---|---:|---|
| `24` | ok | `78.5195G` | rejected, below `80G` |
| `25` | ok | `81.6641G` | accepted |

Formal preflight passed:

```text
device: cuda
train_samples: 25
val_samples: 20
batch_input_shape: (25, 3, 512, 512)
batch_target_shape: (25, 3, 512, 512)
total_params: 189717
loss_total: 0.07665728032588959
val_loss: 0.05260483752936125
val_psnr: 23.06528674039951
```

Startup observation confirmed the memory gate during real training:

```text
observed_gpu_memory_used: about 90.4G
vram_floor_status: passed, above 80G
```

## Completion Status

The run completed cleanly.

```text
tmux session after finish: closed
training process after finish: none
GPU memory after finish: 0 MiB used
final_step: 1500
final_elapsed_minutes: 35.40261516968409
final_train_loss_total: 0.0390656404197216
finish line: training: finished step=1500 run_dir=/root/autodl-tmp/2DGauss/experiments/V1/runs/A_c03
```

Checkpoints present:

- `step_000250.pt`
- `step_000500.pt`
- `step_000750.pt`
- `step_001000.pt`
- `step_001250.pt`
- `step_001500.pt`
- `latest.pt`
- `final.pt`

There was no observed NaN, OOM, data-loader crash, checkpoint write failure, or
early stop.

## Training-Time Full-Image Validation

`A_c03` uses full-image internal validation during training:

```text
val_crop_size = 0
val_limit = 20
max_val_batches = 20
val_interval = 500
```

| Step | Val Loss | Val PSNR |
|---:|---:|---:|
| `500` | `0.02185784773901105` | `28.746070685664325` |
| `1000` | `0.021752151381224393` | `28.79998125115585` |
| `1500` | `0.022075795195996763` | `28.683033152940162` |

The full-image internal-validation selected checkpoint is:

```text
experiments/V1/runs/A_c03/checkpoints/step_001000.pt
```

## Post-Hoc Evaluation

Evaluated checkpoints:

- `step_001000.pt`: internal-val-selected checkpoint
- `final.pt`: final short-run checkpoint

Evaluation protocol:

- split: internal val and official test
- crop: full image, `crop_size=0`
- device: CUDA
- output range: clamped RGB `[0,1]`

| Checkpoint | Split | Samples | Loss | PSNR |
|---|---|---:|---:|---:|
| `step_001000.pt` | full internal val | `20` | `0.021752151381224393` | `28.79998125115585` |
| `final.pt` | full internal val | `20` | `0.022075795195996763` | `28.683033152940162` |
| `step_001000.pt` | official test | `300` | `0.03499144409783184` | `25.101847905535735` |
| `final.pt` | official test | `300` | `0.034676847060521446` | `25.124958465158436` |

## Comparison To Baselines

Current official `UHD_deblur` baseline:

```text
A03 final.pt official full-image test PSNR = 25.1402293880564
```

`A_c03` comparison:

| Run | Checkpoint | Official Test PSNR | Delta vs A03 Final |
|---|---|---:|---:|
| `A03` | `final.pt` | `25.1402293880564` | `0.0000` |
| `A_b03` | `step_000500.pt` | `25.116681468064225` | `-0.0235` |
| `A_c03` | `step_001000.pt` | `25.101847905535735` | `-0.0384` |
| `A_c03` | `final.pt` | `25.124958465158436` | `-0.0153` |

`A_c03` satisfies the new high-VRAM training gate, but it does not replace
`A03` as the official `UHD_deblur` baseline.

## Interpretation

Main conclusions:

1. The high-VRAM gate is feasible under `T01` on `UHD_deblur`: batch `25`
   reaches the required memory region without OOM.
2. Full-image internal validation is stable and usable during short training.
3. The internal-validation-selected checkpoint is `step_001000.pt`, but the
   official test still favors `final.pt`.
4. Neither `step_001000.pt` nor `final.pt` exceeds the `A03 final.pt` official
   baseline.

The useful project conclusion is therefore negative but informative:

```text
Simply raising batch size to satisfy the >=80G memory floor, while keeping the
A_a03/A_b03 capacity and full-image validation protocol, does not improve the
current UHD_deblur official baseline.
```

## Acceptance Decision

`V1/T01/A_c03` is accepted as a completed high-VRAM short-training experiment
and as a valid test of the `>=80G` deployment gate.

It is not accepted as a new official `UHD_deblur` baseline.

The current official baseline remains:

```text
V1/T01/A03 final.pt official full-image test PSNR = 25.1402 dB
```

## Next Steps

- Keep `A_c03` as the high-VRAM protocol result.
- Do not repeat the same configuration as another short run unless a new
  variable is introduced.
- The next useful experiment should change the training signal or architecture
  more directly, rather than only increasing batch size.
- `A_a03` still needs same-protocol official full-image test completion for
  `step_000500.pt` and `final.pt`.
