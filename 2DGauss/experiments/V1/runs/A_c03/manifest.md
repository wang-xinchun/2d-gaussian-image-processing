# Run Manifest: V1/T01/A_c03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_c03`
- Full identifier: `V1/T01/A_c03`

## Current Status

Deployment gate passed after the user added the new VRAM floor:

```text
training VRAM usage must not be below 80G
```

## Purpose

`A_c03` is the high-VRAM replacement for `A_b03`.

`A_b03` completed cleanly and validated the full-image internal validation
protocol, but its training GPU memory usage was about `62.8G`, below the user's
new `80G` floor. Therefore `A_b03` is archived as a protocol probe, while
`A_c03` is the new short-training deployment that must satisfy the memory gate.

Deployment was paused by the user on `2026-05-08 00:22:37 CST` before formal
preflight or launch. The user reauthorized deployment on
`2026-05-08 01:34:55 CST`; the run then completed static checks and formal
preflight and is launch-approved.

## Protocol

Kept from `A_b03`:

- dataset: `UHD_deblur`;
- model capacity: `A_a03/A_b03` settings;
- training crop: random `512x512`;
- internal validation: full-image `crop_size=0`;
- validation limit: first `20` train pairs;
- validation interval: `500`;
- checkpoint interval: `250`;
- official test reserved for post-training evaluation.

Changed from `A_b03`:

- `batch_size=16` -> `batch_size=25`;
- `max_steps=2000` -> `max_steps=1500`, so the higher-memory run stays within
  the short-training tier;
- `seed=20260508`.

## VRAM Calibration

Rejected:

```text
batch_size=24
status=ok
max_memory_reserved_gb=78.51953125
reason=rejected because it is below 80G
```

Accepted target:

```text
batch_size=25
status=ok
max_memory_reserved_gb=81.6640625
avg_total_step_seconds=2.2052947686985136
estimated_steps_60min=1632
```

## Files

- Template config: `experiments/V1/templates/T01/configs/a_c03_short_uhddeblur_fullval_vram80.toml`
- Run-local config: `experiments/V1/runs/A_c03/configs/a_c03_short_uhddeblur_fullval_vram80.toml`
- Gate check: `experiments/V1/runs/A_c03/gate_check.md`
- Manifest: `experiments/V1/runs/A_c03/manifest.md`

## Required Before Launch

1. Syntax-check scripts and TOML identity.
2. Confirm the accepted smoke calibration is recorded in the gate.
3. Run formal preflight with `batch_size=25`.
4. Create and check launch script.
5. Launch only if all gate items pass.

Current non-deployment state:

```text
tmux session: not launched
training process: none
metrics: none expected
checkpoints: none expected
GPU at pause check: 0 MiB used
```

## Gate Completion

Static checks:

```text
scripts/train_v1_t01.py: py_compile ok
scripts/smoke_calibrate_v1_t01.py: py_compile ok
scripts/evaluate_v1_t01.py: py_compile ok
template config full_id: V1/T01/A_c03
run-local config full_id: V1/T01/A_c03
batch_size: 25
max_steps: 1500
val_crop_size: 0
val_limit: 20
launch_tmux.sh: bash -n ok
```

Formal preflight:

```text
preflight: ok
device: cuda
train_samples: 25
val_samples: 20
batch_input_shape: (25, 3, 512, 512)
batch_target_shape: (25, 3, 512, 512)
total_params: 189717
loss_total: 0.07665728032588959
loss_charbonnier: 0.07498067617416382
loss_gradient: 0.03353201970458031
val_batches: 20
val_loss: 0.05260483752936125
val_psnr: 23.06528674039951
```

## Launch Command

```bash
bash experiments/V1/runs/A_c03/launch_tmux.sh
```

## Launch Status

Launched in tmux session `2dgauss_a_c03_uhddeblur_fullval_vram80` on
`2026-05-08 01:37:03 CST`.

```text
session: 2dgauss_a_c03_uhddeblur_fullval_vram80
train log: experiments/V1/runs/A_c03/logs/train_20260508_013703.log
config: experiments/V1/templates/T01/configs/a_c03_short_uhddeblur_fullval_vram80.toml
run_dir: experiments/V1/runs/A_c03
```

Startup-window observation through the first full-image validation:

```text
step 1: elapsed 0.2165 min, loss_total 0.08378
step 100: elapsed 2.4676 min, loss_total 0.04157
step 250: elapsed 6.0317 min, loss_total 0.04271
step 250 checkpoints: step_000250.pt, latest.pt
step 500: elapsed 11.7782 min, loss_total 0.04071
step 500 full-image validation: val_batches 20, val_loss 0.0218578, val_psnr 28.7461
step 500 checkpoint: step_000500.pt
latest observed train step: 510
tmux status: online
observed_gpu_memory_used: about 90.4 GiB
vram_floor_status: passed, above 80G
```

Startup verdict:

```text
passed
```

## Final Runtime Status

`A_c03` finished by reaching `max_steps=1500`.

```text
final_step: 1500
final_elapsed_minutes: 35.40261516968409
final_train_loss_total: 0.0390656404197216
final_validation_step: 1500
final_val_batches: 20
final_val_loss: 0.022075795195996763
final_val_psnr: 28.683033152940162
tmux_session_after_finish: closed
training_process_after_finish: none
gpu_memory_after_finish: 0 MiB used
```

## Acceptance Evaluation Summary

Training/runtime acceptance passed. The run satisfied the `>=80G` memory gate
during startup observation and completed without OOM, NaN, checkpoint write
failure, or data-loader failure.

Full-image internal validation selected `step_001000.pt`:

```text
step_001000.pt full-val: loss_mean=0.021752151381224393, psnr_mean=28.79998125115585
final.pt full-val: loss_mean=0.022075795195996763, psnr_mean=28.683033152940162
```

Official full-image test results:

```text
step_001000.pt official test: loss_mean=0.03499144409783184, psnr_mean=25.101847905535735
final.pt official test: loss_mean=0.034676847060521446, psnr_mean=25.124958465158436
```

Conclusion:

```text
A_c03 is accepted as a completed high-VRAM short-training experiment, but it
does not replace A03 as the official UHD_deblur baseline.
```

Result analysis:

```text
experiments/V1/runs/A_c03/result_analysis.md
```
