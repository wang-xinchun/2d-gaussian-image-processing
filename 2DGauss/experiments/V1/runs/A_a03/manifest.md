# Run Manifest: V1/T01/A_a03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_a03`
- Full identifier: `V1/T01/A_a03`

## Current Status

Deployment gate passed; launch prepared.

This run remains a distinct branch from `A03`:

```text
run: A_a03
full_id: V1/T01/A_a03
run_dir: experiments/V1/runs/A_a03
```

It must not overwrite `experiments/V1/runs/A03`.

## Purpose

`A_a03` is the first planned `UHD_deblur` branch after the accepted `A03`
baseline.

`A03` established the first official `UHD_deblur` baseline under the current
line:

```text
A03 final.pt official-test PSNR = 25.1402 dB
A03 step_000500.pt full-val PSNR = 29.0056 dB
```

The purpose of `A_a03` is to test whether a modest increase in model capacity
and Gaussian expression can improve over that baseline while keeping the dataset,
split policy, crop policy, loss mix, optimizer, validation cadence, checkpoint
cadence, and final evaluation protocol comparable.

This is a branch experiment, not a new template and not a deployment yet.

## Experimental Question

Can `V1/T01` improve the `UHD_deblur` baseline by giving the cross-resolution
Gaussian pathway more representational capacity, without changing the training
template or contaminating the evaluation protocol?

The targeted mechanism is:

1. wider feature channels for stronger local restoration features;
2. deeper high-resolution, low-resolution, and decoder residual stacks;
3. more Gaussian atoms per low-resolution cell;
4. slightly larger affine rank for channel-wise modulation;
5. slightly broader texture carrier range for deblurring edges and motion detail.

## Baseline For Comparison

Baseline run:

- `V1/T01/A03`
- Dataset: `UHD_deblur`
- Training split: `1864` train pairs after reserving first `100` train pairs for internal validation
- Internal validation split: first `100` train pairs
- Official test split: `300` pairs from `test/input300 -> test/gt300`
- Official baseline checkpoint: `experiments/V1/runs/A03/checkpoints/final.pt`
- Official baseline PSNR: `25.1402 dB`

For checkpoint-selection analysis, `A03` also records:

- full-val best checkpoint: `step_000500.pt`
- full-val PSNR: `29.0056 dB`
- `step_000500.pt` official-test PSNR: `25.0978 dB`

This mismatch means `A_a03` must keep the checkpoint-selection policy explicit.

## Checkpoint-Selection Policy

Training-time model selection for `A_a03` should use only the internal
validation split.

Planned policy:

1. During training, record validation on the fixed internal validation split
   every `250` steps with `max_val_batches=100`.
2. After training, evaluate the internal-validation best checkpoint and
   `final.pt` on:
   - full internal validation set;
   - official `300`-pair test split.
3. Record official test numbers for both checkpoints.
4. For deployment/reporting of this branch, compare against `A03` using the same
   rule that is written in the analysis document:
   - primary honest selection: internal-val-selected checkpoint;
   - secondary diagnostic record: `final.pt` official-test result.

Official test must remain post-hoc and must not be used to choose hyperparameters
inside this branch.

## Dataset

- Dataset: `UHD_deblur`
- Train input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/train/input_new`
- Train target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/train/gt_new`
- Internal validation input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/train/input_new`
- Internal validation target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/train/gt_new`
- Internal validation selection: first `100` paired samples from `train`
- Effective training selection: remaining `1864` paired samples from `train`
- Final test input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/test/input300`
- Final test target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/test/gt300`
- Official final test image-pair count: `300`
- Dataset acceptance reference: `docs/unpacked_dataset_acceptance_20260425.md`
- Strategy reference: `docs/next_experiment_dataset_strategy_20260425.md`
- Metric-target reference: `docs/stage_metric_targets_20260505.md`

Important note:

`UHD_deblur/test/gt300` contains extra non-image metadata files such as
`.baiduyun.uploading.cfg` and `.DS_Store`. The current dataloader filters by
image extension, so the effective evaluation count remains `300` image pairs.

## Planned Architecture Delta From A03

Only the model capacity and Gaussian-expression settings change:

| Field | `A03` | Planned `A_a03` | Reason |
|---|---:|---:|---|
| `width` | `32` | `48` | More restoration feature capacity while still lightweight |
| `highres_blocks` | `2` | `3` | Slightly stronger high-resolution detail container |
| `lowres_blocks` | `4` | `6` | Stronger compact low-resolution modeling |
| `decoder_blocks` | `2` | `3` | More capacity after guided feature fusion |
| `affine_rank` | `4` | `6` | More flexible low-rank channel modulation |
| `gaussian_atoms` | `4` | `6` | More local Gaussian primitives per low-res cell |
| `max_frequency` | `4.0` | `6.0` | Broader texture carrier range for edge/motion detail |
| `max_texture_gain` | `1.0` | `1.2` | Slightly stronger texture-aware modulation |

Unchanged:

- `lowres_scale=4`
- `gaussian_param_dim=10`
- loss mix
- optimizer
- dataset split
- train/validation crop policy
- validation and checkpoint intervals
- short-run duration tier

## Planned Runtime

These are design targets only and are not launch-approved:

- Tier target: short
- Duration limit target: `58` minutes
- Max steps target: `2000`
- Batch size target: `16`
- DataLoader workers target: `16`
- Training crop target: `512x512` random patches
- Validation crop target: `512x512` center crops
- Validation interval target: `250`
- Checkpoint interval target: `250`
- Validation batches target: `100`
- AMP target: disabled

The design intentionally reuses the `A03` runtime envelope as the first proposal.
The larger model has now passed its own data-inclusive smoke calibration and
formal preflight.

## Data-Inclusive Smoke Calibration

Smoke calibration on `UHD_deblur` with the final `A_a03` capacity setting:

```text
batch 16, workers 16, steps 5: ok
avg_total_step_seconds: 2.233819863572717
avg_fetch_seconds: 1.7087661772966385
avg_compute_seconds: 0.5250503579154611
max_memory_reserved_gb: 51.759765625
estimated_steps_60min: 1611

batch 16, workers 16, steps 20: ok
avg_total_step_seconds: 0.9746505129616707
avg_fetch_seconds: 0.5232208701316268
avg_compute_seconds: 0.4514271753374487
max_memory_reserved_gb: 51.759765625
estimated_steps_60min: 3693
```

The 20-step smoke is treated as more representative after DataLoader warm-up.
It supports keeping `max_steps=2000` with `wall_clock_minutes=58`.

## Preflight

Formal preflight succeeded with the final selected `A_a03` configuration:

```text
device: cuda
train_samples: 16
val_samples: 100
batch_input_shape: (16, 3, 512, 512)
batch_target_shape: (16, 3, 512, 512)
total_params: 189717
loss_total: 0.19018182158470154
loss_charbonnier: 0.18850398063659668
loss_gradient: 0.033556777983903885
val_batches: 100
val_loss: 0.10965484395623207
val_psnr: 17.758774463709766
```

## Success Criteria

Minimum success:

- smoke calibration and preflight pass in a future deployment-prep turn;
- training reaches at least one validation/checkpoint milestone if launched;
- no NaN, OOM, or data-loader failure.

Baseline success:

- completes the short run and produces full internal validation plus official
  test post-hoc results for the internal-val-selected checkpoint and `final.pt`.

Metric success:

- same-protocol official-test PSNR improves over `A03 final.pt` by at least
  `+0.3 dB`;
- strong target is `+0.5 dB`, consistent with `docs/stage_metric_targets_20260505.md`.

Checkpoint-policy success:

- the branch report explicitly distinguishes internal-val-selected performance
  from post-hoc official-test diagnostics.

## Files

- Template config: `experiments/V1/templates/T01/configs/a_a03_short_uhddeblur_capacity.toml`
- Run-local config: `experiments/V1/runs/A_a03/configs/a_a03_short_uhddeblur_capacity.toml`
- Gate check: `experiments/V1/runs/A_a03/gate_check.md`
- This manifest: `experiments/V1/runs/A_a03/manifest.md`
- Design document: `docs/next_experiment_design_20260507.md`

## Deployment Preparation Record

The following actions were intentionally not performed in the original design
turn:

- no `python3 scripts/smoke_calibrate_v1_t01.py ...`;
- no `python3 scripts/train_v1_t01.py --preflight ...`;
- no `launch_tmux.sh` creation;
- no tmux session;
- no training process;
- no checkpoint or metric generation.

After the user later requested deployment, the following launch-prep steps were
completed:

1. config and script syntax checks;
2. data-inclusive smoke calibration for the `A_a03` capacity setting;
3. formal preflight;
4. gate and manifest update with measured results;
5. version-specific launch script preparation.

## Launch Command

```bash
bash experiments/V1/runs/A_a03/launch_tmux.sh
```

## Launch Status

Launched in tmux session `2dgauss_a_a03_uhddeblur_capacity` on
`2026-05-07 01:01:06 CST`.

Important launch note:

An earlier launch attempt at `2026-05-07 01:00:03 CST` exited before training
because the script used the run-local config path as the launch config. The
training entrypoint then attempted to copy that config into the same
`runs/A_a03/configs/` path and raised `SameFileError`. That attempt produced the
audit log `experiments/V1/runs/A_a03/logs/train_20260507_010003.log`, but no
training metrics or checkpoints. The launch script was corrected to use the
template config:

```text
experiments/V1/templates/T01/configs/a_a03_short_uhddeblur_capacity.toml
```

The corrected launch is the active run:

```text
session: 2dgauss_a_a03_uhddeblur_capacity
train log: experiments/V1/runs/A_a03/logs/train_20260507_010106.log
config: experiments/V1/templates/T01/configs/a_a03_short_uhddeblur_capacity.toml
run_dir: experiments/V1/runs/A_a03
```

Startup-window observation through the first validation/checkpoint:

```text
step 1: elapsed 0.1876 min, loss_total 0.19248
step 10: elapsed 0.2442 min, loss_total 0.05475
step 100: elapsed 1.0057 min, loss_total 0.04290
step 200: elapsed 1.9770 min, loss_total 0.04245
step 250: elapsed 2.5247 min, loss_total 0.04809
step 250 validation: val_batches 100, val_loss 0.0272292, val_psnr 28.5685
checkpoints_present: step_000250.pt, latest.pt
gpu_memory_used: about 53709 MiB
```

Current observed status at `2026-05-07 01:04:20 CST`:

```text
tmux session: online
training process: running
latest observed train step: 340
latest observed train elapsed_minutes: 3.3307
latest observed train loss_total: 0.0355037
status: startup window passed; train/validation/checkpoint loop confirmed
```

## Final Runtime Status

`A_a03` finished by reaching `max_steps=2000`, not by the `58` minute wall-clock
cap.

```text
final_step: 2000
final_elapsed_minutes: 20.225026671091715
final_train_loss_total: 0.023953847587108612
final_validation_step: 2000
final_val_batches: 100
final_val_loss: 0.024598940536379815
final_val_psnr: 28.586646286003294
final_checkpoint: experiments/V1/runs/A_a03/checkpoints/final.pt
latest_checkpoint: experiments/V1/runs/A_a03/checkpoints/latest.pt
last_periodic_checkpoint: experiments/V1/runs/A_a03/checkpoints/step_002000.pt
tmux_session_after_finish: closed
```

## Acceptance Evaluation Summary

Training/runtime acceptance passed.

Post-hoc internal validation and diagnostic `512`-crop test evaluation completed:

```text
step_000500.pt full-val: loss_mean=0.025281552337110044, psnr_mean=29.086066465895943
step_000500.pt test crop512: loss_mean=0.043986560571938754, psnr_mean=23.939908160431653
final.pt full-val: loss_mean=0.024598940536379815, psnr_mean=28.586646286003294
final.pt test crop512: loss_mean=0.043345063224745295, psnr_mean=24.016313494805658
result_analysis: experiments/V1/runs/A_a03/result_analysis.md
```

Official full-image `crop_size=0` test evaluation was later completed under the
same protocol as `A03/A_c03/A_e03`:

```text
step_000500.pt official test: loss_mean=0.03507758348248899, psnr_mean=25.12160752986791
final.pt official test: loss_mean=0.03460686403947572, psnr_mean=25.15848471495631
```

Comparison:

```text
A03 final.pt official full-image test PSNR = 25.1402293880564
A_e03 final.pt official full-image test PSNR = 25.21410079036176
A_a03 final.pt delta vs A03 final.pt = +0.018255326899907942
A_a03 final.pt delta vs A_e03 final.pt = -0.05561607540544955
```

Current official `UHD_deblur` baseline therefore remains:

```text
A_e03 final.pt official full-image test PSNR = 25.2141 dB
```
