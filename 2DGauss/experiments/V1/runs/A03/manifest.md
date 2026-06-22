# Run Manifest: V1/T01/A03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A03`
- Full identifier: `V1/T01/A03`

## Current Status

Launched and healthy in the startup window.

## Purpose

`A03` is the first planned `UHD_deblur` short probe under the current `V1/T01` line.

Its role is not to chase a paper-level absolute number on day one. The immediate goal is:

1. establish a clean, reproducible `UHD_deblur` baseline;
2. verify that `V1/T01` transfers from low-light restoration and dehazing to UHD deblurring;
3. create the baseline against which later `UHD_deblur` branches can target `+0.5 dB` same-protocol improvement.

This goal is aligned with `docs/stage_metric_targets_20260505.md`.

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

`UHD_deblur/test/gt300` contains a few non-image metadata files such as `.baiduyun.uploading.cfg` and `.DS_Store`. The current dataloader filters by image extension, so the effective evaluation count remains `300` image pairs. This must stay explicit in future preflight, evaluation, and reporting records.

## Planned Runtime

The current design intentionally mirrors the conservative, task-transfer-oriented style of `A_a02`, but the runtime envelope has now been refined by `UHD_deblur`-specific data-inclusive smoke calibration.

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

## Data-Inclusive Smoke Calibration

Smoke calibration on `UHD_deblur` with real data-fetch timing produced:

```text
batch 16, workers 8: ok, avg_total_step_seconds 3.0507, fetch 2.5968, compute 0.4540, reserved 25.844G, estimated_steps_60min 1180
batch 24, workers 8: ok, avg_total_step_seconds 4.2877, fetch 3.7583, compute 0.5294, reserved 38.752G, estimated_steps_60min 839
batch 32, workers 8: ok, avg_total_step_seconds 5.4251, fetch 4.7607, compute 0.6644, reserved 51.668G, estimated_steps_60min 663
batch 16, workers 12: ok, avg_total_step_seconds 2.7158, fetch 2.3084, compute 0.4074, reserved 25.844G, estimated_steps_60min 1325
batch 16, workers 16: ok, avg_total_step_seconds 2.6711, fetch 2.2521, compute 0.4190, reserved 25.844G, estimated_steps_60min 1347
```

Chosen deployment setting after smoke:

```text
batch_size: 16
num_workers: 16
max_steps: 2000
wall_clock_minutes: 58
val_interval: 250
checkpoint_interval: 250
```

## Why These Initial Targets

These values are now the selected post-smoke runtime targets, but they are still not launch-approved until formal preflight succeeds.

They were chosen because:

1. `UHD_deblur` is the same broad 4K JPEG paired-restoration regime as `UHD_haze`;
2. the original `A02` lesson showed that GPU-only memory fitting is not enough, and end-to-end data throughput must be checked explicitly;
3. `A_a02` proved that `batch_size=16`, `512` patch training, and `250`-step validation/checkpoint cadence are a safe conservative starting envelope for a UHD short probe.
4. `UHD_deblur`-specific smoke calibration showed that `batch_size=16` remains the best short-run choice and that increasing workers to `16` improves end-to-end throughput over `8` and `12`.

Before launch, `A03` had to pass its own formal preflight. That requirement is now satisfied.

## Preflight

Formal preflight succeeded with the final selected `A03` configuration:

```text
device: cuda
train_samples: 16
val_samples: 100
batch_input_shape: (16, 3, 512, 512)
batch_target_shape: (16, 3, 512, 512)
total_params: 70223
loss_total: 0.10284867137670517
loss_charbonnier: 0.10126832127571106
loss_gradient: 0.031606972217559814
val_batches: 100
val_loss: 0.057967538982629775
val_psnr: 21.87167276766361
```

## Planned Comparison Logic

If deployed later, `A03` should be interpreted against the current stage targets as follows:

- First success condition: the run reaches a valid train/validation/checkpoint loop within the short-training cap.
- Second success condition: it produces a clean `UHD_deblur` baseline that can be post-hoc evaluated on the official `300`-pair test split.
- Third success condition: the resulting baseline is stable enough to justify a later branch run whose target is same-protocol `PSNR +0.5 dB`.

## Files

- Gate check: `gate_check.md`
- Design config in template: `../../templates/T01/configs/a03_short_uhddeblur_patch.toml`
- Run-local config copy: `configs/a03_short_uhddeblur_patch.toml`
- Training script expected at launch time: `scripts/train_v1_t01.py`
- Planned launch script: `launch_tmux.sh`

## Launch Command

```bash
bash experiments/V1/runs/A03/launch_tmux.sh
```

## Launch Status

Launched in tmux session `2dgauss_a03_uhddeblur` on `2026-05-06 01:16:51 CST`.

Startup-window observation through the first validation/checkpoint:

```text
step 1: elapsed 0.1699 min, loss_total 0.10514
step 10: elapsed 0.2019 min, loss_total 0.04293
step 100: elapsed 0.9005 min, loss_total 0.03186
step 200: elapsed 1.7490 min, loss_total 0.03398
step 250: elapsed 2.2487 min, loss_total 0.04022
step 250 validation: val_batches 100, val_loss 0.0256215, val_psnr 28.8894
checkpoints_present: step_000250.pt, latest.pt
gpu_memory_used: about 27171 MiB
```

Current observed status at `2026-05-06 01:20:50 CST`:

```text
tmux session: online
training process: running
train log: experiments/V1/runs/A03/logs/train_20260506_011651.log
latest observed train step: 440
latest observed train elapsed_minutes: 3.9181
latest observed train loss_total: 0.0449156
status: startup window passed; train/validation/checkpoint loop confirmed
```

## Remaining Runtime Goal

The run is now past the startup gate. The next runtime milestones are:

1. continue toward later periodic validation/checkpoint points;
2. finish the short run at `max_steps=2000` or the `58` minute wall-clock cap;
3. perform post-hoc official-test evaluation after training.

## Final Runtime Status

`A03` finished by reaching `max_steps=2000`, not by the `58` minute wall-clock cap.

```text
final_step: 2000
final_elapsed_minutes: 17.7587184270223
final_train_loss_total: 0.0343700647354126
final_validation_step: 2000
final_val_batches: 100
final_val_loss: 0.025943749761208893
final_val_psnr: 28.296640982009375
final_checkpoint: experiments/V1/runs/A03/checkpoints/final.pt
latest_checkpoint: experiments/V1/runs/A03/checkpoints/latest.pt
last_periodic_checkpoint: experiments/V1/runs/A03/checkpoints/step_002000.pt
tmux_session_after_finish: closed
```

## Post-Hoc Full Evaluation

Post-hoc full evaluation was completed on the full internal validation split and
the official `300`-pair test split.

```text
step_000500.pt full-val: loss_mean=0.025427820812910795, psnr_mean=29.00563267209822
step_000500.pt official-test: loss_mean=0.035097659269037344, psnr_mean=25.097786303353434
final.pt full-val: loss_mean=0.025943749761208893, psnr_mean=28.296640982009375
final.pt official-test: loss_mean=0.03511191223127146, psnr_mean=25.1402293880564
result_analysis: experiments/V1/runs/A03/result_analysis.md
```
