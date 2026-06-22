# Run Manifest: V1/T01/A02

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A02`
- Full identifier: `V1/T01/A02`

## Purpose

First cross-task short-training probe for the `V1/T01` template after A01. The goal is to test whether the same paired-restoration model and training code transfer from low-light enhancement to the newly accepted `UHD_haze` UHD dehazing dataset.

## Dataset

- Dataset: `UHD_haze`
- Train input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze/train/input`
- Train target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze/train/gt`
- Internal validation input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze/train/input`
- Internal validation target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze/train/gt`
- Internal validation selection: first `100` paired samples from `train`
- Effective training selection: remaining `2190` paired samples from `train`
- Final test input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze/test/input`
- Final test target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze/test/gt`
- Local final test口径: `231` pairs
- Dataset acceptance reference: `docs/unpacked_dataset_acceptance_20260425.md`
- Strategy reference: `docs/next_experiment_dataset_strategy_20260425.md`

## Planned Runtime

- Tier: short
- Duration limit: `58` minutes
- Training crop: `512x512` random patches
- Validation crop: `512x512` center crops
- Validation batches: `100`
- Batch size: `56`
- Max steps: `3400`
- Validation interval: `500`
- Checkpoint interval: `500`

## Files

- Gate check: `gate_check.md`
- Launch config: `../../templates/T01/configs/a02_short_uhdhaze_patch.toml`
- Training script: `scripts/train_v1_t01.py`
- Archived template snapshot: `template_snapshot/`

## Smoke Calibration

Smoke calibration on real `UHD_haze` batches with `512x512` crops:

```text
batch 48: ok, reserved 77.502G, allocated 73.272G, avg_step_seconds 0.7862
batch 56: ok, reserved 90.418G, allocated 85.480G, avg_step_seconds 0.9131
batch 60: ok but too close to the upper bound, reserved 93.127G, allocated 91.584G, avg_step_seconds 0.9814
```

Chosen deployment setting:

```text
batch_size: 56
train_crop_size: 512
max_steps: 3400
wall_clock_minutes: 58
val_interval: 500
checkpoint_interval: 500
max_val_batches: 100
```

## Preflight

Formal preflight succeeded with the final A02 configuration:

```text
device: cuda
train_samples: 56
val_samples: 100
batch_input_shape: (56, 3, 512, 512)
batch_target_shape: (56, 3, 512, 512)
total_params: 70223
loss_total: 0.3404560089111328
loss_charbonnier: 0.3385476768016815
loss_gradient: 0.038166455924510956
val_batches: 100
val_loss: 0.29253974989056586
val_psnr: 9.839995992396965
```

## Launch Command

```bash
bash experiments/V1/runs/A02/launch_tmux.sh
```

## Launch Status

Launched in tmux session `2dgauss_a02_uhdhaze` on `2026-04-25 17:44:43 CST`.

Startup-window observation:

```text
step 1: elapsed 0.4214 min, loss_total 0.27136
step 10: elapsed 1.9732 min, loss_total 0.19529
step 20: elapsed 3.7640 min, loss_total 0.17042
step 30: elapsed 5.2063 min, loss_total 0.18465
gpu_memory_used: about 93295 MiB
```

Decision:

```text
A02 was stopped during the startup window.
```

Reason:

```text
The run was stable and did not OOM, but real training step time was dominated by
4K JPEG decoding / DataLoader throughput. With val_interval=500 and a 58 minute
wall-clock cap, the run was unlikely to reach the first planned validation and
checkpoint. The original smoke calibration measured GPU compute time only and
underestimated end-to-end training time.
```

Stopped state:

```text
tmux session: closed
checkpoint: none produced
train log: experiments/V1/runs/A02/logs/train_20260425_174443.log
next action: deploy a derivative A02 run with data-inclusive smoke calibration
and adjusted batch/worker/interval settings.
```
