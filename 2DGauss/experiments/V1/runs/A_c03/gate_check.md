# Gate Check: V1/T01/A_c03

Status: deployment gate passed; approved for launch.

## Training Duration Tier

- [x] Tier target: short training.
- [x] Planned wall-clock cap: `58` minutes.
- [x] Purpose: rerun the full-image-validation `UHD_deblur` short branch under
  the user's new VRAM floor constraint: training memory must not be below `80G`.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_c03`.
- [x] Full run identifier: `V1/T01/A_c03`.
- [x] Run folder: `experiments/V1/runs/A_c03`.
- [x] Run folder is independent and does not overwrite `A03`, `A_a03`, or `A_b03`.
- [x] Exact training-template snapshot is configured to be saved under
  `template_snapshot/` at launch.

## Dataset Check

- [x] Dataset name: `UHD_deblur`.
- [x] Dataset root: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur`.
- [x] Train input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/train/input_new`.
- [x] Train target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/train/gt_new`.
- [x] Validation input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/train/input_new`.
- [x] Validation target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/train/gt_new`.
- [x] Validation selection: first `20` train pairs, deterministic filename order.
- [x] Effective training selection: train pairs after the first `100`.
- [x] Official test input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/test/input300`.
- [x] Official test target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/test/gt300`.
- [x] Official test is reserved for post-training evaluation only.
- [x] Training crop policy: random `512x512` patches.
- [x] Validation crop policy: full image, `crop_size=0`.

## Contamination Check

- [x] Config source: `experiments/V1/templates/T01/configs/a_c03_short_uhddeblur_fullval_vram80.toml`.
- [x] Model architecture intentionally matches `A_a03/A_b03` capacity settings.
- [x] Protocol intentionally matches `A_b03` full-image internal validation.
- [x] Main runtime change from `A_b03`: `batch_size=25` to satisfy the new
  `>=80G` VRAM floor.
- [x] Loss terms remain Charbonnier weight `1.0` and gradient loss weight `0.05`.
- [x] Optimizer remains AdamW, lr `0.0002`, weight decay `0.0001`.
- [x] Pretrained weight path: empty; fresh initialization.
- [x] Resume checkpoint path: empty; not a resume.
- [x] Random seed: `20260508`.

## Runtime Check

- [x] Batch size target: `25`.
- [x] Gradient accumulation: `1`.
- [x] Max steps target: `1500`.
- [x] Planned duration cap: `58` minutes.
- [x] Training crop: `512x512`.
- [x] Validation crop: full image, `crop_size=0`.
- [x] Validation interval: every `500` steps.
- [x] Checkpoint interval: every `250` steps.
- [x] Validation batches: `20`.
- [x] DataLoader workers: `8`.
- [x] Device plan: `auto`, CUDA if available.
- [x] AMP: disabled.
- [x] VRAM floor: training `max_memory_reserved_gb` must be at least `80`.
- [x] Syntax check completed.
- [x] Data-inclusive training smoke completed and reserved VRAM is at least `80G`.
- [x] Formal preflight completed.
- [x] Launch script checked.
- [x] Run approved to launch.
- [x] User explicitly paused deployment before launch on `2026-05-08 00:22:37 CST`.
- [x] User explicitly reauthorized deployment on `2026-05-08 01:34:55 CST`.

## Calibration Notes

Rejected calibration:

```text
batch_size=24: ok, max_memory_reserved_gb=78.51953125
reason: below the new 80G floor
```

Accepted target calibration:

```text
batch_size=25: ok, max_memory_reserved_gb=81.6640625
avg_total_step_seconds=2.2052947686985136
estimated_steps_60min=1632
```

No `A_c03` training tmux session has been launched. No `A_c03` metrics or
checkpoints should exist unless the user later explicitly reauthorizes
deployment.

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

Dataset count check:

```text
train/input_new images: 1964
train/gt_new images: 1964
test/input300 images: 300
test/gt300 images: 300
```

Formal preflight result:

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

Launch approval:

```text
approved_at: 2026-05-08 01:35:57 CST
```
