# Gate Check: V1/T01/A_f03

Status: launched; startup window passed through post-validation continuation.

## Training Duration Tier

- [x] Tier target: short training.
- [x] Planned wall-clock cap: `58` minutes.
- [x] Purpose: test whether the accepted `A_e03` edge-guided cache-safe branch
  was limited by the conservative `1500`-step cap rather than by the short-tier
  wall-clock budget.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_f03`.
- [x] Full run identifier: `V1/T01/A_f03`.
- [x] Run folder: `experiments/V1/runs/A_f03`.
- [x] Run folder is independent and does not overwrite `A_d03` or `A_e03`.
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

- [x] Config source: `experiments/V1/templates/T01/configs/a_f03_short_uhddeblur_edge_guided_extended_budget.toml`.
- [x] Model architecture intentionally matches the accepted `A_e03` settings.
- [x] Loss terms intentionally match the accepted `A_e03` settings.
- [x] Runtime hygiene intentionally matches `A_e03`: template-side
  `torch.cuda.empty_cache()` after validation plus launch env
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- [x] Main experimental variable from `A_e03`: `max_steps=1500 -> 2500`.
- [x] Pretrained weight path: empty; fresh initialization.
- [x] Resume checkpoint path: empty; not a resume.
- [x] Random seed: `20260511`.

## Runtime Check

- [x] Batch size target: `20`.
- [x] Gradient accumulation: `1`.
- [x] Max steps target: `2500`.
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

## Calibration Notes

Static checks:

```text
scripts/train_v1_t01.py: py_compile ok
scripts/smoke_calibrate_v1_t01.py: py_compile ok
scripts/evaluate_v1_t01.py: py_compile ok
template config full_id: V1/T01/A_f03
run-local config full_id: V1/T01/A_f03
batch_size: 20
max_steps: 2500
val_crop_size: 0
launch_tmux.sh: bash -n ok
```

Dataset count check:

```text
train/input_new images: 1964
train/gt_new images: 1964
test/input300 images: 300
test/gt300 images: 300
```

Accepted target calibration:

```text
batch_size=20: ok, max_memory_reserved_gb=94.126953125
avg_total_step_seconds=1.3712702706456183
estimated_steps_55min=2406
estimated_steps_60min=2625
```

Formal preflight result:

```text
preflight: ok
device: cuda
train_samples: 20
val_samples: 20
batch_input_shape: (20, 3, 512, 512)
batch_target_shape: (20, 3, 512, 512)
total_params: 231528
loss_total: 0.11356815695762634
loss_charbonnier: 0.1114201620221138
loss_gradient: 0.03171267360448837
loss_laplacian: 0.02811802737414837
val_batches: 20
val_loss: 0.07758434377610683
val_psnr: 20.31078866964017
```

## Launch Approval

```text
approved_at: 2026-05-11 02:09:10 CST
authorization_basis: user requested deployment and explicitly said to run both lines without waiting for further instruction
```

## Launch And Startup

```text
session: 2dgauss_a_f03_uhddeblur_edge_guided_step2500
log: experiments/V1/runs/A_f03/logs/train_20260511_022049.log
launch_env: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Startup-window observations:

```text
step 1 loss_total=0.1244623
step 10 loss_total=0.0545635
step 20 loss_total=0.0510086
step 250 checkpoint written: step_000250.pt and latest.pt
step 500 full-image validation: val_loss=0.0228021, val_psnr=28.5946
step 500 checkpoint written: step_000500.pt and latest.pt
post-validation continuation confirmed through step 550
observed GPU memory: about 97.1GB used
status: startup window passed; extended-budget edge-guided branch crossed the historical A_d03 failure point and is continuing
```
