# Gate Check: V1/T01/A_d03

Status: launched; aborted after step 500 due post-validation CUDA OOM.

## Training Duration Tier

- [x] Tier target: short training.
- [x] Planned wall-clock cap: `58` minutes.
- [x] Purpose: test a new high-resolution edge-guided detail recovery mechanism
  after `A_c03` showed that simply raising batch size did not improve the
  official `UHD_deblur` baseline.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_d03`.
- [x] Full run identifier: `V1/T01/A_d03`.
- [x] Run folder: `experiments/V1/runs/A_d03`.
- [x] Run folder is independent and does not overwrite `A03`, `A_a03`,
  `A_b03`, or `A_c03`.
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

## Mechanism Check

- [x] Baseline protocol: inherits `A_c03` high-VRAM full-image validation setup.
- [x] New architecture mechanism: high-resolution edge-guided detail refiner.
- [x] Edge descriptors: grayscale, Sobel x/y magnitude, gradient magnitude, and
  Laplacian/local-highpass magnitude.
- [x] Detail injection: the refiner fuses decoder features, edge features, and
  Gaussian attention, then adds a bounded RGB refinement residual.
- [x] New training signal: optional Laplacian L1 consistency loss enabled at
  weight `0.02`.
- [x] Old configs remain compatible because the edge refiner and Laplacian loss
  are disabled by default unless explicitly enabled.

## Runtime Check

- [x] Initial batch size target: `24`, rejected after smoke OOM.
- [x] Final batch size target: `20`.
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
- [x] Data-inclusive smoke calibration completed.
- [x] Training reserved VRAM is at least `80G`.
- [x] Formal preflight completed.
- [x] Launch script checked.
- [x] Explicit deployment command received.

Launch authorization:

```text
authorized_at: 2026-05-09 00:00 UTC
user_request: 请你针对问题，设计并部署实验。注意门槛和模板
```

## Calibration Notes

Rejected calibrations:

```text
batch_size=24: oom, max_memory_reserved_gb=94.2910, steps_completed=0
batch_size=23: oom, max_memory_reserved_gb=93.7734, steps_completed=0
batch_size=22: oom, max_memory_reserved_gb=92.9238, steps_completed=0
batch_size=21: oom, max_memory_reserved_gb=93.8066, steps_completed=0
```

Accepted calibration:

```text
batch_size=20
status=ok
steps=20
max_memory_allocated_gb=90.9545
max_memory_reserved_gb=94.1211
avg_total_step_seconds=1.3975
estimated_steps_60min=2575
```

The batch-20 package satisfies the user's `>=80G` training-memory floor while
leaving enough runtime margin for the planned `1500`-step short run.

## Static Checks

```text
scripts/train_v1_t01.py: py_compile ok
scripts/smoke_calibrate_v1_t01.py: py_compile ok
scripts/evaluate_v1_t01.py: py_compile ok
template config full_id: V1/T01/A_d03
run-local config full_id: V1/T01/A_d03
use_edge_guidance: true
batch_size: 20
laplacian_weight: 0.02
launch_tmux.sh: bash -n ok
legacy A_c03 checkpoint compatibility: strict load ok
```

Dataset count check:

```text
train/input_new images: 1964
train/gt_new images: 1964
test/input300 images: 300
test/gt300 images: 300
```

Formal preflight:

```text
preflight: ok
device: cuda
train_samples: 20
val_samples: 20
batch_input_shape: (20, 3, 512, 512)
batch_target_shape: (20, 3, 512, 512)
total_params: 231528
loss_total: 0.09860626608133316
loss_charbonnier: 0.09657278656959534
loss_gradient: 0.029611244797706604
loss_laplacian: 0.027645735070109367
val_batches: 20
val_loss: 0.048894323967397214
val_psnr: 23.718956728936703
```

## Required Before Launch

1. Launch with `experiments/V1/runs/A_d03/launch_tmux.sh`.
2. Confirm startup window through at least the first checkpoint and preferably
   the first full-image validation.

## Deployment Outcome

```text
session: 2dgauss_a_d03_uhddeblur_edge_guided_vram80
log: experiments/V1/runs/A_d03/logs/train_20260509_011214.log
step_000250.pt: written
step_000500.pt: written
step 500 full-image validation: val_loss=0.0235864, val_psnr=28.5929
failure: CUDA OOM on the next backward pass after validation
diagnosis: post-validation CUDA reserved/cache fragmentation
successor: V1/T01/A_e03
```
