# Gate Check: V1/T01/A_b03

Status: deployment gate passed; approved for launch.

## Training Duration Tier

- [x] Tier target: short training.
- [x] Planned wall-clock cap: `58` minutes.
- [x] Purpose: test a checkpoint-selection protocol fix for `UHD_deblur` by
  using full-image internal validation during training.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_b03`.
- [x] Full run identifier: `V1/T01/A_b03`.
- [x] Run folder: `experiments/V1/runs/A_b03`.
- [x] Run folder is independent and does not overwrite `A03` or `A_a03`.
- [x] Exact training-template snapshot is configured to be saved under
  `template_snapshot/` at launch.

## Dataset Check

- [x] Dataset name: `UHD_deblur`.
- [x] Dataset root: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur`.
- [x] Train input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/train/input_new`.
- [x] Train target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/train/gt_new`.
- [x] Validation input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/train/input_new`.
- [x] Validation target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/train/gt_new`.
- [x] Validation selection: first `20` paired samples from train, deterministic filename order.
- [x] Effective training selection: remaining train pairs after the first `100`.
- [x] Official test input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/test/input300`.
- [x] Official test target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/test/gt300`.
- [x] Official test is reserved for post-training evaluation only.
- [x] Accepted inventory reference: `docs/unpacked_dataset_acceptance_20260425.md`.
- [x] Strategy reference: `docs/next_experiment_dataset_strategy_20260425.md`.
- [x] Metric-target reference: `docs/stage_metric_targets_20260505.md`.
- [x] Train image count expected: `1964` pairs.
- [x] Official test image count expected: `300` pairs.
- [x] Training crop policy: random `512x512` patches.
- [x] Validation crop policy: `crop_size=0`, full-image internal validation.

## Contamination Check

- [x] Config source: `experiments/V1/templates/T01/configs/a_b03_short_uhddeblur_fullval.toml`.
- [x] Dataset paths intentionally match `A03/A_a03` for same-task comparison.
- [x] Model architecture intentionally matches `A_a03` capacity settings.
- [x] Main protocol change from `A_a03`: validation is full-image internal
  validation, not `512x512` center crop validation.
- [x] Loss terms remain Charbonnier weight `1.0` and gradient loss weight `0.05`.
- [x] Optimizer remains AdamW, lr `0.0002`, weight decay `0.0001`.
- [x] Evaluation protocol: RGB tensor range `[0,1]`, clamped output, PSNR.
- [x] Pretrained weight path: empty; this is a fresh initialization run.
- [x] Resume checkpoint path: empty; this is not a resume.
- [x] Output directory is new: `experiments/V1/runs/A_b03`.
- [x] Random seed: `20260507`, intentionally matching `A_a03` to isolate
  validation-protocol effects as much as practical.

## Runtime Check

- [x] Batch size target: `16`.
- [x] Gradient accumulation: `1`.
- [x] Max steps target: `2000`.
- [x] Planned duration cap: `58` minutes.
- [x] Training crop: `512x512`.
- [x] Validation crop: full image, `crop_size=0`.
- [x] Validation interval: every `500` steps.
- [x] Checkpoint interval: every `250` steps.
- [x] Validation batches: `20`.
- [x] DataLoader workers: `8`.
- [x] Device plan: `auto`, CUDA if available.
- [x] AMP: disabled for conservatism and comparability.
- [x] Syntax check completed.
- [x] Data-inclusive training smoke completed.
- [x] Formal preflight completed.
- [x] Expected memory risk is acceptable for the selected tier.
- [x] Expected end-to-end throughput is acceptable for the selected tier.

Syntax and identity check:

```text
scripts/train_v1_t01.py: py_compile ok
scripts/smoke_calibrate_v1_t01.py: py_compile ok
scripts/evaluate_v1_t01.py: py_compile ok
template config full_id: V1/T01/A_b03
run-local config full_id: V1/T01/A_b03
val_crop_size: 0
val_limit: 20
val_interval: 500
max_val_batches: 20
launch_tmux.sh: bash -n ok
```

Data count check:

```text
train/input_new images: 1964
train/gt_new images: 1964
test/input300 images: 300
test/gt300 images: 300
```

Data-inclusive smoke calibration:

```text
batch 16, workers 8, crop 512, steps 20: ok
avg_total_step_seconds: 1.3369811685290187
avg_fetch_seconds: 0.8848452201578766
avg_compute_seconds: 0.4521338018588722
estimated_steps_55min: 2468
estimated_steps_60min: 2692
max_memory_allocated_gb: 50.373448848724365
max_memory_reserved_gb: 51.759765625
last_loss: 0.048051316291093826
```

Formal preflight with full-image internal validation:

```text
preflight: ok
device: cuda
train_samples: 16
val_samples: 20
batch_input_shape: (16, 3, 512, 512)
batch_target_shape: (16, 3, 512, 512)
total_params: 189717
loss_total: 0.19018182158470154
loss_charbonnier: 0.18850398063659668
loss_gradient: 0.033556777983903885
val_batches: 20
val_loss: 0.11466717235743999
val_psnr: 17.524048632373642
```

## Launch Decision

- [x] Gate file created.
- [x] Template and run-local configs created.
- [x] User has explicitly requested deployment.
- [x] All required gate checks passed.
- [x] Launch script created and checked.
- [x] Run approved to launch.
