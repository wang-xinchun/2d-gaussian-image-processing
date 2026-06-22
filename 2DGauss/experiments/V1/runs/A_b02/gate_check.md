# Gate Check: V1/T01/A_b02

Status: completed and accepted as a high-VRAM cache-backed IO-fix run; not accepted as a new official UHD_haze baseline.

## Training Duration Tier

- [x] Tier target: short training.
- [x] Planned wall-clock cap: `58` minutes.
- [x] Purpose: repair the confirmed `UHD_haze` 4K JPEG / DataLoader bottleneck by
  moving repeated decode-and-crop work into a deterministic offline patch cache,
  then revalidate whether the original high-VRAM `A02` envelope becomes usable.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_b02`.
- [x] Full run identifier: `V1/T01/A_b02`.
- [x] Run folder: `experiments/V1/runs/A_b02`.
- [x] Run folder is independent and does not overwrite `A02` or `A_a02`.
- [x] Exact training-template snapshot is configured to be saved under
  `template_snapshot/` at launch.

## Dataset Check

- [x] Dataset name: `UHD_haze`.
- [x] Source dataset root: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze`.
- [x] Cache root target:
  `/root/autodl-tmp/2DGauss/datasets/cache/UHD_haze_patch512_p8_png_seed20260511`.
- [x] Source train input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze/train/input`.
- [x] Source train target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze/train/gt`.
- [x] Source validation selection: first `100` paired samples from `train`.
- [x] Source effective training selection: train pairs after the first `100`.
- [x] Source official test input:
  `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze/test/input`.
- [x] Source official test target:
  `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze/test/gt`.
- [x] Cache train policy: deterministic fixed `512x512` patches, `8` patches per
  effective training source image.
- [x] Cache validation policy: deterministic `512x512` center crop for the fixed
  `100`-pair internal validation slice.
- [x] Cache official-test policy: deterministic `512x512` center crop for the
  local `231`-pair test split.
- [x] Official test remains post-training only and remains the same local patch
  protocol as `A_a02`; the cache only changes data access efficiency.

## Contamination Check

- [x] Config source:
  `experiments/V1/templates/T01/configs/a_b02_short_uhdhaze_patch_cache.toml`.
- [x] Model architecture intentionally matches `A02/A_a02` to isolate IO repair.
- [x] Loss terms intentionally match `A02/A_a02`.
- [x] Cache builder path:
  `scripts/build_patch_cache_v1_t01.py`.
- [x] Fresh initialization; no pretrained weight path.
- [x] Fresh run; no resume checkpoint path.
- [x] Random seed: `20260511`.

## Runtime Check

- [x] Target batch size for recalibration starts from the old high-VRAM envelope:
  `56`.
- [x] Gradient accumulation: `1`.
- [x] Planned training crop remains `512x512`.
- [x] Planned validation crop remains `512x512`.
- [x] Max steps target: `3200`.
- [x] Planned duration cap: `58` minutes.
- [x] Validation interval target: every `500` steps.
- [x] Checkpoint interval target: every `500` steps.
- [x] Validation batches target: `100`.
- [x] DataLoader workers target: `8`.
- [x] Device plan: `auto`, CUDA if available.
- [x] AMP: disabled.
- [x] VRAM floor: training `max_memory_reserved_gb` must be at least `80`.
- [x] Cache build completed and counts verified.
- [x] Syntax checks completed.
- [x] Data-inclusive smoke calibration on cache-backed data completed.
- [x] Formal preflight completed.
- [x] Launch script checked.
- [x] Run approved to launch.

## Cache Build Notes

Cache build result:

```text
cache_root: /root/autodl-tmp/2DGauss/datasets/cache/UHD_haze_patch512_p8_png_seed20260511
train cached pairs: 17520
val cached pairs: 100
test cached pairs: 231
train cache bytes: 8013721157
val cache bytes: 54896772
test cache bytes: 99658204
du -sh cache_root: 7.7G
```

Protocol-equivalence check:

```text
A_a02 final.pt cache-backed full-val: loss_mean=0.09807305697351694, psnr_mean=19.960832028478485
A_a02 final.pt cache-backed official test: loss_mean=0.10115578951257648, psnr_mean=19.444372918691123
```

These exactly match the existing `A_a02` records, so the cache changes only data
access efficiency and does not change the local patch-evaluation protocol.

## Calibration Notes

Syntax and static checks:

```text
scripts/build_patch_cache_v1_t01.py: py_compile ok
scripts/train_v1_t01.py: py_compile ok
scripts/smoke_calibrate_v1_t01.py: py_compile ok
scripts/evaluate_v1_t01.py: py_compile ok
template config full_id: V1/T01/A_b02
run-local config full_id: V1/T01/A_b02
launch_tmux.sh: bash -n ok
```

Serial cache-backed smoke calibration:

```text
batch_size=48: ok, max_memory_reserved_gb=74.748046875, avg_total_step_seconds=0.9069910158403218, estimated_steps_60min=3969
batch_size=52: ok, max_memory_reserved_gb=80.978515625, avg_total_step_seconds=0.955974931595847, estimated_steps_60min=3765
batch_size=56: ok, max_memory_reserved_gb=87.208984375, avg_total_step_seconds=1.0165032566990704, estimated_steps_60min=3541
```

Chosen deployment setting:

```text
batch_size: 56
num_workers: 8
max_steps: 3200
wall_clock_minutes: 58
val_interval: 500
checkpoint_interval: 500
```

Formal preflight result:

```text
preflight: ok
device: cuda
train_samples: 56
val_samples: 100
batch_input_shape: (56, 3, 512, 512)
batch_target_shape: (56, 3, 512, 512)
total_params: 70223
loss_total: 0.3798036575317383
loss_charbonnier: 0.3776736855506897
loss_gradient: 0.04259946942329407
val_batches: 100
val_loss: 0.2588256603479385
val_psnr: 10.603145452601757
```

## Launch And Runtime Summary

```text
session: 2dgauss_a_b02_uhdhaze_patch_cache
log: experiments/V1/runs/A_b02/logs/train_20260511_121007.log
startup window: passed
startup GPU memory: about 93309 MiB
historical comparison: decisively crossed the original A02 throughput-failure window
```

## Final Acceptance Summary

```text
final_step: 3200
final_elapsed_minutes: 41.67819548050563
best validation-PSNR checkpoint: step_002500.pt, psnr=21.864663560497057
best official-test checkpoint inside A_b02: final.pt, psnr=19.439466700482857
delta vs A_a02 final.pt official test: -0.004906218208265756 dB
runtime/checkpoint acceptance: passed
cache-backed IO-fix acceptance: passed
official UHD_haze baseline replacement: no
current same-protocol UHD_haze reference remains: A_a02 final.pt, psnr=19.444372918691123
```
