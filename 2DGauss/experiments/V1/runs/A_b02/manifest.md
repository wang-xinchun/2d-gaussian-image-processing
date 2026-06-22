# Run Manifest: V1/T01/A_b02

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_b02`
- Full identifier: `V1/T01/A_b02`

## Purpose

`A_b02` is the first `UHD_haze` branch after the accepted `A_a02` reference run.
Its role is not to change the model first. Instead, it repairs the confirmed
data-pipeline bottleneck by replacing repeated online 4K JPEG decode plus random
crop with a deterministic offline patch cache that the existing `PairedImageDataset`
can read directly with no loader-contract change.

This run is therefore an IO-first engineering branch:

- keep the same `V1/T01` model and loss mix as `A02/A_a02`;
- keep the same local official `UHD_haze` patch-evaluation protocol:
  `231` test pairs, `crop_size=512`;
- change only the training/validation/test data access path so we can test
  whether the original high-VRAM runtime envelope becomes feasible.

## Dataset

- Dataset: `UHD_haze`
- Source train input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze/train/input`
- Source train target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze/train/gt`
- Source test input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze/test/input`
- Source test target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze/test/gt`
- Source train pairs: `2290`
- Source internal validation selection: first `100` train pairs
- Source effective training selection: remaining `2190` train pairs
- Source local official test pairs: `231`
- Cache root:
  `/root/autodl-tmp/2DGauss/datasets/cache/UHD_haze_patch512_p8_png_seed20260511`
- Cache train split design: `8` fixed `512x512` PNG patches per effective train pair
- Cache validation split design: `1` center-crop `512x512` PNG patch per internal-val pair
- Cache official test design: `1` center-crop `512x512` PNG patch per test pair

## Planned Runtime

- Tier: short
- Duration limit: `58` minutes
- Max steps target: `3200`
- Batch size target before smoke confirmation: `56`
- DataLoader workers target before smoke confirmation: `8`
- Training crop: `512x512` random crop over cached `512x512` patches
  (effectively identity crop)
- Validation crop: `512x512` center crop over cached `512x512` patches
  (effectively identity crop)
- Validation interval target: `500`
- Checkpoint interval target: `500`
- Validation batches target: `100`

## Expected Acceptance Logic

`A_b02` is useful only if it satisfies both conditions:

1. cache-backed training recovers the user's `>=80G` VRAM floor; and
2. the end-to-end step time is now good enough to reach repeated
   validation/checkpoint milestones within the short-run envelope.

If both hold, then `A_b02` becomes the first fair high-VRAM `UHD_haze` short run
under the existing local patch protocol.

## Files

- Gate check: `gate_check.md`
- Launch config: `../../templates/T01/configs/a_b02_short_uhdhaze_patch_cache.toml`
- Training script: `scripts/train_v1_t01.py`
- Cache builder: `scripts/build_patch_cache_v1_t01.py`
- Archived template snapshot: `template_snapshot/`

## Current Status

```text
cache built and verified
cache-backed protocol-equivalence check passed
serial high-VRAM smoke passed
formal preflight passed
launch pending
```

## Cache Build Summary

```text
cache_root: /root/autodl-tmp/2DGauss/datasets/cache/UHD_haze_patch512_p8_png_seed20260511
train cached pairs: 17520
val cached pairs: 100
test cached pairs: 231
cache disk usage: about 7.7G
remaining disk after build: about 23G on /root/autodl-tmp
```

## Protocol-Equivalence Check

To ensure that the cache does not silently change the official patch-evaluation
semantics, `A_a02 final.pt` was re-evaluated against the cache-backed splits:

```text
cache-backed full-val: loss_mean=0.09807305697351694, psnr_mean=19.960832028478485
cache-backed official test: loss_mean=0.10115578951257648, psnr_mean=19.444372918691123
```

These exactly match the accepted `A_a02` records, so the cache can be treated as
an IO-path replacement only.

## Data-Inclusive Smoke Calibration

Serial smoke calibration on the cache-backed training split:

```text
batch 48: ok, reserved 74.748G, avg_total_step_seconds 0.9070, estimated_steps_60min 3969
batch 52: ok, reserved 80.979G, avg_total_step_seconds 0.9560, estimated_steps_60min 3765
batch 56: ok, reserved 87.209G, avg_total_step_seconds 1.0165, estimated_steps_60min 3541
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

## Preflight

Formal preflight succeeded with the final `A_b02` configuration:

```text
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

## Launch Command

```bash
bash experiments/V1/runs/A_b02/launch_tmux.sh
```

## Launch Status

Launched in tmux session `2dgauss_a_b02_uhdhaze_patch_cache` on
`2026-05-11 12:10:07 CST`.

Startup-window observation:

```text
step 50: elapsed 0.6962 min, loss_total 0.1465759
step 180: elapsed 2.3747 min, loss_total 0.1598277
step 250: elapsed 3.2786 min, loss_total 0.1214099
step 300: elapsed 3.9242 min, loss_total 0.1310842
gpu_memory_used: about 93309 MiB
status: startup window passed; cache-backed high-VRAM haze branch clearly crossed the historical A02 throughput-failure window
```

## Final Runtime Status

`A_b02` finished by reaching `max_steps=3200`, not by the `58` minute
wall-clock cap.

```text
final_step: 3200
final_elapsed_minutes: 41.67819548050563
final_train_loss_total: 0.087982177734375
final_validation_step: 3000
final_val_batches: 100
final_val_loss: 0.08656642345711589
final_val_psnr: 20.754805010074232
final_checkpoint: experiments/V1/runs/A_b02/checkpoints/final.pt
latest_checkpoint: experiments/V1/runs/A_b02/checkpoints/latest.pt
last_periodic_checkpoint: experiments/V1/runs/A_b02/checkpoints/step_003000.pt
tmux_session_after_finish: closed
```

## Training-Time Validation Summary

```text
step 500: val_loss=0.10760191421955824, val_psnr=18.637189394490655
step 1000: val_loss=0.09774167187511922, val_psnr=19.865953899557827
step 1500: val_loss=0.10641522049903869, val_psnr=18.88077777932586
step 2000: val_loss=0.0750709313340485, val_psnr=21.62895539250113
step 2500: val_loss=0.07516495605930686, val_psnr=21.864663560497057
step 3000: val_loss=0.08656642345711589, val_psnr=20.754805010074232
best validation-PSNR checkpoint: step_002500.pt
lowest validation-loss checkpoint: step_002000.pt
```

## Post-Hoc Evaluation Summary

Completed post-hoc records under the same cache-backed local `UHD_haze`
patch-based official-test protocol (`crop_size=512`, local `231`-pair test
split):

```text
step_002000.pt full-val: loss_mean=0.0750709313340485, psnr_mean=21.62895539250113
step_002000.pt official test: loss_mean=0.1049516647328417, psnr_mean=18.962357169960864
step_002500.pt full-val: loss_mean=0.07516495605930686, psnr_mean=21.864663560497057
step_002500.pt official test: loss_mean=0.10202291427236615, psnr_mean=19.396782047049683
final.pt full-val: loss_mean=0.08637667879462242, psnr_mean=20.796364269935204
final.pt official test: loss_mean=0.10090440487513294, psnr_mean=19.439466700482857
```

Current checkpoint-selection conclusion:

```text
best internal-validation PSNR checkpoint: step_002500.pt
best official-test checkpoint: final.pt
official-test delta final.pt vs step_002500.pt: +0.042684653433173736 dB
official-test delta final.pt vs A_a02 final.pt: -0.004906218208265756 dB
```

Result analysis:

```text
experiments/V1/runs/A_b02/result_analysis.md
```
