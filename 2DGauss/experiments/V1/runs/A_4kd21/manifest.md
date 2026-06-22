# V1/T01/A_4kd21

## Purpose

`A_4kd21` deploys a GPU-accelerated router evaluation after `A_4kd19/A_4kd20`.
The user requested deployment and asked to use GPU because CPU evaluation is too
slow.

The run evaluates `A_4kd08` and `A_4kd15` on a larger non-overlapping current
`4KDehaze` subset using physical GPU3. It reports both the fixed `A_4kd18`
`input_std` rule and the best validation-selected threshold over all collected
router features.

## Data

- View: `/home/ldj/wxc/2DGauss/datasets/current_training_ready/views/4KDehaze_local_90_5_5`
- Validation: offset `200`, limit `200`
- Test: offset `200`, limit `200`
- Crop size: `512`

## Checkpoints

- A branch: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`
- A config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/configs/a_4kd08_fast_4kdehaze_medium_detail_protected_blended_demand_control_probe.toml`
- B branch: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/checkpoints/best_val.pt`
- B config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/configs/a_4kd15_fast_4kdehaze_a08_best_no_haze_guard_ema995_finetune.toml`

## GPU Policy

- Physical GPU3 by default.
- GPU0 is allowed only by explicit override; GPU1/GPU2 remain forbidden.
- Updated by user at `2026-06-12 21:14 CST`: GPU0/GPU3 may be used as long as memory is not full; high utilization alone should not block launch.
- Gate requires free memory at least `12288MiB`; utilization ceiling is relaxed to `100%`.
- `TWODGAUSS_CUDA_MEMORY_FRACTION=0.30`.
- No external processes may be killed or displaced.

## Key Files

- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd21/configs/a_4kd21_a08_a15_gpu_router_offset200_larger_probe.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd21/launch_gpu_router_eval.sh`
- Result analysis: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd21/result_analysis.md`
- Metrics: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd21/metrics/a08_a15_gpu_router_offset200_val200_test200.json`
- Stdout log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd21/logs/a08_a15_gpu_router_offset200_val200_test200.stdout.log`

## Result

Full GPU evaluation completed on physical GPU3 in `673.3705039024353` seconds.
On the offset200 test block, the fixed A18 input-std rule reached
`PSNR=14.64236451612018`, `SSIM=0.6756442764401436`, beating A08
(`14.61942131606054` / `0.6726915469765663`) and A15
(`14.570231632564433` / `0.6721390201151372`). The validation-selected
`pred_a_std` rule was slightly worse on test (`14.634471636481214` PSNR), so the
fixed A18 input-std rule remains the stronger held-out choice.

## Boundary

This is a GPU-accelerated two-checkpoint router diagnostic. It is not a
single-model result and not an official/full benchmark.
