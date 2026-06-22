# A_4kd32 Result Analysis

Updated: 2026-06-14 19:29:38 CST

## Purpose

A32 targeted the A31 failure mode directly. A31 showed useful high-route movement but damaged the low route by transplanting the A15 terminal head into frozen A24 shared features. A32 kept A24 as the base, transplanted only the A08 high route head, froze the low route, sharpened the fixed input_std prior to `softness=0.003`, and trained only `shared`, `high_feature`, and `high_head`.

## Changed Paths

- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd32/configs/a_4kd32_fast_4kdehaze_high_a08_transplant_low_a24_locked.toml`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd32/launch_tmux.sh`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd32/wait_and_launch_a_4kd32_memory_gated_gpu_safe.sh`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd32/manifest.md`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd32/gate_check.md`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd32/result_analysis.md`

No main code path was modified for A32; it used the existing checkpoint-transplant and trainable-prefix infrastructure in `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`.

## Verification

- `py_compile` passed for `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`, `/home/ldj/wxc/2DGauss/src/uhd_gaussian/models/restorer.py`, and `/home/ldj/wxc/2DGauss/src/uhd_gaussian/config.py`.
- TOML parse and checkpoint path checks passed.
- Launcher `bash -n` and GPU3 dry-run gate passed.
- CPU transplant/freeze smoke passed:
  - high head vs A08 `decoder.2` max weight/bias diff `0.0/0.0`;
  - low head vs A24 `decoder.2` max weight/bias diff `0.0/0.0`;
  - trainable params `12899/168406`;
  - low route trainable count `0`;
  - low route gradients `0.0`;
  - initial `low_delta_abs_mean=0.0`.
- GPU3 preflight passed at `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd32/logs/train_20260614_190659.log`, exit code `0`, with `val_psnr=18.296594196429087`.

## Training

Formal training completed on physical GPU3:

- Log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd32/logs/train_20260614_190915.log`
- Exit code: `0`
- Best validation checkpoint: step `25`
- Best validation PSNR: `18.28028052288537`
- Best validation loss: `0.12194366712123156`

Validation degraded as high-route delta grew:

- step25: `val_psnr=18.28028052288537`, high delta `0.01711188313551247`, low delta `0.0`
- step50: `18.266222324393453`, high delta `0.02105612985789776`, low delta `0.0`
- step75: `18.257396540268623`, high delta `0.023996523870155214`, low delta `0.0`
- step100: `18.23277128080543`, high delta `0.029172710105776788`, low delta `0.0`

## Best-Val Test Metrics

- Standard eval: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd32/metrics/test_eval_best_val_gpu.json`
- Task metrics: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd32/metrics/test_task_metrics_best_val_gpu.json`

Results:

- `psnr_mean=17.786780968730323`
- `ssim_mean=0.7034752815961838`
- `loss_mean=0.12026964820921421`
- `total_loss_mean=0.12350073829293251`
- `dark_channel_l1_mean=0.14312145743519067`
- `edge_magnitude_l1_mean=0.15853436321020126`
- `local_contrast_l1_mean=0.02475376288406551`
- `chroma_l1_mean=0.030319798570126295`

A32 does not beat A24 `17.790202882742463`, but it is within `0.003421914012140203 dB`.

## A18 Alignment

Alignment file: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd32/metrics/a18_record_alignment_best_val_gpu.json`

- Overall gap vs A18 router: `-0.060153407084098394 dB`
- Low-route gap vs A18 router: `+0.05380978168030712 dB`
- High-route gap vs A18 router: `-0.211220889864822 dB`

## Route-Regime Diagnostics

Route-regime file: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd32/metrics/input_std_route_regime_best_val_gpu.json`

- Overall selected delta abs mean: `0.006058236313983798`
- High selected delta abs mean: `0.014088921660427437`
- Low selected delta abs mean: `0.0`
- High branch L1 vs A08: `0.23157927844413492`

## Conclusion

A32 validates the low-route protection mechanism: low-route gap recovers to `+0.05380978168030712 dB` and low selected delta stays exactly `0.0`. However, high-route gap regresses to `-0.211220889864822 dB`, meaning the A08-like high-route signal cannot be recovered by A08 terminal head transplant alone. This led directly to A33 checkpoint surgery: combine A31's learned high route with A32/A24 low-route protection.
