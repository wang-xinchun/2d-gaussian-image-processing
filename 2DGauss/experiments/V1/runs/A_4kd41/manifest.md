# A_4kd41 Manifest

时间: 2026-06-16 CST

## Purpose

A41 follows the A40 diagnosis: more high-`input_std` exposure alone did not fix the A08-tail gap. A41 keeps the A37 anchor, freezes low route and `gaussian_control_canvas`, keeps high-balanced sampling, and adds a default-off training-script mechanism that weights high-branch teacher loss by the current crop's A08-vs-A15 oracle advantage against GT.

## Key Paths

- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd41/configs/a_4kd41_fast_4kdehaze_a37_high_route_oracle_weighted.toml`
- Init checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`
- Teacher A config/checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/configs/a_4kd08_fast_4kdehaze_medium_detail_protected_blended_demand_control_probe.toml`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`
- Teacher B config/checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/configs/a_4kd15_fast_4kdehaze_a08_best_no_haze_guard_ema995_finetune.toml`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/checkpoints/best_val.pt`
- Code change: `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`

## Training Intent

- Initialize from A37 best-val, not A39/A40, to keep the better high-route anchor.
- Keep low route locked and selected low delta at `0.0`.
- Keep `gaussian_control_canvas` frozen.
- Use high/A08 branch teacher supervision, but weight it up when A08 is closer than A15 to GT on the current training crop.
- Preserve A40's high-`input_std` balanced sampling and crop targeting.

## Acceptance Focus

- Primary: high-route A18 alignment should beat A37 `-0.11159004428502844 dB` and ideally cross `>-0.10 dB`.
- Safety: low-route selected delta should remain `0.0`; low-route A18 gap should stay positive.
- Aggregate: test PSNR should remain near A37/A40 and not trade away all aggregate progress.

## Boundary

A41 is a fast-probe mechanism diagnostic on `4KDehaze_local_90_5_5_fast_probe_512`, not an official/full benchmark.

## Result Summary

- Formal GPU3 training completed with exit code `0`; best/final step is `120`.
- Best val PSNR: `18.617112345875956`.
- Best-val test PSNR: `17.900201657598004`, which is `+0.010543204159187525 dB` over A37 but `-0.0003612943584414552 dB` below A40.
- A18 alignment: overall gap `+0.05326728178358334 dB`, high-route gap `-0.11813195131188482 dB`, low-route gap `+0.182568457627533 dB`.
- Route-regime diagnostics: high selected delta `0.013213998226578845`, low selected delta `0.0`, high branch L1 vs A08 teacher `0.01348984414754912`.
- Verdict: engineering-complete negative probe. Oracle-weighted high-branch loss slightly improves high-branch teacher L1 relative to A40, but does not improve the high-route A18 gap or reach the `>-0.10 dB` target.

## Result Paths

- Train log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd41/logs/train_20260615_233454.log`
- Standard test eval: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd41/metrics/test_eval_best_val_gpu.json`
- Task metrics: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd41/metrics/test_task_metrics_best_val_gpu.json`
- A18 alignment: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd41/metrics/a18_record_alignment_best_val_gpu.json`
- Route-regime diagnostics: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd41/metrics/input_std_route_regime_best_val_gpu.json`
- Analysis: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd41/result_analysis.md`
