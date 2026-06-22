# A_4kd40 Manifest

时间: 2026-06-15 CST

## Purpose

A40 follows the A37 teacher-compression route, not A39's shared control-canvas expansion. A39 improved aggregate PSNR but worsened the high-route A18 gap, so A40 returns to the A37 anchor, removes `gaussian_control_canvas` from the trainable set, and concentrates training exposure on high-`input_std` samples/crops.

## Key Paths

- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd40/configs/a_4kd40_fast_4kdehaze_a37_high_route_balanced_teacher.toml`
- Init checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`
- Teacher A config/checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/configs/a_4kd08_fast_4kdehaze_medium_detail_protected_blended_demand_control_probe.toml`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`
- Teacher B config/checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/configs/a_4kd15_fast_4kdehaze_a08_best_no_haze_guard_ema995_finetune.toml`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/checkpoints/best_val.pt`

## Training Intent

- Initialize from A37 best-val, not A39.
- Keep low route locked by excluding `input_std_route_decoder_heads.low_*`.
- Keep `gaussian_control_canvas` frozen, because A39 showed shared control-canvas training helps aggregate/low-route behavior but worsens high-route gap.
- Use high/A08 branch teacher supervision at the same conservative `0.10` weight as A39.
- Enable `input_std_route_balanced_sampling` and route-targeted crop sampling so the short run gets dense high-route teacher updates.

## Acceptance Focus

- Primary: high-route A18 alignment should beat A37 `-0.11159004428502844 dB` and ideally cross `>-0.10 dB`.
- Safety: low-route selected delta should remain `0.0`; low-route A18 gap should stay clearly positive.
- Aggregate: test PSNR should remain close to A37/A39 and not trade away the overall milestone.

## Boundary

A40 is a fast-probe mechanism diagnostic on `4KDehaze_local_90_5_5_fast_probe_512`, not an official/full benchmark.

## Result Summary

- Formal GPU3 training completed with exit code `0`; best/final step is `120`.
- Best val PSNR: `18.619183648392536`.
- Best-val test PSNR: `17.900562951956445`, a tiny aggregate improvement over A39 `17.900369585886775` and a `+0.010904498517629 dB` gain over A37.
- A18 alignment: overall gap `+0.0536285761420247 dB`, high-route gap `-0.1180115677202543 dB`, low-route gap `+0.18311149168725271 dB`.
- Verdict: new aggregate numeric best, but not the intended high-route breakthrough because high-route gap remains worse than A37 `-0.11159004428502844 dB` and misses `>-0.10 dB`.

## Result Paths

- Train log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd40/logs/train_20260615_225430.log`
- Standard test eval: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd40/metrics/test_eval_best_val_gpu.json`
- Task metrics: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd40/metrics/test_task_metrics_best_val_gpu.json`
- A18 alignment: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd40/metrics/a18_record_alignment_best_val_gpu.json`
- Route-regime diagnostics: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd40/metrics/input_std_route_regime_best_val_gpu.json`
- Analysis: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd40/result_analysis.md`
