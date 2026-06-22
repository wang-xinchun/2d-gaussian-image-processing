# A_4kd37 Manifest

时间: 2026-06-14 23:23:35 CST

## Purpose

A37 is the first post-A36 compression probe. It keeps A33 as the compact student initialization, uses native-config A08/A15 teachers equivalent to A36's internal high/low experts, and trains only high-route/later shared components to reduce the remaining high-route gap without disturbing the protected low route.

## Key Paths

- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/configs/a_4kd37_fast_4kdehaze_a33_high_route_a36_teacher_compress.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/launch_tmux.sh`
- Memory-gated launcher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/wait_and_launch_a_4kd37_memory_gated_gpu_safe.sh`
- Init checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd33/checkpoints/a31_high_a24_low_surgery.pt`
- Teacher A config/checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/configs/a_4kd08_fast_4kdehaze_medium_detail_protected_blended_demand_control_probe.toml`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`
- Teacher B config/checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/configs/a_4kd15_fast_4kdehaze_a08_best_no_haze_guard_ema995_finetune.toml`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/checkpoints/best_val.pt`

## Training Intent

- Keep natural sampling.
- Keep fixed `input_std` prior.
- Use native-config teachers through `router_distillation_config_a/b`.
- Disable ordinary reconstruction/proxy losses for this first compression probe, so low-route samples do not update the shared late modules. The active route branch loss is high/A08 only and is weighted by the fixed high-route prior.
- Train high route and late shared modules only: `input_std_route_decoder_heads.shared`, `input_std_route_decoder_heads.high_feature`, `input_std_route_decoder_heads.high_head`, `gaussian_feature_refinement`, `gaussian_decoder_guidance`, `decoder.1`.
- Keep low route locked by excluding `input_std_route_decoder_heads.low_*`.

## Acceptance Focus

- Primary: high-route A18 alignment gap should improve materially from A33/A35 around `-0.126 dB`.
- Secondary: aggregate should stay above A33/A35 and move toward A36/A18.
- Safety: low route should not collapse; ideally low-route gap remains positive.

## Outcome

- Status: promoted as new best single-checkpoint fast-probe candidate.
- Formal checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`
- Best validation step: `120`
- Best validation PSNR: `18.56340678071067`
- Test PSNR: `17.889658453438816`
- Comparable standard-loss eval: `loss_mean=0.11911950781941413`, `total_loss_mean=0.12235063299536705`
- Test SSIM: `0.7026848953962326`
- A18 alignment overall gap: `+0.042724077624394126 dB`
- A18 alignment high-route gap: `-0.11159004428502844 dB`
- A18 alignment low-route gap: `+0.1591364853806252 dB`

## Result Paths

- Result analysis: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/result_analysis.md`
- Standard eval: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/metrics/test_eval_best_val_gpu.json`
- Standard-loss eval config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/configs/a_4kd37_eval_standard_loss.toml`
- Standard-loss eval: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/metrics/test_eval_best_val_standard_loss_gpu.json`
- Task metrics: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/metrics/test_task_metrics_best_val_gpu.json`
- A18 alignment: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/metrics/a18_record_alignment_best_val_gpu.json`
- Route-regime diagnostics: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/metrics/input_std_route_regime_best_val_gpu.json`

## Boundary

A37 beats A36/A18 on aggregate PSNR, but it does not fully close the high-input-std gap. The high-route gap improves from A33/A35 around `-0.126 dB` to `-0.11159004428502844 dB`, still slightly short of the `>-0.10 dB` target. The next experiment should preserve A37's low-route surplus while specifically attacking the A08-oracle/high-route tail.
