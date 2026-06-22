# A_4kd36 Manifest

时间: 2026-06-14 22:08:00 CST

## Purpose

Pack A08 and A15 into one checkpoint and evaluate an internal fixed `input_std` router. This validates whether a single checkpoint artifact with complete high/low experts can reproduce the A18 two-checkpoint router.

## Source Inputs

- High/A08 config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/configs/a_4kd08_fast_4kdehaze_medium_detail_protected_blended_demand_control_probe.toml`
- High/A08 checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`
- Low/A15 config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/configs/a_4kd15_fast_4kdehaze_a08_best_no_haze_guard_ema995_finetune.toml`
- Low/A15 checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/checkpoints/best_val.pt`
- A18 records: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd18/metrics/a08_a15_router_val_to_test_cpu.json`
- Dataset config reused for test split: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/configs/a_4kd35_fast_4kdehaze_a33_post_high_adapter.toml`

## Code

- Checkpoint packer: `/home/ldj/wxc/2DGauss/scripts/create_dual_backbone_router_checkpoint_v1_t01.py`
- Evaluator: `/home/ldj/wxc/2DGauss/scripts/evaluate_dual_backbone_router_v1_t01.py`

## Outputs

- Packed checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd36/checkpoints/a08_a15_single_checkpoint_dual_backbone_input_std_router.pt`
- Pack manifest: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd36/metrics/dual_backbone_checkpoint_manifest.json`
- Pack log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd36/logs/create_dual_backbone_checkpoint.stdout.log`
- Smoke metric: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd36/metrics/test_eval_dual_backbone_router_gpu_smoke_limit2.json`
- Smoke log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd36/logs/test_eval_dual_backbone_router_gpu_smoke_limit2.stdout.log`
- Full metric: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd36/metrics/test_eval_dual_backbone_router_gpu.json`
- Full log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd36/logs/test_eval_dual_backbone_router_gpu.stdout.log`
- Full exit code: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd36/logs/test_eval_dual_backbone_router_gpu.exit_code`
- Result analysis: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd36/result_analysis.md`
- Gate check: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd36/gate_check.md`

## Outcome

A36 reaches `psnr_mean=17.8470899392002`, high-route gap `-0.00019952460737376242 dB`, and low-route gap `+0.00042343678412253124 dB` versus A18 records. It is promoted as a single-checkpoint upper-bound milestone.
