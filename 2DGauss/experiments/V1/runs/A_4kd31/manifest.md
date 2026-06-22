# A_4kd31 Manifest

Updated: 2026-06-14 18:53:44 CST

## Run

- ID: `V1/T01/A_4kd31`
- Purpose: high/low route decoder-head checkpoint transplant with route-only training.
- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/configs/a_4kd31_fast_4kdehaze_a08_a15_transplanted_route_heads_route_only.toml`
- Initial checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/checkpoints/best_val.pt`
- High route head source: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`
- Low route head source: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/checkpoints/best_val.pt`

## Code And Launchers

- Modified code: `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/launch_tmux.sh`
- Wait launcher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/wait_and_launch_a_4kd31_memory_gated_gpu_safe.sh`

## Logs

- Preflight: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/logs/train_20260614_183403.log`
- Formal training: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/logs/train_20260614_183538.log`
- Standard eval stdout: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/logs/test_eval_best_val_gpu.stdout.log`
- Task metrics stdout: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/logs/test_task_metrics_best_val_gpu.stdout.log`
- A18 alignment stdout: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/logs/a18_record_alignment_best_val_gpu.stdout.log`
- Route-regime stdout: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/logs/input_std_route_regime_best_val_gpu.stdout.log`

## Metrics

- Best-val summary: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/metrics/best_val_summary.json`
- Transplant info: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/metrics/input_std_route_decoder_heads_transplant.json`
- Parameter policy: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/metrics/parameter_policy.json`
- Training metrics: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/metrics/train_metrics.jsonl`
- Validation metrics: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/metrics/val_metrics.jsonl`
- Standard test: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/metrics/test_eval_best_val_gpu.json`
- Task metrics: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/metrics/test_task_metrics_best_val_gpu.json`
- A18 alignment: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/metrics/a18_record_alignment_best_val_gpu.json`
- Route-regime diagnostics: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd31/metrics/input_std_route_regime_best_val_gpu.json`

## Result

A31 is a negative aggregate result but a positive mechanism diagnostic. It does not beat A24 PSNR, but it materially improves high-route gap and exposes low-route damage from low-head transplant under frozen A24 features.
