# A_4kd35 Manifest

时间: 2026-06-14 22:01:30 CST

## Purpose

Default-off A33 post-route high adapter probe. It keeps A33 decoder-head routing intact and adds a zero-initialized high-route post adapter to test whether small output-space correction can close the remaining A18 high-route gap.

## Key Paths

- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/configs/a_4kd35_fast_4kdehaze_a33_post_high_adapter.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/launch_tmux.sh`
- Memory-gated launcher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/wait_and_launch_a_4kd35_memory_gated_gpu_safe.sh`
- Best checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/checkpoints/best_val.pt`
- Formal train log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/logs/train_20260614_214218.log`
- Standard test: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/metrics/test_eval_best_val_gpu.json`
- Task metrics: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/metrics/test_task_metrics_best_val_gpu.json`
- A18 alignment: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/metrics/a18_record_alignment_best_val_gpu.json`
- Route diagnostics: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/metrics/input_std_route_regime_best_val_gpu.json`
- Result analysis: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/result_analysis.md`
- Gate check: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/gate_check.md`

## Source Code Paths

- `/home/ldj/wxc/2DGauss/src/uhd_gaussian/config.py`
- `/home/ldj/wxc/2DGauss/src/uhd_gaussian/__init__.py`
- `/home/ldj/wxc/2DGauss/src/uhd_gaussian/models/restorer.py`
- `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`

## Outcome

A35 is engineering-complete and numerically a tiny improvement over A33, but it is not promoted as a scientific breakthrough. A33 remains the main promoted single-checkpoint fast-probe candidate until A36/A37 provide stronger evidence.
