# A_4kd38 Gate Check

## Deployment Gate

- Allowed physical GPUs: `0,3`
- Preferred GPU: `3`
- Default minimum free memory: `4096 MiB`
- Default CUDA memory fraction: `0.20`
- Expected memory profile: one student model only; no A08/A15 teacher models during training.

## Required Checks

- Static: `py_compile` for config/model/train/eval scripts.
- Config: TOML parse and checkpoint/data path existence.
- Launcher: `bash -n` for both launch scripts.
- Smoke: load A37 checkpoint into A38 config, verify only high decoder-head parameters are trainable, verify forward aux route keys exist, and verify backward reaches high route while low-route parameters stay frozen.
- GPU preflight: launch with `TWODGAUSS_PREFLIGHT_ONLY=1` before formal training.

## Completed Checks

- Static checks passed for `/home/ldj/wxc/2DGauss/src/uhd_gaussian/config.py`, `/home/ldj/wxc/2DGauss/src/uhd_gaussian/models/restorer.py`, `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`, `/home/ldj/wxc/2DGauss/scripts/evaluate_v1_t01.py`, `/home/ldj/wxc/2DGauss/scripts/evaluate_task_specific_metrics.py`, `/home/ldj/wxc/2DGauss/scripts/evaluate_a18_record_alignment_v1_t01.py`, and `/home/ldj/wxc/2DGauss/scripts/evaluate_input_std_route_regimes_v1_t01.py`.
- TOML/path checks passed for `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd38/configs/a_4kd38_fast_4kdehaze_a37_standard_loss_high_only_short.toml`.
- Launcher `bash -n` and dry-run passed.
- CPU smoke passed: A37 checkpoint strict load missing=0 unexpected=0, trainable parameters are only high route decoder-head parameters, high-route backward gradient is nonzero, low/shared gradients are zero.
- GPU3 preflight passed: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd38/logs/train_20260615_125507.log`, exit code `0`.
- Formal GPU3 short training passed: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd38/logs/train_20260615_125822.log`, exit code `0`.
