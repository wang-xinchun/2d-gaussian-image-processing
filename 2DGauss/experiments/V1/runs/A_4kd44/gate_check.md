# A_4kd44 Gate Check

时间: 2026-06-17 CST

## Planned Checks

- Static compile for train/config/model code.
- TOML parse and required path checks.
- Launcher `bash -n` and GPU dry-run.
- CPU smoke:
  - A37 init loads non-strict with only new residual-gate keys missing;
  - A08/A15 teachers strict-load;
  - residual gate starts exactly at the fixed prior on boundary inputs;
  - oracle-gate loss gives non-zero gradients to `input_std_route_decoder_heads.gate_head`;
  - only gate-head parameters are trainable.
- GPU3 preflight before formal training.

## Intended GPU Policy

- Only physical GPU0/GPU3 are allowed.
- GPU1/GPU2 remain forbidden.
- High utilization alone does not block launch if free memory passes the gate.
- No external process should be killed or displaced.

## Status

- Static compile passed:
  `/home/ldj/wxc/.venvs/2dgauss-py313/bin/python -m py_compile scripts/train_v1_t01.py src/uhd_gaussian/config.py src/uhd_gaussian/models/restorer.py`.
- TOML parse and required dependency paths passed; sampler cache is expected to be generated at runtime.
- Launcher `bash -n` and GPU3 dry-run passed under physical GPU0/GPU3-only policy.
- CPU smoke passed:
  A37 init uses non-strict load with `4` missing keys, all new `input_std_route_decoder_heads.gate_head.*`;
  A08/A15 teachers strict-load with zero missing/unexpected keys;
  only `input_std_route_decoder_heads.gate_head` is trainable (`545` params);
  residual gate starts exactly at fixed prior on boundary inputs (`max_abs_gate_prior=0.0`);
  oracle-gate loss is positive and produces non-zero gate-head gradient.
- GPU3 preflight passed:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd44/logs/train_20260617_010048.log`, `train_exit_code=0`.
- Preflight confirms real checkpoint loading and gates:
  A37 init non-strict missing keys are only new gate-head params; A08/A15 teachers strict-load with zero missing/unexpected keys;
  `input_std_route_teacher_enabled=true`; only `545` gate-head params are trainable; oracle gate is active on sampled low-route
  A08-better crop (`oracle_gate_target_mean=0.800000011920929`, `oracle_gate_positive_fraction=1.0`);
  preflight val uses `20` batches and completes on GPU3 with `TWODGAUSS_CUDA_MEMORY_FRACTION=0.30`.
- Formal training is cleared for GPU3 launch.
