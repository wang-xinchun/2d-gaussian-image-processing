# A_4kd43 Gate Check

时间: 2026-06-17 CST

## Planned Checks

- Static compile for train/config/model code.
- TOML parse and required path checks.
- Launcher `bash -n` and GPU dry-run.
- CPU smoke:
  - A37 non-strict init with only feature-expert missing keys;
  - A08/A15 teachers strict load;
  - A37 trust teacher strict load;
  - strict-positive A08 anchor can be active only on positive advantage;
  - trust anchor can be active only when A37 beats current prediction;
  - low route and `gaussian_control_canvas` remain frozen.
- GPU3 preflight before formal training.

## Intended GPU Policy

- Only physical GPU0/GPU3 are allowed.
- GPU1/GPU2 remain forbidden.
- High utilization alone does not block launch if free memory passes the gate.
- No external process should be killed or displaced.

## Status

- Static compile passed:
  `/home/ldj/wxc/.venvs/2dgauss-py313/bin/python -m py_compile scripts/train_v1_t01.py src/uhd_gaussian/config.py src/uhd_gaussian/models/restorer.py`.
- TOML parse and required path checks passed; required checkpoints/configs all exist.
- Launcher `bash -n` and GPU3 dry-run passed under the physical GPU0/GPU3-only policy.
- CPU smoke passed:
  A37 init uses non-strict load with `34` missing keys, all from newly added `input_std_route_feature_expert.*`;
  A08/A15 teachers strict-load with zero missing/unexpected keys;
  A37 trust teacher strict-loads with zero missing/unexpected keys;
  trainable params are `97923 / 204886`;
  synthetic strict-positive A08 anchor probability `1.0`, trust-anchor probability `1.0`;
  feature/high route gradients are non-zero, while low route and `gaussian_control_canvas` gradients remain zero.
- GPU3 preflight passed:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd43/logs/train_20260617_001854.log`, `train_exit_code=0`.
- Preflight confirms real checkpoint loading and gates:
  A08/A15 teachers strict-load with zero missing/unexpected keys; A37 trust teacher strict-loads with zero missing/unexpected keys.
  A08 strict-positive selected anchor is active on the sampled crop (`probability_mean=0.6636620759963989`, `a_better_fraction=1.0`).
  A37 trust anchor is enabled but inactive at initialization (`probability_mean=0.0`) because the student starts from A37 and the trust-teacher error equals the current prediction error on this crop.
- Formal training is cleared for GPU3 launch.
