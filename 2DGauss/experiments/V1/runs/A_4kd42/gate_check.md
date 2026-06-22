# A_4kd42 Gate Check

时间: 2026-06-16 CST

## Planned Checks

- Static `py_compile` for training/evaluation scripts and model/config code.
- TOML parse and required path checks.
- Launcher `bash -n` and GPU dry-run.
- CPU smoke:
  - non-strict load A37 checkpoint because A42 adds zero-initialized `InputStdRouteFeatureExpert`;
  - native-config A08/A15 teachers load with zero missing/unexpected keys;
  - A08-oracle selected-output anchor is active and finite;
  - feature expert high/shared path receives gradients;
  - low route and `gaussian_control_canvas` remain frozen.
- GPU3 preflight before formal training.
- Formal training only after preflight passes.

## Intended GPU Policy

- Only physical GPU0/GPU3 are allowed.
- GPU1/GPU2 remain forbidden.
- High utilization alone does not block launch if free memory passes the gate.
- No external process should be killed or displaced.

## Status

- Static compile passed after A42 code changes:
  `/home/ldj/wxc/.venvs/2dgauss-py313/bin/python -m py_compile scripts/train_v1_t01.py src/uhd_gaussian/config.py src/uhd_gaussian/models/restorer.py`.
- TOML/path, launcher `bash -n`, GPU3 dry-run, and CPU tensor smoke passed before GPU preflight.
- First GPU3 preflight attempt `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd42/logs/train_20260616_232305.log` stalled in the DataLoader/IO stage with only the launch header written; only the A42-owned tmux/process was stopped. No external process was killed.
- Preflight code was patched to deep-copy config, apply `preflight_val_limit`, and force preflight `dataset.num_workers=0` so the gate uses deterministic single-process loading.
- Second GPU3 preflight passed:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd42/logs/train_20260616_233002.log`, `train_exit_code=0`.
- Preflight confirms A42 mechanisms are active:
  `input_std_route_a08_oracle_anchor_a_better_fraction=1.0`,
  `loss_input_std_route_a08_oracle_anchor=0.010513290762901306`,
  `loss_input_std_route_a08_oracle_anchor_weighted=0.0003679651708807796`,
  `input_std_route_balanced_single_sample_probabilistic=true`,
  `input_std_route_balanced_single_sample_high_probability=0.7`.
- Preflight confirms A37 non-strict init behavior is as intended: missing keys are limited to the newly added `input_std_route_feature_expert.*` parameters; A08/A15 teacher checkpoints load with zero missing/unexpected keys.
- Preflight trainability/freeze policy is as intended:
  `trainable_parameter_count=97923`, `total_parameter_count=204886`; trainable prefixes are the feature expert shared/high path, high decoder head path, Gaussian refinement/guidance, and `decoder.1`.
- Formal training is cleared for GPU3 launch under the physical GPU0/GPU3-only policy.
