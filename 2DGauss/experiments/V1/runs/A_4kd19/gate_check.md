# A_4kd19 Gate Check

## Status

Completed as a CPU-only A08/A15 router generalization evaluation.

## Checks

- Uses existing `A_4kd08` and `A_4kd15` checkpoints.
- Uses `4KDehaze_local_90_5_5` full/current view with offset `100` so the
  `A_4kd18` fast-probe block is excluded.
- No training will be launched.
- GPU use is not required.
- `scripts/evaluate_checkpoint_router_v1_t01.py` `py_compile` passed after
  adding default-preserving `test_offset` and fixed-rule reporting support.
- TOML parse and sample-overlap checks passed: tail val/test subsets have `0`
  overlap with the A18 fast-probe val/test subsets.
- CPU router evaluation completed without GPU use.

## Boundary

The result supports the A18 dynamic-routing hypothesis, but it cannot by itself
establish an official/full benchmark result or single-model breakthrough.
