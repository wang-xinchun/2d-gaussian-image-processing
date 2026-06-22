# A_4kd18 Gate Check

## Status

Completed as a CPU inference-router diagnostic.

## Checks

- `scripts/evaluate_checkpoint_blend_v1_t01.py` added and `py_compile` passed.
- A08/A15 blend smoke on two test samples passed.
- Full CPU blend grid on the 100-sample test split completed.
- `scripts/evaluate_checkpoint_router_v1_t01.py` added and `py_compile` passed.
- Router smoke with three validation and three test samples passed.
- Full CPU val-to-test router search completed.
- Task metrics were aggregated from existing A08/A15 per-sample task metrics
  according to the selected router rule.
- No GPU was used for the CPU blend/router evaluations; no external GPU process
  was killed or displaced.

## Boundary

`A_4kd18` is a two-checkpoint inference wrapper. It is a mechanism diagnostic
and fast-probe candidate, not an official/full benchmark result.
