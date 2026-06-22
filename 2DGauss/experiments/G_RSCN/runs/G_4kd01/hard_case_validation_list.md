# G_4kd01 Hard-Case Validation List

- Created: `2026-06-21 16:43:00 CST`
- Scope: `G_4kd01_smoke01` and later G-RSCN short probes.
- Status: `PASS FOR SMOKE01 - formal promotion still requires post-run metrics`

This list is the minimum guardrail for launching `G_4kd01_smoke01`. It prevents another route-patch loop where aggregate validation looks stable while nominal-low/A08-oracle cases regress.

## Required Hard Positives

These records are the first nominal-low/A08-oracle guard set required by the G-line gate. They must be included in post-smoke alignment diagnostics before any formal run is promoted.

| File | Split | Input | GT | Reason |
| --- | --- | --- | --- | --- |
| `94_9400006.jpg` | test | `datasets/current_training_ready/views/4KDehaze_local_90_5_5_fast_probe_512/test/input/94_9400006.jpg` | `datasets/current_training_ready/views/4KDehaze_local_90_5_5_fast_probe_512/test/gt/94_9400006.jpg` | nominal-low/A08-oracle guard |
| `94_94000065.jpg` | test | `datasets/current_training_ready/views/4KDehaze_local_90_5_5_fast_probe_512/test/input/94_94000065.jpg` | `datasets/current_training_ready/views/4KDehaze_local_90_5_5_fast_probe_512/test/gt/94_94000065.jpg` | nominal-low/A08-oracle guard |
| `94_94000081.jpg` | test | `datasets/current_training_ready/views/4KDehaze_local_90_5_5_fast_probe_512/test/input/94_94000081.jpg` | `datasets/current_training_ready/views/4KDehaze_local_90_5_5_fast_probe_512/test/gt/94_94000081.jpg` | nominal-low/A08-oracle guard |

## Required Hard Negatives

The first smoke run must preserve A37/A15-style low-route stability. Until a wider offline cache is rebuilt for G-RSCN, the hard-negative guard is:

- Do not promote if aggregate validation improves while low-route diagnostics or the three nominal-low guards visibly degrade.
- Do not promote if `input_std_route_gate_mean` moves away from the frozen prior without a paired improvement in selected hard positives.
- Do not promote if `cuda_max_memory_reserved_gb` exceeds the planned single-GPU smoke envelope without clear benefit.

## Smoke01 Approval

- PASS: The three required nominal-low/A08-oracle files exist in both `test/input` and `test/gt`.
- PASS: `G_4kd01_smoke01` remains a short GPU probe, not formal training.
- PASS: Only `gaussian_route_control` is trainable.
- PASS: Post-smoke result analysis must compare the run against A37 and flag hard-positive/hard-negative behavior before any longer run.
