# Gate Check: V1/T01/A_4kd04

Status: completed. Formal preflight, training, checkpointing, CPU test eval,
and CPU task-specific diagnostics passed for the fast-probe scope.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_4kd04`.
- [x] Full run identifier: `V1/T01/A_4kd04`.
- [x] Run folder: `experiments/V1/runs/A_4kd04`.
- [x] Run-local config exists.
- [x] Template config exists.
- [x] This run is independent from `A_4kd03` and does not overwrite it.

## Dataset Check

- [x] Dataset name: `4KDehaze_local_90_5_5_fast_probe_512`.
- [x] Train split expected: `512` paired files.
- [x] Validation split expected: `100` paired files.
- [x] Test split expected: `100` paired files.
- [x] Formal preflight data load.

## Runtime Check

- [x] Planned batch size: `2`.
- [x] Planned gradient accumulation: `4`.
- [x] Planned workers: `1`.
- [x] Planned train crop: `512`.
- [x] Planned validation crop: `512`.
- [x] Planned max steps: `800`.
- [x] Planned wall-clock cap: `58` minutes.
- [x] Only physical GPU `0` or `3` may be used.
- [x] Default launch preference is physical GPU `3`.
- [x] GPU1/GPU2 are forbidden.
- [x] Formal GPU preflight.
- [x] Formal launch.
- [x] First validation/checkpoint.

## Model / Loss Check

- [x] Keeps base lightweight `V1/T01` geometry.
- [x] Uses `pre_guidance` control canvas.
- [x] Uses blended sharing, not hard replacement:
  `gaussian_control_canvas_share_to_guidance_blend = 0.25`.
- [x] Uses reduced feedback strength and budget versus `A_4kd03`.
- [x] Uses `restoration_demand` target only for loss/diagnostics, not forward GT.
- [x] Enables `training.control_diagnostics`.

## Launch Boundary

Use only physical GPU0 or GPU3. Prefer GPU3. Do not kill or displace external
processes. High utilization is acceptable for this low-footprint probe, but
the project must keep its own CUDA memory fraction low and refuse launch below
the configured free-memory floor.

## Completion Check

- [x] Training used physical GPU3 only.
- [x] `CUDA_VISIBLE_DEVICES=3`.
- [x] `TWODGAUSS_CUDA_MEMORY_FRACTION=0.30`.
- [x] `train_exit_code=0`.
- [x] Completed through step `800`.
- [x] Best validation checkpoint exists at step `600`.
- [x] Final checkpoint exists at step `800`.
- [x] CPU standard test eval completed.
- [x] CPU task-specific diagnostics completed.
- [x] No `A_4kd04`, `train_v1_t01.py`,
  `evaluate_v1_t01.py`, `evaluate_task_specific_metrics.py`, or A_4kd tmux
  session remained after completion.

## Result Boundary

`A_4kd04` is accepted as a fast-probe mechanism candidate because it beats
`A_4kd02` on the same `4KDehaze_local_90_5_5_fast_probe_512` validation and
test protocol. It is not an official/full split benchmark and must not be
compared directly to `M04/A_y03` or historical `A_a02` official references.
