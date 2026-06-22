# V1/T01/A_dsh01_predeploy

Status: predeploy waiting for UHD video deshaking dataset.

This package exists to prepare the corrected new task: ultra-high-definition video deshaking / stabilization. It must not be confused with `UHD_haze` or any `A_b02*` package.

## Purpose

- Inspect the incoming 4K/8K deshaking dataset as soon as transfer finishes.
- Establish lightweight temporal proxy metrics for frame sequences.
- Provide a frame-enhancement baseline using frozen `A_y03`.
- Block training launch until deshaking data, supervision, metrics, and config are verified.

## Key Files

- Config: `configs/a_dsh01_predeploy_uhd_video_deshake.toml`
- Data gate: `run_dataset_gate.sh`
- Frame baseline: `run_frame_enhancer_baseline.sh`
- Dataset inspection output: `metrics/dataset_inspection_*.json`
- Frame baseline summary: `metrics/frame_enhancer_*.json`

## Boundary

No training has been launched for this package. `UHD_haze` resource-calibration results do not count as progress for this run.
