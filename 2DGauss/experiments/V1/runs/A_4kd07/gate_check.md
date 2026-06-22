# Gate Check: V1/T01/A_4kd07

Status: completed; formal preflight, training, CPU evals, and closeout checks passed.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_4kd07`.
- [x] Full run identifier: `V1/T01/A_4kd07`.
- [x] Run folder: `experiments/V1/runs/A_4kd07`.
- [x] Run-local config exists.
- [x] Template config exists.
- [x] This run is independent from `A_4kd04`, `A_4kd05`, and `A_4kd06`.

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
- [x] Best checkpoint selection metric: `val_psnr`.
- [x] Only physical GPU `0` or `3` may be used.
- [x] Default launch preference is physical GPU `3`.
- [x] GPU1/GPU2 are forbidden.
- [x] Formal GPU preflight.
- [x] Formal launch.
- [x] First validation/checkpoint.

## Model / Loss Check

- [x] Keeps base lightweight `V1/T01` geometry.
- [x] Keeps `A_4kd05` detail-preserve final strength.
- [x] Keeps `A_4kd05` edge/high-frequency/local-contrast/chroma protection losses.
- [x] Adds only a linear detail-gate schedule.
- [x] Uses `restoration_demand` target only for loss/diagnostics, not forward GT.
- [x] Enables `training.control_diagnostics`.

## Launch Boundary

Use only physical GPU0 or GPU3. Prefer GPU3. Do not kill or displace external
processes. High utilization is acceptable for this low-footprint probe, but
the project must keep its own CUDA memory fraction low and refuse launch below
the configured free-memory floor.

## Completed Evidence

- Formal preflight log:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd07/logs/train_20260610_233805.log`
- Formal training log:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd07/logs/train_20260610_233939.log`
- Preflight exit code: `0`.
- Training exit code: `0`.
- Physical GPU used: `3`.
- `CUDA_VISIBLE_DEVICES`: `3`.
- CUDA memory fraction: `0.30`.
- Peak reserved CUDA memory: about `11.24609375GiB`.
- Validation checkpoints produced through step `800`.
- Best validation checkpoint: step `600`,
  `val_psnr=17.571617667185173`,
  `val_loss=0.13438679296523331`.
- CPU standard test eval:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd07/metrics/test_eval_best_val_cpu.json`
- CPU task-specific diagnostics:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd07/metrics/test_task_metrics_best_val_cpu.json`
- Final process check: no `A_4kd07` train/eval/tmux residue.

## Gate Interpretation

The schedule implementation passed its mechanical check. The blend was uniform
through warmup and reached the intended `A_4kd05` detail-preserve profile at and
after step `600`. The metric outcome does not justify promotion: CPU test PSNR
remains slightly below `A_4kd05`, and validation is lower than both `A_4kd04`
and `A_4kd05`.
