# Gate Check: V1/T01/A_4kd09

Status: completed; formal preflight, training, CPU evals, and closeout checks
passed. Completed as a reliability diagnostic, not promoted as the current
fast-probe candidate.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_4kd09`.
- [x] Full run identifier: `V1/T01/A_4kd09`.
- [x] Run folder: `experiments/V1/runs/A_4kd09`.
- [x] Run-local config exists.
- [x] Template config exists.
- [x] This run is independent from `A_4kd08`.

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
- [x] Keeps `A_4kd08` medium detail-preserve gate.
- [x] Keeps `A_4kd08` protection losses.
- [x] Adds default-off reliability-gate code and enables it only in `A_4kd09`.
- [x] Uses `restoration_demand` target only for loss/diagnostics, not forward GT.
- [x] Enables `training.control_diagnostics`.

## Launch Boundary

Use only physical GPU0 or GPU3. Prefer GPU3. Do not kill or displace external
processes. High utilization is acceptable for this low-footprint probe, but
the project must keep its own CUDA memory fraction low and refuse launch below
the configured free-memory floor.


## Preflight Evidence

- Formal preflight log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd09/logs/train_20260611_093632.log`
- Preflight exit code: `0`
- Physical GPU used: `3`
- `CUDA_VISIBLE_DEVICES`: `3`
- CUDA memory fraction: `0.30`
- Preflight validation: `val_psnr=9.576004736183744`, `val_loss=0.3485717223584652`
- Reliability diagnostics: `val_control_guidance_reliability_mean=0.8250765126943588`, `val_control_guidance_blend_mean=0.16774451807141305`

## Completed Evidence

- Formal training log:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd09/logs/train_20260611_093909.log`
- Training exit code: `0`.
- Physical GPU used: `3`.
- `CUDA_VISIBLE_DEVICES`: `3`.
- CUDA memory fraction: `0.30`.
- Peak reserved CUDA memory: about `11.265625GiB`.
- Best validation checkpoint: step `800`,
  `val_psnr=17.894769024624548`,
  `val_loss=0.11876579158008099`.
- Final validation checkpoint is also the best validation checkpoint.
- CPU standard test eval:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd09/metrics/test_eval_best_val_cpu.json`
- CPU task-specific diagnostics:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd09/metrics/test_task_metrics_best_val_cpu.json`
- Final process check: no `A_4kd09` train/eval/tmux residue.

## Gate Interpretation

The reliability gate behaved as designed: validation reliability stayed at
`1.0` before the schedule opened, then settled near `0.827` late in training,
reducing effective control-to-guidance blend from about `0.2033` to about
`0.1682`.

The mechanism helped stability and dehazing-side proxies, but the current
operating point is too conservative for test fidelity. `A_4kd09` therefore
closes as a useful diagnostic rather than a promoted candidate.
