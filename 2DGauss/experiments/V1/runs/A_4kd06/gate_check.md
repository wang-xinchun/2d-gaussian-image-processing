# Gate Check: V1/T01/A_4kd06

Status: completed; formal preflight, training, CPU test evaluation, diagnostics,
and cleanup checks passed.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_4kd06`.
- [x] Full run identifier: `V1/T01/A_4kd06`.
- [x] Run folder: `experiments/V1/runs/A_4kd06`.
- [x] Run-local config exists.
- [x] Template config exists.
- [x] This run is independent from `A_4kd04` and `A_4kd05`.

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
- [x] Keeps `A_4kd04` blended shared demand-control.
- [x] Adds milder detail-preserve gating around the control-to-guidance blend.
- [x] Does not add the extra `A_4kd05` edge/chroma/edge-weighted protection losses.
- [x] Uses `restoration_demand` target only for loss/diagnostics, not forward GT.
- [x] Enables `training.control_diagnostics`.

## Launch Boundary

Use only physical GPU0 or GPU3. Prefer GPU3. Do not kill or displace external
processes. High utilization is acceptable for this low-footprint probe, but
the project must keep its own CUDA memory fraction low and refuse launch below
the configured free-memory floor.

## Completed Evidence

- Static checks passed: launcher `bash -n`, TOML key checks, and `py_compile`
  for train/eval/restorer/config paths.
- Formal preflight:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd06/logs/train_20260610_225331.log`
- Preflight exit code: `0`
- Preflight batch: `(2, 3, 512, 512)`
- Preflight validation: `val_psnr=9.404449661378553`,
  `val_loss=0.3447077273577452`
- Preflight blend diagnostics: high-detail mean
  `0.18403567776083946`, low-detail mean `0.23201072290539743`,
  detail correlation `-0.8495775282382965`
- Formal training:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd06/logs/train_20260610_225710.log`
- Training exit code: `0`
- Actual CUDA binding: `CUDA_VISIBLE_DEVICES=3`
- CUDA memory fraction: `0.30`
- Best validation: step `600`, `val_psnr=17.674754617905425`,
  `val_loss=0.1261259661614895`
- Final validation: step `800`, `val_psnr=17.121298116103546`,
  `val_loss=0.13434834714978933`
- CPU standard test:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd06/metrics/test_eval_best_val_cpu.json`
- CPU task diagnostics:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd06/metrics/test_task_metrics_best_val_cpu.json`
- Final cleanup: no `A_4kd06`, `train_v1_t01.py`,
  `evaluate_v1_t01.py`, `evaluate_task_specific_metrics.py`, or A_4kd06 tmux
  session remained after completion.
