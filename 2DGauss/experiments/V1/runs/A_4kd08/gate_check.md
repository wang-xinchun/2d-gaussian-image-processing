# Gate Check: V1/T01/A_4kd08

Status: completed; formal preflight, training, CPU evals, and closeout checks passed.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_4kd08`.
- [x] Full run identifier: `V1/T01/A_4kd08`.
- [x] Run folder: `experiments/V1/runs/A_4kd08`.
- [x] Run-local config exists.
- [x] Template config exists.
- [x] This run is independent from `A_4kd05`, `A_4kd06`, and `A_4kd07`.

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
- [x] Keeps `A_4kd05` blended shared demand-control.
- [x] Keeps `A_4kd05` protection losses.
- [x] Reduces final detail-preserve gate from strong to medium.
- [x] Uses `restoration_demand` target only for loss/diagnostics, not forward GT.
- [x] Enables `training.control_diagnostics`.

## Launch Boundary

Use only physical GPU0 or GPU3. Prefer GPU3. Do not kill or displace external
processes. High utilization is acceptable for this low-footprint probe, but
the project must keep its own CUDA memory fraction low and refuse launch below
the configured free-memory floor.

## Completed Evidence

- Formal preflight log:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/logs/train_20260611_002818.log`
- Formal training log:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/logs/train_20260611_002917.log`
- Preflight exit code: `0`.
- Training exit code: `0`.
- Physical GPU used: `3`.
- `CUDA_VISIBLE_DEVICES`: `3`.
- CUDA memory fraction: `0.30`.
- Peak reserved CUDA memory: about `11.24609375GiB`.
- Best validation checkpoint: step `700`,
  `val_psnr=18.163864803283275`,
  `val_loss=0.12629025239497424`.
- CPU standard test eval:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/metrics/test_eval_best_val_cpu.json`
- CPU task-specific diagnostics:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/metrics/test_task_metrics_best_val_cpu.json`
- Final process check: no `A_4kd08` train/eval/tmux residue.

## Gate Interpretation

The medium detail gate behaved as intended. Validation diagnostics show high
detail regions receiving lower effective control-to-guidance blend than
low-detail regions:

- `val_control_guidance_blend_high_detail_mean=0.1767063045501709`
- `val_control_guidance_blend_low_detail_mean=0.23001191183924674`
- `val_control_guidance_blend_detail_corr=-0.8495775324106216`

This is between the strong `A_4kd05` gate and the light `A_4kd06` gate, and it
currently gives the strongest fast-probe validation/test result.
