# Gate Check: V1/T01/A_4kd05

Status: run completed; CPU eval and task diagnostics completed.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_4kd05`.
- [x] Full run identifier: `V1/T01/A_4kd05`.
- [x] Run folder: `experiments/V1/runs/A_4kd05`.
- [x] Run-local config exists.
- [x] Template config exists.
- [x] This run is independent from `A_4kd04` and does not overwrite it.

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

## Completion Evidence

- Formal training log:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd05/logs/train_20260610_183150.log`
- Exit code: `0`
- First validation/checkpoint: step `100`, `val_psnr=12.154267020652284`
- Best validation/checkpoint: step `600`, `val_psnr=17.595543271291344`
- Final validation: step `800`, `val_psnr=17.33071631834636`
- CPU standard test:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd05/metrics/test_eval_best_val_cpu.json`
- CPU task diagnostics:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd05/metrics/test_task_metrics_best_val_cpu.json`
- Final process check: no A_4kd05 train/eval/tmux residue.

## Model / Loss Check

- [x] Keeps base lightweight `V1/T01` geometry.
- [x] Keeps `A_4kd04` blended shared demand-control.
- [x] Adds detail-preserve gating around the control-to-guidance blend.
- [x] Adds small edge/high-frequency/local-contrast/chroma protection terms.
- [x] Uses `restoration_demand` target only for loss/diagnostics, not forward GT.
- [x] Enables `training.control_diagnostics`.

## Launch Boundary

Use only physical GPU0 or GPU3. Prefer GPU3. Do not kill or displace external
processes. High utilization is acceptable for this low-footprint probe, but
the project must keep its own CUDA memory fraction low and refuse launch below
the configured free-memory floor.

## Preflight Evidence

- Timestamp/log: `2026-06-10 18:25:18 CST`,
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd05/logs/train_20260610_182518.log`
- Exit code: `0`
- Physical GPU: `3`
- CUDA memory fraction: `0.30`
- Train batch shape: `(2, 3, 512, 512)`
- Validation samples: `100`
- `val_psnr`: `9.40824378005754`
- `val_loss`: `0.3547653856873512`
- `val_control_guidance_blend_high_detail_mean`: `0.15471820175647735`
- `val_control_guidance_blend_low_detail_mean`: `0.22401548877358438`
- `val_control_guidance_blend_detail_corr`: `-0.8495775413513184`

Gate readout is directionally correct: high-detail regions receive lower
effective control-to-guidance blend than low-detail regions.
