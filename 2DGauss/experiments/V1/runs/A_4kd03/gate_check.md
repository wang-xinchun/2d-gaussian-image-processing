# Gate Check: V1/T01/A_4kd03

Status: stop-loss probe complete; this run is not accepted for continuation.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_4kd03`.
- [x] Full run identifier: `V1/T01/A_4kd03`.
- [x] Run folder: `experiments/V1/runs/A_4kd03`.
- [x] Run-local config exists.
- [x] Template config exists.
- [x] This run is independent from `A_4kd02` and does not overwrite it.

## Dataset Check

- [x] Dataset name: `4KDehaze_local_90_5_5_fast_probe_512`.
- [x] Train split expected: `512` paired files.
- [x] Validation split expected: `100` paired files.
- [x] Test split expected: `100` paired files.
- [x] Same fast-probe view as `A_4kd02` for local comparison.
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
- [x] Launcher must refuse too-low-free-memory states while allowing high
  utilization for this low-footprint squeeze run.
- [x] Formal GPU preflight.
- [ ] Formal launch.
- [x] Formal stop-loss training probe through step `200` validation.
- [ ] CPU test evaluation.
- [ ] Cleanup check.

## Model / Loss Check

- [x] Keeps base lightweight `V1/T01` geometry.
- [x] Enables decoder guidance with decoder-state gate.
- [x] Enables Gaussian feature refinement with detail/confidence/demand gates.
- [x] Enables Gaussian control canvas in `pre_guidance` mode.
- [x] Shares predicted `control_canvas` to guidance paths.
- [x] Uses `restoration_demand` target mode only for auxiliary supervision and
  diagnostics, not as a model input.
- [x] Uses additive bounded feedback plus demand-budget selectivity.
- [x] Enables `training.control_diagnostics`.

## Interpretation Gate

This probe is useful only if the metrics include both restoration quality and
control selectivity. Required readout after preflight/training:

- `loss_control_canvas`
- `train_control_canvas_mean/std`
- `train_control_canvas_l1`
- `train_control_canvas_corr`
- `train_control_feedback_mean/max`
- `val_control_*` equivalents
- PSNR/loss/task-specific metrics

If the control map collapses to all-high, all-low, or low-correlation behavior,
the next action should tune gate/budget/strength instead of extending training.

## Current Launch Boundary

Use only physical GPU0 or GPU3. Prefer GPU3. Do not kill or displace external
processes. High utilization is acceptable for this probe, but the project must
keep its own CUDA memory fraction low and refuse launch below the configured
free-memory floor.

## 2026-06-10 Formal Preflight Gate

- Preflight command used physical GPU3 with `TWODGAUSS_CUDA_MEMORY_FRACTION=0.30`.
- Preflight log:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd03/logs/preflight_20260610_115527.log`.
- Exit code: `0`.
- Batch shape: `(2, 3, 512, 512)`.
- Validation covered `100` batches/samples.
- `val_loss=0.34498191975057124`, `val_psnr=9.40108024020294`.
- Diagnostics present: `preflight_control_*` and `val_control_*`.
- Initial canvas is near-constant before real training:
  `val_control_canvas_std=0.0015831554273609072`,
  `val_control_canvas_corr=0.00371266288235347`.
- Earlier `batch_size=4` preflight attempt hit CUDA OOM under current shared
  GPU3 load. The accepted A_4kd03 launch envelope is therefore
  `batch_size=2`, `grad_accum=4`, `TWODGAUSS_CUDA_MEMORY_FRACTION=0.30`.

## 2026-06-10 Step200 Stop-Loss Gate

- Formal training was stopped after the step `200` validation evidence.
- Step `200`: `val_loss=0.22252442486584187`,
  `val_psnr=13.527238149476778`.
- `A_4kd02` comparison at the same fast-probe scale: step `100`
  `val_psnr=14.261986201609318`, final step `800`
  `val_psnr=17.529067811582326`.
- A_4kd03 control selectivity remained weak:
  `val_control_canvas_std=0.0064538781507872045` versus
  `val_control_target_std=0.10203479953110219`, and
  `val_control_canvas_corr=0.1268156026955694`.
- Conclusion: hard replacement of guidance attention by the early control
  canvas is too aggressive for this setup.
