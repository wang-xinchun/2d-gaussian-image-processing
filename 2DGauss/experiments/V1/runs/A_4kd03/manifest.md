# Run Manifest: V1/T01/A_4kd03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_4kd03`
- Full identifier: `V1/T01/A_4kd03`

## Current Status

Run-local package, formal GPU3 preflight, and a stop-loss training probe are
complete. The run was manually stopped after the step `200` validation evidence
showed weak quality and weak control selectivity. Checkpoints through step
`200` are retained for diagnosis; this is not an accepted candidate.

## Purpose

`A_4kd03` is the first root-cause-driven probe after the recent bottleneck
review. It tests whether a predicted Gaussian control canvas can become a
shared demand-control surface for decoder guidance, feature refinement, and
later correction paths.

The key question is not only whether PSNR improves. The run must also answer
whether `control_canvas` becomes selective: high where restoration demand is
large, restrained where fidelity should be protected, and bounded by the
feedback budget instead of becoming an all-image correction mask.

## Dataset

- Dataset: `4KDehaze_local_90_5_5_fast_probe_512`
- Root data asset: `/home/ldj/wxc/baiduwangpan/2D高斯数据集/4KDehaze/4KDehazing`
- Project split view:
  `/home/ldj/wxc/2DGauss/datasets/current_training_ready/views/4KDehaze_local_90_5_5_fast_probe_512`
- Train pairs: `512`
- Validation pairs: `100`
- Test pairs: `100`
- Boundary: this is the same fast-probe view used by `A_4kd02`, not an
  official/full split benchmark.

## Model Delta

Compared with `A_4kd02`, this run enables a shared demand-control mechanism:

- `use_gaussian_control_canvas = true`
- `gaussian_control_canvas_stage = "pre_guidance"`
- `gaussian_control_canvas_share_to_guidance = true`
- `gaussian_control_canvas_target_mode = "restoration_demand"`
- `gaussian_control_canvas_feedback_mode = "additive"`
- `gaussian_control_canvas_feedback_selectivity_mode = "demand_budget"`
- `gaussian_control_canvas_feedback_demand_source = "canvas"`
- `gaussian_control_canvas_use_confidence_normalization = true`

The model forward pass does not receive GT/target images. Restoration-demand
targets are computed only by training/evaluation auxiliary loss and diagnostics
from `inputs` plus `targets`.

## Diagnostics

`training.control_diagnostics = true` is enabled. Training and validation
metrics should include control diagnostics such as:

- `train_control_canvas_mean/std/min/max`
- `train_control_target_mean/std`
- `train_control_canvas_l1`
- `train_control_canvas_corr`
- `train_control_feedback_mean/std/max`
- `val_control_*` equivalents

These fields are required for interpretation. A PSNR-only readout is not
enough for this probe.

## Runtime Envelope

- Tier target: `short`
- Planned wall-clock cap: `58` minutes
- Planned max steps: `800`
- Planned batch size: `2`
- Gradient accumulation: `4`
- Training crop: `512`
- Validation crop: `512`
- Validation subset for training-time checks: first `100` validation pairs
- DataLoader workers: `1`
- CUDA memory fraction: `0.30`
- Launch gate: `>=8192MiB` free memory; high utilization is allowed for this
  low-footprint squeeze run.
- Allowed physical GPUs: `0` or `3` only
- Default preferred physical GPU: `3`

## Run-Local Config

`/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd03/configs/a_4kd03_fast_4kdehaze_shared_demand_control_probe.toml`

## 2026-06-10 Formal Preflight

- Preflight log:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd03/logs/preflight_20260610_115527.log`.
- Exit code: `train_exit_code=0`.
- Physical GPU: `3`.
- CUDA memory fraction: `0.30`.
- Effective batch shape: `(2, 3, 512, 512)`.
- Train samples: `8`.
- Validation samples: `100`.
- Total parameters: `148688`.
- Preflight validation: `val_loss=0.34498191975057124`,
  `val_psnr=9.40108024020294`.
- Control diagnostics at preflight show the expected untrained near-constant
  canvas: `val_control_canvas_std=0.0015831554273609072`,
  `val_control_canvas_corr=0.00371266288235347`,
  `val_control_canvas_l1=0.08323333244770766`,
  `val_control_feedback_mean=0.225758169144392`.
- Important runtime correction: an earlier `batch_size=4` preflight attempt
  hit CUDA OOM under current shared GPU3 load. The accepted runtime envelope is
  now `batch_size=2`, `grad_accum=4`, preserving effective batch `8` while
  reducing instantaneous memory pressure.

## 2026-06-10 Stop-Loss Training Probe

- Formal training log:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd03/logs/train_20260610_115912.log`.
- Training was launched on physical GPU3 only with `CUDA_VISIBLE_DEVICES=3`
  and `TWODGAUSS_CUDA_MEMORY_FRACTION=0.30`.
- The run was manually stopped after step `200` validation had been recorded;
  train metrics had reached step `240`, but no later validation/checkpoint was
  produced.
- Best retained checkpoint: step `200`, stored as `best_val.pt` and
  `step_000200.pt`.
- Step `100`: `val_loss=0.2597135889157653`,
  `val_psnr=12.1511608526603`,
  `val_control_canvas_std=0.0023506595322396605`,
  `val_control_canvas_corr=0.16863191392738372`.
- Step `200`: `val_loss=0.22252442486584187`,
  `val_psnr=13.527238149476778`,
  `val_control_canvas_std=0.0064538781507872045`,
  `val_control_canvas_corr=0.1268156026955694`.
- Interpretation: the control canvas begins to move away from a fully constant
  map, but remains much less selective than the restoration-demand target
  (`val_control_target_std=0.10203479953110219`) and quality remains materially
  behind the `A_4kd02` fast-probe trajectory.
- Next action: do not extend this hard-replacement setup. Use a conservative
  blended shared-control variant that keeps the original Gaussian attention as
  the main guidance signal.

## Boundary

This run can compare locally against `A_4kd02` because it uses the same
fast-probe data view and runtime envelope. It must not be described as a
direct improvement over historical official `M04/A_y03` or accepted `A_a02`
results.
