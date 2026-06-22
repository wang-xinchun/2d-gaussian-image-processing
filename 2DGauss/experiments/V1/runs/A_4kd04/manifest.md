# Run Manifest: V1/T01/A_4kd04

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_4kd04`
- Full identifier: `V1/T01/A_4kd04`

## Current Status

Completed on 2026-06-10. `A_4kd04` is accepted as a useful fast-probe
mechanism candidate, not as an official/full-protocol benchmark.

The best checkpoint is `checkpoints/best_val.pt` at step `600`. The final
checkpoint exists at `checkpoints/final.pt`, but validation peaked earlier.

## Purpose

`A_4kd04` is the conservative follow-up to the `A_4kd03` stop-loss probe.
`A_4kd03` showed that hard-replacing the main Gaussian attention with an early
control canvas was too aggressive: quality lagged `A_4kd02`, and the control
canvas remained low-variance.

This run keeps the original Gaussian attention as the main guidance path and
only blends in a small fraction of the predicted demand-control canvas.

## Dataset

- Dataset: `4KDehaze_local_90_5_5_fast_probe_512`
- Project split view:
  `/home/ldj/wxc/2DGauss/datasets/current_training_ready/views/4KDehaze_local_90_5_5_fast_probe_512`
- Train pairs: `512`
- Validation pairs: `100`
- Test pairs: `100`
- Boundary: same fast-probe view as `A_4kd02` and `A_4kd03`.

## Model Delta

Relative to `A_4kd03`, this probe is deliberately more conservative:

- `gaussian_control_canvas_share_to_guidance_blend = 0.25`
- `gaussian_control_canvas_strength = 0.06`
- `gaussian_control_canvas_residual_delta_strength = 0.35`
- `gaussian_control_canvas_feedback_budget = 0.20`
- `gaussian_control_canvas_feedback_floor = 0.02`
- `gaussian_control_canvas_feedback_target_mean = 0.10`
- `gaussian_decoder_guidance_strength = 0.08`

The goal is to preserve the proven Gaussian attention signal while testing
whether a weak demand-control overlay improves selectivity without damaging
early restoration quality.

## Diagnostics

`training.control_diagnostics = true` remains enabled. Required readout:

- `val_psnr` and `val_loss`
- `val_control_canvas_std`
- `val_control_canvas_corr`
- `val_control_canvas_l1`
- `val_control_feedback_mean/std/max`

## Completed Result

- Formal preflight: passed on physical GPU3.
- Formal training: completed with `train_exit_code=0`.
- Training log:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd04/logs/train_20260610_122251.log`
- Best validation step: `600`
- Best validation loss: `0.12592033293098212`
- Best validation PSNR: `17.6847619706854`
- Final step `800` validation loss: `0.13507467851042748`
- Final step `800` validation PSNR: `17.09676323141356`
- Best checkpoint:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd04/checkpoints/best_val.pt`
- CPU test eval:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd04/metrics/test_eval_best_val_cpu.json`
- CPU task-specific diagnostics:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd04/metrics/test_task_metrics_best_val_cpu.json`

Compared with `A_4kd02` on the same fast-probe view, `A_4kd04` improves best
validation PSNR from `17.529067811582326` to `17.6847619706854`, and CPU test
PSNR from `15.909636046480303` to `16.901385528055062`. This is a meaningful
fast-probe gain, but it remains a `512/100/100` subset result.

## Runtime Envelope

- Planned wall-clock cap: `58` minutes
- Planned max steps: `800`
- Planned batch size: `2`
- Gradient accumulation: `4`
- Training crop: `512`
- Validation crop: `512`
- Validation subset: `100` samples
- DataLoader workers: `1`
- CUDA memory fraction: `0.30`
- Allowed physical GPUs: `0` or `3` only
- Default preferred physical GPU: `3`

## Run-Local Config

`/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd04/configs/a_4kd04_fast_4kdehaze_blended_demand_control_probe.toml`

## Stop Criteria

Use the same early evidence discipline as `A_4kd03`. If step `100` and step
`200` remain materially behind `A_4kd02` while control diagnostics stay weak,
stop instead of running to `800`.

Outcome: the run was allowed to continue because step `300` and later evidence
showed a clear recovery over `A_4kd03`; best validation arrived at step `600`.
