# Run Manifest: V1/T01/A_4kd10

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_4kd10`
- Full identifier: `V1/T01/A_4kd10`

## Current Status

Prepared for a late-soft reliability breakthrough attempt. Not yet accepted as
a candidate until preflight, training, CPU test evaluation, and diagnostics pass.

## Purpose

`A_4kd10` follows the completed diagnostic `A_4kd09` and the current fast-probe
candidate `A_4kd08`.

`A_4kd09` showed that reliability gating can fix `A_4kd08`'s final-step collapse
and improve dark-channel/saturation diagnostics, but its `strength=0.35`,
`floor=0.70`, schedule `0.40->0.75` setting was too conservative and suppressed
useful guidance. `A_4kd10` tests the narrow hypothesis that reliability should
act only very late and lightly: preserve `A_4kd08` behavior through step `600`,
then apply a small reliability correction for the final phase.

## Dataset

- Dataset: `4KDehaze_local_90_5_5_fast_probe_512`
- Project split view:
  `/home/ldj/wxc/2DGauss/datasets/current_training_ready/views/4KDehaze_local_90_5_5_fast_probe_512`
- Train pairs: `512`
- Validation pairs: `100`
- Test pairs: `100`
- Boundary: same fast-probe view as `A_4kd02` through `A_4kd09`.

## Model Delta

Relative to `A_4kd09`, this run keeps the same medium detail gate and protection
losses, but makes reliability weaker and later:

- `gaussian_control_canvas_share_to_guidance_reliability_gate = true`
- `gaussian_control_canvas_share_to_guidance_reliability_strength = 0.15`
- `gaussian_control_canvas_share_to_guidance_reliability_floor = 0.85`
- `gaussian_control_canvas_share_to_guidance_reliability_kernel_size = 7`
- `gaussian_control_canvas_share_to_guidance_reliability_schedule = "linear"`
- `gaussian_control_canvas_share_to_guidance_reliability_schedule_start = 0.75`
- `gaussian_control_canvas_share_to_guidance_reliability_schedule_end = 1.00`

Relative to `A_4kd08`, this should be almost identical until the final quarter
of training, then lightly reduce control-to-guidance blend where the control
canvas disagrees with the original Gaussian attention.

## Runtime Envelope

- Planned wall-clock cap: `58` minutes
- Planned max steps: `800`
- Planned batch size: `2`
- Gradient accumulation: `4`
- Training crop: `512`
- Validation crop: `512`
- Validation subset: `100` samples
- DataLoader workers: `1`
- Best checkpoint selection: `val_psnr`
- CUDA memory fraction: `0.30`
- Allowed physical GPUs: `0` or `3` only
- Default preferred physical GPU: `3`

## Acceptance Lens

This is a fast-probe mechanism run, not an official/full benchmark. It should be
judged against both `A_4kd08` and `A_4kd09`.

- Useful outcome: retain most of `A_4kd08` CPU test PSNR/SSIM while reducing the
  final-step collapse and improving dark-channel/saturation versus `A_4kd08`.
- Failure mode: repeat `A_4kd09`'s broad test degradation, or behave exactly
  like `A_4kd08` with no stability/proxy improvement.
