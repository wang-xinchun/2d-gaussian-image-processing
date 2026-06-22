# Run Manifest: V1/T01/A_4kd08

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_4kd08`
- Full identifier: `V1/T01/A_4kd08`

## Current Status

Completed and accepted as the current fast-probe candidate.

`A_4kd08` completed formal GPU3 preflight, 800-step low-footprint training,
CPU standard test evaluation, and CPU task-specific diagnostics. It improves
both validation and CPU test PSNR/SSIM over `A_4kd05`, so it supersedes
`A_4kd05` as the current fast-probe candidate. This is still not an
official/full benchmark.

## Purpose

`A_4kd08` continues the short-train mechanism screen after `A_4kd07`.

Current bottleneck: `A_4kd05` improved CPU test PSNR/SSIM and detail proxies,
but paid validation and color/dehazing-proxy costs. `A_4kd06` showed a light
gate alone is insufficient, and `A_4kd07` showed schedule-only tuning is
insufficient. This run tests whether the strong `A_4kd05` gate is the part that
over-protects detail regions and harms validation/color behavior.

The narrow hypothesis is: keep `A_4kd05` protection losses, but reduce the
detail-preserve gate from strong to medium strength. If this keeps most
`A_4kd05` CPU test PSNR/SSIM while improving validation or color/dehazing
proxies, then the bottleneck is gate strength calibration. If it falls back
toward `A_4kd04`/`A_4kd06`, then the next useful move is not weaker gate tuning
but a more explicit control-confidence mechanism.

## Dataset

- Dataset: `4KDehaze_local_90_5_5_fast_probe_512`
- Project split view:
  `/home/ldj/wxc/2DGauss/datasets/current_training_ready/views/4KDehaze_local_90_5_5_fast_probe_512`
- Train pairs: `512`
- Validation pairs: `100`
- Test pairs: `100`
- Boundary: same fast-probe view as `A_4kd02` through `A_4kd07`.

## Model Delta

Relative to `A_4kd05`, this run changes only the detail-preserve gate strength:

- `gaussian_control_canvas_share_to_guidance_detail_preserve_strength = 0.50`
- `gaussian_control_canvas_share_to_guidance_detail_preserve_floor = 0.50`

It keeps the `A_4kd05` protection losses:

- `edge_magnitude_weight = 0.015`
- `high_frequency_weight = 0.025`
- `local_contrast_weight = 0.02`
- `chroma_weight = 0.01`
- `dark_channel_weight = 0.01`
- `use_edge_weighted_reconstruction = true`
- `edge_weighted_reconstruction_strength = 0.01`

There is no schedule in this run.

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

`A_4kd08` is not an official/full benchmark. It should be judged only against
the same fast-probe references:

- `A_4kd04`: validation peak reference
- `A_4kd05`: CPU test PSNR/SSIM and detail-protection reference
- `A_4kd06`: light-gate-alone diagnostic
- `A_4kd07`: schedule-only diagnostic

Useful outcome: approach or exceed `A_4kd05` CPU test PSNR/SSIM while reducing
the `A_4kd05` validation and color/dehazing-proxy costs. Failure mode: lower
test PSNR/SSIM without clear proxy repair, which would mean gate-strength
tuning is not the bottleneck-solving lever.

## Final Result

- Formal preflight: passed on physical GPU3 with `train_exit_code=0`.
- Formal training: completed 800 steps on physical GPU3 with `train_exit_code=0`.
- CUDA memory fraction: `0.30`.
- Peak reserved CUDA memory: about `11.24609375GiB`.
- Best checkpoint selection metric: `val_psnr`.
- Best validation checkpoint: step `700`.
- Best validation: `val_psnr=18.163864803283275`,
  `val_loss=0.12629025239497424`.
- Final step `800`: `val_psnr=14.466338503701529`,
  `val_loss=0.20616793725639582`.
- CPU test eval of `best_val.pt`: `psnr_mean=17.789133503142946`,
  `loss_mean=0.1228142374008894`,
  `auxiliary_loss_mean=0.003231953252106905`,
  `total_loss_mean=0.12604619041085244`.
- CPU task diagnostics: `ssim_mean=0.6947606813907623`,
  `edge_magnitude_l1_mean=0.16361528269946576`,
  `high_frequency_l1_mean=0.02172296936623752`,
  `local_contrast_l1_mean=0.026271848641335964`,
  `dark_channel_l1_mean=0.1491835382580757`,
  `luminance_l1_mean=0.1057051732763648`,
  `chroma_l1_mean=0.030328415371477603`,
  `saturation_l1_mean=0.05636739023029804`.

## Interpretation

The medium gate is the first short-run result that improves both the validation
frontier and the CPU test frontier in this fast-probe family. Relative to
`A_4kd05`, validation improves by about `+0.5683 dB`, CPU test PSNR by about
`+0.7218 dB`, SSIM by about `+0.00870`, and L1 loss improves by about
`-0.00966`.

The useful mechanism lesson is gate calibration: the strong `A_4kd05` gate was
not the right operating point, the light `A_4kd06` gate was too weak by itself,
and the `A_4kd07` schedule did not help. A medium detail gate with the
`A_4kd05` protection losses currently gives the best short-train result.

Remaining caveat: the validation trajectory is volatile, with best at step
`700` and final step `800` falling back sharply. Dark-channel and saturation
proxies also worsen relative to `A_4kd05`. The next bottleneck is therefore not
"can short training find a better point"; it is confirming this point under a
larger/current protocol and stabilizing the control/gate behavior.

## Run-Local Config

`/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/configs/a_4kd08_fast_4kdehaze_medium_detail_protected_blended_demand_control_probe.toml`
