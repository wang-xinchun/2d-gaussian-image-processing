# Run Manifest: V1/T01/A_4kd06

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_4kd06`
- Full identifier: `V1/T01/A_4kd06`

## Current Status

Completed formal preflight, GPU3 low-footprint training, CPU test evaluation,
and task-specific CPU diagnostics on 2026-06-10.

## Purpose

`A_4kd06` is a narrow isolation experiment after `A_4kd04` and `A_4kd05`.
It keeps the successful `A_4kd04` blended shared demand-control path and loss
口径, then adds only a lighter detail-aware control-to-guidance blend gate.

The hypothesis is narrow: `A_4kd05` improved CPU test PSNR/SSIM and repaired
some detail proxies, but it mixed two changes: detail-aware blend gating plus
extra edge/chroma/loss pressure. `A_4kd06` isolates the blend-gate effect by
removing the extra `A_4kd05` protection losses and making the gate less
aggressive.

## Dataset

- Dataset: `4KDehaze_local_90_5_5_fast_probe_512`
- Project split view:
  `/home/ldj/wxc/2DGauss/datasets/current_training_ready/views/4KDehaze_local_90_5_5_fast_probe_512`
- Train pairs: `512`
- Validation pairs: `100`
- Test pairs: `100`
- Boundary: same fast-probe view as `A_4kd02` through `A_4kd05`.

## Model Delta

Relative to `A_4kd04`, this run changes only the control blend map:

- `gaussian_control_canvas_share_to_guidance_detail_preserve = true`
- `gaussian_control_canvas_share_to_guidance_detail_preserve_strength = 0.45`
- `gaussian_control_canvas_share_to_guidance_detail_preserve_floor = 0.55`
- `gaussian_control_canvas_share_to_guidance_detail_preserve_kernel_size = 5`

The base blend remains `0.25`. This is intentionally milder than `A_4kd05`
(`strength=0.65`, `floor=0.35`) and keeps the `A_4kd04` loss settings.

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

`A_4kd06` is not an official/full benchmark. It should be judged only against
the same fast-probe references:

- `A_4kd04`: validation peak reference
- `A_4kd05`: CPU test PSNR/SSIM and detail-protection reference
- `A_4kd02`: older low-footprint fast-probe baseline

Useful outcomes:

- If `A_4kd06` keeps most of `A_4kd05` test PSNR/SSIM while reducing
  dark-channel/chroma/saturation cost, the lighter gate is the cleaner next
  direction.
- If it reverts toward `A_4kd04`, the extra `A_4kd05` loss pressure mattered.
- If it underperforms both, spatial gate tuning should be scheduled or delayed
  rather than applied from step 1.

## Result

`A_4kd06` is a neutral/diagnostic result rather than a new promoted candidate.
It keeps nearly all of the `A_4kd04` validation peak, but its CPU test metrics
also revert almost exactly to `A_4kd04`; it does not reproduce the `A_4kd05`
test PSNR/SSIM improvement.

- Formal preflight log:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd06/logs/train_20260610_225331.log`
- Formal training log:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd06/logs/train_20260610_225710.log`
- Train exit code: `0`
- Actual GPU: physical GPU3 only
- Project GPU1/GPU2 use: none
- Peak reserved CUDA memory: about `11.244140625GiB`
- Best validation checkpoint: step `600`
- Best validation PSNR: `17.674754617905425`
- Best validation loss: `0.1261259661614895`
- Final step `800` validation PSNR: `17.121298116103546`
- CPU test PSNR: `16.89408572072857`
- CPU test loss: `0.12854937952011825`
- CPU test SSIM: `0.6761693635582924`

Relative to `A_4kd04`, best validation is only `-0.0100 dB`, CPU test PSNR is
`-0.0073 dB`, and SSIM is effectively unchanged. Relative to `A_4kd05`, CPU
test PSNR is `-0.1733 dB` and SSIM is `-0.00989`. This means the milder
detail-aware blend gate by itself is too weak to explain the `A_4kd05`
generalization/detail-protection gain.

## Run-Local Config

`/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd06/configs/a_4kd06_fast_4kdehaze_light_detail_blended_demand_control_probe.toml`
