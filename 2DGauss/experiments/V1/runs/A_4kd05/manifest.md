# Run Manifest: V1/T01/A_4kd05

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_4kd05`
- Full identifier: `V1/T01/A_4kd05`

## Current Status

Run completed. `A_4kd05` is accepted as a fast-probe
generalization/detail-protection candidate, not as an official/full benchmark.

## Purpose

`A_4kd05` is the detail-protected follow-up to accepted fast-probe candidate
`A_4kd04`. It keeps the successful blended shared demand-control path, but
adds a local detail-preserve gate to reduce control blending over high-frequency
and edge-rich regions.

The hypothesis is narrow: `A_4kd04` improved dehazing/brightness/SSIM but
worsened edge, high-frequency, local-contrast, and chroma diagnostics. `A_4kd05`
should preserve most of the dehazing gain while reducing those detail losses.

## Dataset

- Dataset: `4KDehaze_local_90_5_5_fast_probe_512`
- Project split view:
  `/home/ldj/wxc/2DGauss/datasets/current_training_ready/views/4KDehaze_local_90_5_5_fast_probe_512`
- Train pairs: `512`
- Validation pairs: `100`
- Test pairs: `100`
- Boundary: same fast-probe view as `A_4kd02`, `A_4kd03`, and `A_4kd04`.

## Model Delta

Relative to `A_4kd04`, this run changes only the control blending/protection
surface and the small task-aware losses:

- `gaussian_control_canvas_share_to_guidance_detail_preserve = true`
- `gaussian_control_canvas_share_to_guidance_detail_preserve_strength = 0.65`
- `gaussian_control_canvas_share_to_guidance_detail_preserve_floor = 0.35`
- `gaussian_control_canvas_share_to_guidance_detail_preserve_kernel_size = 5`
- `edge_magnitude_weight = 0.015`
- `high_frequency_weight = 0.025`
- `local_contrast_weight = 0.02`
- `chroma_weight = 0.01`
- `use_edge_weighted_reconstruction = true`
- `edge_weighted_reconstruction_strength = 0.01`

The uniform `A_4kd04` blend was `0.25`. With detail preservation enabled,
smooth/low-detail regions stay close to that value, while high-detail regions
receive lower effective blend.

## Diagnostics

`training.control_diagnostics = true` remains enabled. New required readout:

- `val_control_guidance_blend_mean/std/min/max`
- `val_control_guidance_blend_high_detail_mean`
- `val_control_guidance_blend_low_detail_mean`
- `val_control_guidance_blend_detail_corr`

Success requires high-detail mean below low-detail mean, while validation/test
metrics stay competitive with or improve over `A_4kd04`.

## Preflight

Formal GPU3 preflight completed at `2026-06-10 18:25:18 CST` with
`TWODGAUSS_CUDA_MEMORY_FRACTION=0.30` and `TWODGAUSS_PREFLIGHT_ONLY=1`.

- Log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd05/logs/train_20260610_182518.log`
- Exit code: `0`
- Validation samples: `100`
- `val_psnr`: `9.40824378005754`
- `val_loss`: `0.3547653856873512`
- `val_control_guidance_blend_high_detail_mean`: `0.15471820175647735`
- `val_control_guidance_blend_low_detail_mean`: `0.22401548877358438`
- `val_control_guidance_blend_detail_corr`: `-0.8495775413513184`

The detail-protection gate is active: effective guidance blend is lower in
high-detail regions than in low-detail regions. Because A_4kd05 adds detail
protection losses, checkpoint selection is by `val_psnr` rather than raw
`val_loss`.

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

## Final Result

- Formal training log:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd05/logs/train_20260610_183150.log`
- Exit code: `0`
- Best validation step: `600`
- Best validation PSNR: `17.595543271291344`
- Final step `800` validation PSNR: `17.33071631834636`
- CPU test PSNR: `17.067359898802888`
- CPU test SSIM: `0.6860608831048012`
- Result analysis:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd05/result_analysis.md`

Relative to `A_4kd04`, validation peak is lower by about `0.0892 dB`, but CPU
test PSNR is higher by about `0.1660 dB` and SSIM is higher by about `0.00988`.
Edge/high-frequency/local-contrast diagnostics are also partially repaired.

## Run-Local Config

`/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd05/configs/a_4kd05_fast_4kdehaze_detail_protected_blended_demand_control_probe.toml`

## Stop Criteria

If step `300` is materially behind `A_4kd04` without clear detail-diagnostic
repair, stop rather than running the full 800 steps. If detail diagnostics
improve and validation is competitive, continue to step `800` and run CPU test
eval on the best checkpoint.
