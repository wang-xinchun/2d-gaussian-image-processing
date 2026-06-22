# Run Manifest: V1/T01/A_aq03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_aq03`
- Full identifier: `V1/T01/A_aq03`

## Current Status

Completed. Short training and post-hoc evaluation are done.

## Purpose

`A_aq03` is the first post-advisor mainline probe for explicit frequency-split Gaussian feedback. It directly addresses the current bottleneck: the previous router family tried to choose roles after mixed features were already entangled, while the advisor asked how the branch entrance separates low-frequency structure and high-frequency details.

## Mechanism

The branch adds a default-off explicit frequency split to `CrossResolutionGaussianRestorer`:

```text
input RGB
  -> low band: avg-pool lowpass -> LR/pixel_unshuffle -> Gaussian parameter/control canvas
  -> high band: RGB - low band -> HR detail projector -> high-resolution feature injection
  -> Gaussian attention feedback: attention_map softly modulates high-band detail feature
```

The intended role split is:

- Low/structure path: cleaner input to LR Gaussian head.
- High/detail path: high-resolution feature branch carries local detail.
- Gaussian feedback: the low-resolution control canvas returns to high-resolution detail recovery rather than trying to synthesize RGB texture by itself.

## Baselines

- Frozen medium baseline: `M04/A_y03`, official test `26.272695829299355`
- Current short leader: `A_z03`, official test `25.385710820365173`
- Stable structured-control candidate: `A_ah03`, official test `25.345361164304613`
- Latest completed run: `A_ap03`, official test `25.233562564509878`

## Protocol

- Dataset: `UHD_deblur`
- Train split: after first `40` pairs
- Internal validation: first `20` train pairs, full image
- Official test: post-hoc, `300` full-image pairs
- Accepted launch geometry: `batch_size=6`, `train_context_size=704`, supervision crop `512`
- Budget: `max_steps=2500`, `wall_clock_minutes=58`
- Reserved VRAM target: `85.0-93.0G`

## Supervision

This branch keeps the fidelity anchor and adds only small deblur proxy weights under the supervision safety guardrails:

```text
charbonnier_weight = 1.0
gradient_weight = 0.05
laplacian_weight = 0.02
edge_weighted_reconstruction_strength = 0.02
edge_magnitude_weight = 0.02
high_frequency_weight = 0.02
local_contrast_weight = 0.01
```

The proxy weights are auxiliary only; promotion requires official fidelity, task diagnostics, and visual panels.

## Gate Status

- [x] Static py_compile
- [x] Explicit split forward/aux check
- [x] Formal preflight: `6x704`, `total_params=318888`, `val_psnr=20.073515259173767`
- [x] True smoke calibration: `6x704`, `max_memory_reserved_gb=92.41015625`, inside `85.0-93.0G`
- [x] Launch only if gates pass

## Launch

- Training finished cleanly at `step=2500`
- Official test: `samples=300`, `crop_size=0`, `device=cuda`, `loss_mean=0.03899502913157145`, `psnr_mean=25.281006597978177`
- Task-specific diagnostics: `ssim_mean=0.7757521148522695`, `edge_magnitude_l1_mean=0.0966632444318384`, `high_frequency_l1_mean=0.012849770373820017`, `local_contrast_l1_mean=0.01624760725651868`, `luminance_l1_mean=0.0318775971342499`, `chroma_l1_mean=0.00515300480183214`, `saturation_l1_mean=0.010168556636199356`, `dark_channel_l1_mean=0.04577985843798767`
- Final mechanism panels: `experiments/V1/runs/A_aq03/diagnostics/mechanism_final_test`
- Final attention diagnosis: `attention_map mean≈0.50023`, `std≈0.00017`

## Decision Rule

Treat `A_aq03` as positive only if it improves or is competitive under same-protocol short official metrics while also improving task-specific diagnostics and visual detail without ringing, halo, over-sharpening, checkerboard artifacts, or color shift. Final judgment: it is a positive short probe, but not a promotion candidate, because the control canvas remains nearly flat and it still trails `A_ah03`, `A_z03`, and frozen `M04/A_y03`.
