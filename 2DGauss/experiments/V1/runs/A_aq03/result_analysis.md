# Result Analysis: V1/T01/A_aq03

Status: completed.

This file is the post-hoc record for the explicit frequency-split Gaussian feedback probe. It reports:

- same-protocol official `UHD_deblur/test300` result;
- task-specific diagnostics from `scripts/evaluate_task_specific_metrics.py`;
- low/high band and Gaussian control visual observations;
- failure-mode checklist: ringing, halo, over-sharpening, color shift, checkerboard, edge break, noise amplification;
- final decision against `A_z03`, `A_ah03`, `A_ap03`, and frozen `M04/A_y03`.

## Official Result

- Checkpoint: `experiments/V1/runs/A_aq03/checkpoints/final.pt`
- Checkpoint step: `2500`
- Split: `test`
- Samples: `300`
- Crop size: `0`
- Device: `cuda`
- `loss_mean`: `0.03899502913157145`
- `psnr_mean`: `25.281006597978177`
- `elapsed_seconds`: `499.73809719085693`

Compared against nearby baselines:

- vs `A_ap03` final: `+0.047443633468299` dB
- vs `A_ah03` final: `-0.064354566826436` dB
- vs `A_z03` final: `-0.104704222387` dB
- vs frozen `M04/A_y03`: `-0.991689231321176` dB

## Task-Specific Diagnostics

- `ssim_mean`: `0.7757521148522695`
- `edge_magnitude_l1_mean`: `0.0966632444318384`
- `high_frequency_l1_mean`: `0.012849770373820017`
- `local_contrast_l1_mean`: `0.01624760725651868`
- `luminance_l1_mean`: `0.0318775971342499`
- `chroma_l1_mean`: `0.00515300480183214`
- `saturation_l1_mean`: `0.010168556636199356`
- `dark_channel_l1_mean`: `0.04577985843798767`

These diagnostics are consistent with a run that improved paired fidelity and sharpness over the input, but they are not a substitute for `FADE`, `NIQE`, `BRISQUE`, `LPIPS`, or `DISTS`, which are still not wired into the codebase.

## Visual Observations

The final mechanism panels show a clean low/high split, but the Gaussian control map remains visually low-entropy and close to a mild texture map rather than a strong spatial controller. The low-band image is a smooth structural carrier, while the high-band and guided-detail channels concentrate fine residual energy. The restored output is visibly better than the input and recovers more edge structure on foliage and branches, but it still reads as soft rather than decisively crisp.

Two representative samples from `mechanism_final_test`:

- `VID_001_00051.jpg`: input `22.15 -> 23.67 dB`, SSIM `0.594 -> 0.674`
- `VID_001_00054.jpg`: input `21.50 -> 22.43 dB`, SSIM `0.546 -> 0.599`

The control canvas remains almost constant across these examples, with `attention_map mean≈0.50023` and `std≈0.00017`.

## Failure-Mode Checklist

- Ringing: not prominent
- Halo: weak at most
- Over-sharpening: not dominant
- Color shift: not prominent
- Checkerboard: not observed
- Edge break: mild on dense foliage and branch boundaries
- Noise amplification: not observed

The main remaining issue is not artifact explosion; it is that the Gaussian control canvas is still too flat to carry a strong, interpretable story about spatial control.

## Final Decision

`A_aq03` is a positive short probe, but not a promotion candidate. It is better than `A_ap03`, yet it still trails `A_ah03`, `A_z03`, and frozen `M04/A_y03`. The run confirms that explicit low/high splitting can help, but the current formulation does not yet make Gaussian control visibly structured enough to be the paper-level endpoint.
