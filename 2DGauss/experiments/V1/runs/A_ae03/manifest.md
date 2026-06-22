# Run Manifest: V1/T01/A_ae03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_ae03`
- Full identifier: `V1/T01/A_ae03`

## Current Status

This branch has now completed smoke calibration, formal preflight, short-budget training, checkpoint-endpoint verification, and same-protocol official post-hoc evaluation. It keeps the accepted `M04/A_y03` short envelope fixed, but strengthens the shared Gaussian control surface by letting the shared prior participate directly in confidence normalization.

## Purpose

`A_ae03` exists to test the following refinement of the current bottleneck hypothesis:

> if Gaussian prior control needs both a shared representation and a stronger notion of where the shared prior should matter, can a shared-prior confidence gate outperform the current local redesigns without increasing the lightweight footprint too much?

## Experimental Question

Can a shared-prior confidence-gated Gaussian control surface beat the current short leaders while staying inside the same short-tier runtime envelope?

## Chosen Mechanism

`A_ae03` keeps:

```text
use_gaussian_decoder_guidance = true
use_gaussian_feature_refinement = true
use_edge_weighted_reconstruction = true
use_edge_guidance = true
use_directional_frequency_guidance = true
use_train_inference_consistency = true
use_gaussian_shared_prior = true
```

and changes:

```text
gaussian_feature_refinement_use_confidence_normalization = true
```

The confidence gate now also receives shared-prior energy summaries so the shared control surface can modulate strength more selectively.

## Baseline For Comparison

- Primary frozen baseline: `M04/A_y03`
- Current strongest completed short continuation: `V1/T01/A_z03`
- Best result from the previous serial scout: `V1/T01/A_ad03`
- Negative confidence-normalized reference: `V1/T01/A_ac03`

## Planned Protocol

Kept from `M04/A_y03` short envelope:

- dataset: `UHD_deblur`
- effective training split: train pairs after the first `40`
- internal validation: first `20` train pairs
- validation crop: full image, `crop_size=0`
- official test: `300` full-image pairs, post-hoc only
- edge-guided refinement
- directional/frequency guidance
- `TLC`-style train-inference consistency
- decoder guidance `strength = 0.15`
- feature refinement `strength = 0.10`
- edge-weighted reconstruction `strength = 0.02`
- short-tier wall-clock target

Changed from nearby short branches:

- no decoder self-detail hint
- no direct edge-aware prior injection
- shared prior now influences confidence gating

## Planned Runtime Envelope

- Tier target: short
- Wall-clock target: `58` minutes
- Max steps target: `2500`
- Accepted batch-size / context geometry: `6x728`
- Gradient accumulation target: `1`
- Train supervision size: `512`
- Validation interval target: `500`
- Checkpoint interval target: `250`
- Validation batches target: `20`
- Target reserved VRAM window: `85.0-93.0G`

## Verification Accepted

Accepted items:

1. branch-local smoke calibration on the true config;
2. formal `train_v1_t01.py --preflight` on the true config;
3. clean short-budget training completion;
4. checkpoint-endpoint equality verification;
5. same-protocol official full-image test on `final.pt`.

## Calibration Notes

Branch-local smoke on the true config shows:

- `6x720`: `reserved_gb=91.3281`, accepted
- `avg_total_step_seconds=2.8192`
- `estimated_steps_58min=1234`

Formal preflight on the final config shows:

- `batch_input_shape=(6, 3, 720, 720)`
- `batch_target_shape=(6, 3, 720, 720)`
- `total_params=306264`
- `val_psnr=18.1336`

Completed run notes:

- `step=500`: `val_psnr=28.5609`
- `step=1000`: `val_psnr=28.3061`
- `step=1500`: `val_psnr=28.6193`
- `step=2000`: `val_psnr=28.6578`
- `step=2500`: `val_psnr=28.7005`
- `final.pt` official test: `25.2533 dB`

## Working Hypothesis

If the current bottleneck is not the absence of shared control but the absence of a shared confidence surface, then the shared prior should improve performance only when it is allowed to influence confidence gating itself.
