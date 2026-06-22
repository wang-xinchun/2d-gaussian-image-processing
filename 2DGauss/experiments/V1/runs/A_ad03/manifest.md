# Run Manifest: V1/T01/A_ad03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_ad03`
- Full identifier: `V1/T01/A_ad03`

## Current Status

This branch has now completed branch-local smoke calibration, formal
preflight, serial short-budget training, checkpoint-endpoint verification,
and same-protocol official post-hoc evaluation. It is the better result
inside the new `A_ac03/A_ad03` serial scout pair, but it is still a globally
insufficient short continuation. The branch keeps the accepted `M04/A_y03`
short envelope fixed while introducing a shared compact prior-control stem
consumed by both Gaussian decoder guidance and Gaussian feature refinement.

## Purpose

`A_ad03` exists to test the second bottleneck hypothesis:

> if the remaining problem is poor coordination between Gaussian decoder
> guidance and Gaussian feature refinement, can a shared prior-control stem
> improve the short-tier line without heavier modeling?

## Experimental Question

Can shared-prior coupling between the two Gaussian control modules outperform
the current short continuation family while staying inside the same short-tier
runtime envelope?

## Chosen Mechanism

`A_ad03` keeps:

```text
use_gaussian_decoder_guidance = true
use_gaussian_feature_refinement = true
use_edge_weighted_reconstruction = true
use_edge_guidance = true
use_directional_frequency_guidance = true
use_train_inference_consistency = true
```

and changes:

```text
use_gaussian_shared_prior = true
gaussian_shared_prior_blocks = 1
```

## Baseline For Comparison

- Primary frozen baseline: `M04/A_y03`
- Current strongest completed short continuation: `V1/T01/A_z03`
- First post-`A_aa03` redesign reference: `V1/T01/A_ab03`
- Serial companion probe: `V1/T01/A_ac03`

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
- add shared prior-control stem consumed by both Gaussian modules

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

1. branch-local smoke calibration on the true `A_ad03` config;
2. formal `train_v1_t01.py --preflight` on the true config;
3. clean serial short-budget completion;
4. checkpoint-endpoint equality verification;
5. same-protocol official full-image test on `final.pt`.

## Calibration Notes

Branch-local smoke on the true `A_ad03` config shows:

- `6x728`: `reserved_gb=86.0547`, accepted
- `avg_total_step_seconds=1.8772`
- `estimated_steps_58min=1853`

Formal preflight on the final config shows:

- `batch_input_shape=(6, 3, 728, 728)`
- `batch_target_shape=(6, 3, 728, 728)`
- `total_params=289992`
- parameter delta vs `M04/A_y03`: `-13920`

Formal parameter comparison note:

- `A_ad03` stays lighter than the other post-`A_aa03` redesign probes.
- Relative to `A_ab03 total_params=306072`, this branch is `-16080`.
- `val_psnr=17.4351`

Final completed status now shows:

- training log:
  `experiments/V1/runs/A_ad03/logs/train_20260518_012147_after_ac03.log`
- checkpoints written:
  `step_000250.pt`, `step_000500.pt`, `step_000750.pt`, `step_001000.pt`,
  `step_001250.pt`, `step_001500.pt`, `step_001750.pt`, `step_002000.pt`,
  `step_002250.pt`, `step_002500.pt`, `latest.pt`, and `final.pt`
- `step=500`: `val_loss=0.0240792`, `val_psnr=28.6524`
- `step=1000`: `val_loss=0.0241585`, `val_psnr=28.6875`
- `step=1500`: `val_loss=0.0240461`, `val_psnr=28.6774`
- `step=2000`: `val_loss=0.0245466`, `val_psnr=28.5878`
- `step=2500`: `val_loss=0.0244863`, `val_psnr=28.6896`
- final training-time best validation PSNR:
  `28.6896 dB @ step=2500`
- training clean finish:
  `step=2500`
- training elapsed minutes:
  `33.6803`
- checkpoint relation:
  `step_002500.pt == latest.pt == final.pt` (model weights)
- official test on `final.pt`:
  `25.2008 dB`

## Working Hypothesis

If the main post-`A_ab03` bottleneck were weak coordination between the two
Gaussian control modules rather than missing cues, then a shared prior-control
stem should have improved coupling without meaningfully increasing runtime
cost.

Outcome read:

- `A_ad03` is runtime-safe, lighter than the nearby redesigns, and more
  stable than `A_ac03`.
- It is also the best result among `A_ab03/A_ac03/A_ad03`.
- But because it still stays below `A_z03` and the older short anchors, this
  current shared-prior design is not yet strong enough for promotion.
