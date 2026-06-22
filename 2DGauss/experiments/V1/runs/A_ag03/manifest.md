# Run Manifest: V1/T01/A_ag03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_ag03`
- Full identifier: `V1/T01/A_ag03`

## Current Status

Prepared but not yet launched. This branch is the second new short probe in the next probe family after `A_ae03`. It keeps the accepted `M04/A_y03` short envelope fixed, but changes feature refinement so it can learn an explicit intervention demand signal rather than only a confidence-suppression signal.

## Purpose

`A_ag03` exists to test a demand-formation hypothesis:

> if confidence normalization and shared-prior gating improved selectivity but did not fully solve the control problem, can feature refinement do better by learning where intervention should be actively requested?

## Experimental Question

Can an intervention-demand gate in Gaussian feature refinement improve the current shared-control surface while staying inside the same short-tier runtime envelope?

## Chosen Mechanism

`A_ag03` keeps:

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
gaussian_feature_refinement_use_intervention_demand = true
```

The refinement gate now has a positive demand path, so the model can learn where to ask for stronger intervention instead of only learning where to damp confidence.

## Baseline For Comparison

- Primary frozen baseline: `M04/A_y03`
- Current strongest completed short continuation: `V1/T01/A_z03`
- Local shared-control result to beat: `V1/T01/A_ae03`
- Confidence-normalized negative reference: `V1/T01/A_ac03`

## Planned Protocol

Kept from the current short envelope:

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

- feature refinement gets an explicit intervention-demand path
- the branch still keeps shared prior enabled
- the branch does not introduce a new cue source

## Planned Runtime Envelope

- Tier target: short
- Wall-clock target: `58` minutes
- Max steps target: `2500`
- Accepted batch-size / context geometry: `6x720`
- Gradient accumulation target: `1`
- Train supervision size: `512`
- Validation interval target: `500`
- Checkpoint interval target: `250`
- Validation batches target: `20`
- Target reserved VRAM window: `85.0-93.0G`

## Verification To Be Performed

Planned items:

1. branch-local smoke calibration on the true config;
2. formal `train_v1_t01.py --preflight` on the true config;
3. clean short-budget training completion;
4. checkpoint-endpoint equality verification;
5. same-protocol official full-image test on `final.pt`.

## Working Hypothesis

If the bottleneck is not only where to suppress intervention but where to request it, then a direct demand signal in feature refinement should improve the shared control surface.
