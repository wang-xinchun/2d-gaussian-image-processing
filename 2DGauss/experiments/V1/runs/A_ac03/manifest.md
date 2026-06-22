# Run Manifest: V1/T01/A_ac03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_ac03`
- Full identifier: `V1/T01/A_ac03`

## Current Status

This branch has now completed branch-local smoke calibration, formal
preflight, serial short-budget training, checkpoint-endpoint verification,
and same-protocol official post-hoc evaluation. It is a cleanly completed but
negative short structural probe. The branch keeps the accepted `M04/A_y03`
short recipe fixed while changing Gaussian feature refinement from uniform
modulation into confidence-normalized modulation, but that formulation does
not outperform the current short leader `A_z03`, nor even the already
negative `A_ab03`.

## Purpose

`A_ac03` exists to test the first literature-guided bottleneck hypothesis:

> if the remaining problem is over-refinement in easy regions rather than
> missing local cues, can a lightweight confidence-normalized Gaussian
> refinement improve short-tier official performance?

## Experimental Question

Can confidence-normalized Gaussian feature refinement outperform the recent
short continuation family while staying inside the same practical lightweight
deployment envelope?

More concretely:

1. keep the full `A_y03/M04` short recipe fixed;
2. keep decoder guidance at `0.15`;
3. keep feature refinement enabled at `0.10`;
4. keep edge-weighted reconstruction enabled at `0.02`;
5. replace uniform refinement strength with confidence-normalized modulation.

## Chosen Mechanism

`A_ac03` keeps:

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
gaussian_feature_refinement_use_confidence_normalization = true
train_context_size = 716
```

## Baseline For Comparison

- Primary frozen baseline: `M04/A_y03`
- Current strongest completed short continuation: `V1/T01/A_z03`
- Companion serial scout: `V1/T01/A_ad03`
- Negative redesign reference: `V1/T01/A_ab03`

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
- add confidence-normalized refinement control

## Planned Runtime Envelope

- Tier target: short
- Wall-clock target: `58` minutes
- Max steps target: `2500`
- Accepted batch-size / context geometry: `6x716`
- Gradient accumulation target: `1`
- Train supervision size: `512`
- Validation interval target: `500`
- Checkpoint interval target: `250`
- Validation batches target: `20`
- Target reserved VRAM window: `85.0-93.0G`

## Verification Accepted

Accepted items:

1. branch-local smoke calibration on the true `A_ac03` config;
2. formal `train_v1_t01.py --preflight` on the true config;
3. clean serial short-budget completion;
4. checkpoint-endpoint equality verification;
5. same-protocol official full-image test on `final.pt`.

## Calibration Notes

Branch-local smoke on the true `A_ac03` config shows:

- `6x716`: `reserved_gb=92.3633`, accepted
- `avg_total_step_seconds=1.9850`
- `estimated_steps_58min=1753`

Formal preflight on the final config shows:

- `batch_input_shape=(6, 3, 716, 716)`
- `batch_target_shape=(6, 3, 716, 716)`
- `total_params=319320`
- parameter delta vs `A_ab03`: `+13248`
- parameter delta vs `M04/A_y03`: `+15408`
- `val_psnr=21.9700`

Final completed status now shows:

- training log:
  `experiments/V1/runs/A_ac03/logs/train_20260518_012147.log`
- checkpoints written:
  `step_000250.pt`, `step_000500.pt`, `step_000750.pt`, `step_001000.pt`,
  `step_001250.pt`, `step_001500.pt`, `step_001750.pt`, `step_002000.pt`,
  `step_002250.pt`, `step_002500.pt`, `latest.pt`, and `final.pt`
- `step=500`: `val_loss=0.0243968`, `val_psnr=28.5750`
- `step=1000`: `val_loss=0.0231508`, `val_psnr=28.8683`
- `step=1500`: `val_loss=0.0235234`, `val_psnr=28.6933`
- `step=2000`: `val_loss=0.0239413`, `val_psnr=28.6603`
- `step=2500`: `val_loss=0.0250348`, `val_psnr=28.3405`
- final training-time best validation PSNR:
  `28.8683 dB @ step=1000`
- training clean finish:
  `step=2500`
- training elapsed minutes:
  `37.2581`
- checkpoint relation:
  `step_002500.pt == latest.pt == final.pt` (model weights)
- official test on `final.pt`:
  `25.1474 dB`

## Working Hypothesis

If the main post-`A_ab03` bottleneck were that Gaussian refinement still
intervenes too uniformly across easy and hard regions, then confidence
normalization should have improved the spatial balance of refinement without
requiring heavier branches or more explicit cue injection.

Outcome read:

- `A_ac03` is runtime-safe and scientifically interpretable.
- But its best internal-validation point appears early and does not convert
  into a stronger official endpoint.
- The correct conclusion is therefore that this **current**
  confidence-normalized refinement design is not the right lightweight upgrade
  path.
