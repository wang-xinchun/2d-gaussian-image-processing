# Run Manifest: V1/T01/A_ab03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_ab03`
- Full identifier: `V1/T01/A_ab03`

## Current Status

This branch has now completed branch-local smoke calibration, formal
preflight, clean tmux launch, short-budget training, and same-protocol
official post-hoc evaluation. It is a cleanly completed but negative first
post-`A_aa03` structural redesign probe. The branch keeps the successful
`M04/A_y03` short envelope fixed while moving the extra Gaussian
feature-refinement signal from decoder self-hints to explicit edge structure,
but that direct edge-aware prior formulation does not outperform the existing
short leader `A_z03`.

## Purpose

`A_ab03` exists to test the highest-priority structural hypothesis after
`A_aa03` failed as a standalone medium-line promotion:

> if Gaussian feature refinement sees explicit high-resolution edge
> descriptors instead of decoder self-detail hints, can it guide feature-space
> correction more effectively?

## Experimental Question

Can an edge-aware prior inside `GaussianGuidedFeatureRefinement` outperform the
current post-`M04` short continuation family while staying inside the same
short-budget runtime envelope?

More concretely:

1. keep the full `A_y03/M04` short envelope fixed;
2. keep `gaussian_feature_refinement_strength = 0.10`;
3. keep `edge_weighted_reconstruction_strength = 0.02`;
4. disable `gaussian_feature_refinement_use_decoder_detail_hint`;
5. enable `gaussian_feature_refinement_use_edge_aware_prior`.

## Chosen Mechanism

`A_ab03` keeps:

```text
use_gaussian_decoder_guidance = true
use_gaussian_feature_refinement = true
use_edge_weighted_reconstruction = true
```

and changes:

```text
gaussian_feature_refinement_use_decoder_detail_hint = false
gaussian_feature_refinement_use_edge_aware_prior = true
```

## Baseline For Comparison

- Primary frozen baseline: `M04/A_y03`
- Matching short-family anchor: `V1/T01/A_z03`
- Recent failed medium promotion: `V1/T01/A_aa03`

## Planned Protocol

Kept from `A_y03/M04` and `A_z03`:

- dataset: `UHD_deblur`
- effective training split: train pairs after the first `40`
- internal validation: first `20` train pairs
- validation crop: full image, `crop_size=0`
- official test: `300` full-image pairs, post-hoc only
- edge-guided refinement
- directional/frequency guidance
- `TLC`-style train-inference consistency
- decoder guidance `strength = 0.15`
- feature refinement enabled
- edge-weighted reconstruction enabled at `0.02`
- prepared geometry target: `6x728`
- short-tier wall-clock target

Changed from `A_z03`:

- remove decoder self-detail hint
- add explicit edge-aware prior inside feature refinement

## Planned Runtime Envelope

- Tier target: short
- Wall-clock target: `58` minutes
- Max steps target: `2500`
- Initial batch-size hypothesis: `6`
- Gradient accumulation target: `1`
- Train context size: `728`
- Train supervision size: `512`
- Validation interval target: `500`
- Checkpoint interval target: `250`
- Validation batches target: `20`
- Target reserved VRAM window: `85.0-93.0G`

## Verification Planned

Accepted so far:

1. branch-local smoke calibration on the true `A_ab03` config;
2. formal `train_v1_t01.py --preflight` on the run-local config.

Accepted so far:

1. branch-local smoke calibration on the true `A_ab03` config;
2. formal `train_v1_t01.py --preflight` on the run-local config;
3. clean-GPU serial tmux launch validation.

Accepted so far:

1. branch-local smoke calibration on the true `A_ab03` config;
2. formal `train_v1_t01.py --preflight` on the run-local config;
3. clean-GPU serial tmux launch validation;
4. short-budget clean finish;
5. same-protocol post-hoc official full-image test on `final.pt`.

## Calibration Notes

Branch-local smoke on the true `A_ab03` config shows:

- `6x728`: `reserved_gb=91.8457`, accepted

Formal preflight on the final run-local config shows:

- `batch_input_shape=(6, 3, 728, 728)`
- `batch_target_shape=(6, 3, 728, 728)`
- `total_params=306072`
- parameter delta vs `A_y03/M04`: `+2160`
- `val_psnr=23.3538`

Live startup status currently shows:

- tmux session:
  `2dgauss_a_ab03_uhddeblur_edgeaware_feature_refinement_probe`
- startup `nvidia-smi`:
  `memory.used≈94507 MiB`, `utilization.gpu=100%`
- training log:
  `experiments/V1/runs/A_ab03/logs/train_20260517_234834.log`
- `train_metrics.jsonl` already writing through at least `step=20`
- early loss trend:
  `step1 loss_total=0.15306 -> step10=0.06798 -> step20=0.05440`

Final completed status now shows:

- checkpoints written:
  `step_000250.pt`, `step_000500.pt`, `step_000750.pt`, `step_001000.pt`,
  `step_001250.pt`, `step_001500.pt`, `step_001750.pt`, `step_002000.pt`,
  `step_002250.pt`, `step_002500.pt`, `latest.pt`, and `final.pt`
- `step=500`: `val_loss=0.0248542`, `val_psnr=28.5939`
- `step=1000`: `val_loss=0.0246931`, `val_psnr=28.5592`
- `step=1500`: `val_loss=0.0239989`, `val_psnr=28.5976`
- `step=2000`: `val_loss=0.0257546`, `val_psnr=28.2654`
- `step=2500`: `val_loss=0.0240276`, `val_psnr=28.7948`
- final training-time best validation PSNR:
  `28.7948 dB @ step=2500`
- official test on `final.pt`:
  `25.1723 dB`

## Working Hypothesis

If the main post-`A_aa03` bottleneck were simply “feature refinement needs
explicit high-resolution geometry instead of decoder self-hints,” then the
edge-aware prior should have outperformed the nearby short family under the
same envelope.

Outcome read:

- `A_ab03` trains cleanly and is runtime-safe.
- But same-protocol official full-image test reaches only `25.1723 dB`, which
  is clearly below `A_z03` and below the older short anchors
  `A_u03/A_t03/A_n03`.
- The correct conclusion is therefore that this **direct** edge-aware prior
  formulation is not the right lightweight upgrade path.
