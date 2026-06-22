# Run Manifest: V1/T01/A_z03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_z03`
- Full identifier: `V1/T01/A_z03`

## Current Status

This branch has now completed branch-local smoke calibration, formal
preflight, formal tmux launch, clean short-budget training, and same-protocol
official post-hoc evaluation. It is now the strongest completed short probe
above frozen milestone `M04/A_y03`, although it still remains far below the
current medium-budget milestone line in absolute performance.

## Purpose

`A_z03` exists to test the narrowest next lightweight upgrade above frozen
milestone `M04/A_y03`.

The immediate question is:

> if Gaussian feature refinement also sees a tiny summary of the current
> decoder detail energy, can it improve high-resolution detail expression
> without breaking the successful light-edge recipe?

## Experimental Question

Can a decoder-detail-aware feature-refinement prior outperform the current
light-edge feature-refinement line while staying inside the same short budget
and the same practical VRAM envelope?

More concretely:

1. keep the full `A_y03/M04` structure fixed;
2. keep `gaussian_feature_refinement_strength = 0.10`;
3. keep `edge_weighted_reconstruction_strength = 0.02`;
4. add only `gaussian_feature_refinement_use_decoder_detail_hint = true`.

## Chosen Mechanism

`A_z03` keeps:

```text
use_gaussian_decoder_guidance = true
use_gaussian_feature_refinement = true
use_edge_weighted_reconstruction = true
```

and changes:

```text
gaussian_feature_refinement_use_decoder_detail_hint = true
```

## Baseline For Comparison

- Primary frozen baseline: `M04/A_y03`
- Matching short-family anchor: `V1/T01/A_u03`
- Earlier successful medium signal: `V1/T01/A_x03`
- Older frozen milestone: `M03/A_k03`

## Planned Protocol

Kept from `A_y03/M04`:

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

Changed from `A_y03/M04`:

- enable decoder-detail-aware hint inside `GaussianGuidedFeatureRefinement`

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

1. branch-local smoke calibration on the true `A_z03` config;
2. formal `train_v1_t01.py --preflight` on the run-local config;
3. clean-GPU serial tmux launch validation;
4. live checkpoint / log / metric writing through `step_002500.pt`;
5. short-budget clean finish;
6. same-protocol post-hoc official full-image test on `final.pt`.

## Calibration Notes

Branch-local smoke on the true `A_z03` config shows:

- `6x728`: `reserved_gb=91.8105`, accepted

Formal preflight on the final run-local config shows:

- `batch_input_shape=(6, 3, 728, 728)`
- `batch_target_shape=(6, 3, 728, 728)`
- `total_params=304776`
- parameter delta vs `A_y03/M04`: `+864`
- `val_psnr=21.7816`

Live training status currently shows:

- tmux session:
  `2dgauss_a_z03_uhddeblur_feature_refinement_detailhint_probe`
- checkpoints written:
  `step_000250.pt`, `step_000500.pt`, `step_000750.pt`, `step_001000.pt`,
  `step_001250.pt`, `step_001500.pt`, `step_001750.pt`, `step_002000.pt`,
  `step_002250.pt`, `step_002500.pt`, `latest.pt`, and `final.pt`
- `step=500`: `val_loss=0.0243871`, `val_psnr=28.4670`
- `step=1000`: `val_loss=0.0240181`, `val_psnr=28.6419`
- `step=1500`: `val_loss=0.0256312`, `val_psnr=28.1106`
- `step=2000`: `val_loss=0.0250846`, `val_psnr=28.5236`
- `step=2500`: `val_loss=0.0231850`, `val_psnr=28.9776`
- final training-time best validation:
  `28.9776 dB @ step=2500`

## Working Hypothesis

If the current main bottleneck is still high-resolution detail under-
expression, then letting feature refinement see a tiny self-conditioned detail
hint should outperform the plain prior-only refinement route. If it does not,
then the next gain likely requires a different lightweight redesign rather than
more local conditioning.

Outcome read:

- `A_z03` did show a mixed mid-run curve, but the branch recovered by the
  final endpoint.
- The clean endpoint `step_002500.pt / latest.pt / final.pt` shares identical
  `model` weights.
- Same-protocol official full-image test on `final.pt` reached `25.3857 dB`,
  which is above the nearby short-probe anchors `A_u03/A_t03/A_v03/A_n03`.
- The branch therefore converts a previously mixed live signal into a real
  positive completed short result.
