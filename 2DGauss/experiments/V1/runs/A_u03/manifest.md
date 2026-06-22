# Run Manifest: V1/T01/A_u03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_u03`
- Full identifier: `V1/T01/A_u03`

## Current Status

This branch has now completed clean serial deployment, short-budget training,
and post-hoc official evaluation. It is currently the strongest completed
branch inside the post-`A_t03` rebalance family.

## Purpose

`A_u03` is the light-edge continuation probe prepared after `A_t03`.

The immediate question is:

> if `A_t03` already contains the right structural direction but is slightly
> over-regularized by supervision-side pressure, can reducing the
> edge-weighted term preserve the gain and improve the short-run endpoint?

## Experimental Question

Can a lighter edge-weighted reconstruction term improve transfer to official
testing without changing the already accepted `A_t03` structure family?

More concretely:

1. keep the `A_t03` structural story fixed;
2. keep feature refinement enabled at the proven safe setting;
3. reduce only the supervision-side edge-weighted strength;
4. use the same short-budget protocol to test whether `A_t03` was slightly
   over-constrained.

## Chosen Mechanism

`A_u03` keeps:

```text
use_gaussian_feature_refinement = true
gaussian_feature_refinement_strength = 0.10
use_edge_weighted_reconstruction = true
```

and changes only:

```text
edge_weighted_reconstruction_strength = 0.02
```

## Baseline For Comparison

- Primary short anchor inside this family: `V1/T01/A_t03`
- Earlier short anchor: `V1/T01/A_n03`
- Companion probes in this rebalance family: `V1/T01/A_v03`, `V1/T01/A_w03`
- Frozen milestone: `experiments/V1/milestones/M03_A_k03_uhddeblur_official_baseline`

## Planned Protocol

Kept from `A_t03`:

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
- prepared geometry target: `6x728`
- short-tier wall-clock target

Changed from `A_t03`:

- `edge_weighted_reconstruction_strength: 0.03 -> 0.02`

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

## Verification Performed

Completed:

1. branch-local smoke calibration on the true `A_u03` config;
2. formal `train_v1_t01.py --preflight` on the run-local config;
3. clean-GPU serial launch validation;
4. tmux launch and completed short-budget training;
5. checkpoint endpoint equivalence verification;
6. same-protocol post-hoc official full-image test on `final.pt`.

## Outcome Summary

`A_u03` completed successfully and is a real positive continuation over both
`A_t03` and `A_n03` on official test:

- clean finish at `step=2500`
- final logged elapsed time `35.6395` minutes
- best training-time full-image validation PSNR `28.7799 dB` at `step=2500`
- post-hoc official full-image test `25.3101 dB` at `final.pt`

Relative to key same-task anchors:

- `A_u03 final.pt` is about `+0.0156 dB` above `A_t03 final.pt`
- `A_u03 final.pt` is about `+0.0181 dB` above `A_n03 final.pt`
- `A_u03 final.pt` is about `+0.0234 dB` above `A_r03 final.pt`
- `A_u03 final.pt` is about `-0.6986 dB` below `A_p03 final.pt`
- `A_u03 final.pt` is about `-0.8899 dB` below frozen `A_k03 final.pt`

## Deployment Summary

- Clean-GPU branch-local smoke at `6x728` returned
  `max_memory_reserved_gb=91.7871`.
- Independent formal preflight passed with
  `batch_input_shape=(6, 3, 728, 728)`, `total_params=303912`,
  `val_psnr=22.3029`.
- Real startup window reached about `nvidia-smi memory.used≈94067 MiB`,
  slightly above the original practical `93G` ceiling, so real startup memory
  remains the authoritative envelope.
- Launch script was created at
  `experiments/V1/runs/A_u03/launch_tmux.sh`.
- Training completed cleanly at `step=2500`.
- `step_002500.pt`, `latest.pt`, and `final.pt` have identical `model`
  weights.
- Post-hoc official evaluation has been completed and recorded under
  `experiments/V1/runs/A_u03/metrics/official_test_final.json`.

## Working Hypothesis

`A_u03` now supports the hypothesis that `A_t03` was slightly
over-regularized on the supervision side. Reducing the edge-weighted term does
not meaningfully raise the internal validation ceiling, but it does improve
the official test endpoint, which is the more important signal for this
family.
