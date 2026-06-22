# Run Manifest: V1/T01/A_n03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_n03`
- Full identifier: `V1/T01/A_n03`

## Current Status

This branch has now completed clean serial deployment, short-budget training,
and post-hoc official evaluation.

## Purpose

`A_n03` is the direct default-strength follow-up to `A_m03`.

Its question is intentionally narrow:

> once low-strength decoder guidance has shown a viable short-run signal, does
> the original default strength `0.15` improve or over-drive that signal under
> the same `6x784` runtime geometry?

## Main Design Choice

Relative to `A_m03`, this branch changes only:

- `gaussian_decoder_guidance_strength = 0.10 -> 0.15`

Preserved on purpose:

- `batch_size = 6`
- `train_context_size = 784`
- `train_supervision_crop_size = 512`
- edge-guided refinement
- directional/frequency guidance
- `TLC`-style train-inference consistency
- the same effective `UHD_deblur` train split after the first `40` pairs

## Planned Runtime Envelope

- Tier target: `short`
- Planned wall-clock cap: `58` minutes
- Planned max steps: `2500`
- Planned batch size: `6`
- Planned training context: `784`
- Planned supervision crop: `512`
- Target reserved VRAM window: `85.0-93.0G`

## Outcome Summary

The branch completed successfully and currently gives the strongest short-run
validation signal in the probe family:

- clean finish at `step=2500`
- final logged elapsed time `33.7058` minutes
- best training-time full-image validation PSNR `28.9612 dB` at `step=2500`
- post-hoc official full-image test `25.2919 dB` at `final.pt`

## Comparison Anchor

- Frozen milestone:
  `experiments/V1/milestones/M03_A_k03_uhddeblur_official_baseline`
- First completed family probe:
  `experiments/V1/runs/A_m03`
- Probe-family design doc:
  `docs/next_experiment_design_20260515_a_m03_a_n03_a_o03_short_scouts_before_5h_window.md`
