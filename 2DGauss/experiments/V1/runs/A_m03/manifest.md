# Run Manifest: V1/T01/A_m03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_m03`
- Full identifier: `V1/T01/A_m03`

## Current Status

This branch has now completed clean serial deployment, short-budget training,
and post-hoc official evaluation.

It is the first completed probe in the post-`M03/A_k03` short-scout family
prepared before the next `5`-hour `UHD_deblur` window.

## Purpose

`A_m03` is the conservative Gaussian-guided decoder modulation scout.

It exists to answer the lowest-risk architectural question first:

> if the newly landed `A_l03` decoder-guidance mechanism is directionally
> useful, can it already show a positive short-run signal at a gentler
> strength before we spend a longer `5`-hour window on it?

## Main Design Choice

Relative to the frozen `A_k03/M03` mainline and the design-only `A_l03`
branch, `A_m03` keeps:

- the same `UHD_deblur` data split after the first `40` train pairs;
- edge-guided refinement;
- directional/frequency guidance;
- `TLC`-style train-inference consistency;
- `train_context_size=784`;
- `train_supervision_crop_size=512`;
- full-image internal validation;
- the lightweight Gaussian-first reconstruction story.

It changes only:

- enable `use_gaussian_decoder_guidance = true`;
- set `gaussian_decoder_guidance_strength = 0.10`;
- use a short-probe runtime envelope `batch_size=6`, `max_steps=2500`,
  `wall_clock_minutes=58`.

## Runtime Envelope

- Tier target: `short`
- Planned wall-clock cap: `58` minutes
- Planned max steps: `2500`
- Planned batch size: `6`
- Planned training context: `784`
- Planned supervision crop: `512`
- Target reserved VRAM window: `85.0-93.0G`

## Comparison Anchor

- Frozen milestone:
  `experiments/V1/milestones/M03_A_k03_uhddeblur_official_baseline`
- Stable milestone link:
  `experiments/V1/milestones/current_uhddeblur_official_baseline`
- Design-only architectural parent:
  `experiments/V1/templates/T01/configs/a_l03_medium_uhddeblur_gaussian_decoder_guidance_design_only.toml`
- Probe-family design doc:
  `docs/next_experiment_design_20260515_a_m03_a_n03_a_o03_short_scouts_before_5h_window.md`

## Outcome Summary

The branch completed successfully and produced the first positive short-run
signal for Gaussian-guided decoder modulation:

- clean finish at `step=2500`
- final logged elapsed time `33.7139` minutes
- best training-time full-image validation PSNR `28.8486 dB` at `step=2500`
- post-hoc official full-image test PSNR `25.2690 dB` at `final.pt`

Under the probe-family gate rules, this makes `A_m03` a strong keep candidate
for long-run promotion rather than a baseline-replacement attempt.
