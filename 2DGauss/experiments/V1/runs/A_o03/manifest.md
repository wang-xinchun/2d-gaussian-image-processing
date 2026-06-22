# Run Manifest: V1/T01/A_o03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_o03`
- Full identifier: `V1/T01/A_o03`

## Current Status

This branch has now completed clean serial deployment, short-budget training,
and post-hoc official evaluation.

## Purpose

`A_o03` is the high-fill runtime scout rather than a third independent
architecture vote.

Its question is:

> can the new decoder-guidance mechanism keep a near-`A_k03` deployment shape
> while remaining inside the required `85-93G` runtime window?

## Main Design Choice

Relative to `A_n03`, this branch keeps the default strength `0.15` but changes
launch geometry to:

- `batch_size = 7`
- initial context probe target `train_context_size = 776`

Serial branch-local calibration has already narrowed the accepted high-fill
launch shape:

- `776`: immediate OOM
- `772`: immediate OOM
- `768`: OOM after several steps
- `760`: runs but exceeds the `93G` upper bound
- `752`: runs but still exceeds the `93G` upper bound
- `748`: first accepted `batch=7` point inside `85-93G`
- `744`: also accepted, but farther from the original near-`A_k03` target

So the final selected launch candidate for this branch is:

- `batch_size = 7`
- `train_context_size = 748`

Preserved on purpose:

- `train_supervision_crop_size = 512`
- edge-guided refinement
- directional/frequency guidance
- `TLC`-style train-inference consistency
- the same effective `UHD_deblur` train split after the first `40` pairs

## Runtime Envelope

- Tier target: `short`
- Planned wall-clock cap: `58` minutes
- Planned max steps: `2500`
- Planned batch size: `7`
- Initial training context hypothesis: `776`
- Final launch context after calibration: `748`
- Planned supervision crop: `512`
- Target reserved VRAM window: `85.0-93.0G`

## Comparison Anchor

- Frozen milestone:
  `experiments/V1/milestones/M03_A_k03_uhddeblur_official_baseline`
- Design-only architectural parent:
  `experiments/V1/templates/T01/configs/a_l03_medium_uhddeblur_gaussian_decoder_guidance_design_only.toml`
- Probe-family design doc:
  `docs/next_experiment_design_20260515_a_m03_a_n03_a_o03_short_scouts_before_5h_window.md`

## Outcome Summary

The branch completed successfully as a runtime scout:

- clean finish at `step=2500`
- final logged elapsed time `34.9640` minutes
- best training-time full-image validation PSNR `28.8394 dB`
- post-hoc official full-image test `25.2809 dB`

At the same time, the branch also exposed the key practical limitation that
prevents direct promotion:

- accepted smoke at `7x748` reports `reserved_gb=92.5859`
- but real startup observation rises to about `95.5G`, above the user’s
  practical `93G` upper bound

This means `A_o03` is accepted as useful deployment-envelope evidence, but not
as the preferred `5`-hour anchor.
