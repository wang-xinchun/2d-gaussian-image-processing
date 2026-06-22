# Run Manifest: V1/T01/A_q03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_q03`
- Full identifier: `V1/T01/A_q03`

## Current Status

This branch has now completed smoke calibration, formal preflight, short-budget
training, and post-hoc official evaluation. It is now recorded as a completed
negative-quality short probe rather than an active run or a baseline-replacement
candidate.

## Purpose

`A_q03` exists to test the current bottleneck diagnosis directly and has now
finished that experiment:

> if the real gap now sits at the high-resolution terminal reconstruction end,
> can a lightweight Gaussian-guided terminal residual head improve the final
> short-run signal without changing the backbone or abandoning the accepted
> `A_n03` / `A_k03` mainline?

The answer is:

- the branch trains cleanly and is engineering-safe;
- it does **not** improve same-protocol official performance.

## Main Design Choice

Relative to `A_n03`, this branch keeps:

- edge-guided refinement
- directional/frequency guidance
- `TLC`-style train-inference consistency
- default decoder guidance `strength = 0.15`
- effective `UHD_deblur` training split after the first `40` pairs
- lightweight width/depth and Gaussian-first architecture story
- short-budget `6x744` launch geometry after branch-local recalibration

It changes:

- enables a default-off lightweight Gaussian-guided terminal residual head
- adds direct Gaussian prior control on the final RGB residual path

## Planned Runtime Envelope

- Tier target: `short`
- Planned wall-clock cap: `58` minutes
- Planned max steps: `2500`
- Planned batch size: `6`
- Planned training context: `744`
- Planned supervision crop: `512`
- Target reserved VRAM window: `85.0-93.0G`

## Comparison Anchor

- Frozen milestone:
  `experiments/V1/milestones/M03_A_k03_uhddeblur_official_baseline`
- Latest decoder-guidance short winner:
  `experiments/V1/runs/A_n03`
- Latest medium follow-up:
  `experiments/V1/runs/A_p03`

## Outcome Summary

The branch completed successfully but produced a negative-quality result:

- clean finish at `step=2500`
- final logged elapsed time `34.4349` minutes
- best training-time full-image validation PSNR `28.7026 dB` at `step=1500`
- post-hoc official full-image test `25.0953 dB` at `step_001500.pt`
- post-hoc official full-image test `25.1826 dB` at `final.pt`

Relative to the nearest short and medium anchors:

- `A_q03 final.pt` is about `-0.1094 dB` below `A_n03 final.pt`
- `A_q03 final.pt` is about `-0.8262 dB` below `A_p03 final.pt`
- `A_q03 final.pt` is about `-1.0174 dB` below frozen `A_k03 final.pt`

## Expected Gate Logic

This branch should be judged as a short probe, not as a baseline-replacement
run. It must first pass:

1. clean GPU check
2. branch-local smoke calibration inside `85-93G`
3. formal preflight
4. clean short-budget completion
5. post-hoc official test only after training completes

## Deployment Summary

- Branch-local smoke calibration completed at `6x736` and `6x744`.
- Accepted launch geometry: `batch_size=6`, `train_context_size=744`.
- Formal preflight passed with `batch_input_shape=(6, 3, 744, 744)`, `total_params=290091`, `val_psnr=22.6355`.
- Launch script was prepared at `experiments/V1/runs/A_q03/launch_tmux.sh`.
- Training then completed cleanly at `step=2500`.
- `step_002500.pt`, `latest.pt`, and `final.pt` have identical model weights.
- Post-hoc official evaluation has been completed and recorded under
  `experiments/V1/runs/A_q03/metrics/`.
