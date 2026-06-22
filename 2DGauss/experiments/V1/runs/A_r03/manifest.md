# Run Manifest: V1/T01/A_r03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_r03`
- Full identifier: `V1/T01/A_r03`

## Current Status

This branch has now completed smoke recalibration, independent formal
preflight, short-budget training, and post-hoc official evaluation. It is now
recorded as a completed near-neutral short feature-refinement probe rather
than an active run.

## Purpose

`A_r03` is the first post-`A_q03` short probe prepared to keep the
high-resolution bottleneck focus while avoiding direct terminal RGB control.

Current frozen baseline:

```text
V1/T01/A_k03 final.pt official full-image test PSNR = 26.2000 dB
```

The immediate question is:

> if Gaussian priors remain useful as a structural controller, can a very
> lightweight feature-space refinement stage improve final reconstruction more
> cleanly than the negative `A_q03` terminal residual path?

## Experimental Question

Can we beat the current short-probe band by refining high-resolution decoder
features before final RGB projection, while keeping the accepted decoder-guided
mainline intact?

More concretely:

1. keep the completed `A_n03` short-run structure as the main anchor;
2. preserve edge-guided refinement, directional/frequency guidance, and
   train-inference consistency;
3. avoid direct RGB residual control;
4. change only the final high-resolution feature expression mechanism.

## Chosen Mechanism

`A_r03` enables a new default-off feature-space module:

```text
use_gaussian_feature_refinement = true
gaussian_feature_refinement_blocks = 1
gaussian_feature_refinement_strength = 0.10
```

Implementation meaning:

1. Gaussian attention, local contrast, and optional directional priors are
   encoded into a tiny prior feature;
2. that prior is fused with the decoder feature map before the last RGB conv;
3. the module predicts a bounded scale/bias style feature correction;
4. Gaussian still controls structure and routing rather than directly writing
   terminal RGB pixels.

## Baseline For Comparison

- Short-structure anchor: `V1/T01/A_n03`
- Negative terminal-control reference: `V1/T01/A_q03`
- Frozen milestone: `experiments/V1/milestones/M03_A_k03_uhddeblur_official_baseline`
- Stable milestone link: `experiments/V1/milestones/current_uhddeblur_official_baseline`

## Planned Protocol

Kept from `A_n03`:

- dataset: `UHD_deblur`
- effective training split: after the first `40` train pairs
- internal validation: first `20` train pairs
- validation crop: full image, `crop_size=0`
- official test: `300` full-image pairs, post-hoc only
- edge-guided refinement
- directional/frequency guidance
- `TLC`-style train-inference consistency
- decoder guidance `strength = 0.15`
- short-tier wall-clock target

Changed from `A_n03`:

- new default-off config switch:
  `use_gaussian_feature_refinement = true`
- feature refinement strength:
  `gaussian_feature_refinement_strength = 0.10`
- prepared launch geometry is provisionally tightened from `6x784` to `6x728`
  because the new feature-refinement path raises peak activation memory.

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

## Code Prepared In This Turn

Adjusted code:

- `src/uhd_gaussian/models/feature_refinement.py`
- `src/uhd_gaussian/models/restorer.py`
- `src/uhd_gaussian/config.py`
- `src/uhd_gaussian/__init__.py`
- `scripts/train_v1_t01.py`

Prepared experiment package:

- `experiments/V1/templates/T01/configs/a_r03_short_uhddeblur_gaussian_feature_refinement_probe.toml`
- `experiments/V1/runs/A_r03/configs/a_r03_short_uhddeblur_gaussian_feature_refinement_probe.toml`
- `experiments/V1/runs/A_r03/manifest.md`
- `experiments/V1/runs/A_r03/gate_check.md`

## Verification Performed

Completed:

1. `py_compile` on modified source and training entrypoint files.
2. branch-local smoke calibration at `6x736` and `6x728` to locate a safe
   preflight geometry.
3. true-config formal preflight on the prepared `A_r03` config.
4. clean-GPU revalidation smoke at `6x728`.
5. independent formal preflight rerun.
6. tmux launch and completed short training.
7. post-hoc official full-image test on `final.pt`.

## Outcome Summary

The branch completed successfully and is the strongest result so far inside the
post-`A_q03` short family, but it is still slightly below `A_n03`:

- clean finish at `step=2500`
- final logged elapsed time `35.6173` minutes
- best training-time full-image validation PSNR `28.7519 dB` at `step=2500`
- post-hoc official full-image test `25.2867 dB` at `final.pt`

Relative to key same-task anchors:

- `A_r03 final.pt` is about `+0.0300 dB` above `A_s03 final.pt`
- `A_r03 final.pt` is about `+0.1041 dB` above `A_q03 final.pt`
- `A_r03 final.pt` is about `-0.0052 dB` below `A_n03 final.pt`
- `A_r03 final.pt` is about `-0.7220 dB` below `A_p03 final.pt`
- `A_r03 final.pt` is about `-0.9133 dB` below frozen `A_k03 final.pt`

## Deployment Summary

- Clean-GPU branch-local smoke at `6x728` returned
  `max_memory_reserved_gb=91.7871`.
- Independent formal preflight passed with
  `batch_input_shape=(6, 3, 728, 728)`, `total_params=303912`,
  `val_psnr=22.3030`.
- Launch script was created at
  `experiments/V1/runs/A_r03/launch_tmux.sh`.
- Training then completed cleanly at `step=2500`.
- `step_002500.pt`, `latest.pt`, and `final.pt` have identical `model`
  weights.
- Post-hoc official evaluation has been completed and recorded under
  `experiments/V1/runs/A_r03/metrics/official_test_final.json`.

## Working Hypothesis

If the current deblur bottleneck is mainly about how high-resolution detail is
expressed, then feature-space refinement should outperform the negative
`A_q03` terminal-residual path while staying lightweight and Gaussian-centered.
