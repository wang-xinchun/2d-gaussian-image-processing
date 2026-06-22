# Run Manifest: V1/T01/A_t03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_t03`
- Full identifier: `V1/T01/A_t03`

## Current Status

This branch has now completed clean serial deployment, short-budget training,
and post-hoc official evaluation. It is now recorded as the strongest member
of the post-`A_q03` short family, but only by a very small margin.

## Purpose

`A_t03` is the combination probe for the post-`A_q03` short family.

The immediate question is:

> if both mechanism-side feature refinement and loss-side edge emphasis are
> individually plausible, does their combination become complementary or does
> it over-correct under the same lightweight budget?

## Experimental Question

Can the combined probe give a better early signal than either single-variable
probe, while still remaining lightweight and structurally interpretable?

More concretely:

1. start from the same `A_n03` mainline anchor;
2. add feature-space refinement before final RGB projection;
3. add a small edge-weighted reconstruction term;
4. test complementarity before spending any longer budget.

## Chosen Mechanism

`A_t03` combines:

```text
use_gaussian_feature_refinement = true
gaussian_feature_refinement_strength = 0.10
use_edge_weighted_reconstruction = true
edge_weighted_reconstruction_strength = 0.03
```

The branch is intentionally narrow:

1. feature refinement handles mechanism-side high-resolution expression;
2. edge-weighted supervision handles optimization-side detail emphasis;
3. no direct terminal RGB residual path is reintroduced.

## Baseline For Comparison

- Short-structure anchor: `V1/T01/A_n03`
- Single-variable companions: `V1/T01/A_r03`, `V1/T01/A_s03`
- Negative terminal-control reference: `V1/T01/A_q03`
- Frozen milestone: `experiments/V1/milestones/M03_A_k03_uhddeblur_official_baseline`

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

- enables feature refinement
- enables edge-weighted reconstruction
- prepared launch geometry is provisionally tightened from `6x784` to `6x728`
  because feature refinement raises peak activation memory.

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

1. `py_compile` on modified source and training entrypoint files.
2. branch-local smoke calibration at `6x728` after reducing the prepared
   geometry from the original `6x784` anchor.
3. true-config formal preflight on the prepared `A_t03` config after reducing
   the prepared geometry to `6x728`.
4. clean-GPU serial launch validation.
5. tmux launch and completed short training.
6. same-protocol post-hoc official full-image test on `final.pt`.

## Outcome Summary

The branch completed successfully and is the strongest result so far inside the
post-`A_q03` short family, but the gain over `A_n03` is tiny:

- clean finish at `step=2500`
- final logged elapsed time `35.6375` minutes
- best training-time full-image validation PSNR `28.7877 dB` at `step=2500`
- post-hoc official full-image test `25.2944 dB` at `final.pt`

Relative to key same-task anchors:

- `A_t03 final.pt` is about `+0.0025 dB` above `A_n03 final.pt`
- `A_t03 final.pt` is about `+0.0078 dB` above `A_r03 final.pt`
- `A_t03 final.pt` is about `+0.0377 dB` above `A_s03 final.pt`
- `A_t03 final.pt` is about `-0.7143 dB` below `A_p03 final.pt`
- `A_t03 final.pt` is about `-0.9055 dB` below frozen `A_k03 final.pt`

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
  `experiments/V1/runs/A_t03/launch_tmux.sh`.
- The first tmux launch exposed a run-local config self-copy startup bug in
  `scripts/train_v1_t01.py`; the second launch succeeded after the bug fix.
- Training completed cleanly at `step=2500`.
- `step_002500.pt`, `latest.pt`, and `final.pt` have identical `model`
  weights.
- Post-hoc official evaluation has been completed and recorded under
  `experiments/V1/runs/A_t03/metrics/official_test_final.json`.
## Code Prepared In This Turn

Adjusted code:

- `src/uhd_gaussian/models/feature_refinement.py`
- `src/uhd_gaussian/models/restorer.py`
- `src/uhd_gaussian/config.py`
- `src/uhd_gaussian/__init__.py`
- `scripts/train_v1_t01.py`

Prepared experiment package:

- `experiments/V1/templates/T01/configs/a_t03_short_uhddeblur_feature_refinement_plus_edge_weighted_probe.toml`
- `experiments/V1/runs/A_t03/configs/a_t03_short_uhddeblur_feature_refinement_plus_edge_weighted_probe.toml`
- `experiments/V1/runs/A_t03/manifest.md`
- `experiments/V1/runs/A_t03/gate_check.md`

## Working Hypothesis

If the current bottleneck needs both a better expression path and more
detail-aware supervision, `A_t03` should be the strongest of the three short
probes. The completed run now supports that hypothesis, but only weakly: the
combination is slightly positive, not decisively better.
