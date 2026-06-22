# Run Manifest: V1/T01/A_s03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_s03`
- Full identifier: `V1/T01/A_s03`

## Current Status

This branch has now completed smoke calibration, formal preflight, short-budget
training, and post-hoc official evaluation. It is now recorded as a completed
negative-quality supervision-only probe rather than an active run.

## Purpose

`A_s03` is the loss-side companion probe in the post-`A_q03` short family.

The immediate question is:

> if the bottleneck is partly about high-frequency detail under-weighting, can
> a tiny edge-weighted reconstruction term improve the short-run signal without
> changing the accepted model structure?

## Experimental Question

Can a supervision-only intervention help the `A_n03` mainline close some of the
high-resolution detail gap, without adding parameters or changing the backbone?

More concretely:

1. keep the entire `A_n03` architecture fixed;
2. preserve edge-guided refinement, directional/frequency guidance, and
   train-inference consistency;
3. modify only the reconstruction objective;
4. test whether the bottleneck is partly optimization-side rather than purely
   mechanism-side.

## Chosen Mechanism

`A_s03` adds a new default-off edge-weighted reconstruction term:

```text
use_edge_weighted_reconstruction = true
edge_weighted_reconstruction_strength = 0.03
edge_weighted_reconstruction_floor = 1.0
```

Implementation meaning:

1. target-image Sobel magnitude defines a normalized edge emphasis map;
2. the auxiliary term reweights reconstruction error toward sharper regions;
3. the global weight remains intentionally small to preserve short-run stability;
4. no extra model parameters are introduced.

## Baseline For Comparison

- Short-structure anchor: `V1/T01/A_n03`
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

- new default-off loss switch:
  `use_edge_weighted_reconstruction = true`
- auxiliary loss weight:
  `edge_weighted_reconstruction_strength = 0.03`

## Planned Runtime Envelope

- Tier target: short
- Wall-clock target: `58` minutes
- Max steps target: `2500`
- Initial batch-size hypothesis: `6`
- Gradient accumulation target: `1`
- Train context size: `784`
- Train supervision size: `512`
- Validation interval target: `500`
- Checkpoint interval target: `250`
- Validation batches target: `20`
- Target reserved VRAM window: `85.0-93.0G`

## Code Prepared In This Turn

Adjusted code:

- `scripts/train_v1_t01.py`
- `src/uhd_gaussian/config.py`
- `src/uhd_gaussian/__init__.py`

Prepared experiment package:

- `experiments/V1/templates/T01/configs/a_s03_short_uhddeblur_edge_weighted_reconstruction_probe.toml`
- `experiments/V1/runs/A_s03/configs/a_s03_short_uhddeblur_edge_weighted_reconstruction_probe.toml`
- `experiments/V1/runs/A_s03/manifest.md`
- `experiments/V1/runs/A_s03/gate_check.md`

## Verification Performed

Completed:

1. `py_compile` on modified source and training entrypoint files.
2. serial branch-local smoke calibration at `6x784`.
2. true-config formal preflight on the prepared `A_s03` config.
3. tmux launch of the short run under session
   `2dgauss_a_s03_uhddeblur_edge_weighted_reconstruction_probe`.
4. post-hoc official full-image test on `final.pt`.

## Outcome Summary

The branch completed successfully but produced a negative-quality result
relative to the current short and medium anchors:

- clean finish at `step=2500`
- final logged elapsed time `33.7827` minutes
- best training-time full-image validation PSNR `28.8699 dB` at `step=2500`
- post-hoc official full-image test `25.2567 dB` at `final.pt`

Relative to key same-task anchors:

- `A_s03 final.pt` is about `-0.0352 dB` below `A_n03 final.pt`
- `A_s03 final.pt` is about `-0.7520 dB` below `A_p03 final.pt`
- `A_s03 final.pt` is about `-0.9432 dB` below frozen `A_k03 final.pt`
- `A_s03 final.pt` is about `+0.0742 dB` above `A_q03 final.pt`

## Deployment Summary

- Branch-local smoke calibration completed at `6x784`.
- Formal preflight passed with `batch_input_shape=(6, 3, 784, 784)`,
  `total_params=267864`, `val_psnr=21.7247`.
- Training then completed cleanly at `step=2500`.
- `step_002500.pt`, `latest.pt`, and `final.pt` have identical `model`
  weights.
- Post-hoc official evaluation has been completed and recorded under
  `experiments/V1/runs/A_s03/metrics/official_test_final.json`.

## Working Hypothesis

If the current bottleneck is partly loss-side, `A_s03` should improve the
detail-sensitive short-run signal with almost zero parameter cost. If it does
not, the main bottleneck is more likely to remain structural.
