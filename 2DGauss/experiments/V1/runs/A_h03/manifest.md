# Run Manifest: V1/T01/A_h03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_h03`
- Full identifier: `V1/T01/A_h03`

## Current Status

Design package prepared. This branch has **not** been smoke-calibrated,
preflighted, launched, or trained in this turn.

## Purpose

`A_h03` is the first post-milestone `UHD_deblur` branch prepared after the
accepted and frozen `A_g03` baseline.

Current frozen baseline:

```text
V1/T01/A_g03 final.pt official full-image test PSNR = 25.3801 dB
```

`A_g03` proved that Gaussian-derived directional/frequency priors are useful,
but the gain remained modest and the internal-val / official-test mismatch
persisted. The next highest-priority question is therefore no longer
"can we squeeze a little more from the same directional idea", but:

> is the current ceiling partly caused by the patch-train / full-image-test
> mismatch itself?

## Experimental Question

Can we improve same-protocol official full-image `UHD_deblur` performance by
making training-time context more consistent with full-image inference, while
keeping the accepted `A_g03` model story intact?

More concretely:

1. keep the accepted `A_g03` architecture fixed as the comparison anchor;
2. preserve edge-guided refinement and directional/frequency guidance;
3. change only the training-time context seen by the model;
4. keep official test strictly post-hoc and unchanged.

## Chosen Mechanism

`A_h03` introduces a default-off train-inference consistency mechanism:

```text
use_train_inference_consistency = true
train_context_size = 768
train_supervision_crop_size = 512
```

Implementation meaning:

1. training no longer feeds only a random `512x512` patch;
2. instead, the model sees a random `768x768` context crop;
3. supervision is applied only to the centered `512x512` region;
4. validation and official test remain the same as the accepted `A_g03`
   protocol: full-image `crop_size=0`.

This is intentionally a `TLC`-style **paradigm borrowing**, not a direct TLC
backbone port:

- no main-backbone replacement;
- no change to the `lowres -> gaussian -> highres` method story;
- no official-test peeking;
- no inference-time benchmark protocol change.

## Why This Placement Fits The Current V1 Line

The current model has several places that can be affected by local-only patch
training:

1. low-resolution compact feature extraction before Gaussian parameter
   prediction;
2. Gaussian attention modulation of the high-resolution branch;
3. edge-guided directional-frequency detail refinement.

Giving these modules a wider training-time context is the smallest coherent
way to test the project strategy's `TLC`-style hypothesis without abandoning
the current mainline.

## Baseline For Comparison

- Official baseline: `V1/T01/A_g03`
- Official checkpoint: `experiments/V1/runs/A_g03/checkpoints/final.pt`
- Official PSNR: `25.3800849048404`
- Frozen milestone: `experiments/V1/milestones/M01_A_g03_uhddeblur_official_baseline`
- Stable milestone link: `experiments/V1/milestones/current_uhddeblur_official_baseline`

## Planned Protocol

Kept from `A_g03`:

- dataset: `UHD_deblur`
- effective training split: after the first `100` train pairs
- internal validation: first `20` train pairs
- validation crop: full image, `crop_size=0`
- official test: `300` full-image pairs, post-hoc only
- edge-guided refinement
- directional/frequency guidance
- Laplacian consistency loss
- short-tier wall-clock target

Changed from `A_g03`:

- training input context: `512 -> 768`
- supervised center region: fixed at `512`
- new default-off config switch:
  `use_train_inference_consistency = true`

## Planned Runtime Envelope

These are design targets only. They are **not** launch-approved yet:

- Tier target: short
- Wall-clock target: `58` minutes
- Max steps target: `2500`
- Initial batch-size hypothesis: `8`
- Gradient accumulation target: `1`
- Train context size: `768`
- Train supervision size: `512`
- Validation interval target: `500`
- Checkpoint interval target: `250`
- Validation batches target: `20`

The initial batch-size hypothesis `8` is derived analytically from the accepted
`A_g03` envelope: `batch=18` at `512x512` already occupied about `93G`
reserved memory, and `768x768` carries about `2.25x` the pixel area of
`512x512`.

## Code Prepared In This Turn

Adjusted code:

- `src/uhd_gaussian/config.py`
- `src/uhd_gaussian/__init__.py`
- `src/uhd_gaussian/data/paired_image_dataset.py`
- `scripts/train_v1_t01.py`
- `scripts/smoke_calibrate_v1_t01.py`

Prepared experiment package:

- `experiments/V1/templates/T01/configs/a_h03_short_uhddeblur_tlc_consistency.toml`
- `experiments/V1/runs/A_h03/configs/a_h03_short_uhddeblur_tlc_consistency.toml`
- `experiments/V1/runs/A_h03/manifest.md`
- `experiments/V1/runs/A_h03/gate_check.md`
- `docs/next_experiment_design_20260513_a_h03_tlc_consistency.md`

## Verification Performed In This Turn

Completed:

1. `py_compile` on modified source files and training/calibration entrypoints.
2. CPU random-input forward verification on the accepted `A_g03` path with the
   new default-off consistency code still disabled.
3. CPU random-input forward verification on the new consistency-enabled path
   with `768x768` input.
4. Real-dataset sample check confirming that the new training loader can
   produce a `768x768` paired crop from `UHD_deblur`.

Intentionally **not** done in this turn:

1. branch-specific smoke calibration;
2. formal preflight;
3. launch script creation;
4. tmux launch;
5. training.

## Required Before Any Future Deployment

1. Run serial smoke calibration specifically for `A_h03`, starting around
   `batch_size=8`, `crop_size=768`.
2. Reconfirm that the resulting launch setting still satisfies the user's
   `>=80G` VRAM floor.
3. Run formal preflight under a clean GPU state.
4. Only after those gates pass, create `launch_tmux.sh` and consider
   deployment.

## Working Hypothesis

If the current `UHD_deblur` ceiling is meaningfully constrained by
patch-train / full-image-test mismatch, then `A_h03` should outperform
`A_g03 final.pt` on the same official full-image test protocol.

If it fails, the next priority should shift away from this protocol variable
and more aggressively toward stronger upstream low-resolution representation or
other modeling-side upgrades.
