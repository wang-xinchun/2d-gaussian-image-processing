# Next Experiment Design: V1/T01/A_g03

Date: 2026-05-13 UTC

## Boundary

This turn continues the main line by adjusting the model and preparing the next
branch design, but it intentionally does **not** deploy training.

Per the user's instruction:

- analyze the next step based on the latest accepted experiment;
- adjust the model;
- do not start smoke calibration, formal branch preflight, tmux launch, or
  training.

## Current Anchor

The current accepted `UHD_deblur` baseline is:

```text
V1/T01/A_f03 final.pt official full-image test PSNR = 25.3145 dB
```

The most important lesson from `A_f03` is not only that more budget helped.
It is that:

1. the accepted edge-guided cache-safe branch still had headroom;
2. the gain appeared on the official full-image test rather than on the small
   internal full-image validation slice;
3. the same mechanism is therefore valid but not yet strong enough to create a
   clearly meaningful jump.

## Why Not Just Run A Longer Replay

`A_f03` already answered the budget-extension question. The result was positive
but modest:

```text
A_f03 final.pt vs A_e03 final.pt = +0.1004 dB
```

That is enough to refresh the baseline, but not enough to justify another
nearly identical replay as the highest-value next step.

The project strategy already narrowed the next deblur move to two classes:

1. training-inference consistency repair;
2. stronger directional/frequency structure.

For this turn, the cleaner model-side move is the second one, because it can be
added without changing the training template or deployment contract.

## Chosen Adjustment

The new branch design is:

```text
V1/T01/A_g03
```

Core idea:

> keep the accepted `A_f03` edge-guided cache-safe branch fixed, but inject a
> Gaussian-derived directional/frequency prior into the high-resolution detail
> refiner.

This design follows the project strategy directly:

- `Stripformer`-style insight: blur carries directional structure;
- `FFTformer`-style insight: restoration quality depends on selecting which
  frequencies to amplify or suppress;
- project constraint: do not replace the current
  `lowres -> gaussian -> highres reinjection` story with a generic backbone
  swap.

## Model Delta

### What stays the same

- low-resolution compact feature extraction;
- texture/frequency descriptor fusion on the low-resolution branch;
- Gaussian parameter prediction;
- Gaussian-rendered scalar attention map;
- low-rank affine modulation;
- high-resolution edge-guided detail refiner;
- Laplacian consistency loss;
- `A_f03` data protocol and short-tier runtime target.

### What changes

New default-off mechanism:

```text
use_directional_frequency_guidance = true
```

Implementation summary:

1. Reuse the already predicted Gaussian parameters.
2. Summarize atom-wise amplitude-weighted direction and frequency statistics:
   - dominant direction `(cos(theta), sin(theta))`
   - mean frequency
   - mean texture gain
   - direction coherence
   - spectral decay summary
3. Upsample the resulting compact prior to high resolution.
4. Project it into the edge-guided refiner and fuse it with:
   - decoder feature
   - edge feature
   - Gaussian scalar attention

This means the high-resolution refinement branch is no longer conditioned only
on generic edge energy and scalar attention. It also sees a structured prior
that says, in effect:

- what dominant blur direction is implied downstream of the low-resolution
  Gaussian model;
- how oscillatory the local texture trend is;
- how coherent that direction is.

## Why This Placement Is Better Than A Backbone Swap

The project is not trying to become:

- a generic deblurring backbone paper;
- a full `Stripformer` replacement;
- a full `FFTformer` replacement.

This placement is useful precisely because it keeps the method narrative intact:

- Gaussian parameters are still the compact low-resolution representation;
- high-resolution restoration is still done through structured reinjection;
- the new information is injected where `A_f03` is currently weakest:
  task-specific fine-detail recovery.

## Code Prepared In This Turn

Adjusted core code:

- `src/uhd_gaussian/config.py`
- `src/uhd_gaussian/models/restorer.py`
- `src/uhd_gaussian/models/edge.py`
- `src/uhd_gaussian/models/directional.py`
- `src/uhd_gaussian/__init__.py`
- `scripts/train_v1_t01.py`

Prepared design package:

- `experiments/V1/templates/T01/configs/a_g03_short_uhddeblur_directional_frequency.toml`
- `experiments/V1/runs/A_g03/configs/a_g03_short_uhddeblur_directional_frequency.toml`
- `experiments/V1/runs/A_g03/gate_check.md`
- `experiments/V1/runs/A_g03/manifest.md`

## Verification Performed This Turn

Completed:

1. Python syntax check on modified source files and training entrypoint.
2. CPU random-input forward verification on the accepted old config
   `A_f03`:
   - output shape correct
   - `directional_prior` remains `None`
3. CPU random-input forward verification on the new config `A_g03`:
   - output shape correct
   - `directional_prior` is produced

Observed but not treated as a code failure:

- GPU `--preflight` attempts during this turn hit CUDA OOM because the device
  was already heavily occupied by other processes. That result is not suitable
  for interpreting branch memory safety.

Therefore the correct project conclusion is:

> logic compatibility is verified, but branch-specific memory calibration still
> needs to be done later under a controlled GPU state.

## Required Before Any Future Deployment

1. Run branch-specific smoke calibration for `A_g03`.
2. Re-measure whether `batch_size=20` still satisfies the `>=80G` gate.
3. Run formal branch preflight under a clean GPU state.
4. Only then decide whether `A_g03` is launch-worthy.

## Working Hypothesis

If the current deblur ceiling is partly caused by weak directional/frequency
conditioning in the high-resolution detail path, then `A_g03` should improve
official full-image PSNR over `A_f03`.

If it does not, the next priority should shift more aggressively toward the
other strategic branch:

```text
TLC-style training-inference consistency repair
```
