# Run Manifest: V1/T01/A_i03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_i03`
- Full identifier: `V1/T01/A_i03`

## Current Status

This branch has now completed training and post-hoc evaluation. It launched
with `batch_size=7`, `train_context_size=784`, stayed inside the user's
`85.0-93.0G` runtime window, finished cleanly at `step=12000`, and improved
the official `UHD_deblur` full-image test result beyond the frozen `A_g03`
milestone. Read this file together with `gate_check.md`,
`result_analysis.md`, and the post-hoc metric artifacts.

## Purpose

`A_i03` is the medium-budget follow-up to the prepared-but-undeployed
`A_h03` consistency branch.

It keeps the accepted and frozen `A_g03` model story fixed:

```text
edge-guided refinement + directional/frequency guidance + TLC-style
train-inference consistency
```

but changes the runtime envelope from short probe to a more serious
medium-budget test:

1. wall-clock budget expands from about `58` minutes to `300` minutes;
2. the deployment target is no longer generic `>=80G`, but a tighter reserved
   VRAM window of `85-93G`;
3. the effective training set is enlarged by reducing `train_offset`
   from `100` to `40`, so the branch trains on `1924` pairs instead of `1864`.

## Why This Branch Exists

`A_g03` established the current official `UHD_deblur` baseline at:

```text
V1/T01/A_g03 final.pt official full-image test PSNR = 25.3801 dB
```

`A_h03` then prepared the first post-milestone consistency-repair idea, but it
was only a short-run design package.

The user has now explicitly requested a stronger test:

1. still do not deploy in this turn;
2. prepare a branch for longer training, up to `5` hours;
3. keep training-memory usage inside `85-93G`;
4. use more training data to see whether the consistency idea benefits from a
   broader effective train split.

`A_i03` is therefore the first medium-budget version of the `TLC`-style
consistency line.

## Experimental Question

Can the post-milestone consistency-repair idea show clearer gains when we give
it both:

1. a longer `5h` training budget; and
2. a larger effective train split;

while keeping the same official full-image evaluation protocol and the same
accepted `A_g03` architectural story?

## Main Design Choices

Preserved from `A_g03` / `A_h03`:

- dataset: `UHD_deblur`
- edge-guided refinement
- directional/frequency guidance
- Laplacian consistency loss
- train-inference consistency mechanism
- full-image validation protocol with `crop_size=0`
- official full-image test remains post-hoc only

Changed relative to `A_h03`:

- tier: `short -> medium`
- wall-clock cap: `58 -> 300` minutes
- max steps target: `2500 -> 12000`
- validation interval: `500 -> 1000`
- checkpoint interval: `250 -> 500`
- effective training split: after first `100` pairs -> after first `40` pairs
- explicit target reserved VRAM window: `85.0-93.0G`

## Dataset Plan

- dataset: `UHD_deblur`
- total train pairs available: `1964`
- internal validation slice: first `20` train pairs
- reserved training-time holdout prefix: first `40` train pairs
- effective training split: train pairs after the first `40`
- effective training size: `1964 - 40 = 1924`
- official test split: `300` full-image pairs

This choice enlarges the effective training set by `60` pairs compared with
the accepted `A_g03/A_h03` convention while still preserving a deterministic
non-test validation prefix.

## Planned Runtime Envelope

These started as design targets and are now concretely instantiated by the
active launch:

- Tier target: `medium`
- Planned wall-clock cap: `300` minutes
- Max steps target: `12000`
- Initial batch-size hypothesis: `8`
- Gradient accumulation target: `1`
- Final train context size: `784`
- Train supervision size: `512`
- Validation interval target: `1000`
- Checkpoint interval target: `500`
- Validation batches target: `20`
- Reserved VRAM target window: `85.0-93.0G`

The current calibrated launch candidate is `batch_size=7`,
`train_context_size=784`. This setting preserves the consistency-repair idea
while landing inside the tighter `85-93G` window with a little more headroom
against real-launch drift than the earlier `batch=8`, `context=752` trial.

Current static runtime evidence:

1. a small-sample `A_i03` preflight with the earlier `batch_size=8` candidate
   completed successfully and produced a real large-context training batch;
2. the old `A_h03` config still carrying `batch_size=18` cannot be treated as
   a valid launch candidate for `768x768` context training, because a later
   preflight attempt under that setting OOMed on GPU before any branch-specific
   recalibration was done.
3. serial smoke calibration established the practical envelope:
   - `batch=7, context=768` -> reserved `84.4570G`, below target window;
   - `batch=8, context=768` -> reserved `93.1113G`, slightly above target window;
   - `batch=8, context=752` -> reserved `91.9004G`, inside smoke window, but a
     later real launch drifted above the user's `93G` upper bound and was
     intentionally stopped;
   - `batch=7, context=784` -> reserved `88.0137G`, inside target window;
   - estimated steps at `300` minutes under the selected setting: `12244`.

4. formal preflight on the final setting also passes:
   - `device=cuda`
   - `train_samples=7`, `val_samples=20`
   - `batch_input_shape=(7, 3, 784, 784)`
   - `loss_total=0.09533`
   - `val_loss=0.06981`
   - `val_psnr=21.7638`
5. clean tmux launch succeeded:
   - session: `2dgauss_a_i03_uhddeblur_tlc_medium`
   - log: `experiments/V1/runs/A_i03/logs/train_20260514_022908.log`
   - startup memory: `90833 MiB`
   - startup metrics written through at least `step=70`

This means deployment had to recalibrate from the low-batch side rather than
inherit `A_g03`'s `batch=18` setting, and the resulting candidate has now
survived the real startup window.

## Completion Summary

- training finish step: `12000`
- final logged elapsed minutes: `156.1359`
- post-hoc official result, `step_011000.pt`: `25.7934 dB`
- post-hoc official result, `final.pt`: `25.8506 dB`
- current branch winner: `final.pt`

This branch has therefore completed the full loop:

1. design and gating
2. real-launch deployment under the strict memory window
3. clean training completion
4. post-hoc full-image validation
5. post-hoc official full-image test

## Milestone Isolation

- Frozen baseline remains:
  `experiments/V1/milestones/M01_A_g03_uhddeblur_official_baseline`
- Stable milestone link remains:
  `experiments/V1/milestones/current_uhddeblur_official_baseline`
- This branch does not modify the frozen milestone directory or its weights.
