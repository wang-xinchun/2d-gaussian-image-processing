# Next Experiment Design: V1/T01/A_h03

Date: 2026-05-13 UTC

## Boundary

This turn prepares the next `UHD_deblur` branch after the frozen `A_g03`
milestone, but intentionally does **not** deploy training.

Per the user's instruction:

- continue from the milestone mindset;
- prepare the next experiment for better absolute performance;
- do not run smoke calibration, formal preflight, tmux launch, or training in
  this turn.

## Current Anchor

The current accepted and frozen `UHD_deblur` baseline is:

```text
V1/T01/A_g03 final.pt official full-image test PSNR = 25.3800849048404
```

Key lessons from `A_g03`:

1. directional/frequency guidance is now validated on the official protocol;
2. the gain is still modest (`+0.0656 dB` over `A_f03`);
3. the internal-val / official-test mismatch still persists.

So the next branch should not be another near-identical replay of the same
idea.

## Why Shift To Consistency Repair Now

The project strategy had already narrowed the post-baseline choices to:

1. `TLC`-style training-inference consistency repair;
2. stronger upstream low-resolution representation change.

Given the current evidence, the first branch is cleaner and higher priority:

- it directly targets the already documented patch-train / full-image-test
  mismatch;
- it does not abandon the accepted `A_g03` mainline;
- it tests a different failure mode than the now-validated
  directional/frequency branch.

## Chosen Adjustment

The new prepared branch is:

```text
V1/T01/A_h03
```

Core idea:

> keep the accepted `A_g03` architecture fixed, but make training-time context
> more consistent with full-image inference by feeding a larger patch and only
> supervising the center region.

This is a `TLC`-style **paradigm** adaptation, not a direct TLC
implementation transplant.

## Mechanism Summary

New default-off mechanism:

```text
use_train_inference_consistency = true
train_context_size = 768
train_supervision_crop_size = 512
```

Training behavior with the mechanism enabled:

1. sample a random `768x768` training crop;
2. run the full model on that larger context;
3. crop predictions and targets to the centered `512x512` region for loss;
4. leave validation and official test unchanged.

This means:

- low-resolution compact modeling sees a broader neighborhood before Gaussian
  parameter prediction;
- Gaussian-rendered attention is computed under a less myopic local context;
- the edge-guided directional-frequency high-resolution branch also sees a
  larger degraded neighborhood during training.

What stays the same:

- edge-guided detail refiner;
- directional/frequency guidance;
- Gaussian attention and affine modulation;
- full-image validation and full-image official test;
- post-hoc-only official test rule.

## Why This Fits Better Than A Backbone Swap

The project is not trying to become:

- a generic `TLC` backbone reproduction;
- another pure deblurring backbone paper;
- a method whose story is "we replaced V1 with a stronger network".

This placement keeps the current contribution narrative intact:

- Gaussian parameters remain the compact low-resolution representation;
- high-resolution restoration remains structured reinjection;
- the new branch tests whether the current training protocol itself is still
  holding back the accepted architecture.

## Code Prepared In This Turn

Adjusted core code:

- `src/uhd_gaussian/config.py`
- `src/uhd_gaussian/__init__.py`
- `src/uhd_gaussian/data/paired_image_dataset.py`
- `scripts/train_v1_t01.py`
- `scripts/smoke_calibrate_v1_t01.py`

Prepared design package:

- `experiments/V1/templates/T01/configs/a_h03_short_uhddeblur_tlc_consistency.toml`
- `experiments/V1/runs/A_h03/configs/a_h03_short_uhddeblur_tlc_consistency.toml`
- `experiments/V1/runs/A_h03/manifest.md`
- `experiments/V1/runs/A_h03/gate_check.md`

## Verification Performed In This Turn

Completed:

1. `py_compile` on modified Python files.
2. CPU random-input forward verification with the old accepted path unchanged.
3. CPU random-input forward verification with the new consistency-enabled path.
4. Real-dataset sample verification that the new training crop logic can
   produce `768x768` context crops.

Intentionally skipped:

1. real-GPU smoke calibration;
2. formal preflight;
3. launch script creation;
4. tmux launch.

That boundary is deliberate and matches the user's instruction for this turn.

## Required Before Deployment

1. Run serial smoke calibration for `A_h03`, likely starting near
   `batch_size=8`, `crop_size=768`.
2. Reconfirm the user's `>=80G` VRAM floor under the final calibrated setting.
3. Run formal preflight on clean GPU.
4. Only then create a launch script and decide whether to deploy.

## Working Hypothesis

If the current `UHD_deblur` ceiling is partly caused by the mismatch between
patch training and full-image inference, `A_h03` should improve official
full-image PSNR over `A_g03`.

If it does not, the next priority should shift more decisively toward stronger
upstream representation changes rather than another protocol-focused branch.
