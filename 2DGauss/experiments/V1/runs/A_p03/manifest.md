# Run Manifest: V1/T01/A_p03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_p03`
- Full identifier: `V1/T01/A_p03`

## Current Status

This branch has now completed smoke calibration, formal preflight, formal
tmux launch, full training, and official post-hoc evaluation. It is the
completed `5`-hour medium-budget promotion run for the Gaussian-guided
decoder-modulation line after the completed `A_m03/A_n03/A_o03` short-scout
family.

## Purpose

`A_p03` exists to answer the next clean promotion question and has now
finished that experiment:

> if we keep the strongest short-scout structure (`A_n03`) fixed and give it a
> full `5`-hour window under a safe high-VRAM geometry, can it beat the frozen
> `A_k03/M03` official baseline?

The answer is:

- it improves on the short-scout family;
- it does **not** beat the frozen `A_k03/M03` official baseline.

## Main Design Choice

Relative to `A_n03`, this branch keeps:

- `gaussian_decoder_guidance_strength = 0.15`
- edge-guided refinement
- directional/frequency guidance
- `TLC`-style train-inference consistency
- effective `UHD_deblur` training split after the first `40` pairs
- lightweight width/depth and Gaussian-first architecture story

It changes:

- training tier `short -> medium`
- wall-clock cap `58 -> 300` minutes
- max steps `2500 -> 24000`
- launch geometry `6x784 -> 6x800`

## Why This Branch Exists

The completed short-scout family now gives a clear ranking:

- `A_n03` best short-run val `28.9612`, official test `25.2919`
- `A_m03` best short-run val `28.8486`, official test `25.2690`
- `A_o03` best short-run val `28.8394`, official test `25.2809`

So `A_n03` is the best structural anchor.

At the same time, `A_o03` gave an equally important deployment warning:

- its `batch=7` near-`A_k03` launch shape can be made trainable,
  but real startup memory rises above `93G`.

`A_p03` therefore combines the winning short-scout structure with a new
branch-local medium-run geometry that stays inside the target memory window
without reusing the riskier `A_o03` high-fill envelope.

## Runtime Envelope

- Tier target: `medium`
- Planned wall-clock cap: `300` minutes
- Planned max steps: `24000`
- Planned batch size: `6`
- Planned training context: `800`
- Planned supervision crop: `512`
- Target reserved VRAM window: `85.0-93.0G`

## Comparison Anchor

- Frozen milestone:
  `experiments/V1/milestones/M03_A_k03_uhddeblur_official_baseline`
- Stable milestone link:
  `experiments/V1/milestones/current_uhddeblur_official_baseline`
- Short-run nomination winner:
  `experiments/V1/runs/A_n03`
- High-fill runtime scout:
  `experiments/V1/runs/A_o03`
- Design doc:
  `docs/next_experiment_design_20260515_a_p03_medium_gaussian_decoder_guidance_default_5h.md`

## Deployment Summary

The branch has now passed the required launch gates:

1. branch-local smoke calibration:
   - `6x792`: `reserved_gb=89.5762`
   - `6x800`: `reserved_gb=91.3105`
2. final selected launch geometry: `batch_size=6`, `train_context_size=800`
3. formal preflight passed with:
   - `batch_input_shape=(6, 3, 800, 800)`
   - `total_params=267864`
   - `val_psnr=21.6963`
4. clean tmux launch succeeded under session:
   - `2dgauss_a_p03_uhddeblur_gaussian_decoder_guidance_default_5h`
5. startup-window `nvidia-smi` shows:
   - `memory.used=93751 MiB`
   - `utilization.gpu=100%`
6. `metrics/train_metrics.jsonl` advanced through the full five-hour window and
   the run completed cleanly at `step=21820`.
7. `metrics/official_test_final.json` records the final official-test result:
   `PSNR=26.0087`.
