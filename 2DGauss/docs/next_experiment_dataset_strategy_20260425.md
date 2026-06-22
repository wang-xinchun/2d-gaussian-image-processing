# Next Experiment Dataset Strategy

## Record Time

- Time: `2026-04-25 17:32:21 CST`
- Context: after accepting `UHD_haze`, `UHD_deblur`, and `4K`, the next-run dataset plan should no longer be based only on the old `UHD-LL_down4` view.

## Current Constraint

`V1/T01` currently uses one paired image dataset per run:

- `train_input`
- `train_target`
- `val_input`
- `val_target`

The current `PairedImageDataset` is suitable for same-resolution paired restoration tasks, because it pairs files by name or stem and reads RGB images.

Suitable for immediate `T01` runs:

- `UHD-LL`
- `UHD-LL_down4`
- `UHD_haze`
- `UHD_deblur`
- `4K-Rain13k`
- `LOL-v1`
- `LOL-v2`

Not suitable for immediate `T01` training without a template change:

- `4K`
- `8K`

Reason: these are super-resolution datasets with LR/HR size mismatch. The current V1/T01 model outputs the input resolution, and the current paired loader crops input/target to the shared minimum size. That is not a valid LR-to-HR super-resolution setup. 4K/8K should wait for a dedicated SR-capable template.

## Adjustment To The Previous A02 Idea

The previous `A02` idea was to rerun `UHD-LL_down4` with better validation. That is still useful, but after adding `UHD_haze` and `UHD_deblur`, it should no longer be the only next step.

The adjusted plan is:

1. Use newly accepted same-resolution UHD restoration datasets first.
2. Keep experiments separate by task instead of immediately mixing datasets.
3. Keep `UHD-LL_down4` as a calibration/fallback dataset, not the only next training target.

## Recommended Run Order

### V1/T01/A02

- Dataset: `UHD_haze`
- Task: UHD dehazing
- Train split: `UHD_haze/train/input` -> `UHD_haze/train/gt`
- Internal validation: fixed subset from `UHD_haze/train`, not official test
- Official test: `UHD_haze/test`, local `231` pairs
- Training tier: short
- Crop policy: patch training, not full 4K batch
- Purpose: test whether V1/T01 transfers from low-light restoration to a newly accepted UHD dehazing task.

Rationale:

- It is a highest-priority paper-scope dataset.
- It has `2290` train pairs and `231` local test pairs.
- It is same-resolution paired restoration, so it fits current T01 with config changes.

### V1/T01/A03

- Dataset: `UHD_deblur`
- Task: UHD deblurring
- Train split: `UHD_deblur/train/input_new` -> `UHD_deblur/train/gt_new`
- Internal validation: fixed subset from `UHD_deblur/train`, not official test
- Official test: `UHD_deblur/test/input300` -> `UHD_deblur/test/gt300`
- Training tier: short
- Crop policy: patch training
- Purpose: check whether the same V1/T01 idea works on blur restoration after the dehazing probe.

### V1/T01/A04 Or Branch Run

- Dataset: `UHD-LL_down4`
- Purpose: protocol-clean replay of A01 with full internal validation.

This run is still useful, but it becomes a control/reproducibility run rather than the only next experiment.

## Why Not Mix All Datasets Immediately

Immediate mixed training is not recommended for the next short run because:

- T01 currently has no multi-dataset sampler.
- T01 has no task-conditioning field.
- Low-light, dehazing, deblurring, and deraining degradation distributions are different.
- If a mixed run succeeds or fails, it will be hard to tell which dataset or task drove the result.

Mixed training can become a later template, for example `T02`, after single-task short probes confirm that V1 is viable across several tasks.

## Practical A02 Gate Notes

Before launching `A02`, the deployment gate should explicitly check:

- dataset paths point to `UHD_haze`, not stale `UHD-LL_down4`;
- official `UHD_haze/test` is reserved for final evaluation;
- local `UHD_haze` test口径 is `231` pairs;
- validation subset is carved from train split;
- crop size and batch size are recalibrated, because full 4K batch 28 is not safe;
- `T01` is still a single-dataset paired-restoration template.

## Current Conclusion

Use the new datasets, but use them cleanly.

The next practical step should be:

> `V1/T01/A02`: short, patch-based `UHD_haze` run with a fixed internal validation subset.

Then:

> `V1/T01/A03`: short, patch-based `UHD_deblur` run.

This gives stronger evidence about V1's cross-task viability than repeating only `UHD-LL_down4`, while still keeping each run interpretable.
