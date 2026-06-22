# Evaluation Split Policy

## Record Time

- Time: `2026-04-24 20:04:03 CST`

## Terms

- Training set: samples used for gradient updates.
- Acceptance/validation set: samples used during training to check whether a run is healthy, choose checkpoints, and decide whether to move from short training to medium or long training.
- Test set: official or held-out samples used only after a run is complete for final evaluation.

## Core Rule

Do not use the official test split for training-time acceptance unless explicitly marked as a temporary debugging exception.

For standard runs:

- carve a fixed internal acceptance/validation subset from the official training split;
- train on the remaining training samples;
- keep the official test split untouched until final evaluation.

## A01 Split

For `V1/T01/A01` on `UHD-LL_down4`:

- Internal acceptance/validation set:
  - source: `UHD-LL_down4/training_set`
  - selection: first `100` paired samples in deterministic filename order
  - purpose: training-time validation, short-run acceptance, checkpoint trend checking
- Training set:
  - source: `UHD-LL_down4/training_set`
  - selection: remaining `1900` paired samples after the first `100`
  - purpose: gradient updates
- Test set:
  - source: `UHD-LL_down4/testing_set`
  - size: `150` paired samples
  - purpose: final post-training evaluation only

For fast preflight, it is acceptable to evaluate only a small subset of the internal acceptance set, such as `20` samples, because preflight is not a reported metric.

## Recommended Future Splits

- `UHD-LL`: reserve a fixed internal validation subset from `training_set`; keep `testing_set` for final evaluation.
- `4K-Rain13k`: reserve a fixed internal validation subset from `train`; keep `test` for final evaluation.
- `8K`: reserve a fixed internal validation subset from `train`; keep `test` for final evaluation.
- `LOL-v1`: reserve an internal validation subset from `our485`; keep `eval15` for final evaluation.
- `LOL-v2`: reserve an internal validation subset from `Train`; keep `Test` for final evaluation.

The exact count for each future split should be written into the run-specific gate and manifest before launch.
