# Training Deployment Gate

This file is the mandatory pre-launch checklist for every training deployment.

Deployment means any fresh training run, resumed training run, branch run, or rerun that produces logs, metrics, or checkpoints.

Before launch, copy this checklist into the run folder as `gate_check.md`, fill it out, and keep it with the run archive.

## Training Duration Tiers

Use the smallest tier that can answer the current question.

- Short training: no longer than 1 hour. Used to validate or falsify ideas, data loading, loss behavior, memory use, and basic output quality.
- Medium training: no longer than 24 hours. Used after a short run shows value, to test stability, hyperparameters, validation behavior, and readiness for long training.
- Long training: measured in days. Used only when the short and medium evidence is strong enough to justify full investment.

## Required Run Identity

- [ ] Architecture version is specified, for example `V1`.
- [ ] Training template version is specified, for example `T01`.
- [ ] Training run version is specified, for example `A01` or `A_a01`.
- [ ] Full run identifier is written, for example `V1/T01/A01`.
- [ ] Run folder is independent and will not overwrite any previous run.
- [ ] The exact training-template snapshot will be saved under the run folder.

## Dataset Check

- [ ] Dataset name is explicit.
- [ ] Dataset root path is explicit.
- [ ] Train/validation/test split paths are explicit.
- [ ] Validation/acceptance samples are not taken from the official test split unless explicitly marked as a temporary debugging exception.
- [ ] Official test split is reserved for final evaluation after training.
- [ ] Paired input/target folders are correct when supervised paired training is used.
- [ ] Sample counts are checked against the accepted dataset inventory.
- [ ] Image resolution or crop policy is explicit.
- [ ] No test split is used for training.
- [ ] Any synthetic degradation, downsampling, or augmentation policy is explicit.
- [ ] The run manifest records the dataset version/status document used for verification.

## Contamination Check

This section prevents stale settings from the previous run from leaking into the new run.

- [ ] Config file was created from the intended template, not copied blindly from an old run.
- [ ] Dataset paths match this run's plan.
- [ ] Model architecture parameters match this run's plan.
- [ ] Loss terms and loss weights match this run's plan.
- [ ] Optimizer and scheduler settings match this run's plan.
- [ ] Evaluation protocol, metric names, crop border, color space, and data range are explicit.
- [ ] Pretrained weight path is empty unless this run intentionally uses one.
- [ ] Resume checkpoint path is empty unless this run is intentionally a resume.
- [ ] If resuming, optimizer, scheduler, step count, and epoch count recovery policy is explicit.
- [ ] Output directory is new or intentionally tied to the resume run.
- [ ] Random seed policy is explicit.

## Runtime Check

- [ ] Training tier is selected: short, medium, or long.
- [ ] Planned wall-clock duration is written.
- [ ] Batch size is written.
- [ ] Gradient accumulation setting is written.
- [ ] Patch size or full-resolution policy is written.
- [ ] Number of epochs or max steps is written.
- [ ] Validation interval is written.
- [ ] Checkpoint save interval is written.
- [ ] GPU/device plan is written.
- [ ] Expected memory risk is acceptable for the selected tier.

## Launch Decision

- [ ] All required checks above are complete.
- [ ] Any exception is written and justified in the run manifest.
- [ ] The run is approved to launch.

If any required item is unchecked, do not launch the training deployment until it is resolved or explicitly approved and logged.
