# Versioning Policy

## Current Architecture

- Architecture version: `V1`
- Meaning: first executable architecture version of the project.
- Code baseline: `src/uhd_gaussian/`
- Top-level model: `CrossResolutionGaussianRestorer`

## Training Run Names

All training runs under `V1` use `A**` identifiers:

- `A01`
- `A02`
- `A03`

Derived or branch runs use subgroup identifiers:

- `A_a01`
- `A_a02`
- `A_b03`

## Training Template Names

Training templates are reusable code baselines for runs under the same architecture.

Use `T**` identifiers for templates:

- `T01`
- `T02`
- `T03`

The first complete training pipeline for `V1` should become `V1/T01`. Later template changes, such as a new loss mix, scheduler policy, dataset loader contract, logging layout, or distributed-training wrapper, should create a new template version instead of silently mutating past runs.

## Recommended Full Identifier

Use the combined form when naming checkpoints, experiment folders, and reports:

```text
V1/A01
V1/A02
V1/A_a01
V1/A_b03
V1/T01
```

For filesystem paths, use a slash-safe form:

```text
V1_A01
V1_A02
V1_A_a01
V1_A_b03
V1_T01
```

## Management Rule

Architecture versions and training versions are separate:

- `V1`, `V2`, etc. describe model architecture lineage.
- `T01`, `T02`, etc. describe reusable training-template lineage under an architecture.
- `A01`, `A02`, `A_a01`, etc. describe training/checkpoint lineage under an architecture.

A checkpoint or experiment record should always state the architecture version, training-template version, and training-run version.

## Archival Rule

Every training run must keep an independent, restorable record. Do not rely only on the current working tree.

Each run should preserve:

- exact training-template code snapshot used by the run
- config files and command line
- dataset paths and dataset split/version notes
- environment notes when available
- tensorboard/log files/stdout summaries
- all useful checkpoints and final weights
- metrics and validation outputs

Recommended run layout:

```text
experiments/
  V1/
    templates/
      T01/
    runs/
      A01/
        template_snapshot/
        configs/
        gate_check.md
        checkpoints/
        logs/
        metrics/
        manifest.md
```

Historical runs should remain available for resuming training, reproducing results, ablation comparison, and later paper analysis.

## First Planned Run

The first prepared run is:

```text
V1/T01/A01
```

It is a short-training run on `UHD-LL_down4`, with its run-specific gate at:

```text
experiments/V1/runs/A01/gate_check.md
```

The split policy for A01 is recorded in:

```text
docs/evaluation_split_policy.md
```
