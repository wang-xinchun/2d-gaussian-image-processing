# First Training Plan: V1/T01/A01

## Role

`A01` is the first short-training run for the `V1` architecture using the `T01` training template.

## Dataset

Use `UHD-LL_down4` first because it is paired, accepted, smaller than full UHD, and suitable for fast train-loop validation.

Split policy:

- Training-time acceptance/validation: first `100` pairs from `UHD-LL_down4/training_set`.
- Effective training: remaining `1900` pairs from `UHD-LL_down4/training_set`.
- Final test: all `150` pairs from `UHD-LL_down4/testing_set`.
- The official test split is not used for checkpoint selection or short-run acceptance.

## Goal

The goal is not to claim final performance. The goal is to prove or falsify the first trainable path:

- paired dataloader works
- V1 forward/backward works on real image patches
- loss decreases enough to justify follow-up
- validation and checkpointing work
- archive layout is restorable

## Deployment Gate

The run-specific gate is:

```text
experiments/V1/runs/A01/gate_check.md
```

The reusable gate template is:

```text
docs/training_deployment_gate.md
```

## Launch

Preflight first:

```bash
python3 scripts/train_v1_t01.py --config experiments/V1/templates/T01/configs/a01_short_uhdll_down4.toml --preflight
```

Current preflight status: corrected preflight passed after the validation/test split correction.

Deployment smoke status: passed. Use batch size `28` and full-image `540x960` training to target about `89.453G` peak reserved GPU memory. A01 is configured for `4000` max steps with a `58` minute wall-clock cap.

Launch short training only after preflight passes:

```bash
python3 scripts/train_v1_t01.py --config experiments/V1/templates/T01/configs/a01_short_uhdll_down4.toml
```
