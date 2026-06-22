# Next Experiment Design: V1/T01/A_a03

Date: 2026-05-07 CST

## Boundary

This document designs the next experiment only. It does not authorize deployment.

Per the user instruction, this design turn intentionally does not run smoke
calibration, formal preflight, or training, and it does not create a launch
script.

## Current Baseline

The latest accepted `UHD_deblur` baseline is `V1/T01/A03`.

Key numbers:

| Run | Checkpoint | Split | PSNR |
|---|---|---|---:|
| `A03` | `step_000500.pt` | full internal val | `29.0056` |
| `A03` | `step_000500.pt` | official test | `25.0978` |
| `A03` | `final.pt` | full internal val | `28.2966` |
| `A03` | `final.pt` | official test | `25.1402` |

The official baseline record is therefore:

```text
V1/T01/A03 final.pt official-test PSNR = 25.1402 dB
```

`A03` also revealed a checkpoint-selection issue: internal validation favored
`step_000500.pt`, while official test marginally favored `final.pt`. This must
be made explicit in the next branch.

## Proposed Next Run

- Run: `V1/T01/A_a03`
- Dataset: `UHD_deblur`
- Type: branch run after `A03`
- Status: design package only
- Core idea: modest capacity and Gaussian-expression increase under the same
  `T01` template and same data protocol.

## Why This Branch

At this point, another identical `A03` replay would tell us little. The useful
question is whether the current method line benefits from a little more
representational capacity before we redesign the training template.

This branch therefore changes only model-capacity and Gaussian-expression
settings, while holding the following fixed:

- dataset and split policy;
- crop policy;
- loss mix;
- optimizer;
- validation cadence;
- checkpoint cadence;
- short-run tier;
- official test reservation.

This keeps the result interpretable: if `A_a03` improves, the first explanation
is that `UHD_deblur` benefits from stronger cross-resolution Gaussian-guided
representation, not from a different dataset, loss, schedule, or split.

## Planned Architecture Delta

| Field | `A03` | `A_a03` |
|---|---:|---:|
| `width` | `32` | `48` |
| `highres_blocks` | `2` | `3` |
| `lowres_blocks` | `4` | `6` |
| `decoder_blocks` | `2` | `3` |
| `affine_rank` | `4` | `6` |
| `gaussian_atoms` | `4` | `6` |
| `max_frequency` | `4.0` | `6.0` |
| `max_texture_gain` | `1.0` | `1.2` |

Unchanged:

- `lowres_scale=4`
- `gaussian_param_dim=10`
- Charbonnier weight `1.0`
- gradient loss weight `0.05`
- AdamW lr `0.0002`
- weight decay `0.0001`

## Planned Data Protocol

Same as `A03`:

- train input: `UHD_deblur/train/input_new`
- train target: `UHD_deblur/train/gt_new`
- internal validation: first `100` train pairs
- effective training: remaining `1864` train pairs
- official test: `test/input300 -> test/gt300`
- official test count: `300` image pairs

The official test split is reserved for post-training evaluation only.

## Planned Runtime Envelope

Initial design values:

```text
batch_size = 16
num_workers = 16
train_crop_size = 512
val_crop_size = 512
max_steps = 2000
wall_clock_minutes = 58
val_interval = 250
checkpoint_interval = 250
max_val_batches = 100
amp = false
```

These values are not launch-approved. They are only a starting envelope copied
from the successful `A03` run. Because `A_a03` is larger, it must pass a new
data-inclusive smoke calibration before launch.

## Checkpoint Policy

`A_a03` should not choose a model by peeking at official test.

The planned analysis protocol is:

1. Select the best checkpoint by full internal validation.
2. Also evaluate `final.pt`.
3. Run official test post-hoc for both the internal-val-selected checkpoint and
   `final.pt`.
4. Report both numbers.
5. Compare the primary branch result to `A03` using the written rule in the
   result analysis.

This directly addresses the `A03` mismatch between internal-val and official
test preference.

## Success Criteria

Minimum:

- future smoke/preflight pass;
- if launched, reaches validation/checkpoint milestones without runtime failure.

Useful:

- completes `2000` steps inside the short-run cap;
- produces full-val and official-test post-hoc evaluations.

Metric:

- meaningful target: official-test PSNR at least `A03 + 0.3 dB`;
- strong target: official-test PSNR at least `A03 + 0.5 dB`.

Given `A03 final.pt official-test PSNR=25.1402`, this means:

```text
meaningful target: >= 25.4402 dB
strong target:     >= 25.6402 dB
```

## Prepared Files

- `experiments/V1/templates/T01/configs/a_a03_short_uhddeblur_capacity.toml`
- `experiments/V1/runs/A_a03/configs/a_a03_short_uhddeblur_capacity.toml`
- `experiments/V1/runs/A_a03/gate_check.md`
- `experiments/V1/runs/A_a03/manifest.md`

## Required Before Any Future Deployment

1. Syntax-check the new config and scripts.
2. Run `scripts/smoke_calibrate_v1_t01.py` on real `UHD_deblur` batches for the
   `A_a03` capacity setting.
3. Confirm memory and end-to-end throughput.
4. Run formal `scripts/train_v1_t01.py --preflight`.
5. Update `gate_check.md` and `manifest.md` with measured results.
6. Ask for explicit user approval to launch.
