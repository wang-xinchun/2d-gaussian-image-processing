# Run Manifest: V1/T01/A_k03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_k03`
- Full identifier: `V1/T01/A_k03`

## Current Status

This branch has now completed training and post-hoc evaluation. It is the
immediate post-`A_j03` long-budget follow-up to the frozen `A_i03/M02`
baseline, and it has now been accepted as the new official `UHD_deblur`
baseline under the current local protocol.

## Purpose

`A_k03` exists to answer the cleanest remaining question after `A_i03` and
`A_j03`:

> if we keep the accepted `A_i03/M02` setting fixed and only remove the
> premature step-cap bottleneck, does the deblur line keep improving?

This branch therefore does not promote `A_j03`'s `864` context, does not
change the effective train split, and does not introduce any new mechanism.

## Why This Branch Exists

The line now has two strong pieces of evidence:

1. `A_i03` is the accepted official baseline and the current frozen milestone;
2. `A_j03` is a negative short probe showing that simply enlarging training
   context to `864` under a short budget is not the right next upgrade
   direction.

At the same time, `A_i03` did not actually use the full intended `300`-minute
budget. It finished at:

```text
step=12000
elapsed=156.1359 minutes
```

That means the clearest next long-budget action is not another structural
change, but to stay on the accepted `A_i03` anchor and let the same branch use
substantially more optimization steps.

## Experimental Question

Can the accepted `A_i03` setting improve further on the official full-image
protocol if we:

1. keep `batch_size=7`;
2. keep `train_context_size=784`;
3. keep `train_offset=40`;
4. keep the same edge-guided + directional/frequency + consistency stack;
5. keep the same `85-93G` runtime envelope;
6. but raise the step cap high enough that the run is no longer stopped early
   by `max_steps=12000`?

## Main Design Choices

Preserved from `A_i03/M02`:

- dataset: `UHD_deblur`
- effective training split: train pairs after the first `40`
- edge-guided refinement
- directional/frequency guidance
- `TLC`-style train-inference consistency
- `train_context_size=784`
- `train_supervision_crop_size=512`
- official full-image validation/test protocol
- target real-launch VRAM window: `85.0-93.0G`

Changed relative to `A_i03`:

- run identity: `A_i03 -> A_k03`
- task tag: baseline extension instead of baseline establishment
- `max_steps`: `12000 -> 24000`

Unchanged on purpose:

- `wall_clock_minutes = 300`
- `batch_size = 7`
- `val_interval = 1000`
- `checkpoint_interval = 500`
- optimizer and loss settings
- random seed

## Runtime Strategy

`A_k03` is intentionally a step-cap extension branch, not a context or
mechanism branch.

The working assumption is:

- `A_i03` already proved the `7x784` setting is scientifically valid and
  deployable;
- `A_j03` already ruled out immediate promotion of `864` context;
- the most likely remaining upside comes from allowing the accepted branch to
  actually consume the runtime budget it was prepared for.

## Comparison Anchor

- Frozen milestone:
  `experiments/V1/milestones/M02_A_i03_uhddeblur_official_baseline`
- Stable milestone link:
  `experiments/V1/milestones/current_uhddeblur_official_baseline`
- Negative short probe:
  `V1/T01/A_j03`

## Deployment And Completion Summary

The branch completed the required launch gate before training:

1. branch-local smoke calibration on `7x784` passes with
   `max_memory_reserved_gb=88.4258` and `estimated_steps_target_wall_clock=13108`;
2. formal preflight passes on CUDA with `batch_input_shape=(7, 3, 784, 784)`,
   `total_params=246888`, `loss_total=0.0953299`, and `val_psnr=21.7638`;
3. clean tmux launch succeeds under session
   `2dgauss_a_k03_uhddeblur_tlc_stepcap_extension`;
4. startup-window `nvidia-smi` shows `memory.used=90833 MiB`, which remains
   inside the required `85-93G` real-launch window;
5. `metrics/train_metrics.jsonl` advanced normally from startup and training
   remained stable through the full run.

The branch has now also completed the full training and evaluation loop:

1. training clean-finished at `step=23016`;
2. final logged elapsed time is `299.9313` minutes;
3. periodic checkpoints were written through `step_023000.pt`, followed by a
   distinct `final.pt`;
4. post-hoc internal-validation winner is `step_022000.pt` with
   `PSNR=29.4266`;
5. post-hoc official-test winner is `final.pt` with `PSNR=26.2000`;
6. relative to the frozen `A_i03/M02` baseline (`25.8506 dB`), the official
   gain is `+0.3493 dB`.

This means the branch answers its motivating question positively:

> keeping the accepted `A_i03` setting fixed and only removing the earlier
> step cap does further improve same-protocol official test performance.

At the same time, the branch does not yet cross the current lightweight
`UHD-Blur` `90%` threshold of `26.5743 dB`; the remaining gap is about
`0.3743 dB`.
