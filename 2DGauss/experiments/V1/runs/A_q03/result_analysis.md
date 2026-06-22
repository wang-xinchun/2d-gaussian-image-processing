# Result Analysis: V1/T01/A_q03

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_q03`
- Dataset: `UHD_deblur`
- Training split: `1924` pairs from `train`
- Internal validation split: `20` pairs from `train`
- Official test split: `300` pairs from `test`

## Completion Status

- Training finished successfully.
- Final step: `2500`
- Final logged elapsed time: `34.4349` minutes
- Final checkpoint: `experiments/V1/runs/A_q03/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A_q03/checkpoints/step_002500.pt`
- Training log: `experiments/V1/runs/A_q03/logs/train_20260515_092930.log`
- Train metrics: `experiments/V1/runs/A_q03/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A_q03/metrics/val_metrics.jsonl`

The run completed by reaching `max_steps=2500`, not by the `wall_clock_minutes=58`
cap. No NaN, data-loader failure, checkpoint failure, or post-validation OOM
occurred.

Technical note:

- `step_002500.pt`, `latest.pt`, and `final.pt` have identical `model`
  weights.
- This means the accepted official result of `final.pt` is also the effective
  result of the true step-`2500` endpoint.

## Training Behavior

The first logged training loss was:

```text
step 1 loss_total = 0.12407676130533218
```

The final logged training loss before clean finish was:

```text
step 2500 loss_total = 0.026448819786310196
```

Interpretation:

- The lightweight terminal-residual branch is fully deployable under the
  calibrated `6x744` high-VRAM envelope.
- So the branch answers the engineering question positively: the mechanism can
  train stably and does not introduce runtime regressions by itself.
- The real question therefore remains quality, not trainability.

## Training-Time Full-Image Validation

Training-time validation used the first `20` train pairs with full-image
`crop_size=0`.

Validation trajectory:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 500 | 0.024028 | 28.4375 |
| 1000 | 0.023606 | 28.4604 |
| 1500 | 0.022684 | 28.7026 |
| 2000 | 0.023619 | 28.4306 |
| 2500 | 0.023003 | 28.6758 |

The best training-time validation loss is:

```text
step 1500
val_loss = 0.022684021014720203
val_psnr = 28.70263263899879
```

The best training-time validation PSNR is the same point:

```text
step 1500
val_loss = 0.022684021014720203
val_psnr = 28.70263263899879
```

Interpretation:

- The branch shows a healthy short-run validation signal and peaks in the mid
  phase rather than only at the endpoint.
- But this validation scale is still weaker than the promoted decoder-guidance
  short winner `A_n03` (`28.9612 dB`).
- So even before official testing, the branch does not look like a stronger
  short-run anchor than the existing promoted family winner.

## Post-Hoc Full Evaluation

Post-hoc evaluation was run on:

- official test set: `300` samples, `crop_size=0`

Evaluated checkpoints:

- `experiments/V1/runs/A_q03/checkpoints/step_001500.pt`
- `experiments/V1/runs/A_q03/checkpoints/final.pt`

Results:

| Checkpoint | Split | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|
| `step_001500.pt` | official test | 300 | 0.035354 | 25.0953 |
| `final.pt` | official test | 300 | 0.035314 | 25.1826 |

Interpretation:

- The final checkpoint is again better than the internal-validation winner on
  the official test, so the project-wide endpoint-preference lesson still
  holds.
- But the terminal-residual branch as a whole is weak on the official
  protocol: even its best endpoint remains below every completed
  decoder-guidance short probe in the same family window.

## Comparison To Existing UHD Deblur Anchors

Frozen milestone before this run:

```text
A_k03 final.pt official full-image test PSNR = 26.199959289578075
```

Relevant same-task same-protocol short and medium references:

```text
A_m03 final.pt official full-image test PSNR = 25.269007130575343
A_n03 final.pt official full-image test PSNR = 25.291929097577473
A_o03 final.pt official full-image test PSNR = 25.280874529877735
A_p03 final.pt official full-image test PSNR = 26.008706971730295
```

Same-task same-protocol comparison:

| Run | Checkpoint | Official Test PSNR | Delta vs A_k03 Final |
|---|---|---:|---:|
| `A_k03` | `final.pt` | 26.2000 | 0.0000 |
| `A_p03` | `final.pt` | 26.0087 | -0.1913 |
| `A_n03` | `final.pt` | 25.2919 | -0.9080 |
| `A_o03` | `final.pt` | 25.2809 | -0.9191 |
| `A_m03` | `final.pt` | 25.2690 | -0.9310 |
| `A_q03` | `step_001500.pt` | 25.0953 | -1.1047 |
| `A_q03` | `final.pt` | 25.1826 | -1.0174 |

Additional deltas:

```text
A_q03 final.pt vs A_q03 step_001500.pt = +0.08728309913108673 dB
A_q03 final.pt vs A_m03 final.pt       = -0.08645303293846256 dB
A_q03 final.pt vs A_n03 final.pt       = -0.10937500000059375 dB
A_q03 final.pt vs A_o03 final.pt       = -0.09832043224085538 dB
A_q03 final.pt vs A_p03 final.pt       = -0.8261528740934148 dB
A_q03 final.pt vs A_k03 final.pt       = -1.0174051919411945 dB
```

Interpretation:

- This is not a near-tie negative. The branch is materially below the current
  short-run promoted structure and far below the frozen `M03/A_k03` baseline.
- The result therefore argues against promoting the current terminal-residual
  design directly into a longer-budget run.
- The bottleneck diagnosis itself may still be directionally right, but this
  specific lightweight implementation does not convert that diagnosis into a
  positive same-protocol gain.

## Main Conclusion

`V1/T01/A_q03` is accepted as a successfully completed short terminal-control
probe, but it is a **negative-quality result** rather than a new baseline
candidate or a new preferred promotion anchor.

It proves:

- the lightweight `Gaussian-guided terminal residual` mechanism is runtime-safe
  and trainable under the required `85-93G` window;
- final endpoints still outperform earlier internal-validation winners on the
  official test;
- the current implementation of direct terminal residual control does **not**
  improve same-protocol official performance over the existing
  decoder-guidance short line.

It does not prove:

- that terminal control is irrelevant in general;
- that no future high-resolution end-stage refinement should be explored.

What it does strongly suggest is:

- the next deblur branch should not simply replay `A_q03` for longer;
- future lightweight upgrades should keep targeting the final reconstruction
  gap, but with a different mechanism or a better-coupled implementation than
  the current terminal residual head.

## Acceptance Decision

Acceptance status:

```text
runtime/checkpoint acceptance: passed
short-probe completion: passed
official baseline replacement: no
preferred next promotion anchor: stay with A_k03/M03 and A_n03-derived mainline, not A_q03
```

## Recommended Next Action

The highest-value next step on the current run line is now:

1. record `A_q03` as a clean but negative short probe rather than a winner;
2. preserve the broader “final reconstruction still matters” diagnosis, but
   reject this specific terminal-residual implementation as the immediate
   promotion path;
3. design the next lightweight deblur upgrade so it continues targeting the
   high-resolution end of reconstruction without simply replaying `A_q03`.
