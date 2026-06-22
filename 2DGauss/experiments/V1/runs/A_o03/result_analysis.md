# Result Analysis: V1/T01/A_o03

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_o03`
- Dataset: `UHD_deblur`
- Training split: `1924` pairs from `train`
- Internal validation split: `20` pairs from `train`
- Official test split: `300` pairs from `test`

## Completion Status

- Training finished successfully.
- Final step: `2500`
- Final logged elapsed time: `34.9640` minutes
- Final checkpoint: `experiments/V1/runs/A_o03/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A_o03/checkpoints/step_002500.pt`
- Training log: `experiments/V1/runs/A_o03/logs/train_20260515_serial.log`
- Train metrics: `experiments/V1/runs/A_o03/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A_o03/metrics/val_metrics.jsonl`

The run completed by reaching `max_steps=2500`, not by the `58`-minute
wall-clock cap. No NaN, checkpoint failure, or post-validation OOM occurred
after the accepted launch geometry was selected.

## Training Behavior

The first logged training loss was:

```text
step 1 loss_total = 0.12385682016611099
```

The final logged training loss before clean finish was:

```text
step 2500 loss_total = 0.03820892423391342
```

Interpretation:

- The decoder-guidance branch can train cleanly even under a near-`A_k03`
  high-fill launch attempt after branch-local context trimming.
- The engineering risk of this branch is therefore not “cannot train”, but
  “real startup memory is too close to or above the user’s upper bound”.

## High-Fill Calibration Outcome

This branch’s main job is deployment-shape scouting.

Observed serial calibration:

```text
batch7 context776: immediate OOM, reserved_gb=93.9688
batch7 context772: immediate OOM, reserved_gb=93.6875
batch7 context768: OOM after 5 steps, reserved_gb=94.2813
batch7 context760: ok but over upper bound, reserved_gb=93.9082
batch7 context752: ok but over upper bound, reserved_gb=93.5742
batch7 context748: accepted, reserved_gb=92.5859
batch7 context744: accepted, reserved_gb=91.6055
```

Final chosen launch geometry:

- `batch_size = 7`
- `train_context_size = 748`

Critical runtime note:

- despite the accepted smoke result at `7x748`, real startup observation
  reached about `95.5G`, which exceeds the user’s practical `93G` upper bound.

Interpretation:

- `A_o03` answers the high-fill question only partially positively.
- A near-`A_k03` geometry can be made trainable for this branch, but it is not
  yet a practical safe anchor for the next `5`-hour run.

## Training-Time Full-Image Validation

Training-time validation used the first `20` train pairs with full-image
`crop_size=0`.

Validation trajectory:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 500 | 0.023264 | 28.5758 |
| 1000 | 0.024318 | 28.4420 |
| 1500 | 0.023372 | 28.5986 |
| 2000 | 0.023886 | 28.5425 |
| 2500 | 0.022345 | 28.8394 |

The best training-time validation PSNR is:

```text
step 2500
val_loss = 0.022345110028982162
val_psnr = 28.83941029392428
```

Interpretation:

- `A_o03` clears the family strong-keep threshold `28.80 dB`.
- But it does **not** outrank `A_n03` on the primary short-run validation
  criterion.

## Post-Hoc Official Evaluation

Post-hoc evaluation was run on:

- official test set: `300` samples, `crop_size=0`

Evaluated checkpoint:

- `experiments/V1/runs/A_o03/checkpoints/final.pt`

Result:

| Checkpoint | Split | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|
| `final.pt` | official test | 300 | 0.034571 | 25.2809 |

Interpretation:

- This corroborative official result also remains below `A_n03 final.pt`
  (`25.2919 dB`).
- So `A_o03` does not win either the short-run validation gate or the
  corroborative official-test comparison inside the scout family.

## Comparison To A_m03 And A_n03

Relevant family summary:

| Run | Best short-run val PSNR | Official test PSNR |
|---|---:|---:|
| `A_m03` | 28.8486 | 25.2690 |
| `A_n03` | 28.9612 | 25.2919 |
| `A_o03` | 28.8394 | 25.2809 |

Interpretation:

- `A_o03` is competitive, but not the family winner.
- Its main value is the launch-shape evidence, not promotion priority.

## Main Conclusion

`V1/T01/A_o03` is accepted as a successful completed high-fill runtime scout
inside the `A_m03/A_n03/A_o03` short-probe family.

It proves:

- the default-strength decoder-guidance branch can train under a trimmed
  near-`A_k03` `batch=7` geometry;
- the branch remains quality-competitive enough to pass the strong-keep gate;
- the family now has concrete evidence about where the high-fill memory edge
  starts to become unsafe.

It does not prove:

- that it should be promoted over `A_n03` into the next `5`-hour window;
- that its accepted smoke geometry is safe enough as a practical long-run
  launch anchor under the user’s real startup memory constraint.

## Promotion Decision

Promotion status:

```text
family completion gate: passed
family strong-keep gate: passed
family top rank by short-run val: no
family top rank by official corroboration: no
practical 5h anchor recommendation: no
```

The next `5`-hour medium run should therefore inherit the **structure** of
`A_n03`, while treating `A_o03` mainly as a deployment-envelope reference.
