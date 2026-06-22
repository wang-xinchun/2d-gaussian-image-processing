# Result Analysis: V1/T01/A_ap03

Status: completed and closed out, but not a promotion candidate.

`A_ap03` is the demand-conditioned residual soft-router mainline after the negative `A_am03` result. Final interpretation uses same-protocol official full-image testing on `final.pt` after clean short-run completion.

## Completion

- Training finished cleanly at `step=2500`.
- `step_002500.pt`, `latest.pt`, and `final.pt` share identical model weights.
- Final train metric: `loss_total=0.02709762565791607`.

## Validation Trajectory

| step | val_loss | val_psnr |
| ---: | ---: | ---: |
| 500 | 0.025119276158511637 | 28.512026467667624 |
| 1000 | 0.02411575373262167 | 28.665201331008205 |
| 1500 | 0.025638479460030793 | 28.306285066358477 |
| 2000 | 0.024406322557479144 | 28.602542903189573 |
| 2500 | 0.023809175845235585 | 28.759788226347787 |

## Official Test

```text
checkpoint = experiments/V1/runs/A_ap03/checkpoints/final.pt
split = test
samples = 300
crop_size = 0
device = cuda
loss_mean = 0.03691591784358025
psnr_mean = 25.233562564509878
```

## Comparison

| run | official test PSNR | delta vs A_ap03 final |
| --- | ---: | ---: |
| `A_ap03 final.pt` | 25.233562564509878 | 0 |
| `A_am03 final.pt` | 25.169950648899793 | +0.06361191561008523 |
| `A_aj03 final.pt` | 25.2308542283191 | +0.002708336190778134 |
| `A_ah03 final.pt` | 25.345361164304613 | -0.11179859979473462 |
| `A_z03 final.pt` | 25.385710820365173 | -0.15214825585529515 |

## Interpretation

The demand-conditioned residual soft router is a real step up over `A_am03`. The new mechanism clearly restored some stability, and the validation curve is healthier: the final checkpoint is the best validation point, not a late degradation.

That said, the gain is still insufficient. `A_ap03` remains below `A_ah03` and `A_z03` on the official test split, which means the residual-soft routing idea improves the hard-competition failure mode but does not fully solve the cross-split selectivity problem.

The practical bottleneck diagnosis after `A_ap03` is:

- hard competition was too blunt;
- demand-conditioned soft routing is better and more stable;
- but the current control surface still lacks enough structured specialization to overtake the best short leader.

## Decision

Keep the demand-conditioned residual soft-routing lesson, but do not promote `A_ap03`. The next mainline should likely keep the same cooperative-first principle while exploring either a stronger stage-aware demand schedule, a multi-stage controller, or a more expressive shared-control scaffold that can preserve cooperation early and specialization late without relying on a single global demand gate.
