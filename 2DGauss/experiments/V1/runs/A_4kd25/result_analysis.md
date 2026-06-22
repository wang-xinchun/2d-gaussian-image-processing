# Result Analysis: V1/T01/A_4kd25

- Run: `V1/T01/A_4kd25`
- Status: stopped early on `2026-06-13 CST`
- Purpose: Test file-level high/low `input_std` balanced sampling on top of A24.
- Training log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd25/logs/train_20260613_205414.log`

## Outcome

`A_4kd25` was intentionally stopped after early validation showed it was not on
the A24 trajectory.

| Step | val_psnr | val_loss | Selection |
| ---: | ---: | ---: | --- |
| 50 | 18.231016319544015 | 0.12287615351378918 | best |
| 100 | 18.198464752448107 | 0.12331412188708782 | no |

Best early validation was far below `A_4kd24` final/best validation
`18.349088096641417`.

## Interpretation

The sampler itself was enabled and wrote metadata showing `429` high-file and
`83` low-file samples. However, file-level `input_std` does not guarantee
crop-level routing after random crop. Training logs still showed batches with
`input_std_route_high_hard_mean=0.0`, so the mechanism did not reliably provide
per-step high/low branch supervision.

## Decision

Do not continue or promote `A_4kd25`. Keep it as a negative mechanism diagnostic
that motivated `A_4kd26` crop-aware route sampling.
