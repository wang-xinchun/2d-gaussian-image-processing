# Result Analysis: V1/T01/A_4kd26

- Run: `V1/T01/A_4kd26`
- Status: stopped early on `2026-06-13 CST`
- Purpose: Make A25's route balancing crop-aware by passing high/low targets to the dataset cropper.
- Training log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd26/logs/train_20260613_210916.log`

## Outcome

`A_4kd26` fixed the A25 mechanism problem but hurt validation quality, so it was
stopped early.

| Step | val_psnr | val_loss | Selection |
| ---: | ---: | ---: | --- |
| 50 | 18.233808405468825 | 0.12289005909115076 | best |
| 100 | 18.236443073291603 | 0.12283354680985212 | best |
| 150 | 18.2230662628357 | 0.12306080736219883 | no |

Best early validation remained far below `A_4kd24` final/best validation
`18.349088096641417`.

## Interpretation

The crop-aware mechanism worked technically: training logs kept
`input_std_route_high_hard_mean=0.5` across the observed steps, so each batch
actually contained one high-route and one low-route crop. The negative result is
therefore meaningful: forcing a 50/50 route distribution changes the natural
training distribution too aggressively and degrades the A24 trajectory.

## Decision

Do not continue or promote `A_4kd26`. Keep `A_4kd24` as the best current
single-checkpoint candidate. The next mechanism should preserve natural sample
frequency while strengthening the route capacity, for example a stronger
route-conditioned expert path or route-specific terminal decoder heads, rather
than hard 50/50 balanced sampling.
