# Manifest: V1/T01/A_4kd25

- Created: `2026-06-13 CST`
- Run: `V1/T01/A_4kd25`
- Purpose: Strengthen A24 by enforcing high/low `input_std` balance in each training batch.
- Baseline/Init: `A_4kd15` best-val, loaded with `init_checkpoint_strict=false`.
- Teacher A/high route: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`
- Teacher B/low route: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/checkpoints/best_val.pt`
- Run-local config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd25/configs/a_4kd25_fast_4kdehaze_balanced_input_std_dual_route_residual.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd25/launch_tmux.sh`
- Wait launcher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd25/wait_and_launch_a_4kd25_memory_gated_gpu_safe.sh`

## Hypothesis

`A_4kd24` is the first positive single-checkpoint structural conversion result,
but it still trails the A18 two-checkpoint router on PSNR. Training logs show
many batches are all-high or all-low route samples, which makes high/low branch
updates uneven.

`A_4kd25` keeps the A24 dual-route residual mechanism and adds a default-off
training sampler:

```text
batch_size=2 -> 1 high-input_std sample + 1 low-input_std sample per batch
threshold = 0.07026463374495506
```

This targets the remaining bottleneck directly: both route branches should get
teacher supervision every optimization step.

## Narrow Variables

- Same fast 4KDehaze crop512 data view as A24.
- Same A15 init and A08/A15 branch teachers.
- Same gate/branch losses as A24.
- Enable `training.input_std_route_balanced_sampling=true`.
- Increase route residual strength from `0.20` to `0.24`.
- Extend training from `400` to `500` steps.

## Success Criteria

- Preflight must report sampler metadata under
  `input_std_route_balanced_sampling=true` with both high and low sample counts.
- Training logs should show each batch has mixed route targets instead of
  repeatedly all-high/all-low.
- Best-val fast-test should beat A24 best-val `17.790202882742463` PSNR and
  move closer to A18 router `17.84693437641316` without losing A24's SSIM/loss
  improvement.
