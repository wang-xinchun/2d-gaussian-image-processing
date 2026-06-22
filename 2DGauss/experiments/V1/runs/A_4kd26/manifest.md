# Manifest: V1/T01/A_4kd26

- Created: `2026-06-13 CST`
- Run: `V1/T01/A_4kd26`
- Purpose: Fix A25's file-level balancing miss by making route balancing crop-aware.
- Baseline/Init: `A_4kd15` best-val, loaded with `init_checkpoint_strict=false`.
- Teacher A/high route: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`
- Teacher B/low route: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/checkpoints/best_val.pt`
- Run-local config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd26/configs/a_4kd26_fast_4kdehaze_crop_balanced_input_std_dual_route_residual.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd26/launch_tmux.sh`
- Wait launcher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd26/wait_and_launch_a_4kd26_memory_gated_gpu_safe.sh`

## Hypothesis

`A_4kd25` showed that file-level `input_std` balancing is insufficient: random
crop selection can still produce all-low or all-high route batches. `A_4kd26`
therefore passes route targets from the batch sampler into the dataset and lets
the random cropper retry up to `32` times to match the requested high/low
`input_std` side.

## Narrow Variables

- Same dual-route residual mechanism as A25.
- Same A15 init and A08/A15 branch teachers.
- Enable both file-level balanced batch selection and crop-aware route retry:
  `input_std_route_balanced_crop_sampling=true`.

## Success Criteria

- Training logs should show `input_std_route_high_hard_mean` near `0.5` much
  more consistently than A25.
- Validation should recover from A25's early drop and ideally beat A24's
  single-checkpoint fast-test PSNR after training.
