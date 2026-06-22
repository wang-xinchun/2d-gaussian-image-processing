# Manifest: V1/T01/A_4kd24

- Created: `2026-06-13 CST`
- Run: `V1/T01/A_4kd24`
- Purpose: Replace A23's weak whole-output router distillation with a structural input-std dual-route residual head.
- Baseline/Init: `A_4kd15` best-val, loaded with `init_checkpoint_strict=false` because A24 adds new route-head parameters.
- Teacher A/high route: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`
- Teacher B/low route: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/checkpoints/best_val.pt`
- Run-local config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/configs/a_4kd24_fast_4kdehaze_a15_input_std_dual_route_residual.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/launch_tmux.sh`
- Wait launcher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/wait_and_launch_a_4kd24_memory_gated_gpu_safe.sh`

## Hypothesis

`A_4kd23` failed because it tried to compress the A18 two-checkpoint router into
one final-output target. The student stayed A15-like and did not form a distinct
A08 high-input-std behavior.

`A_4kd24` changes the mechanism:

```text
input_std >  0.07026463374495506 -> high route -> A_4kd08 teacher
input_std <= 0.07026463374495506 -> low route  -> A_4kd15 teacher
```

The route head is a default-off, zero-initialized residual module attached near
the terminal residual path. It returns high/low branch outputs separately, so
training can supervise each branch instead of only supervising their mixture.

## Narrow Variables

- Keep the A15 backbone and fast 4KDehaze crop512 data view.
- Add `use_input_std_route_residual=true` and the new branch/gate losses.
- Set `router_distillation_weight=0.0`; two teachers are loaded for branch
  supervision only.
- Use `input_std_route_gate_weight=0.04` to anchor the learned gate to the A18
  input-std rule.
- Use `input_std_route_branch_weight=0.08` to push high branch toward A08 and
  low branch toward A15.

## Success Criteria

- Preflight must non-strict-load A15 init and both teachers, then report route
  metrics with nonzero `loss_input_std_route_*` values.
- Training logs must show route diagnostics such as `train_route_gate_mean`,
  `train_route_high_soft_mean`, and high/low branch losses.
- Best-val checkpoint should be evaluated against A08/A15/A18-router/A23. The
  immediate bar is to recover evidence that a single checkpoint can preserve
  the two-route complement rather than collapsing to A15.

## Boundary

This is still a fast-probe conversion experiment. It is not a full benchmark
claim unless the fast-test result clears the A18 router bar and later survives a
larger evaluation split.
