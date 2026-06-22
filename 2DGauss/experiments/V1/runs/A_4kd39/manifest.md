# A_4kd39 Manifest

时间: 2026-06-15 CST

## Purpose

A39 follows the A37 route-aware teacher compression line instead of the A38 standard-loss high-head-only line. It starts from A37 best-val and keeps the low route locked, then restores native-config A08/A15 online teachers with high-route branch supervision.

The narrow mechanism change from A37 is to let `gaussian_control_canvas` join the trainable set. This tests whether a slightly earlier control/feature modulation path can reduce the remaining high-route/A08-tail gap without giving back A37's low-route surplus.

## Key Paths

- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd39/configs/a_4kd39_fast_4kdehaze_a37_teacher_control_refine.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd39/launch_tmux.sh`
- Memory-gated launcher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd39/wait_and_launch_a_4kd39_memory_gated_gpu_safe.sh`
- Init checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`
- Teacher A config/checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/configs/a_4kd08_fast_4kdehaze_medium_detail_protected_blended_demand_control_probe.toml`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`
- Teacher B config/checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/configs/a_4kd15_fast_4kdehaze_a08_best_no_haze_guard_ema995_finetune.toml`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/checkpoints/best_val.pt`

## Training Intent

- Keep natural sampling and the fixed A18 `input_std` prior.
- Initialize from A37 best-val, not A38 final.
- Disable ordinary reconstruction/proxy losses to avoid reopening low-route/base drift.
- Use high/A08 route branch teacher supervision with a slightly higher high-branch weight than A37.
- Keep low route locked by excluding `input_std_route_decoder_heads.low_*`.
- Train A37 late shared/high modules plus `gaussian_control_canvas`.

## Acceptance Focus

- Primary: high-route A18 alignment gap should improve from A37 `-0.11159004428502844 dB` toward or beyond `>-0.10 dB`.
- Safety: low-route gap should remain strongly positive, ideally `>= +0.12 dB`.
- Aggregate: test PSNR should stay near A37/A38, ideally `>= 17.88`.

## Boundary

A39 is still a fast-probe experiment on `4KDehaze_local_90_5_5_fast_probe_512`, not an official/full benchmark. It is a mechanism probe for route-conditioned capacity placement after A37.

## Completed Outcome

- Formal GPU3 training completed with exit code `0`.
- Best/final step: `120`.
- Best validation PSNR: `18.62166244241818`.
- Test PSNR: `17.900369585886775`, the best aggregate fast-probe number so far.
- A18 overall gap: `+0.053435210072354966`.
- A18 low-route gap: `+0.18320352678672946`.
- A18 high-route gap: `-0.11858325626995542`.

Interpretation: trainable `gaussian_control_canvas` improved aggregate/low-route behavior but did not solve the high-route/A08-tail bottleneck. The next run should return to the A37 anchor and focus high-route pressure without expanding this shared control path.
