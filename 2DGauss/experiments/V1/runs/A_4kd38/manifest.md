# A_4kd38 Manifest

## Purpose

A38 is a short, low-memory probe from A37 best-val. It tests whether reintroducing standard reconstruction/proxy loss while training only the high input-std decoder-head path can close the remaining high-route gap without disturbing the low route.

## Primary Paths

- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd38/configs/a_4kd38_fast_4kdehaze_a37_standard_loss_high_only_short.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd38/launch_tmux.sh`
- Wait launcher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd38/wait_and_launch_a_4kd38_memory_gated_gpu_safe.sh`
- Initial checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`
- Output root: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd38`

## Training Boundary

- No route teachers are loaded during training because all route distillation and branch weights are zero.
- Trainable prefixes are restricted to `input_std_route_decoder_heads.high_feature` and `input_std_route_decoder_heads.high_head`.
- The low route, shared route encoder, Gaussian/control modules, and base decoder remain frozen.
- Natural sampling is preserved; balanced route sampling remains disabled.

## Acceptance Lens

- Preserve A37 aggregate line: PSNR should stay near or above `17.88`.
- Target the remaining A37 high-route gap `-0.11159004428502844 dB`; a useful probe should move it toward `>-0.10 dB`.
- Preserve A37 low-route surplus; low-route gap should remain comfortably positive.
