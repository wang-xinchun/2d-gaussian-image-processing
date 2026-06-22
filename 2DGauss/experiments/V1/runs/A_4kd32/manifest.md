# A_4kd32 Manifest

Updated: 2026-06-14 19:02:00 CST

## Run

- ID: `V1/T01/A_4kd32`
- Purpose: high-route-only A08 decoder-head transplant while preserving the A24 low route.
- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd32/configs/a_4kd32_fast_4kdehaze_high_a08_transplant_low_a24_locked.toml`
- Initial checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/checkpoints/best_val.pt`
- High route head source: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`
- Low route head source: A24 base decoder head copied during init and then frozen.

## Mechanism

A31 proved that A08 terminal-head transplant can move the high-route bottleneck, but the A15 low-head transplant damaged the low route under frozen A24 features. A32 keeps the useful high-route transplant, removes the A15 low-head transplant, freezes low route parameters, sharpens the fixed input_std prior, and trains only:

- `input_std_route_decoder_heads.shared`
- `input_std_route_decoder_heads.high_feature`
- `input_std_route_decoder_heads.high_head`

## Code And Launchers

- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd32/configs/a_4kd32_fast_4kdehaze_high_a08_transplant_low_a24_locked.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd32/launch_tmux.sh`
- Wait launcher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd32/wait_and_launch_a_4kd32_memory_gated_gpu_safe.sh`

## Gate Targets

- Beat A24 single-checkpoint fast-test PSNR: `17.790202882742463`.
- First breakthrough target: `psnr_mean >= 17.82`.
- Milestone target: approach A18 router `17.84693437641316`.
- High-route gap target: improve A31 `-0.14121316850824578 dB` toward `>-0.10 dB`.
- Low-route floor: restore low-route gap to at least `+0.03 dB`.
