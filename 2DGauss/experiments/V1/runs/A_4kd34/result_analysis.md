# A_4kd34 Result Analysis

更新时间: 2026-06-14 20:44:54 CST

## Summary

A34 successfully deployed the A33 surgery checkpoint as a formal trainable run. The run loaded `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd33/checkpoints/a31_high_a24_low_surgery.pt` with route decoder-head reinitialization disabled, froze all low-route parameters, and trained only `input_std_route_decoder_heads.shared/high_feature/high_head`.

It is not promoted. A34 keeps the A33/A32 low-route protection, but the best checkpoint does not exceed A33.

## Deployment

- Code change: `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`
- New run config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd34/configs/a_4kd34_fast_4kdehaze_a33_high_refine_low_locked.toml`
- GPU launcher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd34/launch_tmux.sh`
- Memory-gated launcher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd34/wait_and_launch_a_4kd34_memory_gated_gpu_safe.sh`
- Preflight log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd34/logs/train_20260614_202945.log`
- Formal training log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd34/logs/train_20260614_203136.log`

## Checks

- `py_compile` passed for `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`, `/home/ldj/wxc/2DGauss/src/uhd_gaussian/models/restorer.py`, and `/home/ldj/wxc/2DGauss/src/uhd_gaussian/config.py`.
- TOML parse and checkpoint path checks passed.
- Launcher `bash -n` passed.
- GPU3 dry-run gate passed with free memory `13041 MiB`, util `100`, memory fraction `0.35`.
- CPU smoke confirmed A33 route decoder-head weights were preserved exactly after init: shared/high/low key max diff `0.0`.
- CPU smoke confirmed `init_checkpoint_route_decoder_heads_reinitialized_from_base=false`, `low_delta_abs_mean=0.0`, low trainable params `0`, low grad abs sum `0.0`, high/shared trainable params `12899`.
- GPU3 preflight passed with exit code `0`, `val_psnr=18.31555500051223`, `val_route_low_delta_abs_mean=0.0`.

## Training

Formal GPU3 training completed with exit code `0`.

- Best-val checkpoint: step `10`
- `best_val_psnr=18.311850790485888`
- `best_val_loss=0.12204197365790606`
- Validation after step10 did not recover:
  - step20 `18.30490060018016`
  - step30 `18.304562493863173`
  - step40 `18.30540387123757`
  - step50 `18.305566591960357`
  - step60 `18.30550528666232`
- Validation low-route delta stayed `0.0` for every validation point.
- Validation high-route delta increased from step10 `0.025077012898400426` to about `0.026` later, while PSNR remained below step10.

## Test Metrics

Standard fast-test on `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd34/checkpoints/best_val.pt`:

- `psnr_mean=17.820851508384713`
- `loss_mean=0.1203028042241931`
- `total_loss_mean=0.12353389412164688`
- samples `100`

Task metrics:

- `ssim_mean=0.7054232949018479`
- `dark_channel_l1_mean=0.14506833117455245`
- `edge_magnitude_l1_mean=0.15835020028054714`
- `local_contrast_l1_mean=0.02475977601483464`
- `chroma_l1_mean=0.02959987794980407`

## A18 Alignment

Compared with A18 router records:

- Overall gap vs A18 router: `-0.026082867429706925 dB`
- Low-route gap vs A18 router: `+0.053032394140660195 dB`
- High-route gap vs A18 router: `-0.1309565862555424 dB`
- A08-oracle group gap vs router: `-0.08089136622024616 dB`
- A15-oracle group gap vs router: `+0.02060585376223391 dB`

## Route Diagnostics

Route-regime diagnostics:

- Overall selected delta abs mean: `0.008751856992021204`
- High-route selected delta abs mean: `0.020353155795398148`
- Low-route selected delta abs mean: `0.0`
- High-route branch L1 vs A08 teacher: `0.32904680800992386`
- Low-route branch L1 vs A15 teacher: `0.2585565087042357`

## Comparison

A34 remains above the old A24 single-checkpoint baseline but below A33:

- A24 fast-test PSNR: `17.790202882742463`
- A33 fast-test PSNR: `17.823062311698262`
- A34 fast-test PSNR: `17.820851508384713`
- A18 router PSNR: `17.84693437641316`

A34 vs A33:

- PSNR: `-0.002210803313549`
- Overall A18 gap: A33 `-0.023872064116158373`, A34 `-0.026082867429706925`
- Low-route gap: A33 `+0.053754617905617835`, A34 `+0.053032394140660195`
- High-route gap: A33 `-0.12677254958688497`, A34 `-0.1309565862555424`

## Decision

A34 is deployment-complete but not promoted. It validates the deployment mechanism and confirms that low-route locking is robust, but even gentle high/shared micro-tuning from A33 slightly worsens the aggregate PSNR and high-route gap.

The important bottleneck conclusion is that A33's high/shared route state is already close to a narrow optimum. Additional high-branch teacher pressure and small-LR micro-tuning do not repair the remaining high-route gap; they tend to move the A33 surgery solution away from the useful balance. The next serious route should not be more A33 micro-tuning. It should change the mechanism again: preserve A33 as the base, then add a separate high-route adapter with zero-impact initialization or per-regime selection, or move route specialization into an upstream high-route-only feature/control branch while keeping A33 output as a trust anchor.
