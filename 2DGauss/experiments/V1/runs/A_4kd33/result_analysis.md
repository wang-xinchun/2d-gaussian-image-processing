# A_4kd33 Result Analysis

Updated: 2026-06-14 19:29:38 CST

## Purpose

A33 is a weight-level synthesis designed after A31 and A32 isolated complementary effects:

- A31 improved high-route gap but damaged the low route.
- A32 preserved the low route but lost most high-route improvement.

A33 starts from the A32 route-decoder-head structure, copies A31 `input_std_route_decoder_heads.shared`, `high_feature`, and `high_head`, restores A24 `decoder.2` into the low route head, and zeros the low route feature delta output. This keeps a single checkpoint and fixed input_std prior while combining the useful parts of A31 and A32.

## Changed Paths

- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd33/configs/a_4kd33_fast_4kdehaze_a31_high_a24_low_surgery.toml`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd33/checkpoints/a31_high_a24_low_surgery.pt`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd33/metrics/checkpoint_surgery.json`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd33/manifest.md`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd33/gate_check.md`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd33/result_analysis.md`

No main training or model code was modified for A33.

## Surgery Verification

Checkpoint surgery file: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd33/metrics/checkpoint_surgery.json`

- Copied A31 high/shared route key count: `22`
- High head diff vs A31: weight `0.0`, bias `0.0`
- Low head diff vs A24 `decoder.2`: weight `0.0`, bias `0.0`
- Zeroed keys:
  - `input_std_route_decoder_heads.low_feature.1.weight`
  - `input_std_route_decoder_heads.low_feature.1.bias`
- Forward smoke passed with required aux keys and `input_std_route_low_delta.abs().mean()=0.0`.

## Standard Test

Standard eval: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd33/metrics/test_eval_surgery_gpu.json`

- `psnr_mean=17.823062311698262`
- `loss_mean=0.12025987021625043`
- `total_loss_mean=0.12349095992743969`
- checkpoint step: `50`

A33 exceeds A24 single-checkpoint PSNR `17.790202882742463` by `+0.03285942895579969 dB` and clears the target `17.82`. It remains below A18 router `17.84693437641316` by `-0.0238720647148984 dB`.

## Task Metrics

Task metrics: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd33/metrics/test_task_metrics_surgery_gpu.json`

- `ssim_mean=0.7052153688669205`
- `loss_mean=0.12025987021625043`
- `dark_channel_l1_mean=0.1449299818277359`
- `edge_magnitude_l1_mean=0.15836596436798572`
- `local_contrast_l1_mean=0.02476189107634127`
- `chroma_l1_mean=0.02959902461618185`

The SSIM/loss gates pass. Dark-channel and edge proxies are slightly higher than A18/A24-like preferred values, but not enough to invalidate the PSNR/SSIM breakthrough.

## A18 Alignment

Alignment file: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd33/metrics/a18_record_alignment_surgery_gpu.json`

- Overall gap vs A18 router: `-0.023872064116158373 dB`
- Low-route gap vs A18 router: `+0.053754617905617835 dB`
- High-route gap vs A18 router: `-0.12677254958688497 dB`
- A08-oracle group gap vs router: `-0.08633674234915452 dB`
- A15-oracle group gap vs router: `+0.029338587711949447 dB`

A33 preserves the low-route floor and substantially improves A32 high-route gap (`-0.211220889864822` to `-0.12677254958688497`). It does not yet meet the high-route target `>-0.10 dB`.

## Route-Regime Diagnostics

Route-regime file: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd33/metrics/input_std_route_regime_surgery_gpu.json`

- Overall selected delta abs mean: `0.008283325210213662`
- High selected delta abs mean: `0.019263547000496886`
- Low selected delta abs mean: `0.0`
- High route PSNR: `19.42790980135041`
- Low route PSNR: `16.612387889679976`

## Conclusion

A33 is the first post-A24 single-checkpoint candidate to clear the `17.82` target and is the current best single-checkpoint fast-probe checkpoint. It is a mechanism breakthrough, not yet a full A18 replacement: the high-route gap is improved but still `-0.12677254958688497 dB`, short of the `>-0.10 dB` target and still below the A18 router by `0.0238720647 dB`.

The scientific conclusion is strong: the high-route benefit resides in learned high-route shared/high-feature/head weights, while the low-route must remain A24/base-like. The next step should formalize this surgery into a trainable A34 mechanism: initialize high route from A31/A08-like weights, lock or softly regularize low route to A24, and train only high/shared modules with a stricter high-route early-stop or distillation budget.
