# A_4kd33 Manifest

Updated: 2026-06-14 19:29:38 CST

## Run

- ID: `V1/T01/A_4kd33`
- Purpose: checkpoint surgery that combines A31's useful high-route learned modules with A32's low-route protection.
- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd33/configs/a_4kd33_fast_4kdehaze_a31_high_a24_low_surgery.toml`
- Surgery checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd33/checkpoints/a31_high_a24_low_surgery.pt`

## Mechanism

A31 improved high-route gap but damaged low route. A32 preserved low route but lost most high-route improvement. A33 is an eval-only weight surgery:

- Start from A32 best-val checkpoint structure.
- Copy A31 best-val `input_std_route_decoder_heads.shared`, `high_feature`, and `high_head`.
- Restore A24 `decoder.2` into `input_std_route_decoder_heads.low_head`.
- Zero the low route feature delta output so `low_delta_abs_mean` starts at zero.
- Keep the sharper A32/A33 fixed input_std prior (`softness=0.003`).

## Gate Targets

- Beat A24 single-checkpoint fast-test PSNR: `17.790202882742463`.
- Preserve low-route gap at least `+0.03 dB`.
- Improve high-route gap beyond A32 `-0.211220889864822 dB`, ideally toward A31 `-0.14121316850824578 dB` or better.

## Result

A33 is promoted as the current best single-checkpoint fast-probe candidate:

- Standard test PSNR: `17.823062311698262`
- SSIM: `0.7052153688669205`
- Loss: `0.12025987021625043`
- A18 overall gap: `-0.023872064116158373 dB`
- Low-route gap: `+0.053754617905617835 dB`
- High-route gap: `-0.12677254958688497 dB`
