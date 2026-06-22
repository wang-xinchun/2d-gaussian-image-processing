# V1/T01/A_4kd34

## Purpose

A34 deploys the A33 breakthrough as a formal trainable run. It starts from `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd33/checkpoints/a31_high_a24_low_surgery.pt`, preserves the low route by freezing all low-route decoder-head parameters, and only allows the shared/high route decoder-head path to move.

## Mechanism

- Keep fixed `input_std` routing with threshold `0.07026463374495506`, direction `gt`, and softness `0.003`.
- Keep `use_input_std_route_decoder_heads=true`.
- Disable init-time route-head reinitialization from the base decoder, so the A33 surgery weights are not overwritten.
- Trainable prefixes:
  - `input_std_route_decoder_heads.shared`
  - `input_std_route_decoder_heads.high_feature`
  - `input_std_route_decoder_heads.high_head`
- Keep low route locked with low branch loss `0.0` and no trainable low prefixes.
- Use gentle high-route refinement: lr `1e-5`, high branch weight `0.02`, delta trust weight `0.10`, delta budget `0.020`, max steps `60`, validation every `10` steps.

## Acceptance Focus

- Preserve A33/A24 low-route protection.
- Try to reduce the A18 high-route gap from A33 `-0.12677254958688497 dB` toward `>-0.10 dB`.
- Do not promote unless A34 beats A33 `17.823062311698262` or materially improves high-route gap without low-route collapse.

## Result

A34 completed GPU3 preflight, GPU3 formal training, standard fast-test, task metrics, A18 alignment, and route-regime diagnostics.

- Best-val step: `10`
- Fast-test PSNR: `17.820851508384713`
- SSIM: `0.7054232949018479`
- Loss: `0.1203028042241931`
- A18 overall gap: `-0.026082867429706925 dB`
- A18 low-route gap: `+0.053032394140660195 dB`
- A18 high-route gap: `-0.1309565862555424 dB`
- Route-regime high selected delta: `0.020353155795398148`
- Route-regime low selected delta: `0.0`

A34 is not promoted because it is slightly below A33 PSNR `17.823062311698262` and slightly worsens the high-route gap. A33 remains the current best single-checkpoint fast-probe candidate.
