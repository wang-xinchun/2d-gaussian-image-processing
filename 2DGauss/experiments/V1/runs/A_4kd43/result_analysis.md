# A_4kd43 Result Analysis

时间: 2026-06-17 CST

## Status

Completed. GPU3 preflight and formal training both exited with code `0`.

Formal training log:
`/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd43/logs/train_20260617_002105.log`

Best/final checkpoint:
`/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd43/checkpoints/best_val.pt`

Best/final validation step: `140`, `val_psnr=18.617040557582847`.

## Hypothesis

A42 failed because its selected-output A08 anchor could fire even when A08 was not strictly better than A15. A43 tests whether stricter oracle gating plus an A37 trust anchor can keep the useful feature-expert capacity while preventing high-tail/A08-oracle collapse.

## Promotion Criteria

- high-route gap better than A37 `-0.11159004428502844 dB`;
- A08-oracle gap better than A42 `-0.31086879727087796 dB` and preferably better than A40/A41 near `-0.238 dB`;
- low-route gap remains positive;
- aggregate PSNR is interpreted only after high-tail metrics.

## Results

- Standard test:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd43/metrics/test_eval_best_val_gpu.json`
  gives `psnr_mean=17.895968476137174`, `samples=100`, checkpoint step `140`.
- A18 record alignment:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd43/metrics/a18_record_alignment_best_val_gpu.json`
  gives router-gap overall `+0.04903410032275456 dB`, high-route `-0.12314298527548243 dB`,
  low-route `+0.17892207717756492 dB`, A08-oracle `-0.24640859619400587 dB`,
  and A15-oracle `+0.3007075084666616 dB`.
- Oracle-gap diagnostic in the same file gives overall `+0.012412681050638242 dB`,
  high-route `-0.1326023781356385 dB`, low-route `+0.1218100064016891 dB`,
  A08-oracle `-0.317177901285852 dB`, and A15-oracle `+0.29317502896690767 dB`.
- Route-regime diagnostics:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd43/metrics/input_std_route_regime_best_val_gpu.json`
  gives overall selected delta `0.005622951323166489`, high selected delta `0.013076630984108115`,
  low selected delta `0.0`, high branch L1 vs A08 `0.013162859867131987`,
  low branch L1 vs A15 `0.014308017907304722`, and overall gate mean `0.42893063667252734`.

## Comparison

- A43 aggregate PSNR is above A37 (`+0.00631002269835796 dB`) but below A42 (`-0.002796220643883884 dB`)
  and below A40 aggregate best (`-0.004594475819271055 dB`).
- A43 high-route router gap `-0.12314298527548243 dB` is worse than A37 `-0.11159004428502844 dB`
  and worse than A42 `-0.11934718931008992 dB`.
- A43 A08-oracle router gap `-0.24640859619400587 dB` is slightly worse than A42
  `-0.24009949217903187 dB` and still far worse than A37 `-0.19984098726138502 dB`.
- Worst router-gap records remain the same nominal-low/A08-oracle boundary cases:
  `94_9400006.jpg`, `94_94000065.jpg`, `94_94000081.jpg`.

## Interpretation

A43 is an important negative control. The strict-positive A08 anchor and A37 trust anchor are mechanically correct:
training logs show A08 pressure is zero when A08 is worse than A15, and the trust anchor only activates when A37 is
closer to GT than the current student. Validation PSNR rises smoothly to `18.617040557582847`, and high branch L1 remains
near the A42 improved range.

The scientific bottleneck does not move because A43 still inherits the fixed `input_std` route boundary. The worst cases
are not merely high-`input_std` samples with weak A08 branch fitting; many are below the threshold and are routed to A15
even though A08 is the oracle winner. Since low-route selected delta remains exactly `0.0`, A43 has no mechanism to rescue
those nominal-low/A08-oracle samples.

## Decision

A43 is not promoted. It disproves the hypothesis that strict selected-output anchoring plus an A37 trust anchor is
sufficient to recover the high-tail. The next experiment should attack the route boundary itself: learn a small, prior-
initialized gate residual under A08-vs-A15 oracle supervision, especially for `input_std < threshold` boundary samples
where A08 is closer to GT than A15.
