# A_4kd44 Result Analysis

时间: 2026-06-17 CST

## Status

Completed. Negative as a breakthrough, but useful as a mechanism diagnostic.

## Hypothesis

A43 failed because strict anchoring still left the fixed `input_std` route boundary unchanged. A44 tested whether a
prior-initialized learned gate, supervised by A08-vs-A15 oracle teacher error, could rescue nominal-low/A08-oracle
boundary samples without modifying the frozen A37 high/low route experts.

## Training Result

- Formal training log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd44/logs/train_20260617_010259.log`.
- Exit code: `0`.
- Best checkpoint: step `10`.
- Best validation PSNR: `18.563404644343226`.
- Final step: `120`.
- Final validation PSNR: `18.563403042004886`.

Validation peaked immediately and then moved down monotonically by a tiny amount. This means A44 did not discover a
useful learned route correction after initialization.

## Test Metrics

- Standard test PSNR: `17.889657071833376`.
- A37 standard test PSNR: `17.889658453438816`.
- A43 standard test PSNR: `17.895968476137174`.

A44 is effectively tied with A37 at aggregate level and below A43/A40. It is not a numeric improvement.

## A18 Alignment

- Overall router gap: `+0.04272269601895694 dB`.
- High-route router gap: `-0.11159031808549981 dB`.
- Low-route router gap: `+0.15913426806266992 dB`.
- A08-oracle router gap: `-0.19984133741155483 dB`.
- A15-oracle router gap: `+0.24935131708939287 dB`.

The high-route and A08-oracle gaps are essentially A37-level, not improved. The worst router-gap records remain led by
`94_9400006.jpg`, `95_95000033.jpg`, and `95_9500003.jpg`; the nominal-low/A08-oracle record `94_9400006.jpg` remains
bad at `-0.5319216747385873 dB`.

## Route-Regime Diagnostics

- Overall gate mean: `0.42893156833639295`.
- Overall prior mean: `0.42893063667252734`.
- Overall selected delta: `0.0062688171397894624`.
- High-route selected delta: `0.014578644511138284`.
- Low-route selected delta: `0.0`.
- High branch L1 vs A08 teacher: `0.014566886629909277`.

The learned gate changed the prior by only about `9.3e-7` on average. Route behavior therefore remained the fixed
`input_std` prior.

## Failure Mechanism

A44 exposed a parameterization failure, not a lack of oracle signal. On low-prior crops where the oracle target says A08
should win, the residual-logit gate remained saturated:

- Step 40: target `0.8`, prior `2.8012205888217068e-09`, gate `1.000038901111111e-05`, oracle gate loss `9.210311889648438`.
- Step 60: target `0.8`, prior `2.18894102843592e-09`, gate `1.000075826596003e-05`, oracle gate loss `9.210281372070312`.
- Step 120: target `0.8`, prior `5.255986616248265e-05`, gate `5.25665600434877e-05`, oracle gate loss `7.882754325866699`.

The residual-logit form starts from the fixed prior, but moving from a near-zero prior to a high gate requires a very
large logit residual. With only the tiny gate head trainable and LR `8e-6`, the model cannot move far enough during the
short probe. This also interacts with BCE clamping near `1e-5`, making the lowest-prior cases especially hard to update.

## Conclusion

A44 rejects the residual-logit gate parameterization for this bottleneck. It does not reject oracle-supervised route
learning itself. The next experiment should keep prior-initialized behavior but use a bounded additive residual gate
that has direct gradient and direct movement room when `prior` is near zero.

## Next

A45 should implement `input_std_route_learned_gate_mode = "bounded_residual"`:

- zero-initialized, so initial gate equals the clamped fixed prior;
- bounded inside `[0, 1]`;
- direct upward gradient for nominal-low/A08-oracle cases;
- stricter negative target ceiling so the gate does not lift true A15/low-route samples.
