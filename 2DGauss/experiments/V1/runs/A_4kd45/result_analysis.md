# A_4kd45 Result Analysis

时间: 2026-06-17 CST

## Status

Completed. Negative as a breakthrough, but it resolves the A44 parameterization question.

## Hypothesis

A44 failed because residual-logit route gating was saturated near `prior = 0`. A45 tested whether a zero-initialized
bounded residual gate could actually move nominal-low/A08-oracle samples toward the high/A08 route while preserving low
route samples.

## Training Result

- Formal training log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd45/logs/train_20260617_013440.log`.
- Exit code: `0`.
- Best checkpoint: step `70`.
- Best validation PSNR: `18.563397419824184`.
- Final step: `120`.
- Final validation PSNR: `18.56339710733686`.

The bounded gate is trainable and stable. Validation recovered after an initial dip, but never exceeded A44/A37.

## Test Metrics

- Standard test PSNR: `17.88963097090341`.
- A44 standard test PSNR: `17.889657071833376`.
- A37 standard test PSNR: `17.889658453438816`.

A45 is slightly below A44/A37 and is not an aggregate improvement.

## A18 Alignment

- Overall router gap: `+0.042696595088990194 dB`.
- High-route router gap: `-0.111655554804118 dB`.
- Low-route router gap: `+0.1591376906223876 dB`.
- A08-oracle router gap: `-0.19997947735713625 dB`.
- A15-oracle router gap: `+0.24942065680235717 dB`.

A45 gives a tiny low/A15 improvement but worsens the target A08/high side. It therefore fails the high-tail promotion
criteria.

## Route-Regime Diagnostics

- Overall gate mean: `0.4284928046321966`.
- Overall prior mean: `0.42893063667252734`.
- High-route gate mean: `0.9652301585951517`.
- High-route prior mean: `0.966229241947795`.
- Low-route gate mean: `0.023585677958388323`.
- Low-route prior mean: `0.023600109885921854`.
- Overall selected delta: `0.0062688171397894624`.
- Low-route selected delta: `0.0`.

Unlike A44, the gate did move. But the learned movement is mostly a conservative downward adjustment from the fixed
prior, not the desired upward correction for nominal-low/A08-oracle boundary samples.

## Failure Mechanism

A45 solves A44's immobility problem, but exposes the next bottleneck: the training distribution and focus rule rarely
present positive low-route A08-oracle samples before checkpoint selection.

Training rows show this clearly:

- Step 1: low-route negative sample, target `0.0`, weight `1.0`, gate `1e-5`.
- Step 30/40/60/70/80/90/100/120: mostly low-route negative samples with target `0.0`, weight `1.0`.
- Step 20/50: A08-positive/high-prior samples had oracle weight `0.0` because low-route focus excluded them.
- Step 110: first logged low-route positive sample, target `0.8`, weight `1.0`, gate `9.9868e-6`; but best checkpoint was already step `70`.

So the new gate can learn, but the objective mostly taught it to keep or lower the low-route gate. The intended hard
case, fixed-low but A08-better, was too sparse under the current sampler and checkpoint selection.

## Conclusion

A45 rejects the current low-focus online oracle-gate training recipe, not the bounded residual parameterization. The
next serious run should stop relying on random online exposure to rare A08-positive low-route cases. It should either:

- build an offline hard-case cache of A08-positive low-route samples and force them into each epoch/batch; or
- add an oracle-positive sampler keyed by A08-vs-A15 teacher advantage, not only by `input_std`.

The gate architecture is now adequate enough for this test; the bottleneck has moved to hard-case sampling and target
selection.
