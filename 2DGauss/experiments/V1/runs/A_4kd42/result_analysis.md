# A_4kd42 Result Analysis

时间: 2026-06-16 CST

## Status

Completed. GPU3 preflight and formal training both exited with code `0`.

Formal training log:
`/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd42/logs/train_20260616_233200.log`

Best/final checkpoint:
`/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd42/checkpoints/best_val.pt`

Best/final validation step: `140`, `val_psnr=18.61766014960455`.

## Hypothesis

A41 failed because it only reweighted high-branch teacher loss and still missed nominal-low A08-oracle hard cases. A42 should be a stronger probe because it:

- adds earlier route-conditioned feature capacity;
- uses an A08-oracle anchor on the selected output, not only branch output;
- keeps nominal low-route crops in training through probabilistic sampling.

## Promotion Criteria

- high-route gap better than A37 `-0.11159004428502844 dB`;
- A08-oracle gap better than A37 `-0.19984098726138502 dB`, or at minimum not near A40/A41 `~ -0.238 dB`;
- low-route gap remains positive;
- aggregate PSNR is secondary unless high-tail metrics also improve.

## Results

- Standard test:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd42/metrics/test_eval_best_val_gpu.json`
  gives `psnr_mean=17.898764696781058`, `samples=100`.
- A18 record alignment:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd42/metrics/a18_record_alignment_best_val_gpu.json`
  gives overall router gap `+0.05183032096663528 dB`, high-route gap `-0.11934718931008992 dB`,
  low-route gap `+0.18096423222802446 dB`, A08-oracle gap `-0.31086879727087796 dB`,
  and A15-oracle gap `+0.29297879340578303 dB`.
- Route-regime diagnostics:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd42/metrics/input_std_route_regime_best_val_gpu.json`
  gives selected delta `0.0056412214320152995`, high selected delta `0.013119119609337907`,
  low selected delta `0.0`, and high branch L1 vs A08 `0.01301031275977229`.

## Comparison

- A42 beats A37 aggregate by `+0.00910624334224182 dB`, but remains below A40 by `-0.001798255175386367 dB`.
- A42 high-route gap `-0.11934718931008992 dB` is worse than A37 `-0.11159004428502844 dB`,
  A40 `-0.1180115677202543 dB`, and A41 `-0.11813195131188482 dB`.
- A42 A08-oracle gap `-0.31086879727087796 dB` is the clearest regression:
  worse than A37 `-0.19984098726138502 dB`, A40 `-0.23833660113789967 dB`, and A41 `-0.23811165869770295 dB`.
- The worst router-gap records are still nominal-low A08-oracle hard cases:
  `94_9400006.jpg`, `94_94000065.jpg`, `94_94000081.jpg`.

## Interpretation

A42 is an important negative result. The new feature expert and selected-output A08-oracle anchor are mechanically active,
and high branch L1 vs A08 improves strongly versus A37/A40/A41, but that lower teacher L1 does not translate into
high-route or A08-oracle PSNR.

The likely failure mode is the selected-output A08 anchor trigger. With `margin=0.0`, `temperature=0.006`, and
`probability_floor=0.35`, the sigmoid can still assign non-zero anchor probability when A08 is slightly worse than A15.
This pulls nominal-low/A15-favored samples toward A08 and damages the exact A08-oracle tail that A42 was meant to protect.

## Decision

A42 is not promoted. It confirms that earlier route-conditioned feature capacity alone is not enough, and that oracle
anchors must be gated by strict positive teacher advantage or protected by a previous-best trust anchor.

The next run should keep the useful A42 infrastructure but change the objective:

- require strictly positive A08-vs-A15 advantage before applying the A08 selected-output anchor;
- add an A37 trust anchor so the model is penalized only where the A37 checkpoint is closer to GT than the current
  prediction;
- avoid treating high branch L1 improvement as a promotion criterion unless A18 high-route/A08-oracle gaps improve.
