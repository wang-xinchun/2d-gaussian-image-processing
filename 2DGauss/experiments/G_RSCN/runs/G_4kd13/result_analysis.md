# G_4kd13 Result Analysis

- Created: `2026-06-22 21:15:00 CST`
- Status: `smoke01 complete; not promoted`

## Smoke01 Question

`G_4kd13_smoke01` tests the user's diagnosis that the current G-RSCN bottleneck is not yet truly decoupled.

The specific hypothesis is:

```text
shared gaussian_route_control prior feature
=> high/A08 and low/A15 gradients still interfere
```

G13 changes this to:

```text
separate high/low Gaussian-conditioned prior encoders
=> high-preserve and low-dehaze deltas get separate route-control capacity
```

## Expected Useful Outcomes

- A full transfer-gate candidate appears where G12 had only protect-only movement.
- Projection accepts a candidate segment that satisfies both high/A08 target and low/A15 protection.
- Final or best-val passes A37-aligned high-route and A08-oracle without spending low-route and A15-oracle.

## Negative Outcomes

- All accepted movement remains protect-only.
- High/A08 improves but low/A15 regresses.
- Low/A15 improves but high/A08 regresses.
- No update passes transfer-aware target/protect gates despite the split prior capacity.

## Promotion Rule

Do not promote from transfer-validation behavior alone. Run A37-aligned GPU evaluation when a checkpoint contains a real new weight signal. Promotion requires high-route, A08-oracle, low-route, and A15-oracle at or above A37 on test-side evaluation.

## Smoke01 Training Result

`G_4kd13_smoke01` completed on physical GPU3 with train exit code `0`.

Static and preflight checks passed:

- Python compile for changed model/config and evaluation scripts;
- TOML parse and bash syntax checks;
- launcher dry-run;
- GPU3-bound preflight with `TWODGAUSS_CUDA_MEMORY_FRACTION=0.35`;
- A37 initialization with only expected `gaussian_route_control.*` missing keys and no unexpected keys;
- trainable prefix only `gaussian_route_control`, trainable params `50114`;
- A18 `val_records` alignment `100/100`.

Training summary:

```text
final step: 20
guard checks: 10
accepted updates: 0
rolled back updates: 10
full-gate passes: 0
protect-gate passes: 0
target-gate passes: 0
projection-accepted updates: 0
best_val step: 0
final gaussian_route_control_high_delta_abs_mean: 0.0
final gaussian_route_control_low_delta_abs_mean: 0.0
```

Last guard check at step `20`:

```text
baseline high-route transfer gap: 0.03710367872908309
step20   high-route transfer gap: 0.037082491694269956

baseline A08-oracle transfer gap: 0.07303136937758502
step20   A08-oracle transfer gap: 0.07301728865679573

baseline low-route transfer gap: 0.417457762218657
step20   low-route transfer gap: 0.4173693127102322

baseline A15-oracle transfer gap: 0.3613124905210553
step20   A15-oracle transfer gap: 0.36122343756935815

baseline low-A15 transfer gap: 0.4313206589012815
step20   low-A15 transfer gap: 0.43123176641631544
```

Every checked candidate was below the transfer baseline on all five guard metrics. Because each candidate was rolled back, the final checkpoint remained A37-equivalent with zero visible GRC delta. A37-aligned GPU evaluation was intentionally skipped: `best_val` is the A37 parent and `final` has no real new adapter signal.

## Conclusion

G13 proves the first structural split was not enough by itself. The dual high/low prior-control bank is present, but zero-init outputs plus strict every-2-step rollback create a second bottleneck: the adapter cannot accumulate a measurable direction before it is reset.

The useful next probe is not another scalar threshold tweak. It should keep the dual adapter but decouple the learning window from the final transfer gate, so we can see whether the split capacity can form a real direction before judging it. This is the purpose of `G_4kd14_smoke01`.
