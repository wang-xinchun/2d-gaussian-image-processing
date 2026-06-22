# G_4kd09 Result Analysis

- Created: `2026-06-22 00:07:34 CST`
- Updated: `2026-06-22 12:35:19 CST`
- Status: `not promoted; positive-margin guard rejected every update`

## Smoke01 Result

`G_4kd09_smoke01` restarted from A37 and added positive validation margins after G08 showed validation non-regression was not enough for test transfer.

Guard behavior:

- guard checks: `8`;
- accepted updates: `0`;
- rolled back updates: `8`, steps `2, 4, 6, 8, 10, 12, 14, 16`;
- test evaluation: skipped because best/final/latest contain no accepted update signal.

Positive-margin thresholds:

- high-route `-0.013097625732421875`;
- A08-oracle `-0.10020664672851562`;
- low-route `0.4329318639119466`;
- A15-oracle `0.2581212246287953`;
- low-A15 `0.4329318639119466`.

Closest useful candidate:

- step `4`;
- high-route margin `+0.00017597452799479087`;
- A08-oracle margin `-0.00002738494873047448`;
- A15-oracle margin `+0.00020400536277076498`;
- low/low-A15 also had margin-positive candidates in the same short window.

## Interpretation

G_4kd09 narrowed the immediate bottleneck: strict validation margins can protect low/A15, but the A08 margin was just too tight for the available update. This justified one more narrow probe that relaxed only A08 while preserving the high/low/A15 margins.

## Decision

`G_4kd09_smoke01` is a clean guard negative. It is not promotable and must not be extended into formal long training.
