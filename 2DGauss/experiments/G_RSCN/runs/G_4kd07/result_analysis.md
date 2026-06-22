# G_4kd07 Result Analysis

- Created: `2026-06-21 22:20:07 CST`
- Updated: `2026-06-22 12:35:19 CST`
- Status: `not promoted; all updates rolled back`

## Smoke01 Result

`G_4kd07_smoke01` tested a G_4kd05-final continuation with high/A08 micro-lift and a five-metric full-validation rollback guard.

Observed guard behavior:

- guard checks: `6`;
- accepted updates: `0`;
- rolled back updates: `6`, steps `2, 4, 6, 8, 10, 12`;
- test evaluation: skipped because best/final/latest contain no accepted update signal.

Best metric-side movements:

- low-route best: `0.4335018157958984`, margin `+0.0006199518839518081`;
- A15-oracle best: `0.25836606459184125`, margin `+0.000344839963045962`;
- high-route best: `-0.013554467095269097`, margin `-0.00020684136284722203`;
- A08-oracle best: `-0.10048103332519531`, margin `-0.0000743865966796875`.

## Interpretation

G_4kd07 protected low/A15 but could not move high/A08 above the guard. Since every update was rolled back, extending the same recipe would only repeat the rejected trajectory. The next useful probe needed either an A37 restart or a different validation-margin design, not a longer continuation from G_4kd05 final.

## Decision

`G_4kd07_smoke01` is a clean negative. It is not promotable and must not be extended into formal long training.
