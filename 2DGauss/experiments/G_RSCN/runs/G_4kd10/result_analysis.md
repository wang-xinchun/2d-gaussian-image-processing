# G_4kd10 Result Analysis

- Created: `2026-06-22 00:15:44 CST`
- Updated: `2026-06-22 12:35:19 CST`
- Status: `not promoted; all candidate updates rolled back`

## Smoke01 Result

`G_4kd10_smoke01` tested whether the G09 near-pass could be accepted by relaxing only the A08 validation margin while keeping high/low/A15 margins strict.

Deployment:

- preflight: PASS on physical GPU3;
- training: exit code `0`;
- final step: `18`;
- trainable prefix: `gaussian_route_control` only;
- trainable parameters: `37698`;
- peak CUDA reserved: `3.095703125GiB`;
- log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd10/logs/smoke01_20260622_122138.log`;
- summary: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd10_smoke01/metrics/g_4kd10_smoke01_guard_summary.json`.

Guard behavior:

- guard checks: `9`;
- accepted updates: `0`;
- rolled back updates: `9`, steps `2, 4, 6, 8, 10, 12, 14, 16, 18`;
- A37-aligned GPU evaluation: skipped because no accepted update exists.

Best gate-count candidate:

- step `4`;
- high-route margin `+0.00030991278754340275`;
- A08-oracle margin `+0.00005888366699219327`;
- low-route margin `-0.000497400410970017`;
- A15-oracle margin `-0.00019558417580345377`;
- low-A15 margin `-0.000497400410970017`.

Closest all-gate candidate:

- step `18`;
- high-route margin `-0.00010419379340277819`;
- A08-oracle margin `-0.00010037994384765048`;
- low-route margin `-0.00016272430419922435`;
- A15-oracle margin `-0.00012232464877043325`;
- low-A15 margin `-0.00016272430419922435`.

## Interpretation

G10 answered the G09 question: the A08 margin was one blocker, but relaxing it does not produce a simultaneous safe update. The update direction separates into two shapes:

- early candidates can pass high/A08 but spend low/A15;
- late candidates become balanced but remain slightly below every positive margin.

This means the next bottleneck is not a single scalar A08 threshold. The next probe must change how candidate updates are applied or selected, for example with a two-phase acceptance window or route-projected update that only keeps high/A08 lift when low/A15 surplus remains above margin.

## Decision

`G_4kd10_smoke01` is a clean negative and must not be extended into formal long training. The next deployable short probe should be a new G-line version, not a G10 rerun.
