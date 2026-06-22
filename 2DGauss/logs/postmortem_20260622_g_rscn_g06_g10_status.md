# G-RSCN G06-G10 Postmortem And Current Status

- Created: `2026-06-22 12:46:39 CST`
- Scope:案后梳理 `G_4kd06_smoke01` through `G_4kd10_smoke01`
- Current frontier: `G_4kd10_smoke01`
- Next intended version: `G_4kd11`

## Project Identity

The project is building a Gaussian-preserved route-specific control architecture for UHD image restoration. The current new-architecture line is `G-RSCN`, stored under:

```text
/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kdXX
```

This is intentionally separate from the old A-series route-patch line:

```text
/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kdXX
```

The Gaussian core remains immutable:

- `GaussianParameterHead`;
- `TextureAwareGaussianRenderer`;
- Gaussian attention/control decoder, refinement, and control paths;
- cross-resolution Gaussian restoration identity.

New mechanisms must be Gaussian-conditioned route-specific control. `input_std` is a prior/sampler signal only, not the architecture identity.

## Promotion Gate

Aggregate PSNR alone cannot promote a G-RSCN run.

Promotion requires A37-aligned test-side non-regression or improvement on all required axes:

- high-route at or above A37;
- A08-oracle at or above A37;
- low-route at or above A37;
- A15-oracle at or above A37.

Validation-only movement is diagnostic unless it transfers to the test-side A37-aligned gate.

## Latest Run: G_4kd10_smoke01

Artifacts:

- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd10/configs/g_4kd10_smoke01_a37_asymmetric_a08_margin_short.toml`
- Train log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd10/logs/smoke01_20260622_122138.log`
- Summary: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd10_smoke01/metrics/g_4kd10_smoke01_guard_summary.json`

Deployment:

- physical GPU3 only;
- `CUDA_VISIBLE_DEVICES=3`;
- `TWODGAUSS_CUDA_MEMORY_FRACTION=0.35`;
- temporary util exception used per user instruction;
- no CPU fallback;
- train exit code `0`;
- final step `18`;
- peak CUDA reserved `3.095703125GiB`.

Result:

- guard checks: `9`;
- accepted updates: `0`;
- rolled back updates: `9`;
- A37-aligned test eval skipped because no accepted update created a new weight signal.

Important candidates:

- Step `4` passed high/A08 after A08-margin relaxation:
  - high-route margin `+0.00030991278754340275`;
  - A08-oracle margin `+0.00005888366699219327`;
  - low-route margin `-0.000497400410970017`;
  - A15-oracle margin `-0.00019558417580345377`;
  - low-A15 margin `-0.000497400410970017`.
- Step `18` was closest overall:
  - high-route margin `-0.00010419379340277819`;
  - A08-oracle margin `-0.00010037994384765048`;
  - low-route margin `-0.00016272430419922435`;
  - A15-oracle margin `-0.00012232464877043325`;
  - low-A15 margin `-0.00016272430419922435`.

Verdict: `G_4kd10_smoke01` is a clean negative. It must not be extended into formal long training.

## What G06-G10 Proved

`G_4kd06_smoke01`:

- Accepted late updates from the G_4kd05-final parent.
- A37-aligned test still failed high-route and A08-oracle.
- It preserved the near-miss shape but did not close the decisive target gap.

`G_4kd07_smoke01`:

- Continued from G_4kd05 final with five validation gates.
- Rolled back all `6/6` updates.
- Showed that the G_4kd05-final continuation route cannot pass the stricter guard.

`G_4kd08_smoke01`:

- Restarted from A37 and accepted `2/6` updates.
- Validation five-gate pass did not transfer to A37-aligned test.
- Best checkpoint still failed PSNR, high-route, and A08-oracle vs A37.

`G_4kd09_smoke01`:

- Added positive validation margins.
- Rolled back all `8/8` updates.
- Closest candidate missed only the A08 margin by `2.738494873047448e-05`.
- This made A08 look like the narrowest immediate blocker.

`G_4kd10_smoke01`:

- Relaxed only the A08 margin.
- Rolled back all `9/9` updates.
- Proved A08 was not the only blocker: early high/A08-positive updates spend low/A15, while late balanced updates are slightly below every margin.

## Current Bottleneck

The bottleneck is no longer:

- GPU availability;
- basic G-RSCN mechanism wiring;
- `gaussian_route_control` trainability;
- A08 teacher plumbing;
- A15 teacher plumbing;
- a single A08 validation threshold;
- simply needing more steps.

The bottleneck is update selection/application under competing route constraints. The model can move toward high/A08, and it can remain balanced, but current updates do not produce a simultaneously safe high/A08 lift while preserving low/A15 surplus.

## Next Probe: G_4kd11

`G_4kd11` should be a new G-line version, not a G10 rerun.

Required invariants:

- initialize from A37 best-val;
- train only `gaussian_route_control`;
- keep five positive validation margins;
- keep full 24-batch `step_no_harm_guard`;
- keep GPU0/3-only single-GPU policy;
- keep `TWODGAUSS_CUDA_MEMORY_FRACTION=0.35`;
- no DDP, no `CUDA_VISIBLE_DEVICES=0,3`, no CPU fallback.

Recommended mechanism:

- two-phase or projected update selection;
- first require low-route/A15/low-A15 surplus above margin;
- then permit high/A08 lift only when the actual or projected update keeps low/A15 above guard;
- reject high/A08-positive movement that spends low/A15, even if A08 passes.

Deployment gate:

- write `manifest.md`, `gate_check.md`, `result_analysis.md`;
- add clear G11 config/script names separate from old A-line and prior G-line versions;
- preflight before training;
- only run A37-aligned GPU evaluation if at least one update is accepted or a checkpoint contains a real new weight signal.

## Current Operational State

- No tmux training session is active after G10.
- GPU policy remains: physical GPUs `0` and `3` only, with short-run util exception allowed only if memory is safe and no one is evicted.
- Formal long training remains blocked.
- The handoff entry is `/home/ldj/wxc/2DGauss/logs/current_state.md`.
