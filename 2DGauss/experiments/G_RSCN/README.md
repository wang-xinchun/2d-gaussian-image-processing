# G-RSCN Experiment Line

- Created: `2026-06-21 16:06:13 CST`
- Scope: new architecture line after the A-series route-patch stop-loss.
- First run: `G_4kd01`.
- Current diagnostic frontier: `G_4kd12_smoke01` completed; next planned version is structural `G_4kd13` or `G_4kd12b`.

## Naming Rule

The new architecture line is separated from the old V1/T01 A-line at both directory and run-ID levels.

- Old architecture / patch line: `experiments/V1/runs/A_4kdXX`.
- New Gaussian-preserved route-specific control line: `experiments/G_RSCN/runs/G_4kdXX`.
- First run: `G_4kd01`, also referred to in design notes as `G/B_4kd01` when connecting to the earlier B-series discussion.

Do not reuse `A_4kdXX` identifiers for G-RSCN training. Any A48-style run may only be a bounded falsification probe, not the mainline.

## Architecture Boundary

G-RSCN keeps the Gaussian core immutable:

- `GaussianParameterHead`;
- `TextureAwareGaussianRenderer`;
- cross-resolution Gaussian attention/control path;
- Gaussian-guided decoder/refinement/control injection before RGB output;
- A08/A15 conditional evidence and high-tail-first evaluation.

Allowed new mechanisms must consume, modulate, or protect Gaussian priors. Generic non-Gaussian adapters, post-head-only fixes, gate-only fixes, bigger shared-backbone-only changes, or non-Gaussian replacements are not valid G-RSCN mainline mechanisms.

## Deployment Rule

Every G-RSCN run must ship with:

- `manifest.md`;
- `gate_check.md`;
- `result_analysis.md`;
- a config file under `configs/`;
- a launch script that refuses formal training until mechanism and GPU gates are explicitly passed;
- a wait-and-launch script that checks the approved physical GPU policy.

No G-RSCN formal training may start until the run-local `gate_check.md` says the mechanism gate and GPU gate are both passed.

## Current Status

As of `2026-06-22 17:33:53 CST`, `G_4kd12_smoke01` is the latest deployed short run. It completed on physical GPU3 with train exit code `0`; A37-aligned GPU evaluation for `best_val` and `final` also completed with exit code `0`.

G12 kept the G11 model structure and added a transfer-aware validation guard based on A18 `val_records=100`. Training produced `8` guard checks: steps `2/4` rolled back, steps `6/8/10/12/14/16` were accepted as `protect_only`, and no update passed the full transfer gate. Projection acceptance remained `0`. `best_val` stayed at step `0`, exactly the A37 parent. The trained `final` checkpoint improved aggregate PSNR by `+0.000042957497897`, low-route by `+0.000115119498806`, and A15-oracle by `+0.000274946950254`, but regressed high-route by `-0.000052699107953` and A08-oracle by `-0.000229377946168`.

The current bottleneck has shifted from validation-to-test transfer detection to route-control capacity. The transfer-aware guard works and prevents unsafe best-val promotion, but same-structure shared `gaussian_route_control` can only find protect-only movement; it cannot produce a full high/A08 plus low/A15 checkpoint. The next G-RSCN version should split Gaussian-conditioned high-preserve and low-dehaze route-control capacity rather than lengthening G12 or tuning scalar thresholds.

Current handoff: `/home/ldj/wxc/2DGauss/logs/current_state.md`.
