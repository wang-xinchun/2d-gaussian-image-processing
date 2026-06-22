# G-RSCN Experiment Line

- Created: `2026-06-21 16:06:13 CST`
- Scope: new architecture line after the A-series route-patch stop-loss.
- First run: `G_4kd01`.
- Current diagnostic frontier: `G_4kd15_smoke01` completed; next planned version should change branch objective or acceptance geometry beyond warmup scheduling.

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

As of `2026-06-22 22:44:00 CST`, `G_4kd15_smoke01` is the latest deployed short run. `G_4kd13_smoke01`, `G_4kd14_smoke01`, and `G_4kd15_smoke01` all completed on physical GPU3 with exit code `0`; none is promoted.

G13 implemented the first structural split of `gaussian_route_control` into a shared gate/confidence feature plus separate high/low Gaussian-conditioned prior-control banks. It kept A37 initialization and the A18 transfer-aware step guard. Training ran `20` steps and produced `10/10` rollbacks, `0` accepted updates, and final GRC high/low deltas of `0.0/0.0`. This proved the dual adapter was wired, but zero-init plus strict step-level rollback prevented signal accumulation.

G14 kept the dual adapter but disabled step-level rollback, using a deferred transfer gate at step `20`. It produced non-zero final GRC deltas (`2.574286554590799e-05` high, `1.9299128325656056e-05` low), proving the adapter can learn when not reset from zero. A37-aligned test evaluation showed the direction is high/A08-positive but low/A15-negative: final high-route improved by `+0.000121429641323` and A08-oracle by `+0.000564839120677`, while PSNR regressed by `-0.000220353899213`, low-route by `-0.000478190605231`, and A15-oracle by `-0.000889222027268`.

G15 then added `training.step_no_harm_guard_start_step`, kept the dual adapter, warmed up until step `10`, and enabled A18 transfer-aware guard/projection every `2` steps. It completed with `6/6` delayed guard rollbacks, `0` accepted updates, and best-val step `0`. Final A37-aligned evaluation was close to A37 but below every promotion gate: PSNR `-0.000069120880880`, high-route `-0.000051010594676`, A08-oracle `-0.000092875257996`, low-route `-0.000082783026609`, and A15-oracle `-0.000048885670740` versus A37.

The current bottleneck is therefore not just structural capacity or warmup scheduling. The next G-RSCN version should keep dual high/low Gaussian route-control capacity but change the branch objective or acceptance geometry so that smaller projected candidate segments can be tested before target-gate passage, or so low/A15 anti-regression is enforced directly while high/A08 pressure is applied. Formal long training remains blocked.

Current handoff: `/home/ldj/wxc/2DGauss/logs/current_state.md`.
