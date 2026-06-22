# G_4kd11 Manifest

- Created: `2026-06-22 16:05:00 CST`
- Architecture line: `G-RSCN`
- Run package: `experiments/G_RSCN/runs/G_4kd11`
- First smoke run: `G_4kd11_smoke01`
- Status: `smoke01 completed; not promoted; formal training blocked`

## Difference From G_4kd10

`G_4kd10_smoke01` proved that relaxing the A08 validation margin alone is insufficient: an early candidate passed high/A08 but spent low/A15, while the closest later candidate missed every positive margin slightly. `G_4kd11` therefore changes update application rather than scalar thresholds.

Kept from G10:

- initialize from `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`;
- train only `gaussian_route_control`;
- keep the immutable Gaussian core;
- keep A37-aligned high-route, A08-oracle, low-route, A15-oracle, and low-A15 promotion gates;
- keep full `24` validation batches for the step guard;
- keep single approved GPU policy with no CPU fallback.

New in G11:

- `step_no_harm_guard_two_phase_enabled = true`;
- low/A15 protect gate may accept low/A15-surplus-building updates even if high/A08 has not yet passed;
- high/A08 target gate is tracked separately;
- when a full candidate passes high/A08 but fails low/A15, projection tests interpolation ratios `[0.75, 0.50, 0.25]` between the previous accepted state and the candidate;
- a projected state is accepted only if it preserves low/A15 and improves high/A08.

## Smoke01 Artifacts

- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd11/configs/g_4kd11_smoke01_a37_two_phase_projection_short.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd11/launch_smoke01_gpu.sh`
- Evaluator: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd11/evaluate_smoke01_gpu.sh`
- Parent checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`
- Scope: physical GPU0/3 only, single visible GPU, `max_steps=20`, `batch_size=1`, `grad_accum=2`, `max_val_batches=24`.

Completed smoke01 artifacts:

- Run dir: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd11_smoke01`
- Train log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd11/logs/smoke01_20260622_161856.log`
- Best-val eval log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd11/logs/eval_smoke01_best_val_20260622_162732.log`
- Final eval log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd11/logs/eval_smoke01_final_20260622_163217.log`
- Best checkpoint: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd11_smoke01/checkpoints/best_val.pt`
- Final checkpoint: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd11_smoke01/checkpoints/final.pt`

## Promotion Boundary

`G_4kd11_smoke01` is promotable only if A37-aligned GPU evaluation passes all route/oracle gates simultaneously:

```text
high-route >= A37
A08-oracle >= A37
low-route >= A37
A15-oracle >= A37
```

Aggregate PSNR alone cannot promote. Formal long training remains blocked until the short-run result passes the A37-aligned gate.

## Smoke01 Outcome

`G_4kd11_smoke01` completed successfully on physical GPU3 with train exit code `0` and eval exit code `0` for both `best_val` and `final`.

Training accepted all `10/10` step guard checks with `accepted_mode=full`, produced `0` rollbacks, and selected step `6` as `best_val` by validation A08-oracle gap. Projection was implemented and enabled, but was not exercised because every checked candidate passed the full five validation gates.

A37-aligned test-side evaluation did not pass the promotion boundary:

- `best_val`: PSNR `17.889756173256146` (`+0.000097719817330` vs A37), high-route `-0.111665766148642` (`-0.000075721863613`), low-route `0.159365047167789` (`+0.000228561787164`), A08-oracle `-0.200214915987967` (`-0.000373928726582`), A15-oracle `0.249853071844794` (`+0.000499494502884`).
- `final`: PSNR `17.889620811442292` (`-0.000037641996524` vs A37), high-route `-0.111663922610018` (`-0.000073878324990`), low-route `0.159126179561716` (`-0.000010305818909`), A08-oracle `-0.200062964555102` (`-0.000221977293717`), A15-oracle `0.249472961709662` (`+0.000119384367752`).

Conclusion: G11 is useful as a validation-safe update diagnostic, but it is not a breakthrough. The next probe must address validation-to-test transfer, preferably with a test-transfer-aware guard or route-decoupled Gaussian-conditioned capacity rather than scalar threshold tuning.
