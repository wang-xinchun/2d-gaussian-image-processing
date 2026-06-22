# G_4kd03 Manifest

- Created: `2026-06-21 19:45:00 CST`
- Architecture line: `G-RSCN`
- Run package: `experiments/G_RSCN/runs/G_4kd03`
- First smoke run: `G_4kd03_smoke01`
- Status: `smoke01 deployed/evaluated; protected selector passed as guard; not promoted`

## Difference From G_4kd02

`G_4kd02_smoke01` proved branch-isolated Gaussian route-control supervision is live and improves high/A08, but it still selected checkpoints that spent low/A15 surplus. `G_4kd03` changes the deployment rule, not the Gaussian core:

- validation now records router-aligned high/low/A08/A15 gap metrics when enabled;
- best checkpoint selection can require no-regression gates versus the initialized parent baseline;
- smoke01 selects by `val_router_a08_oracle_gap_mean` only if low-route, A15-oracle, and low-A15-oracle gaps do not drop versus the step-0 baseline;
- training loss also raises low/A15 branch and selected-output protection while reducing A08 pressure.

The Gaussian core remains unchanged.

## Smoke01

- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd03/configs/g_4kd03_smoke01_a15_guarded_selector_short.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd03/launch_smoke01_gpu.sh`
- Evaluator: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd03/evaluate_smoke01_gpu.sh`
- Parent checkpoint: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd02_smoke01/checkpoints/best_val.pt`
- Scope: physical GPU0/3 only, single visible GPU, `max_steps=40`, `batch_size=1`, `grad_accum=2`, `max_val_batches=24`.
- Trainable prefix: `gaussian_route_control` only.
- Successful train log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd03/logs/smoke01_20260621_200941.log`
- Best eval log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd03/logs/eval_smoke01_best_val_20260621_201216.log`
- Final eval log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd03/logs/eval_smoke01_final_20260621_201506.log`
- Summary JSON: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd03_smoke01/metrics/g_4kd03_smoke01_a37_aligned_summary.json`

## Smoke01 Outcome

- GPU: physical GPU3 only, `CUDA_VISIBLE_DEVICES=3`, `TWODGAUSS_CUDA_MEMORY_FRACTION=0.35`, no CPU fallback.
- Training exit code: `0`.
- Final step: `40`.
- Saved best step: `0`, intentionally initialized from the parent baseline because all trained candidate checkpoints failed the low/A15 validation gates.
- Peak CUDA reserved: `3.130859375GiB`.
- Best-val checkpoint A37 deltas: high-route `+0.0002355462360844468`, A08-oracle `+0.0018106951341312072`, low-route `-0.0017883107046233748`, A15-oracle `-0.0032425407070728884`.
- Final checkpoint A37 deltas: high-route `+0.00046075554710511835`, A08-oracle `+0.004295594447264128`, low-route `-0.005180837680822031`, A15-oracle `-0.008760974145471945`.
- Verdict: selector is useful as a guard, but the training mechanism is still a high/A08 versus low/A15 trade-off. Formal long training remains blocked.

## Promotion Boundary

`G_4kd03_smoke01` is promotable only if A37-aligned GPU evaluation improves high-route and A08-oracle over A37 while keeping low-route and A15-oracle at or above A37. The protected validation selector is necessary but not sufficient; aggregate PSNR alone cannot promote.
