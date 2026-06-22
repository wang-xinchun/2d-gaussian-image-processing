# Manifest: V1/T01/A_4kd23

- Created: `2026-06-12 CST`
- Run: `V1/T01/A_4kd23`
- Purpose: Convert the A18/A19/A21 two-checkpoint input-std routing milestone into a single student checkpoint through constrained router distillation.
- Baseline/Init: `A_4kd15` best-val.
- Teacher A: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`
- Teacher B: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/checkpoints/best_val.pt`
- Run-local config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd23/configs/a_4kd23_fast_4kdehaze_a15_router_distill_input_std_single_model.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd23/launch_tmux.sh`
- Wait launcher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd23/wait_and_launch_a_4kd23_memory_gated_gpu_safe.sh`

## Hypothesis

`A_4kd22` shows that unconstrained saved-record logistic routers do not beat the
fixed A18 `input_std` rule across A18/A19/A21 blocks. `A_4kd23` therefore keeps
the stable rule as a prior and trains one student model to approximate the
two-checkpoint router target:

```text
input_std <= 0.07026463374495506 -> A_4kd15 teacher
input_std  > 0.07026463374495506 -> A_4kd08 teacher
```

The distillation target is softened with `router_distillation_softness=0.005`
and a conservative `router_distillation_weight=0.03`.

## Narrow Variables

- Start from `A_4kd15` best-val instead of `A_4kd08`.
- Keep the A15 architecture, data view, reconstruction/task-aware losses, EMA,
  and `val_psnr` checkpoint selection.
- Add only the default-off `loss.router_distillation_*` branch to the training
  entrypoint and enable it for this run.
- Use GPU memory gating on physical GPU `3` by default; physical GPU `0` is also
  allowed when explicitly selected. GPU1/GPU2 remain forbidden.

## Success Criteria

- Preflight must load both teachers strictly, run one forward/backward, and log
  nonzero router-distillation diagnostics.
- A23 best-val should be evaluated against A08/A15/A18-router baselines after
  training. The immediate bar is to determine whether the two-checkpoint routing
  signal can survive conversion into one checkpoint.
- If A23 does not beat A15/A08, inspect whether router-distillation loss is too
  weak/strong before adding richer router features.

## Boundary

This is a single-checkpoint conversion probe on the fast 4KDehaze view. It is
not yet an official/full benchmark claim.

## Completion

- Status: completed on `2026-06-12 CST`.
- Preflight log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd23/logs/train_20260612_224738.log`, exit code `0`.
- Formal training log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd23/logs/train_20260612_225300.log`, exit code `0`.
- Best checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd23/checkpoints/best_val.pt`, step `50`.
- Final checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd23/checkpoints/final.pt`, step `300`.
- Result analysis: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd23/result_analysis.md`.

## Results

Validation peaked at step `50` with `val_psnr=18.256583743058567` and
`val_loss=0.12251933488994837`, then drifted down to final
`val_psnr=18.154180648966136`.

Fast-test results on the 100-sample crop512 protocol:

- `A_4kd23` best-val step50: `psnr_mean=17.78064128762362`,
  `ssim_mean=0.6971213883161544`, `loss_mean=0.12059733912348747`,
  `total_loss_mean=0.12382887728512287`.
- `A_4kd23` final step300: `psnr_mean=17.035365402315897`,
  `loss_mean=0.13115204714238643`,
  `total_loss_mean=0.13438382618129252`.

Conclusion: `A_4kd23` is an engineering/deployment success for default-off
router-distillation infrastructure, but it is a negative single-checkpoint
conversion result. It does not supersede `A_4kd08`, `A_4kd15`, or the
`A_4kd18` two-checkpoint router.
