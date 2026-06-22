# Current State

更新时间: 2026-06-22 22:44:00 CST

## Active Goal

- 用户判断新架构瓶颈仍未真正解耦，要求部署实验继续突破。
- 当前路线是新架构 `G-RSCN`，不是旧 `A_4kdXX` route-patch 线。
- 最新短训：`G_4kd15_smoke01` 已完成；未 promoted。
- 正式长训仍然禁止。

## Hard Invariant

- 旧架构/route-patch 线保留为 `experiments/V1/runs/A_4kdXX`。
- 新架构线必须使用 `experiments/G_RSCN/runs/G_4kdXX`。
- `GaussianParameterHead`、`TextureAwareGaussianRenderer`、Gaussian attention/control 的 decoder/refinement/control path 必须保留。
- 新增机制只能作为 Gaussian-conditioned route-specific control；`input_std` 只能作 prior/sampler。

## Latest Result: G13-G15 Decoupling Probes

### G_4kd13_smoke01

- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd13/configs/g_4kd13_smoke01_a37_dual_prior_transfer_guard_short.toml`
- Train log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd13/logs/smoke01_20260622_212808.log`
- Run-local docs: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd13/manifest.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd13/gate_check.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd13/result_analysis.md`

G13 implemented the first structural split:

- `gaussian_route_control_adapter_count = 2`;
- shared route feature retained for gate/confidence;
- separate `high_prior_encoder -> high_control -> high_delta`;
- separate `low_prior_encoder -> low_control -> low_delta`;
- trainable prefix only `gaussian_route_control`;
- immutable Gaussian core unchanged.

Training result:

- train exit code `0`;
- final step `20`;
- trainable params `50114`;
- A18 transfer-val records matched `100/100`;
- step guard checks `10`;
- accepted updates `0`;
- rollbacks `10`;
- full/protect/target gate passes `0/0/0`;
- best-val step `0`;
- final `gaussian_route_control_high_delta_abs_mean = 0.0`;
- final `gaussian_route_control_low_delta_abs_mean = 0.0`.

Interpretation: G13 structurally wired dual high/low route-control capacity, but zero-init output plus every-2-step transfer-aware rollback reset every candidate before the adapter could accumulate signal. G13 is not a promotion candidate and A37-aligned eval was skipped because `best_val` is A37 and `final` has no new adapter signal.

### G_4kd14_smoke01

- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd14/configs/g_4kd14_smoke01_a37_dual_prior_deferred_transfer_gate_short.toml`
- Train log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd14/logs/smoke01_20260622_215226.log`
- Final eval log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd14/logs/eval_smoke01_final_20260622_215551.log`
- Step-10 eval log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd14/logs/eval_smoke01_step_000010_20260622_220150.log`
- Run-local docs: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd14/manifest.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd14/gate_check.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd14/result_analysis.md`

G14 kept G13 dual adapter capacity but disabled step-level rollback:

- `step_no_harm_guard_enabled = false`;
- `val_interval = 20`;
- best-val still requires A18 transfer full gate;
- final records the unconstrained 20-step adapter direction.

Training result:

- train exit code `0`;
- final step `20`;
- trainable params `50114`;
- step20 transfer full gate failed;
- best-val step `0`;
- final `gaussian_route_control_high_delta_abs_mean = 2.574286554590799e-05`;
- final `gaussian_route_control_low_delta_abs_mean = 1.9299128325656056e-05`.

Step-20 transfer validation missed all five full-gate metrics:

```text
baseline high-route transfer gap: 0.03710367872908309
step20   high-route transfer gap: 0.036676999684926624

baseline A08-oracle transfer gap: 0.07303136937758502
step20   A08-oracle transfer gap: 0.07297538308536305

baseline low-route transfer gap: 0.417457762218657
step20   low-route transfer gap: 0.41615719265407985

baseline A15-oracle transfer gap: 0.3613124905210553
step20   A15-oracle transfer gap: 0.35986068032004614

baseline low-A15 transfer gap: 0.4313206589012815
step20   low-A15 transfer gap: 0.42999906707228275
```

A37-aligned test baseline:

```text
PSNR 17.889658453438816
high-route -0.11159004428502844
A08-oracle -0.19984098726138502
low-route 0.1591364853806252
A15-oracle 0.2493535773419097
```

G14 `step_000010`:

```text
PSNR 17.88950629774918 (-0.000152155689637)
high-route -0.11149740290558754 (+0.000092641379441)
A08-oracle -0.19942862311188653 (+0.000412364149498)
low-route 0.15879965821783007 (-0.000336827162795)
A15-oracle 0.24872053438190425 (-0.000633042960005)
```

G14 `final`:

```text
PSNR 17.889438099539603 (-0.000220353899213)
high-route -0.11146861464370558 (+0.000121429641323)
A08-oracle -0.19927614814070793 (+0.000564839120677)
low-route 0.15865829477539373 (-0.000478190605231)
A15-oracle 0.24846435531464198 (-0.000889222027268)
```

Interpretation: G14 proves the deferred learning window works mechanically: the dual adapter can grow non-zero deltas. It also proves the current unconstrained direction is still not promotable: it improves high/A08 while spending low/A15 and aggregate PSNR.

### G_4kd15_smoke01

- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd15/configs/g_4kd15_smoke01_a37_dual_prior_warmup_projection_guard_short.toml`
- Train log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd15/logs/smoke01_20260622_222651.log`
- Final eval log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd15/logs/eval_smoke01_final_20260622_223732.log`
- Run-local docs: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd15/manifest.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd15/gate_check.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd15/result_analysis.md`

G15 added `training.step_no_harm_guard_start_step` and used warmup then transfer guard/projection:

- `step_no_harm_guard_start_step = 10`;
- `step_no_harm_guard_interval = 2`;
- `step_no_harm_guard_projection_ratios = [0.75, 0.50, 0.25, 0.125]`;
- protect-only acceptance disabled;
- trainable prefix only `gaussian_route_control`.

Training result:

- train exit code `0`;
- final step `20`;
- trainable params `50114`;
- delayed guard checks `6`;
- accepted updates `0`;
- rollbacks `6`;
- projection attempts/accepts `0/0`;
- best-val step `0`.

Step-20 transfer candidate:

```text
high-route transfer: 0.03732513737034153 vs baseline 0.03710367872908309
A08-oracle transfer: 0.07288708406336167 vs baseline 0.07303136937758502
low-route transfer: 0.4171813329060872 vs baseline 0.417457762218657
A15-oracle transfer: 0.36124710603193805 vs baseline 0.3613124905210553
low-A15 transfer: 0.43102905206512987 vs baseline 0.4313206589012815
```

Final A37-aligned GPU evaluation:

```text
PSNR 17.889589332557936 (-0.000069120880880)
high-route -0.11164105487970442 (-0.000051010594676)
A08-oracle -0.19993386251938058 (-0.000092875257996)
low-route 0.15905370235401622 (-0.000082783026609)
A15-oracle 0.24930469167116928 (-0.000048885670740)
```

Interpretation: G15 proves warmup scheduling alone is insufficient. It keeps damage small compared with G14 but still misses all A37-aligned promotion gates; every post-warmup update remains unsafe under the A18 transfer guard.

## Current Bottleneck

- G12 proved the A18 transfer-aware guard works.
- G13 proved simple dual prior/control capacity plus strict rollback from zero cannot accumulate signal.
- G14 proved signal can accumulate when rollback is deferred, but the branch objective still follows a high/A08-positive and low/A15-negative tradeoff.
- G15 proved delaying guard after warmup prevents large drift but still cannot accept a simultaneous high/A08 and low/A15-safe update.
- The bottleneck is now more precise: branch-objective/acceptance geometry, not just structural capacity or warmup timing.

## Next Step

- Do not promote G13, G14, or G15.
- Do not lengthen G14/G15 as-is.
- Do not use aggregate PSNR or high/A08-only improvement as promotion.
- Next experiment should keep dual adapter capacity but change branch-specific objective/acceptance:
  - keep A37 initialization;
  - keep A18 transfer-aware best selection;
  - keep A37-aligned test gates unchanged;
  - test smaller projected candidate segments before requiring full target-gate passage, or add direct low/A15 anti-regression while high/A08 pressure is applied;
  - require any accepted movement to retain low-route and A15-oracle while preserving or improving high/A08.
- Formal long training remains blocked until a checkpoint passes A37-aligned high-route, A08-oracle, low-route, and A15-oracle simultaneously.

## GitHub / Privacy State

- GitHub mirror is private at `git@github.com:wang-xinchun/2d-gaussian-image-processing.git`.
- Latest pushed privacy/audit commit remains `3b59adcf12367371721f91f621bbdbf123907356`.
- Current G13/G14/G15 run outputs are local and include checkpoints/metrics/logs; do not `git add .`.
- Any future sync must stage only lightweight source/config/docs after checking datasets, checkpoints, metrics, raw logs, images, videos, private paths, large files, and secrets.

## GPU Rule

- 只允许物理 GPU `0` 和 `3`；禁止 GPU `1/2`。
- 默认物理 GPU 为 `3`。
- G-RSCN 禁止 DDP 和 `CUDA_VISIBLE_DEVICES=0,3`。
- 默认门槛：`TWODGAUSS_MIN_FREE_MIB=8192`、`TWODGAUSS_MAX_GPU_UTIL=20`、`TWODGAUSS_CUDA_MEMORY_FRACTION=0.35`、`num_workers<=1`。
- 用户授权短训/评估可以不干等：GPU0/3 只要显存合适且不会把别人挤下去，可以临时放宽 util 等待，但必须保持单 GPU、显存比例 `0.35`、无 CPU fallback。

## Logging Rules

- 新窗口先读本文件，再按 `G_4kd13`、`G_4kd14`、`G_4kd15`、`DEC-000230`、`OP-000779`、`OP-000780`、`OP-000781`、`OP-000782`-`OP-000785` 定点检索 run-local docs 和 raw logs。
- 新实验必须维护 run-local `manifest.md`、`gate_check.md`、`result_analysis.md`，并更新全局 `current_state.md`、`operations`，必要时追加 `DEC-*`。
