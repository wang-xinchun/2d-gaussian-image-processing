# Current State

更新时间: 2026-06-22 19:28:00 CST

## Active Goal

- 用户要求认真分析新架构和实验瓶颈，并继续短训尝试突破。
- 当前路线是新架构 `G-RSCN`，不是旧 `A_4kdXX` route-patch 线。
- 最新短训 `G_4kd12_smoke01` 已完成训练和 A37-aligned GPU 评测；未 promoted。
- 正式长训仍然禁止。

## Hard Invariant

- 旧架构/route-patch 线保留为 `experiments/V1/runs/A_4kdXX`。
- 新架构线必须使用 `experiments/G_RSCN/runs/G_4kdXX`。
- `GaussianParameterHead`、`TextureAwareGaussianRenderer`、Gaussian attention/control 的 decoder/refinement/control path 必须保留。
- 新增机制只能作为 Gaussian-conditioned route-specific control；`input_std` 只能作 prior/sampler。

## Latest Result: G_4kd12_smoke01

- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd12/configs/g_4kd12_smoke01_a37_transfer_guard_short.toml`
- Train log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd12/logs/smoke01_20260622_170257.log`
- Best-val eval log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd12/logs/eval_smoke01_best_val_20260622_172216.log`
- Final eval log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd12/logs/eval_smoke01_final_20260622_172803.log`
- Run-local docs: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd12/manifest.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd12/gate_check.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd12/result_analysis.md`

Training:

- physical GPU3 only;
- train exit code `0`;
- final step `16`;
- trainable params `37698`, prefix only `gaussian_route_control`;
- peak CUDA reserved `3.15234375GiB`;
- no CPU fallback;
- A18 `val_records` transfer validation matched `100/100` records.

Guard behavior:

- Step no-harm guard max batches: ordinary val `24` plus full A18 transfer-val `100`.
- Guard checks: `8`.
- Rollbacks: `2` at steps `2` and `4`.
- Accepted updates: `6`, all `protect_only` at steps `6/8/10/12/14/16`.
- Full transfer-gate accepted updates: `0`.
- Projection-accepted updates: `0`.
- Best validation checkpoint stayed at step `0` because no candidate passed the full transfer gate.
- Best transfer-val A08 candidate was step `8`: `0.07311837813433479`, below required full-gate `0.07313136937758502`.

A37-aligned test result:

- A37 baseline: PSNR `17.889658453438816`, high-route `-0.11159004428502844`, low-route `0.1591364853806252`, A08-oracle `-0.19984098726138502`, A15-oracle `0.2493535773419097`.
- `best_val`: exactly A37 parent, all five gates equal baseline.
- `final`: PSNR `17.889701410936713` (`+0.000042957497897`), high-route `-0.11164274339298112` (`-0.000052699107953`), low-route `0.15925160487943105` (`+0.000115119498806`), A08-oracle `-0.2000703652075532` (`-0.000229377946168`), A15-oracle `0.2496285242921635` (`+0.000274946950254`).

## Interpretation

- G12 implemented the transfer-aware guard requested for `G_4kd12_smoke01` and validated it on the full A18 `val_records=100` surface.
- The transfer guard prevented unsafe candidates from being promoted to `best_val`; `best_val` therefore remains the A37 parent.
- Protect-only movement is possible and improves final PSNR/low/A15 slightly, but it still regresses high-route and A08-oracle on A37-aligned test.
- The bottleneck is no longer ordinary validation acceptance, threshold choice, projection plumbing, GPU, teacher plumbing, hard-case exposure, or missing transfer metrics.
- The current same-structure shared `gaussian_route_control` capacity cannot reliably produce a checkpoint that improves/protects high/A08 and low/A15 simultaneously.

## Next Step

- Privacy rule: all project GitHub credentials, wrappers, and user-installed skills must stay under `/home/ldj/wxc`; do not put them in `~/.ssh`, `~/.gitconfig`, or global Codex skill directories on the shared server.
- Current GitHub SSH key is stored under `/home/ldj/wxc/.private/github/`; current repository-local SSH dispatcher is `/home/ldj/wxc/.private/ssh/git-ssh-dispatch`; both are ignored by `/home/ldj/wxc/.gitignore`.
- User-installed skills were moved to `/home/ldj/wxc/.codex/skills/`; `.codex/` is ignored.
- GitHub SSH auth succeeds as `wang-xinchun` through the project-local dispatcher. Private GitHub remote `github` is configured as `git@github.com:wang-xinchun/2d-gaussian-image-processing.git`; remote `HEAD` points to `main`.
- GitHub 首次 source mirror push 已完成：commit `03296272d2ff7635f92ad45400c10873a4b73874` pushed to `github/main`；对应操作记录 `OP-000773`。
- V1/A-series 旧架构分支已补入 GitHub mirror：commit `4734fec278049e1d1ec70bfdc4cac8e43893bc8e` pushed to `github/main`；远端核验 `G_RSCN=101` files、`V1=730` files、forbidden artifact scan empty；对应用户要求 `MSG-000728`、操作记录 `OP-000774`。
- GitHub 端完整性/隐私终检已完成：commit `3b59adcf12367371721f91f621bbdbf123907356` pushed to `github/main`，补入轻量研究报告 `2DGauss/deep-research-report.md` 并硬化 `.gitignore`；远端只有 `main`，共 `979` files，`src=24`、`scripts=36`、`docs=67`、`logs=17`、`G_RSCN=101`、`V1=730`、`deep_report=1`；当前树与 `github/main` 全历史均无 >1MiB blob、权重/数据/输出/日志/图片/视频/私密路径命中，真实密钥/token 扫描为空；对应用户要求 `MSG-000729`、操作记录 `OP-000775`。
- Tool docs: public usage guide `/home/ldj/wxc/GITHUB_TOOL_USAGE.md`; private key-handling guide `/home/ldj/wxc/.private/github/README.md` (ignored).
- 当前 GitHub 镜像包含轻量代码/文档、新架构 `experiments/G_RSCN` 和旧架构 `experiments/V1` 的 configs/manifests/gate checks/result analyses/scripts；服务器本地仍保留完整 datasets、checkpoints、metrics、raw logs、papers 和实验输出。
- 后续仍不要 `git add .`；同步前必须继续审计 `.gitignore`、staged 文件、large files、datasets、checkpoints、archives、logs 和 secrets。
- Do not lengthen G12 or rerun it with only scalar/threshold tweaks.
- Do not promote aggregate PSNR-only or protect-only movement.
- Next probe should be structural: `G_4kd13` or `G_4kd12b` should split Gaussian-conditioned high-preserve and low-dehaze route-control capacity while keeping the immutable Gaussian core.
- Keep the A18 transfer-aware guard as the training-time selection surface and keep A37-aligned test promotion gates unchanged.
- Formal long training remains blocked until a checkpoint passes A37-aligned high-route, A08-oracle, low-route, and A15-oracle simultaneously.

## GPU Rule

- 只允许物理 GPU `0` 和 `3`；禁止 GPU `1/2`。
- 默认物理 GPU 为 `3`。
- G-RSCN 禁止 DDP 和 `CUDA_VISIBLE_DEVICES=0,3`。
- 默认门槛：`TWODGAUSS_MIN_FREE_MIB=8192`、`TWODGAUSS_MAX_GPU_UTIL=20`、`TWODGAUSS_CUDA_MEMORY_FRACTION=0.35`、`num_workers<=1`。
- 用户授权短训/评估可以不干等：GPU0/3 只要显存合适且不会把别人挤下去，可以临时放宽 util 等待，但必须保持单 GPU、显存比例 `0.35`、无 CPU fallback。

## Logging Rules

- 新窗口先读本文件，再按 `G_4kd12`、`G_4kd12_smoke01`、`DEC-000229`、`OP-000762`、`OP-000763`、`postmortem_20260622_g_rscn_g06_g10_status.md` 和 `grscn_breakthrough_strategy_20260622.md` 定点检索 raw logs。
- 最新用户要求是更新 GitHub 端，确保没有隐私或权重文件泄露，同时项目完整：`MSG-000729`。
- 新实验必须维护 run-local `manifest.md`、`gate_check.md`、`result_analysis.md`，并更新全局 `current_state.md`、`operations`，必要时追加 `DEC-*`。
