# 2DGauss Logging Rules

- Project root: `/home/ldj/wxc/2DGauss`
- Logging start time (Beijing): `2026-04-23 17:34:38 CST`
- Time zone: `Asia/Shanghai` (UTC+8)
- Research direction: ultra-high-resolution image restoration
- Long-term goal: build a fully traceable project history that supports top-tier paper work

## Files

- `current_state.md`: short handoff card for new windows. This is the default startup entry.
- `messages.md`: human-readable conversation archive.
- `messages.jsonl`: machine-readable message stream, one JSON object per line.
- `operations.md`: human-readable operation trace.
- `operations.jsonl`: machine-readable operation trace.
- `decisions.md`: important research, scope, and workflow decisions.
- `../scripts/append_message.sh`: append a conversation record with a Beijing timestamp.
- `../scripts/append_operation.sh`: append an operation record with a Beijing timestamp.

## Logging Conventions

- All timestamps must use `TZ=Asia/Shanghai`.
- Dialogue records should preserve the original user request and assistant response as much as possible.
- Operation records should include intent, command or action, related path, result, and status.
- Logs are append-only. If a correction is needed, add a new correction entry instead of deleting history.
- When future work involves code, experiments, or datasets, the affected files or directories should be recorded in the relevant log entry.

## Layered Logging Policy

The logging system is split by responsibility:

- Raw audit: `messages.md/jsonl`, `operations.md/jsonl`, and `decisions.md`. These files preserve the full trace and are not optimized for startup token cost.
- Current state: `current_state.md`. This file is the short, current-only handoff card for a new window.
- Targeted retrieval: read raw audit logs only when a concrete question needs evidence, using stable keys such as experiment IDs, decision IDs, operation IDs, file paths, script names, dates, and run directories.

New windows should read `current_state.md` first. They should not start by reading the full raw audit logs unless the current task specifically requires historical evidence.

`current_state.md` should stay short and should be overwritten when the active goal, phase, conclusion, plan, blocker, or next step changes. It should not become a stage summary or another long log.

`Milestones` are optional for now. Add a milestone layer only after a phase has clearly closed and there is a stable conclusion worth compressing for medium-term review.

## Current Layer Index

Updated: `2026-06-22 19:28:00 CST`

- Startup handoff: `/home/ldj/wxc/2DGauss/logs/current_state.md`.
- Raw conversation: `/home/ldj/wxc/2DGauss/logs/messages.md` and `/home/ldj/wxc/2DGauss/logs/messages.jsonl`; latest recorded user request is `MSG-000729`.
- Operation audit: `/home/ldj/wxc/2DGauss/logs/operations.md` and `/home/ldj/wxc/2DGauss/logs/operations.jsonl`; latest operation is `OP-000775`.
- Decision audit: `/home/ldj/wxc/2DGauss/logs/decisions.md`; latest decision is `DEC-000229`.
- Current G-RSCN experiment line: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/README.md`.
- Current G-RSCN run: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd12/manifest.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd12/gate_check.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd12/result_analysis.md`.
- Current G-RSCN postmortem: `/home/ldj/wxc/2DGauss/logs/postmortem_20260622_g_rscn_g06_g10_status.md`.
- Current G-RSCN breakthrough strategy: `/home/ldj/wxc/2DGauss/logs/grscn_breakthrough_strategy_20260622.md`.
- Current G-RSCN GPU policy: `/home/ldj/wxc/2DGauss/logs/gpu_usage_policy_20260621_g_rscn.md`.
- Current G-RSCN architecture lessons: `/home/ldj/wxc/2DGauss/logs/architecture_lessons_for_g_rscn_20260617.md`.
- Current Gaussian-core preservation review: `/home/ldj/wxc/2DGauss/logs/gaussian_core_preservation_architecture_review_20260617.md`.
- Current architecture recommendation: `/home/ldj/wxc/2DGauss/logs/new_architecture_recommendation_20260617_b_series.md`.
- Current mechanism audit: `/home/ldj/wxc/2DGauss/logs/mechanism_feasibility_audit_20260617.md`.
- Current stop-loss review: `/home/ldj/wxc/2DGauss/logs/architecture_stoploss_review_20260617_route_patch_trap.md`.
- Current standalone report: `/home/ldj/wxc/2DGauss/logs/model_architecture_and_breakthrough_experiment_report_20260616_cn.md`.
- English technical report: `/home/ldj/wxc/2DGauss/logs/model_architecture_and_breakthrough_experiment_report_20260616.md`.
- Current retrospective: `/home/ldj/wxc/2DGauss/logs/retrospective_20260616_a37_a41_high_route_bottleneck.md`.
- Earlier retrospective: `/home/ldj/wxc/2DGauss/logs/retrospective_20260613_no_single_checkpoint_breakthrough.md`.
- Architecture review: `/home/ldj/wxc/2DGauss/logs/architecture_review_20260613_model_architecture.md`.
- Performance shortfall review: `/home/ldj/wxc/2DGauss/logs/performance_shortfall_analysis_20260613_arch_data_theory.md`.

## Experiment-Level Logs

Recent active mainline run directories carry their own local log layers:

- G_4kd12: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd12/manifest.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd12/gate_check.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd12/result_analysis.md`.
- G_4kd11: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd11/manifest.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd11/gate_check.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd11/result_analysis.md`.
- G_4kd10: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd10/manifest.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd10/gate_check.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd10/result_analysis.md`.
- G_4kd09: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd09/manifest.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd09/gate_check.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd09/result_analysis.md`.
- G_4kd08: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd08/manifest.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd08/gate_check.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd08/result_analysis.md`.
- G_4kd07: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd07/manifest.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd07/gate_check.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd07/result_analysis.md`.
- G_4kd06: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd06/manifest.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd06/gate_check.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd06/result_analysis.md`.
- G_4kd05: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd05/manifest.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd05/gate_check.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd05/result_analysis.md`.
- G_4kd04: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd04/manifest.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd04/gate_check.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd04/result_analysis.md`.
- G_4kd03: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd03/manifest.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd03/gate_check.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd03/result_analysis.md`.
- G_4kd02: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd02/manifest.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd02/gate_check.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd02/result_analysis.md`.
- G_4kd01: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/manifest.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/gate_check.md`, `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/result_analysis.md`.
- A37: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/manifest.md`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/gate_check.md`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/result_analysis.md`.
- A39: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd39/manifest.md`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd39/gate_check.md`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd39/result_analysis.md`.
- A40: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd40/manifest.md`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd40/gate_check.md`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd40/result_analysis.md`.
- A41: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd41/manifest.md`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd41/gate_check.md`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd41/result_analysis.md`.
- A42: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd42/manifest.md`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd42/gate_check.md`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd42/result_analysis.md`.
- A43: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd43/manifest.md`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd43/gate_check.md`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd43/result_analysis.md`.
- A44: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd44/manifest.md`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd44/gate_check.md`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd44/result_analysis.md`.
- A45: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd45/manifest.md`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd45/gate_check.md`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd45/result_analysis.md`.
- A46: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd46/manifest.md`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd46/gate_check.md`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd46/result_analysis.md`.
- A47: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd47/manifest.md`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd47/gate_check.md`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd47/result_analysis.md`.

When deploying a new experiment, update all relevant layers:

- Record the user's exact request with `scripts/append_message.sh`.
- Record major actions and final outcomes with `scripts/append_operation.sh`.
- Add or update a `DEC-*` entry when the run changes project strategy or promotion status.
- Update `current_state.md` when the active conclusion or next step changes.
- Add run-local `manifest.md`, `gate_check.md`, and `result_analysis.md`.
- Add a retrospective only when a phase-level conclusion has emerged.

## ID Rules

- Turn ID: `T0001`, `T0002`, ...
- Message ID: `MSG-000001`, `MSG-000002`, ...
- Operation ID: `OP-000001`, `OP-000002`, ...
- Decision ID: `DEC-000001`, `DEC-000002`, ...
