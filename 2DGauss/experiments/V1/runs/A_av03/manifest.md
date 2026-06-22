# Run Manifest: V1/T01/A_av03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_av03`
- Full identifier: `V1/T01/A_av03`

## Current Status

Completed and evaluated.

## Purpose

`A_av03` is the direct bottleneck probe after `A_au03`. `A_au03` was directionally correct: residual-demand canvas supervision improved fidelity and mechanism structure. Its remaining failure was feedback-gate saturation (`control_feedback_map mean≈0.98474`), which made the protected additive回流 too close to a global correction.

## Mechanism

Compared with `A_au03`, this run keeps the restoration-demand canvas and additive residual feedback, but adds a default-off selective budget mechanism now enabled in this run:

- Disable the old confidence normalization that tended to push gates toward `1.0`.
- Recenter the raw feedback gate spatially before applying it.
- Reweight feedback by the `restoration_demand` target so already-correct regions are less likely to receive回流.
- Cap the spatial mean of the feedback gate with `gaussian_control_canvas_feedback_budget=0.45`.
- Keep a small `feedback_floor=0.05` so the model is not forced into a hard binary mask at startup.

This run uses config: `experiments/V1/runs/A_av03/configs/a_av03_short_uhddeblur_demand_budget_selective_feedback_probe.toml`.

## Baselines

- Frozen medium baseline: `M04/A_y03`, official test `26.272695829299355`
- Current short leader: `A_z03`, official test `25.385710820365173`
- Latest demand-feedback probe: `A_au03`, official test `25.361614657535245`, `control_canvas std≈0.01699`, `control_feedback_map mean≈0.98474`, `attention_map std≈0.00024`

## Gate Status

- [x] Static py_compile
- [x] Config parse/model forward
- [x] Formal preflight at accepted geometry
- [x] True smoke calibration inside the memory window
- [x] Launch only if gates pass and no conflicting active training is present

## Launch Evidence

- Static py_compile passed for the touched config/model/training/evaluation scripts.
- CPU small tensor forward/backward passed; selective feedback sanity check produced `feedback_mean≈0.24183`, `feedback_std≈0.15412`.
- Formal preflight passed at `batch_size=6`, `train_context_size=640`, supervision crop `512`: `total_params=374761`, `val_psnr=23.197261183431607`, `loss_total=0.11129850894212723`.
- True smoke passed at `6x640`: `max_memory_reserved_gb=91.5078125`, `max_memory_allocated_gb=88.21524906158447`, `avg_total_step_seconds=2.0858878704408803`, `estimated_steps_target_wall_clock=1668`, `within_target_reserved_memory_window=true`.
- Launched tmux session: `2dgauss_a_av03_uhddeblur_demand_budget_selective_feedback_probe`.
- Active log: `experiments/V1/runs/A_av03/logs/train_20260520_120727.log`.
- Startup check: metrics have written through at least `step=60`; GPU memory is about `94411 MiB` with high utilization; no early NaN/OOM observed.

## Final Acceptance

- Training finished cleanly at `step=2500`.
- Final validation: `val_loss=0.025011458806693553`, `val_psnr=28.94858555400976`.
- Same-protocol official test: `loss_mean=0.03904216613465299`, `psnr_mean=25.32983078774204`, `auxiliary_loss_mean=0.006300219018012285`, `total_loss_mean=0.045342385297020275`.
- Task-specific diagnostics (`limit=20`): `psnr_mean=23.02967504761376`, `ssim_mean=0.6918422877788544`, `edge_magnitude_l1_mean=0.1320122629404068`, `high_frequency_l1_mean=0.017329713702201842`, `local_contrast_l1_mean=0.021890324354171754`, `luminance_l1_mean=0.04286699257791042`, `chroma_l1_mean=0.005025674449279904`, `saturation_l1_mean=0.009798495518043637`, `dark_channel_l1_mean=0.04197766873985529`.
- Mechanism panels: `experiments/V1/runs/A_av03/diagnostics/mechanism_final_test`.
- Mechanism diagnosis: `control_feedback_map mean≈0.2415`, `std≈0.1429`, `control_canvas std≈0.0167`, `control_canvas_target std≈0.1800`, `attention_map std≈0.000061`.

## Decision Rule

`A_av03` is useful only if it preserves or improves the official fidelity of `A_au03` while reducing feedback saturation. Promotion evidence requires same-protocol official test, task-specific diagnostics, and mechanism panels showing a lower, more selective feedback map without collapsing the restoration-demand canvas or creating visual artifacts.

Final judgment: `A_av03` is not a promotion candidate. It successfully fixed feedback saturation, but the selective feedback became underpowered and official fidelity fell below both `A_au03` and `A_z03`. The next mechanism should preserve selectivity while calibrating average correction strength.
