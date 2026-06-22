# Result Analysis: V1/T01/A_av03

Status: completed and evaluated.

## Pre-Launch Rationale

`A_au03` was on-topic but incomplete: the residual-demand target made the canvas meaningful and recovered official fidelity, but the feedback gate was almost always open (`control_feedback_map mean≈0.98474`). `A_av03` keeps the restoration-demand canvas and additive residual feedback, but disables the old confidence normalization and adds demand-budget selective feedback.

## Gate Evidence

- CPU small tensor forward/backward passed; initial selective feedback sanity check: `feedback_mean≈0.24183`, `feedback_std≈0.15412`.
- Formal preflight passed at `6x640`: `total_params=374761`, `val_psnr=23.197261183431607`.
- True smoke passed at `6x640`: `max_memory_reserved_gb=91.5078125`, `within_target_reserved_memory_window=true`.
- Training launched as tmux session `2dgauss_a_av03_uhddeblur_demand_budget_selective_feedback_probe`.
- Active log: `experiments/V1/runs/A_av03/logs/train_20260520_120727.log`.

## Mid-Run Notes

- `step=500`: `val_loss=0.025735499151051045`, `val_psnr=28.709412065424157`
- `step=1000`: `val_loss=0.025895099248737098`, `val_psnr=28.723837828997652`
- Latest observed train metric before stopping foreground monitoring: around `step=1500`, with no NaN/OOM signal.
- User requested not to守着训练. The run continues in tmux; no foreground wait loop should be kept open.

## Final Acceptance

- Training finished at `step=2500`.
- Final validation: `val_loss=0.025011458806693553`, `val_psnr=28.94858555400976`.
- Final checkpoint: `experiments/V1/runs/A_av03/checkpoints/final.pt`.
- Official full-image test (`UHD_deblur/test300`, `crop_size=0`): `loss_mean=0.03904216613465299`, `psnr_mean=25.32983078774204`, `auxiliary_loss_mean=0.006300219018012285`, `total_loss_mean=0.045342385297020275`.
- Task-specific diagnostics (`limit=20`): `psnr_mean=23.02967504761376`, `ssim_mean=0.6918422877788544`, `edge_magnitude_l1_mean=0.1320122629404068`, `high_frequency_l1_mean=0.017329713702201842`, `local_contrast_l1_mean=0.021890324354171754`, `luminance_l1_mean=0.04286699257791042`, `chroma_l1_mean=0.005025674449279904`, `saturation_l1_mean=0.009798495518043637`, `dark_channel_l1_mean=0.04197766873985529`.
- Mechanism panel export: `experiments/V1/runs/A_av03/diagnostics/mechanism_final_test`.

## Diagnosis

`A_av03` directly targeted the `A_au03` saturation failure and succeeded on that mechanism axis: `control_feedback_map mean` fell from about `0.98474` to about `0.2415`, with `std≈0.1429`. However, the official result dropped below both `A_au03` and the short leader `A_z03`, and the limit20 task-specific fidelity also declined.

The root cause is therefore over-shrunk selective reinjection. The gate is no longer saturated, but the mean correction strength is too low. The next mechanism should keep the non-saturated selective gate while calibrating its average correction strength upward, instead of simply lowering budget further.

Final judgment: on-topic diagnostic result, not a promotion candidate.
