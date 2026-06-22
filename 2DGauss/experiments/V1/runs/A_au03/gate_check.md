# Gate Check: V1/T01/A_au03

## Branch Identity

- Run: `V1/T01/A_au03`
- Config: `experiments/V1/runs/A_au03/configs/a_au03_short_uhddeblur_residual_demand_protected_feedback_probe.toml`
- Mechanism family: residual-demand aware, fidelity-protected Gaussian reinjection

## Static Gate

- [x] Config exists
- [x] Python compile/import check passed
- [x] Model forward/aux check passed
- [x] Parameter count checked: `total_params=374761`

## Smoke / Preflight Gate

- [x] Formal preflight passed at `batch_size=6`, `train_context_size=640`, supervision crop `512`
- [x] True smoke calibration passed
- [x] Reserved VRAM inside `85.0-93.0G`: `max_memory_reserved_gb=88.181640625`
- [x] Average smoke step time recorded: `avg_total_step_seconds=1.00056027257815`
- [x] Estimated steps in 58 minutes recorded: `3478`

## Launch Gate

- [x] No conflicting active tmux training session
- [x] Training launched only after gates pass
- [x] Startup metrics writing

## Result Gate

- [x] Training finished cleanly: `step=2500`
- [x] Checkpoint endpoint verified: `experiments/V1/runs/A_au03/checkpoints/final.pt`
- [x] Same-protocol official test completed: `metrics/official_test_final.json`
- [x] Task-specific diagnostics generated: `metrics/task_specific_test_limit20.json`
- [x] Mechanism panels generated: `diagnostics/mechanism_final_test`

## Result Summary

- Official test: `samples=300`, `crop_size=0`, `loss_mean=0.039005936284859974`, `psnr_mean=25.361614657535245`
- Task diagnostics (`limit=20`): `psnr_mean=23.115990186771846`, `ssim_mean=0.692804542183876`, `edge_magnitude_l1_mean=0.13188842125236988`, `high_frequency_l1_mean=0.017282098811119796`, `local_contrast_l1_mean=0.022034568060189486`
- Mechanism panels (`limit=4`): `control_canvas std≈0.01699`, `control_canvas_target std≈0.18005`, `control_feedback_map mean≈0.98474`, `std≈0.01373`, `attention_map std≈0.00024`

Gate verdict: all acceptance artifacts are complete. The run is diagnostically positive because the residual-demand mechanism recovers fidelity relative to `A_ar03/A_as03` and improves over `A_aq03`, but it is not promotion-ready because it remains below `A_z03/M04` and the feedback gate is nearly saturated.
