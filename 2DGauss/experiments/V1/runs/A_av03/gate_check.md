# Gate Check: V1/T01/A_av03

## Branch Identity

- Run: `V1/T01/A_av03`
- Config: `experiments/V1/runs/A_av03/configs/a_av03_short_uhddeblur_demand_budget_selective_feedback_probe.toml`
- Mechanism family: residual-demand aware Gaussian reinjection with demand-budget selective feedback

## Static Gate

- [x] Config exists
- [x] Python compile/import check passed
- [x] Model forward/aux check passed
- [x] Parameter count checked: `total_params=374761`

## Smoke / Preflight Gate

- [x] Formal preflight passed
- [x] True smoke calibration passed
- [x] Reserved VRAM inside `85.0-93.0G`: `max_memory_reserved_gb=91.5078125`
- [x] Average smoke step time recorded: `avg_total_step_seconds=2.0858878704408803`
- [x] Estimated steps in 58 minutes recorded: `1668`

## Launch Gate

- [x] No conflicting active tmux training session
- [x] Training launched only after gates pass
- [x] Startup metrics writing through at least `step=60`

## Result Gate

- [x] Training finished cleanly
- [x] Checkpoint endpoint verified
- [x] Same-protocol official test completed
- [x] Task-specific diagnostics generated
- [x] Mechanism panels generated

## Startup Summary

- Preflight: `val_loss=0.06175124198198319`, `val_psnr=23.197261183431607`, `loss_total=0.11129850894212723`, `loss_control_canvas=0.20819306373596191`
- True smoke: `status=ok`, `batch_shape=(6, 3, 640, 640)`, `loss=0.10973247140645981`
- Training log: `experiments/V1/runs/A_av03/logs/train_20260520_120727.log`
- Initial training: `step=1 loss_total=0.11549046635627747`; `step=10 loss_total=0.07164347916841507`; `step=60 loss_total=0.07206574082374573`

## Final Result Summary

- Finished at `step=2500`; final validation `val_loss=0.025011458806693553`, `val_psnr=28.94858555400976`.
- Official test: `psnr_mean=25.32983078774204`, below `A_au03=25.361614657535245` and `A_z03=25.385710820365173`.
- Task-specific limit20: `psnr_mean=23.02967504761376`, `ssim_mean=0.6918422877788544`.
- Mechanism panels: feedback saturation fixed (`control_feedback_map mean≈0.2415`, `std≈0.1429`), but correction strength became too weak for fidelity recovery.
