# Gate Check: V1/T01/A_aq03

## Branch Identity

- Run: `V1/T01/A_aq03`
- Config: `experiments/V1/runs/A_aq03/configs/a_aq03_short_uhddeblur_explicit_frequency_split_gaussian_feedback_probe.toml`
- Mechanism: explicit frequency-split Gaussian feedback

## Static Gate

- [x] Config exists
- [x] Python compile/import check passed
- [x] Explicit split forward/aux check passed
- [x] Parameter count checked: `318888`

## Smoke / Preflight Gate

- [x] Formal preflight passed at `6x704`, `val_psnr=20.073515259173767`
- [x] True smoke calibration passed at `6x704`
- [x] Reserved VRAM inside `85.0-93.0G`: `92.41015625G`
- [x] Average smoke step time recorded: `3.093231681113442s`
- [x] Estimated steps in 58 minutes recorded: `1125`

## Launch Gate

- [x] No conflicting active tmux training session
- [x] Training launched
- [x] Startup metrics writing through at least `step=40`

## Result Gate

- [x] Training finished cleanly: `step=2500`
- [x] Checkpoint endpoint verified: `final.pt`, `latest.pt`, and `step_002500.pt` share identical weights
- [x] Same-protocol official test completed: `loss_mean=0.03899502913157145`, `psnr_mean=25.281006597978177`
- [x] Task-specific diagnostics generated: `experiments/V1/runs/A_aq03/metrics/task_specific_test_final.json`
- [x] Visual/auxiliary panels generated: `experiments/V1/runs/A_aq03/diagnostics/mechanism_final_test`

## Final Decision

Completed. This branch directly tested whether explicit low/high responsibility splitting gives the Gaussian control canvas a cleaner structure input and a more useful high-detail feedback target. The result is positive but not promotion-grade: official PSNR improved to `25.281006597978177`, yet the Gaussian control canvas stayed nearly flat and the run remains below `A_ah03`, `A_z03`, and frozen `M04/A_y03`.
