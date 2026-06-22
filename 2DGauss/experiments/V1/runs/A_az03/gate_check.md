# Gate Check: V1/T01/A_az03

## Branch Identity

- Run: `V1/T01/A_az03`
- Config: `experiments/V1/runs/A_az03/configs/a_az03_short_uhddeblur_control_only_attention_demand_bridge_probe.toml`
- Mechanism family: control-only Gaussian attention demand bridge plus calibrated selective feedback

## Static Gate

- [x] Config exists
- [x] Python compile/import check passed
- [x] CPU forward/backward check passed
- [x] Control-only attention routing verified
- [x] Parameter count checked: `408458`

## Smoke / Preflight Gate

- [x] Formal preflight passed at selected geometry (`6x624`, supervision crop `512`)
- [x] True smoke calibration passed
- [x] Reserved VRAM inside `85.0-93.0G`: `92.4921875`
- [x] Average smoke step time recorded: `1.3882675596202414`
- [x] Estimated steps in 58 minutes recorded: `2506`

## Launch Gate

- [x] No conflicting active training/evaluation job before launch
- [x] Training launched in tmux
- [x] Startup metrics writing through initial steps

## Result Gate

- [x] Training finished cleanly
- [x] Checkpoint endpoint verified
- [x] Same-protocol official test completed
- [x] Task-specific diagnostics generated
- [x] Mechanism panels generated

## Pre-Launch Hypothesis

`A_ay03` failed because the bridge was applied to the whole Gaussian attention path. `A_az03` tests a tighter version: let the bridge provide restoration-demand structure only to the control canvas/feedback path, while preserving raw attention for the main fidelity-sensitive restoration backbone.

## Gate Summary

- CPU sanity: `attention_std=raw_attention_std≈3.647e-05`, `control_attention_std≈0.0009169`, `bridge_std≈0.003054`, `main_raw_l1=0.0`, `control_raw_l1≈0.01177`.
- Formal CUDA preflight: `val_psnr=18.975287419921493`, `loss_total=0.13312028348445892`, `loss_control_canvas=0.20398859679698944`, `loss_attention_bridge=0.19957368075847626`.
- True smoke: `max_memory_allocated_gb=89.4190526008606`, `max_memory_reserved_gb=92.4921875`, `within_target_reserved_memory_window=true`.
- Launch: tmux session `2dgauss_a_az03_uhddeblur_control_only_attention_demand_bridge_probe`, log `experiments/V1/runs/A_az03/logs/train_20260520_225643.log`, startup metrics through at least `step=130`, GPU about `95419 MiB/100%`.
- Endpoint: no active `A_az03` training process at acceptance time, GPU idle, `final.pt/latest.pt/step_002500.pt` present, final train step `2500`.
- Final validation: `val_loss=0.025261803343892098`, `val_psnr=28.92787912557181`.
- Official test: `experiments/V1/runs/A_az03/metrics/official_test_final.json`, `psnr_mean=25.30935139353653`, `loss_mean=0.03883044661798825`.
- Task-specific diagnostics: `experiments/V1/runs/A_az03/metrics/task_specific_test_limit20.json`, `psnr_mean=23.132057715603047`, `ssim_mean=0.6942389011383057`.
- Mechanism panels: `experiments/V1/runs/A_az03/diagnostics/mechanism_final_test`; `attention_bridge_map std≈0.070`, `control_attention_map std≈0.021`, raw/main `attention_map std≈0.00066`, `control_canvas std≈0.055`, `control_feedback_map mean≈0.423`, `std≈0.241`.
- Verdict: gate completed; run is on-topic and partially recovers `A_ay03`, but it remains below `A_aw03/A_z03/M04`, so it is not a promotion candidate.
