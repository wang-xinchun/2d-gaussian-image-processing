# Run Manifest: V1/T01/A_az03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_az03`
- Full identifier: `V1/T01/A_az03`

## Current Status

Completed and evaluated. `A_az03` is a useful partial-recovery diagnostic, but not a promotion candidate.

## Purpose

`A_ay03` proved the demand bridge can make the effective attention map highly structured, but the same bridged attention was fed into every downstream Gaussian-controlled path and official fidelity collapsed to `psnr_mean=24.998189217540585`.

`A_az03` keeps the demand bridge as an interpretable control signal, but protects the main restoration backbone: raw Gaussian attention remains the primary attention for high-frequency split, highpass modulation, decoder guidance, feature refinement, terminal residual, and edge refinement. The bridged attention is used only by the Gaussian control canvas/feedback module.

## Mechanism

- Enable `use_gaussian_attention_demand_bridge = true`.
- Set `gaussian_attention_demand_bridge_apply_to = "control"`.
- Keep `gaussian_attention_demand_bridge_strength = 0.30` and `attention_bridge_weight = 0.04`.
- Keep `A_aw03` target-sourced calibrated selective additive feedback: `feedback_budget=0.45`, `feedback_target_mean=0.45`, `feedback_mean_boost_cap=2.0`.
- Keep `gaussian_attention_weight = 0.0`.
- Use the `A_ay03` accepted geometry: `batch_size=6`, `train_context_size=624`, supervision crop `512`.

## Baselines

- Frozen medium baseline: `M04/A_y03`, official `psnr_mean=26.272695829299355`
- Current short leader: `A_z03`, official `psnr_mean=25.385710820365173`
- `A_aw03`: official `25.380666832181568`, best calibrated feedback run
- `A_ax03`: official `25.233892769220702`, failed weak attention supervision
- `A_ay03`: official `24.998189217540585`, demand bridge activated attention but damaged fidelity

## Decision Rule

`A_az03` is useful only if it recovers at least above `A_ax03` and ideally approaches `A_aw03/A_z03`, while mechanism panels show raw main attention remains stable and the control-only bridge gives the feedback/canvas a structured demand signal without feedback saturation.

## Gate Evidence

- Static py_compile passed for touched code.
- CPU forward/backward passed and verified control-only routing: `attention_std=raw_attention_std≈3.647e-05`, `control_attention_std≈0.0009169`, `bridge_std≈0.003054`, `main_raw_l1=0.0`, `control_raw_l1≈0.01177`.
- Formal CUDA preflight passed at `6x624`: `total_params=408458`, `val_psnr=18.975287419921493`, `loss_total=0.13312028348445892`, `loss_control_canvas=0.20398859679698944`, `loss_attention_bridge=0.19957368075847626`.
- True smoke passed at `6x624`: `max_memory_reserved_gb=92.4921875`, `max_memory_allocated_gb=89.4190526008606`, `avg_total_step_seconds=1.3882675596202414`, `estimated_steps_target_wall_clock=2506`, `within_target_reserved_memory_window=true`.

## Launch Evidence

- Tmux session: `2dgauss_a_az03_uhddeblur_control_only_attention_demand_bridge_probe`
- Train log: `experiments/V1/runs/A_az03/logs/train_20260520_225643.log`
- Startup metrics reached at least `step=130`.
- GPU during startup: about `95419 MiB`, `100%` utilization.
- Startup log contains the inherited `OMP_NUM_THREADS=0` warning from the environment, but JSON metrics are writing normally; no early NaN/OOM observed.

## Training Endpoint

- Training finished cleanly at `step=2500`.
- Final validation: `val_loss=0.025261803343892098`, `val_psnr=28.92787912557181`.
- Final checkpoint: `experiments/V1/runs/A_az03/checkpoints/final.pt`.
- No active `A_az03` / `train_v1_t01.py` process was present at acceptance time, and GPU was idle before evaluation.

## Result Evidence

- Official full-image UHD_deblur/test300: `loss_mean=0.03883044661798825`, `psnr_mean=25.30935139353653`, `auxiliary_loss_mean=0.011713128695264459`, `total_loss_mean=0.050543575460712115`, `samples=300`, `crop_size=0`.
- Task-specific diagnostics on the stable `limit=20` subset: `psnr_mean=23.132057715603047`, `ssim_mean=0.6942389011383057`, `edge_magnitude_l1_mean=0.12970237173140048`, `high_frequency_l1_mean=0.0174344873521477`, `local_contrast_l1_mean=0.021073565259575842`, `luminance_l1_mean=0.04221528498455882`, `chroma_l1_mean=0.005514730582945049`, `saturation_l1_mean=0.010305101098492742`, `dark_channel_l1_mean=0.04557064678519964`.
- Mechanism panels: `experiments/V1/runs/A_az03/diagnostics/mechanism_final_test`.
- Mechanism diagnosis on the four exported samples: raw/main `attention_map` stayed protected and near-flat (`std≈0.00066`), `attention_bridge_map` remained structured (`std≈0.070`), `control_attention_map` carried demand structure (`std≈0.021`), `control_canvas` became much more structured (`std≈0.055`), and feedback stayed in the non-saturated calibrated regime (`control_feedback_map mean≈0.423`, `std≈0.241`).

## Final Judgment

`A_az03` fixes the worst `A_ay03` failure mode: control-only routing recovers official PSNR from `24.998189217540585` to `25.30935139353653` and restores task-specific limit20 PSNR from `22.975469413149693` to `23.132057715603047`.

It still does not break through. It beats `A_ax03=25.233892769220702`, but remains below `A_aw03=25.380666832181568`, `A_z03=25.385710820365173`, and frozen `M04/A_y03=26.272695829299355`. The bottleneck has shifted from over-invasive bridge damage to insufficient conversion of the structured control-only demand signal into effective restoration gain.
