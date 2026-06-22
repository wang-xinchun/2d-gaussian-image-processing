# Run Manifest: V1/T01/A_ay03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_ay03`
- Full identifier: `V1/T01/A_ay03`

## Current Status

Completed and evaluated. Not a promotion candidate.

## Purpose

`A_ax03` showed that a weak direct L1 loss on the rendered Gaussian `attention_map` does not solve the heatmap bypass. Official fidelity fell to `psnr_mean=25.233892769220702`, and the final `attention_map std` dropped to only about `3.6e-05` to `4.2e-05`, below `A_aw03≈0.00017`.

`A_ay03` keeps the best calibrated feedback recipe from `A_aw03`, but replaces the failed weak attention supervision with a stronger forward-path mechanism: a default-off Gaussian attention demand bridge.

## Mechanism

- Enable `use_gaussian_attention_demand_bridge = true`.
- Predict a HR `attention_bridge_map` from high-resolution features, the raw rendered Gaussian attention, local attention contrast, and input structure.
- Blend it into the primary `attention_map` with `gaussian_attention_demand_bridge_strength = 0.30`.
- Supervise the bridge map with `attention_bridge_weight = 0.04` against the restoration-demand target.
- Keep `gaussian_attention_weight = 0.0`, because `A_ax03` showed weak endpoint L1 on rendered attention is not effective.
- Keep `A_aw03` calibrated selective additive feedback with target-sourced demand: `feedback_budget=0.45`, `feedback_target_mean=0.45`, `feedback_mean_boost_cap=2.0`.
- Use launch geometry `batch_size=6`, `train_context_size=624`, supervision crop `512`, after `6x640` smoke was slightly over the reserved VRAM gate.

## Baselines

- Frozen medium baseline: `M04/A_y03`, official `psnr_mean=26.272695829299355`
- Current short leader: `A_z03`, official `psnr_mean=25.385710820365173`
- `A_aw03`: official `25.380666832181568`, best demand-control run so far, calibrated feedback but flat main attention
- `A_ax03`: official `25.233892769220702`, failed to activate main attention and degraded fidelity

## Gate Evidence

- Static py_compile passed for touched code.
- CPU forward/backward passed; raw attention remained near-flat while bridge/effective attention became structured (`raw_attention_std≈2.84e-05`, `attention_bridge_std≈0.00273`, effective `attention_std≈0.000818`).
- Formal CUDA preflight at `6x624` passed: `total_params=408458`, `val_psnr=20.264334175626225`, `loss_total=0.22732174396514893`, `loss_control_canvas=0.17585372924804688`, `loss_attention_bridge=0.16498465836048126`.
- True smoke at `6x624` passed: `max_memory_reserved_gb=92.4921875`, `max_memory_allocated_gb=89.41702651977539`, `avg_total_step_seconds=1.9396179039031267`, `estimated_steps_target_wall_clock=1794`, `within_target_reserved_memory_window=true`.

## Launch Evidence

- Tmux session: `2dgauss_a_ay03_uhddeblur_attention_demand_bridge_probe`
- Train log: `experiments/V1/runs/A_ay03/logs/train_20260520_180410.log`
- Startup train metrics reached at least `step=130`.
- GPU during startup: about `95419 MiB`, `100%` utilization.
- Early metrics include `loss_attention_bridge` and `loss_control_canvas`; no early NaN/OOM observed.

## Acceptance Evidence

- Training finished cleanly at `step=2500`.
- Final validation degraded late: `step=500 val_psnr=28.675151131465366`, `step=1000 val_psnr=28.72393012958687`, `step=1500 val_psnr=28.658803866275264`, `step=2000 val_psnr=28.432735276230538`, `step=2500 val_psnr=27.634088171325992`.
- Official final full-image test: `loss_mean=0.040016187620349225`, `psnr_mean=24.998189217540585`, `auxiliary_loss_mean=0.011719125838329394`, `total_loss_mean=0.05173531337330739`, `samples=300`, `checkpoint_step=2500`.
- Task-specific limit20 diagnostics: `psnr_mean=22.975469413149693`, `ssim_mean=0.6872823059558868`, `edge_magnitude_l1_mean=0.13152226880192758`, `high_frequency_l1_mean=0.017945018922910096`, `local_contrast_l1_mean=0.021368088200688362`, `luminance_l1_mean=0.04324529198929668`, `chroma_l1_mean=0.005130211869254708`, `saturation_l1_mean=0.009870028588920832`, `dark_channel_l1_mean=0.039017872884869576`.
- Diagnostic `step_001000.pt` official test: `psnr_mean=25.062951734772415`, still below `A_ax03=25.233892769220702`, so early checkpoint selection does not explain away the failure.
- Mechanism panels: `raw_attention_map std≈0.000337-0.000341`, `attention_bridge_map std≈0.0738-0.0742`, effective `attention_map std≈0.0221-0.0223`, `control_feedback_map mean≈0.416-0.417`, `std≈0.214`.

## Decision Rule

`A_ay03` is a negative diagnostic. It succeeded at making the effective attention map structured, but failed the accepted restoration test: final official PSNR fell below `A_ax03`, `A_aw03`, and `A_z03`, and the `step_001000.pt` diagnostic remained far below the competitive band. The result says the bridge should not be applied to the whole main attention path; the next test must protect fidelity-sensitive backbone paths and restrict demand-bridged attention to the control/feedback route.
