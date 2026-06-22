# Run Manifest: V1/T01/A_ax03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_ax03`
- Full identifier: `V1/T01/A_ax03`

## Current Status

Completed and accepted as a negative diagnostic run. It is not a promotion candidate.

## Purpose

`A_aw03` recovered the correction strength lost by `A_av03`, but it still did not beat the short leader `A_z03`, and mechanism panels showed the main Gaussian `attention_map` remained nearly flat. `A_ax03` is the next stronger mechanism probe: it tries to make the predicted restoration-demand canvas and the primary Gaussian heatmap participate more directly in feedback, rather than continuing to tune only the mean feedback gate.

## Mechanism

`A_ax03` inherits the `A_aw03` architecture and geometry, with two targeted changes:

- `gaussian_control_canvas_feedback_demand_source = "canvas"`: demand-budget feedback is driven by the predicted `control_canvas`, not the input-derived fallback target.
- `gaussian_attention_weight = 0.02`: a small auxiliary L1 alignment term pulls the main Gaussian `attention_map` toward the restoration-demand target.

The hypothesis is that the feedback map should stay in the non-saturated calibrated regime while the primary Gaussian heatmap gains meaningful spatial structure.

## Baselines

- Frozen medium baseline: `M04/A_y03`, official `psnr_mean=26.272695829299355`
- Current short leader: `A_z03`, official `psnr_mean=25.385710820365173`
- `A_au03`: official `25.361614657535245`, saturated feedback (`mean≈0.98474`)
- `A_av03`: official `25.32983078774204`, over-shrunk feedback (`mean≈0.2415`)
- `A_aw03`: official `25.380666832181568`, calibrated feedback (`mean≈0.42420`) but flat main attention (`std≈0.00017`)

## Gate Evidence

- Static py_compile passed for touched code.
- CPU small-tensor forward/backward passed with both `loss_control_canvas` and `loss_gaussian_attention`.
- Formal CUDA preflight at `6x640` passed: `total_params=374761`, `val_psnr=18.752572677892744`, `loss_total=0.21266134083271027`, `loss_control_canvas=0.19127671420574188`, `loss_gaussian_attention=0.19207045435905457`.
- True smoke at `6x640` passed: `max_memory_reserved_gb=91.947265625`, `max_memory_allocated_gb=88.66405534744263`, `avg_total_step_seconds=1.836618307357033`, `estimated_steps_target_wall_clock=1894`, `within_target_reserved_memory_window=true`.

## Launch Evidence

- Tmux session: `2dgauss_a_ax03_uhddeblur_canvas_budget_attention_aligned_feedback_probe`
- Train log: `experiments/V1/runs/A_ax03/logs/train_20260520_151434.log`
- Startup metrics reached at least `step=60`
- GPU during startup: about `94861 MiB`, `100%` utilization
- No early NaN/OOM observed

## Training Endpoint

- Finished cleanly at `step=2500`
- Final validation: `val_loss=0.026070117484778166`, `val_psnr=28.688878115482986`
- Checkpoints: `final.pt`, `latest.pt`, and `step_002500.pt` exist

## Final Evidence

- Official same-protocol UHD_deblur/test300: `psnr_mean=25.233892769220702`, `loss_mean=0.03905327895656228`, `auxiliary_loss_mean=0.009553199315754076`, `total_loss_mean=0.0486064781807363`
- Task-specific diagnostics on test `limit=20`: `psnr_mean=23.12751232307337`, `ssim_mean=0.6937903940677643`, `edge_magnitude_l1_mean=0.1309316113591194`, `high_frequency_l1_mean=0.017599273147061466`, `local_contrast_l1_mean=0.02132912538945675`, `luminance_l1_mean=0.04230853011831641`, `chroma_l1_mean=0.005110498261637985`, `saturation_l1_mean=0.010020384145900608`, `dark_channel_l1_mean=0.04433827875182032`
- Mechanism panels: `experiments/V1/runs/A_ax03/diagnostics/mechanism_final_test`
- Mechanism evidence from four panels: `attention_map std≈3.6e-05` to `4.2e-05`, `control_canvas std≈0.0135` to `0.0138`, `control_canvas_target std≈0.1788` to `0.1815`, `control_feedback_map mean≈0.443`, `std≈0.159`

## Final Judgment

`A_ax03` failed the breakthrough test. It preserved non-saturated feedback, but it did not make the main Gaussian heatmap carry restoration demand. The effective `attention_map` became even flatter than `A_aw03`, and official fidelity fell well below `A_aw03=25.380666832181568` and `A_z03=25.385710820365173`.

The next mechanism should not simply increase `gaussian_attention_weight`. The evidence says weak endpoint L1 supervision on the rendered attention is a poor control handle. A stronger forward-path bridge is needed if the effective Gaussian heatmap is to become physically meaningful.

## Decision Rule

`A_ax03` did not satisfy its decision rule and must remain a negative diagnostic reference.
