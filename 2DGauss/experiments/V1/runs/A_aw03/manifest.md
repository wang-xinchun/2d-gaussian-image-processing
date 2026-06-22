# Run Manifest: V1/T01/A_aw03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_aw03`
- Full identifier: `V1/T01/A_aw03`

## Current Status

Completed and accepted as a diagnostic run. It is not a promotion candidate.

## Purpose

`A_aw03` was the direct follow-up to `A_av03`. `A_av03` fixed the `A_au03` feedback-saturation failure, dropping `control_feedback_map mean` from about `0.98474` to about `0.2415`, but its correction became too weak and fidelity fell. `A_aw03` tested whether target-mean calibration could keep demand-budget selectivity while restoring enough correction energy.

## Mechanism

This run kept `A_av03` restoration-demand canvas and demand-budget selective additive feedback, with:

- `gaussian_control_canvas_feedback_budget = 0.45`
- `gaussian_control_canvas_feedback_target_mean = 0.45`
- `gaussian_control_canvas_feedback_mean_boost_cap = 2.0`
- `gaussian_control_canvas_residual_delta_strength = 0.80`

The intended target was a middle regime: more selective than `A_au03`, stronger than `A_av03`.

## Training Endpoint

- Train log: `experiments/V1/runs/A_aw03/logs/train_20260520_141523.log`
- Finished at `step=2500`
- Final validation: `val_loss=0.024846228305250406`, `val_psnr=29.035750230659232`
- Checkpoints: `final.pt`, `latest.pt`, and `step_002500.pt` exist

## Final Evidence

- Official same-protocol UHD_deblur/test300: `psnr_mean=25.380666832181568`, `loss_mean=0.03877821452294787`, `auxiliary_loss_mean=0.006292636484528581`, `total_loss_mean=0.04507085102920731`
- Task-specific diagnostics on test `limit=20`: `psnr_mean=23.18663806081702`, `ssim_mean=0.6963050037622451`, `high_frequency_l1_mean=0.017245512548834087`, `luminance_l1_mean=0.04201342137530446`, `dark_channel_l1_mean=0.04317469717934728`
- Mechanism panels: `experiments/V1/runs/A_aw03/diagnostics/mechanism_final_test`
- Mechanism averages from four panels: `control_feedback_map mean≈0.42420`, `std≈0.23441`; `control_canvas std≈0.01511`; `control_canvas_target std≈0.18005`; `attention_map std≈0.00017`

## Judgment

`A_aw03` is a positive mechanism diagnostic: target-mean calibration recovered correction strength and beat both `A_av03=25.32983078774204` and `A_au03=25.361614657535245`. However, it still narrowly trails the short leader `A_z03=25.385710820365173` by about `0.00504 dB`, and it remains far below frozen `M04/A_y03=26.272695829299355`.

The key residual bottleneck is that the main Gaussian `attention_map` is still nearly constant, so the restoration-demand control role is mostly handled by the auxiliary `control_canvas/feedback_map` rather than the primary Gaussian heatmap. This result motivated `A_ax03`, which tests canvas-sourced feedback budget plus direct low-weight attention-demand alignment.
