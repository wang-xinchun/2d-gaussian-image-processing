# Result Analysis: V1/T01/A_aw03

Status: completed, evaluated, not promotion-grade.

## What Was Tested

`A_aw03` tested the direct bottleneck exposed by `A_av03`: demand-budget selectivity fixed feedback saturation, but it suppressed correction too much. The change was target-mean calibration:

- keep `feedback_budget=0.45`
- add `feedback_target_mean=0.45`
- cap the boost with `feedback_mean_boost_cap=2.0`

This was meant to test whether Gaussian-guided reinjection can be both selective and strong enough.

## Official Result

Same-protocol UHD_deblur/test300 full-image evaluation:

- `samples=300`
- `crop_size=0`
- `checkpoint_step=2500`
- `loss_mean=0.03877821452294787`
- `psnr_mean=25.380666832181568`
- `auxiliary_loss_mean=0.006292636484528581`
- `total_loss_mean=0.04507085102920731`

Comparison:

- Beats `A_av03=25.32983078774204`
- Beats `A_au03=25.361614657535245`
- Still below `A_z03=25.385710820365173` by about `0.00504 dB`
- Far below frozen `M04/A_y03=26.272695829299355`

## Task-Specific Diagnostics

On the stable test `limit=20` subset:

- `psnr_mean=23.18663806081702`
- `ssim_mean=0.6963050037622451`
- `edge_magnitude_l1_mean=0.13199830800294876`
- `high_frequency_l1_mean=0.017245512548834087`
- `local_contrast_l1_mean=0.021899627521634102`
- `luminance_l1_mean=0.04201342137530446`
- `chroma_l1_mean=0.005635173711925745`
- `saturation_l1_mean=0.010972261475399137`
- `dark_channel_l1_mean=0.04317469717934728`

This is stronger than both `A_au03` and `A_av03` on the paired PSNR/SSIM subset. It supports the interpretation that the calibration recovered useful correction strength.

## Mechanism Diagnosis

Mechanism panels were exported to:

`experiments/V1/runs/A_aw03/diagnostics/mechanism_final_test`

Four-panel averages:

- `control_feedback_map mean≈0.42420`, `std≈0.23441`
- `control_canvas mean≈0.45155`, `std≈0.01511`
- `control_canvas_target std≈0.18005`
- `attention_map std≈0.00017`
- panel mean `restored_psnr=23.98870875`, `delta_psnr=1.56463175`

The feedback map is finally in the intended middle regime: neither saturated nor underpowered. The remaining bottleneck is that the primary Gaussian attention map is still near constant. The model has learned a useful auxiliary demand canvas, but the original Gaussian heatmap still does not carry the main restoration-control story.

## Verdict

`A_aw03` is directionally correct and validates target-mean calibrated selective feedback, but it is not a breakthrough. The next short probe should stop tuning only the feedback mean and instead force the main Gaussian heatmap into the restoration-demand loop. This is why `A_ax03` changes the demand-budget source to predicted `control_canvas` and adds a small `gaussian_attention_weight=0.02` auxiliary alignment term.
