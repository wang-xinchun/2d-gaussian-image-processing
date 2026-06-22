# Run Manifest: V1/T01/A_au03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_au03`
- Full identifier: `V1/T01/A_au03`

## Current Status

Completed and evaluated.

Training reached `step=2500` cleanly and wrote `checkpoints/final.pt`, `latest.pt`, and `step_002500.pt`. The acceptance chain followed the training completion immediately: same-protocol official test, `limit=20` task-specific diagnostics, and test mechanism panels have all been generated.

## Purpose

`A_au03` is the post-`A_as03` bottleneck-breaking short probe. It responds to the root-cause finding that structured Gaussian control canvas alone is not enough: the canvas must represent restoration demand, and feedback must preserve fidelity.

## Mechanism

Compared with `A_as03`, this run keeps the mature deblur ingredients but changes the control-canvas mechanism:

- Canvas supervision target changes from generic structure to `restoration_demand`, computed from input-vs-GT structure, high-frequency, and color residuals.
- Gaussian control feedback changes from multiplicative feature scaling to bounded additive feature correction.
- Feedback uses confidence normalization so the demand gate should stay selective rather than becoming a broad global enhancer.
- Feedback strength is reduced to `0.12` with `residual_delta_strength=0.80` to protect the identity path.

This run uses config: `experiments/V1/runs/A_au03/configs/a_au03_short_uhddeblur_residual_demand_protected_feedback_probe.toml`.

## Baselines

- Frozen medium baseline: `M04/A_y03`, official test `26.272695829299355`
- Current short leader: `A_z03`, official test `25.385710820365173`
- Latest split probe: `A_aq03`, official test `25.281006597978177`, attention std `≈0.00017`
- Latest closed-loop canvas probe: `A_as03`, official test `25.19097600105495`, `control_canvas std≈0.01112`, `control_feedback_map std≈0.00724`, `attention_map std≈0.00021`

## Gate Status

- [x] Static py_compile
- [x] Config parse/model forward
- [x] Formal preflight at accepted geometry
- [x] True smoke calibration inside the memory window
- [x] Launch only if gates pass and no conflicting active training is present
- [x] Training finished cleanly at `step=2500`
- [x] Same-protocol official test completed
- [x] Task-specific diagnostics generated
- [x] Mechanism panels generated

## Final Evidence

- Training log: `experiments/V1/runs/A_au03/logs/train_20260520_104916.log`
- Final checkpoint: `experiments/V1/runs/A_au03/checkpoints/final.pt`
- Official test: `experiments/V1/runs/A_au03/metrics/official_test_final.json`
- Task-specific diagnostics: `experiments/V1/runs/A_au03/metrics/task_specific_test_limit20.json`
- Mechanism panels: `experiments/V1/runs/A_au03/diagnostics/mechanism_final_test`

Official UHD_deblur/test300 full-image result:

- `samples=300`
- `crop_size=0`
- `checkpoint_step=2500`
- `loss_mean=0.039005936284859974`
- `psnr_mean=25.361614657535245`
- `auxiliary_loss_mean=0.006336619006469845`
- `total_loss_mean=0.04534255526339014`

Task-specific diagnostics on the stable `limit=20` subset:

- `psnr_mean=23.115990186771846`
- `ssim_mean=0.692804542183876`
- `edge_magnitude_l1_mean=0.13188842125236988`
- `high_frequency_l1_mean=0.017282098811119796`
- `local_contrast_l1_mean=0.022034568060189486`
- `luminance_l1_mean=0.04261171650141478`
- `chroma_l1_mean=0.0056723413523286585`
- `saturation_l1_mean=0.010556848114356398`
- `dark_channel_l1_mean=0.04656200259923935`

Mechanism-panel average over four full-image test samples:

- `control_canvas std≈0.01699`
- `control_canvas_target std≈0.18005`
- `control_feedback_map mean≈0.98474`
- `control_feedback_map std≈0.01373`
- `attention_map std≈0.00024`
- Panel mean `restored_psnr=23.67263975`, `delta_psnr=1.248563`

## Decision Rule

`A_au03` is useful only if it improves the structure-fidelity alignment story. Promotion evidence requires same-protocol official test, task-specific diagnostics, and mechanism panels showing a selective demand canvas / feedback map without fidelity collapse, ringing, halo, over-sharpening, checkerboard artifacts, or color shift. Canvas/demand metrics alone are not promotion evidence.

## Final Judgment

`A_au03` is a positive mechanism diagnosis but not yet a promotion candidate. It recovers most of the fidelity lost by `A_ar03/A_as03` and beats `A_aq03` on official PSNR (`25.361614657535245` vs `25.281006597978177`), but it is still slightly below the short leader `A_z03` (`25.385710820365173`) and far below frozen `M04/A_y03` (`26.272695829299355`).

The new residual-demand canvas is directionally useful: the canvas is more structured than `A_as03`, and the stable subset task diagnostics also improve. The remaining bottleneck is feedback selectivity. The `control_feedback_map` is spatially non-flat but has `mean≈0.98474`, which indicates that the additive回流 gate is almost always open. The next mechanism should preserve the restoration-demand target while adding stronger sparsity, centering, or residual budget control so the Gaussian回流 becomes selective rather than a near-global correction.
