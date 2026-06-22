# Result Analysis: V1/T01/A_au03

Status: completed, evaluated, diagnostic-positive but not a promotion candidate.

## Training Endpoint

- Train log: `experiments/V1/runs/A_au03/logs/train_20260520_104916.log`
- Final checkpoint: `experiments/V1/runs/A_au03/checkpoints/final.pt`
- Finished step: `2500`
- Final validation: `val_loss=0.025665613915771247`, `val_psnr=28.79189041625805`

## Official Test

Same-protocol UHD_deblur/test300 full-image evaluation:

- File: `experiments/V1/runs/A_au03/metrics/official_test_final.json`
- `samples=300`
- `crop_size=0`
- `checkpoint_step=2500`
- `loss_mean=0.039005936284859974`
- `psnr_mean=25.361614657535245`
- `auxiliary_loss_mean=0.006336619006469845`
- `total_loss_mean=0.04534255526339014`
- `elapsed_seconds=458.27933716773987`

Comparison:

- `A_ar03`: `psnr_mean=25.104431332416773`
- `A_as03`: `psnr_mean=25.19097600105495`
- `A_aq03`: `psnr_mean=25.281006597978177`
- `A_z03`: `psnr_mean=25.385710820365173`
- Frozen `M04/A_y03`: `psnr_mean=26.272695829299355`

Interpretation: `A_au03` successfully recovers fidelity relative to the supervised-canvas and closed-loop-canvas probes and slightly beats `A_aq03`, but it does not surpass the current short leader `A_z03` or the frozen medium milestone `M04/A_y03`.

## Task-Specific Diagnostics

Stable test subset (`limit=20`):

- File: `experiments/V1/runs/A_au03/metrics/task_specific_test_limit20.json`
- `psnr_mean=23.115990186771846`
- `ssim_mean=0.692804542183876`
- `edge_magnitude_l1_mean=0.13188842125236988`
- `high_frequency_l1_mean=0.017282098811119796`
- `local_contrast_l1_mean=0.022034568060189486`
- `luminance_l1_mean=0.04261171650141478`
- `chroma_l1_mean=0.0056723413523286585`
- `saturation_l1_mean=0.010556848114356398`
- `dark_channel_l1_mean=0.04656200259923935`

Relative to `A_as03` on the same 20-sample convention, `A_au03` improves PSNR, SSIM, edge magnitude L1, high-frequency L1, luminance L1, and chroma L1. Local contrast and dark-channel diagnostics remain effectively tied/slightly worse, so this should be read as a deblur fidelity/detail improvement rather than a dehaze-quality claim.

## Mechanism Panels

- Folder: `experiments/V1/runs/A_au03/diagnostics/mechanism_final_test`
- Samples: `4`
- Panel mean: `restored_psnr=23.67263975`, `delta_psnr=1.248563`, `restored_ssim=0.72268`, `delta_ssim=0.03752425`
- Average `control_canvas std≈0.01699`
- Average `control_canvas_target std≈0.18005`
- Average `control_feedback_map mean≈0.98474`
- Average `control_feedback_map std≈0.01373`
- Average `attention_map std≈0.00024`

Visual/mechanism read:

- The residual-demand target is meaningful and spatially structured.
- The learned control canvas is more structured than `A_as03` (`std≈0.01699` vs `≈0.01112`).
- The mechanism panel gives a clearer story: low/high split separates coarse image content from high-frequency detail; demand canvas highlights restoration-needed regions; residual output/error panels show where correction is being applied.
- The main Gaussian `attention_map` remains nearly flat, so the interpretable control burden is still mostly carried by the auxiliary canvas rather than the original Gaussian heatmap.
- The feedback map is spatially non-flat but almost saturated (`mean≈0.98474`), which means the bounded additive回流 is still too close to a global correction gate.

## Judgment

`A_au03` is the best result in the supervised/demand-control short-probe family so far. It validates the root-cause correction that generic structure supervision was the wrong target and that residual-demand supervision with protected additive回流 is more faithful.

It is not enough for promotion. The run remains below `A_z03` by about `0.0241` PSNR and below frozen `M04/A_y03` by about `0.9111` PSNR. More importantly, the mechanism bottleneck has moved: the demand canvas is alive, but the feedback gate is too open and the main Gaussian attention is still flat.

Next recommended mechanism: keep `restoration_demand` supervision, but constrain回流 selectivity with a centered/sparse demand gate, residual budget, top-k or percentile-normalized demand, or a separate identity-protection penalty so that already-correct regions are not globally rewritten.
