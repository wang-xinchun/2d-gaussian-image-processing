# A_dsh17 Result Analysis

更新时间: 2026-06-06 17:45:15 CST

## Summary

A_dsh17 completed the context-guided GPU3-only training safely after A_dsh16 failed with a project-local CUDA OOM at `batch_size=4`. A_dsh17 reduced `batch_size` to `2`, kept the same context-guided model line and verified BVI-AOM synthetic paired-frame data, and trained to `1600` steps with `train_exit_code=0`.

This is a successful deployment and safety result. It is not a new model baseline: under the same independent 32-pair validation eval, A_dsh17 best-val is slightly worse than A_dsh15 final.

## Safety

- Real training used physical GPU3 only through `CUDA_VISIBLE_DEVICES=3`.
- GPU0 was avoided while external work was present; GPU1/2 were not used.
- A_dsh17's first dry-run correctly refused to launch when a new GPU3 process appeared.
- No external process was killed or displaced.
- Final cleanup found no A_dsh17 training/tmux residue and GPU3 returned to `0MiB` used.

## Training Metrics

- Train steps: `1600`
- Train metric records: `321`
- Validation records: `32`
- Checkpoints: `best_val.pt`, `final.pt`, `latest.pt`, and step checkpoints through `step_001600.pt`
- Best validation loss: step `1350`, `val_loss=0.09807008923962712`, `val_psnr=18.343223704028027`
- Final validation: step `1600`, `val_loss=0.09814217837993056`, `val_psnr=18.34113527708859`

## Independent Eval

Same eval script and validation view as A_dsh15 were used:

- Final checkpoint: `prediction_psnr_mean=18.34113541614135`, `psnr_gain_mean=1.8340331879357459`, `prediction_ssim_mean=0.4930315390229225`, `ssim_gain_mean=0.11407482577487826`
- Best-val checkpoint: `prediction_psnr_mean=18.343223705783423`, `psnr_gain_mean=1.836121477577819`, `prediction_ssim_mean=0.4939711121842265`, `ssim_gain_mean=0.11501439893618226`
- A_dsh15 final eval remains stronger: `prediction_psnr_mean=18.504268639458388`, `prediction_ssim_mean=0.5011140927672386`

Relative to A_dsh15 final, A_dsh17 best-val is `-0.1610449336749653 dB` PSNR and `-0.007142980583012104` SSIM, with `+0.0019246686715632677` higher prediction L1.

## Decision

A_dsh17 is accepted as a correctly deployed, safely isolated experiment. It is not promoted as the current best model. A_dsh15 remains the accepted paired-frame deshaking baseline.
