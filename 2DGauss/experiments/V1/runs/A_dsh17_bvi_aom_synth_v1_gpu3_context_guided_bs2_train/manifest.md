# A_dsh17 Manifest

- Full id: `V1/T01/A_dsh17_bvi_aom_synth_v1_gpu3_context_guided_bs2_train`
- Status: completed and independently evaluated; safe deployment succeeded, but A_dsh15 remains the accepted baseline because A_dsh17 did not outperform it under the same eval protocol.
- Purpose: deploy the context-guided deshaking branch without displacing external GPU0 training or using GPU1/2.
- Dataset root: `/home/ldj/wxc/4K_8K/BVI-AOM_synth_deshake_v1`
- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/deshake_ddp_0_3_pending/configs/a_dsh17_bvi_aom_synth_v1_gpu3_context_guided_bs2_train.toml`
- Run-local config mirror: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_dsh17_bvi_aom_synth_v1_gpu3_context_guided_bs2_train/configs/a_dsh17_bvi_aom_synth_v1_gpu3_context_guided_bs2_train.toml`
- Training log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/deshake_ddp_0_3_pending/logs/train_deshake_gpu3_train_20260606_173118_784237.log`

## Safety Delta

- Keeps the A_dsh16 structural model line: `width=48`, edge guidance, directional-frequency guidance, train-inference consistency, and Gaussian decoder guidance.
- Lowers single-GPU batch size from `4` to `2` after the A_dsh16 OOM.
- Keeps physical GPU3 only. GPU0 remains avoided while external training is active; GPU1/2 remain disallowed.
- The first A_dsh17 dry-run was correctly blocked when a new process appeared on GPU3; the real launch happened only after GPU3 returned to idle.
- No external process was killed or displaced. Post-run cleanup found no A_dsh17 train/tmux residue and GPU3 returned to `0MiB` used.

## Training Result

- Completed `1600` steps with `train_exit_code=0`.
- Produced `321` train metric records and `32` full-validation records.
- Saved `best_val.pt`, `final.pt`, `latest.pt`, and step checkpoints through `step_001600.pt`.
- Best validation loss was at step `1350`: `val_loss=0.09807008923962712`, `val_psnr=18.343223704028027`.
- Final step `1600`: `val_loss=0.09814217837993056`, `val_psnr=18.34113527708859`.

## Independent Eval

- Final checkpoint: input baseline `16.507102228205603` PSNR / `0.37895671324804425` SSIM; prediction `18.34113541614135` PSNR / `0.4930315390229225` SSIM; gain `+1.8340331879357459 dB` / `+0.11407482577487826` SSIM.
- Best-val checkpoint: prediction `18.343223705783423` PSNR / `0.4939711121842265` SSIM; gain `+1.836121477577819 dB` / `+0.11501439893618226` SSIM.
- Compared with A_dsh15 final eval, A_dsh17 best-val is lower by `0.1610449336749653 dB` PSNR and `0.007142980583012104` SSIM, with slightly higher L1. Therefore this run is recorded as a successful safe deployment but not accepted as the new best baseline.
