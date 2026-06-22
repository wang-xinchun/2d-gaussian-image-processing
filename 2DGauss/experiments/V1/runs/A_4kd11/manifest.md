# Run Manifest: V1/T01/A_4kd11

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_4kd11`
- Full identifier: `V1/T01/A_4kd11`

## Purpose

`A_4kd11` is a bottleneck-targeted continuation from `A_4kd08` best-val
checkpoint. It avoids the reliability-gate family that suppressed useful
guidance in `A_4kd09/A_4kd10`, and instead tests whether a low-LR optimizer
stabilizer plus light dehazing-side guards can preserve or improve the strong
`A_4kd08` fast-probe point.

## Narrow Variable

Compared with `A_4kd08`, this run:

- strictly loads `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`;
- keeps medium detail-preserve shared demand-control unchanged;
- keeps reliability gating disabled;
- changes LR from `2e-4` to `3e-5`;
- runs only `300` continuation steps;
- adds `saturation_weight=0.004`;
- changes `dark_channel_weight` from `0.01` to `0.012`;
- enables linear LR decay from progress `0.50` to `1.00`, final factor `0.25`;
- enables `grad_clip_norm=1.0`.

## Candidate Rule

This run can supersede `A_4kd08` only if CPU fast-probe test PSNR/SSIM remain at
or above the `A_4kd08` level while dark-channel/saturation diagnostics improve,
or if it provides a clearly stronger validation/test tradeoff. Otherwise it is
only a continuation diagnostic.

## GPU Boundary

Only physical GPU0/GPU3 are allowed, defaulting to GPU3. GPU1/GPU2 are
forbidden. Do not kill or displace external processes.

## Result Summary

- Formal GPU3 preflight passed, then the first formal launch failed before step1 due to a local `base_lrs` scope bug in the new LR schedule patch. The failed log is retained as audit evidence.
- After patching `scripts/train_v1_t01.py`, formal training completed `300` steps on physical GPU3 only.
- Best validation: step `200`, `val_psnr=18.456089733874084`, `val_loss=0.11976054381579161`.
- CPU test: `psnr_mean=17.36895205643582`, `loss_mean=0.12583659052848817`, `ssim_mean=0.6967320647835732`.
- Conclusion: diagnostic only. It improves validation and task proxies but loses too much CPU test PSNR relative to `A_4kd08`.
