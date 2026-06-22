# Gate Check: V1/T01/A_ax03

## Branch Identity

- Run: `V1/T01/A_ax03`
- Config: `experiments/V1/runs/A_ax03/configs/a_ax03_short_uhddeblur_canvas_budget_attention_aligned_feedback_probe.toml`
- Mechanism family: canvas-sourced feedback budget plus low-weight Gaussian attention-demand alignment

## Static Gate

- [x] Config exists
- [x] Python compile/import check passed
- [x] Model forward/auxiliary check passed
- [x] New auxiliary attention loss records correctly
- [x] Parameter count checked: `total_params=374761`

## Smoke / Preflight Gate

- [x] Formal preflight passed at `batch_size=6`, `train_context_size=640`, supervision crop `512`
- [x] True smoke calibration passed
- [x] Reserved VRAM inside `85.0-93.0G`: `max_memory_reserved_gb=91.947265625`
- [x] Average smoke step time recorded: `avg_total_step_seconds=1.836618307357033`
- [x] Estimated steps in 58 minutes recorded: `1894`

## Launch Gate

- [x] No conflicting active training/evaluation job before launch
- [x] Training launched in tmux
- [x] Startup metrics writing through initial steps

## Result Gate

- [x] Training finished cleanly
- [x] Checkpoint endpoint verified
- [x] Same-protocol official test completed
- [x] Task-specific diagnostics generated
- [x] Mechanism panels generated

## Pre-Launch Hypothesis

`A_aw03` proved that calibrated selective feedback is useful, but it left the main Gaussian heatmap nearly constant. `A_ax03` tests whether budget routing through predicted `control_canvas` and a small direct `attention_map` demand-alignment loss can make the primary Gaussian heatmap carry restoration demand without collapsing fidelity.

## Startup Summary

- Preflight: `val_loss=0.09545606002211571`, `val_psnr=18.752572677892744`, `loss_total=0.21266134083271027`.
- True smoke: `status=ok`, `batch_shape=(6, 3, 640, 640)`, `loss=0.08137731999158859`.
- Startup train metrics reached at least `step=60`; the log records both `loss_control_canvas` and `loss_gaussian_attention`.
- Active log: `experiments/V1/runs/A_ax03/logs/train_20260520_151434.log`.

## Acceptance Summary

- Training ended at `step=2500`; final validation `val_loss=0.026070117484778166`, `val_psnr=28.688878115482986`.
- Official test300: `psnr_mean=25.233892769220702`, `loss_mean=0.03905327895656228`.
- Task-specific limit20: `psnr_mean=23.12751232307337`, `ssim_mean=0.6937903940677643`.
- Mechanism panels show feedback stayed non-saturated (`control_feedback_map mean≈0.443`, `std≈0.159`) but the main `attention_map` stayed near constant (`std≈3.6e-05` to `4.2e-05`).
- Final gate judgment: not a promotion candidate and not a successful heatmap-activation mechanism.
