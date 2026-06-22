# Gate Check: V1/T01/A_ay03

## Branch Identity

- Run: `V1/T01/A_ay03`
- Config: `experiments/V1/runs/A_ay03/configs/a_ay03_short_uhddeblur_attention_demand_bridge_probe.toml`
- Mechanism family: Gaussian attention demand bridge plus calibrated selective feedback

## Static Gate

- [x] Config exists
- [x] Python compile/import check passed
- [x] CPU forward/backward check passed
- [x] Bridge auxiliary loss records correctly
- [x] Parameter count checked: `total_params=408458`

## Smoke / Preflight Gate

- [x] Formal preflight passed at selected geometry (`6x624`, supervision crop `512`)
- [x] True smoke calibration passed
- [x] Reserved VRAM inside `85.0-93.0G`: `max_memory_reserved_gb=92.4921875`
- [x] Average smoke step time recorded: `avg_total_step_seconds=1.9396179039031267`
- [x] Estimated steps in 58 minutes recorded: `1794`

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

`A_ax03` failed because weak endpoint supervision on the rendered `attention_map` neither made the map structured nor preserved fidelity. `A_ay03` injects a learned demand bridge into the effective Gaussian control path so downstream high-frequency/detail modules see a structured attention signal during forward propagation.

## Geometry Note

Initial `6x640` true smoke ran but reserved `93.416015625G`, slightly above the `93.0G` gate. The selected geometry is therefore `batch_size=6`, `train_context_size=624`, supervision crop `512`.

## Gate Summary

- CPU sanity: `raw_attention_std≈2.84e-05`, `attention_bridge_std≈0.00273`, effective `attention_std≈0.000818`, showing the bridge can activate the effective attention path while the raw renderer remains stable.
- Formal CUDA preflight at `6x624`: `loss_total=0.22732174396514893`, `loss_control_canvas=0.17585372924804688`, `loss_attention_bridge=0.16498465836048126`, `val_psnr=20.264334175626225`.
- True smoke at `6x624`: `max_memory_allocated_gb=89.41702651977539`, `max_memory_reserved_gb=92.4921875`, `estimated_steps_target_wall_clock=1794`, `within_target_reserved_memory_window=true`.
- Launch: tmux session `2dgauss_a_ay03_uhddeblur_attention_demand_bridge_probe`, log `experiments/V1/runs/A_ay03/logs/train_20260520_180410.log`, startup metrics through at least `step=130`, GPU about `95419 MiB/100%`.

## Result Summary

- Endpoint: clean training finish at `step=2500`; `final.pt`, `latest.pt`, and `step_002500.pt` exist.
- Validation trend: best observed validation was `step=1000 val_psnr=28.72393012958687`; final validation dropped to `27.634088171325992`.
- Official final test: `psnr_mean=24.998189217540585`, `loss_mean=0.040016187620349225`, below `A_ax03=25.233892769220702`, `A_aw03=25.380666832181568`, and `A_z03=25.385710820365173`.
- Task-specific limit20: `psnr_mean=22.975469413149693`, `ssim_mean=0.6872823059558868`, below `A_aw03=23.18663806081702/0.6963050037622451`.
- Diagnostic `step_001000.pt`: official `psnr_mean=25.062951734772415`; not an accepted final replacement and still below `A_ax03`.
- Mechanism: bridge activated the effective heatmap (`attention_map std≈0.022`) while raw attention stayed near-flat (`raw_attention_map std≈0.00034`), but this activation damaged fidelity rather than improving restoration.
