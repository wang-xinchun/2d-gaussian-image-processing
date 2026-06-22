# Result Analysis: V1/T01/A_az03

Status: completed and evaluated. Final judgment: on-topic partial recovery, not a promotion candidate.

## Why This Run Exists

`A_ay03` answered one important question: the demand bridge can make the effective Gaussian heatmap structured. Mechanism panels showed `attention_bridge_map std≈0.074` and effective `attention_map std≈0.022`, far above the near-flat `A_aw03/A_ax03` regime.

But `A_ay03` failed the restoration objective. Official full-image test fell to `psnr_mean=24.998189217540585`, task-specific limit20 fell to `psnr_mean=22.975469413149693` and `ssim_mean=0.6872823059558868`, and a diagnostic `step_001000.pt` official test only recovered to `25.062951734772415`. The issue is not just late overtraining. The bridge was too invasive.

## Mechanism Change From A_ay03

- Keep the bridge and its auxiliary supervision.
- Do not replace the main `attention_map` used by the high-frequency split, highpass modulation, decoder guidance, feature refinement, terminal residual, or edge refinement.
- Use the bridged attention only as `control_attention_map` for the Gaussian control canvas/feedback module.
- Expose `control_attention_map` in mechanism panels, so the run can separately inspect raw main attention, demand bridge, main Gaussian attention, and control-only attention.

## Acceptance Questions

1. Does official fidelity recover above `A_ax03=25.233892769220702` and preferably near `A_aw03=25.380666832181568` / `A_z03=25.385710820365173`?
2. Does task-specific limit20 recover above `A_ay03=22.975469413149693` and ideally near `A_aw03=23.18663806081702`?
3. Does the control path receive a structured demand signal while raw/main attention remains stable?
4. Does calibrated feedback remain non-saturated rather than returning to the `A_au03` all-open regime?

## Gate And Launch Notes

- Static py_compile passed.
- CPU sanity verified the routing change: main `attention_map` remains identical to raw attention (`main_raw_l1=0.0`), while `control_attention_map` is bridged (`control_attention_std≈0.0009169` versus `raw_attention_std≈3.647e-05`).
- Formal preflight at `6x624` passed with `total_params=408458`, `val_psnr=18.975287419921493`, and `loss_total=0.13312028348445892`.
- True smoke at `6x624` passed with `max_memory_reserved_gb=92.4921875`, `max_memory_allocated_gb=89.4190526008606`, and estimated `2506` steps in 58 minutes.
- Training launched in tmux session `2dgauss_a_az03_uhddeblur_control_only_attention_demand_bridge_probe`; log is `experiments/V1/runs/A_az03/logs/train_20260520_225643.log`.
- Startup metrics are writing through at least `step=130`; the inherited `OMP_NUM_THREADS=0` environment warning appears, but training is active and no early NaN/OOM was observed.

## Final Training Endpoint

- Training finished at `step=2500`.
- Final validation: `val_loss=0.025261803343892098`, `val_psnr=28.92787912557181`.
- Final train record: `loss_total=0.0524655357003212`, `loss_control_canvas=0.15785270929336548`, `loss_attention_bridge=0.15465877950191498`.
- Checkpoints present: `final.pt`, `latest.pt`, and `step_002500.pt`.
- Acceptance started after confirming no active `A_az03`/`train_v1_t01.py` process and idle GPU.

## Official Result

Same-protocol full-image UHD_deblur/test300:

- Output: `experiments/V1/runs/A_az03/metrics/official_test_final.json`
- `samples=300`, `crop_size=0`, `checkpoint_step=2500`
- `loss_mean=0.03883044661798825`
- `psnr_mean=25.30935139353653`
- `auxiliary_loss_mean=0.011713128695264459`
- `total_loss_mean=0.050543575460712115`

This recovers substantially over `A_ay03=24.998189217540585` and clears `A_ax03=25.233892769220702`, but it remains below `A_aw03=25.380666832181568`, short leader `A_z03=25.385710820365173`, and frozen `M04/A_y03=26.272695829299355`.

## Task-Specific Diagnostics

Stable `limit=20` test subset:

- Output: `experiments/V1/runs/A_az03/metrics/task_specific_test_limit20.json`
- `psnr_mean=23.132057715603047`
- `ssim_mean=0.6942389011383057`
- `edge_magnitude_l1_mean=0.12970237173140048`
- `high_frequency_l1_mean=0.0174344873521477`
- `local_contrast_l1_mean=0.021073565259575842`
- `luminance_l1_mean=0.04221528498455882`
- `chroma_l1_mean=0.005514730582945049`
- `saturation_l1_mean=0.010305101098492742`
- `dark_channel_l1_mean=0.04557064678519964`

This is a recovery over `A_ay03` (`psnr_mean=22.975469413149693`, `ssim_mean=0.6872823059558868`) but still below the stronger calibrated-feedback reference `A_aw03` (`psnr_mean=23.18663806081702`, `ssim_mean=0.6963050037622451`).

## Mechanism Diagnosis

Mechanism panels were exported to `experiments/V1/runs/A_az03/diagnostics/mechanism_final_test`.

Four-sample diagnostic averages and aux stats show:

- `attention_bridge_map std≈0.069-0.071`: the demand bridge remains structured.
- raw/main `attention_map std≈0.00066`: the main fidelity-sensitive path is no longer driven by the high-variance bridged map.
- `control_attention_map std≈0.0208-0.0213`: the control path receives a structured demand signal.
- `control_canvas std≈0.054-0.056`: the control canvas is much more structured than earlier near-flat regimes.
- `control_feedback_map mean≈0.423`, `std≈0.241`: feedback stays in the calibrated non-saturated middle regime, close to the intended `A_aw03` behavior rather than the saturated `A_au03` behavior.
- Panel mean: `restored_psnr=23.89168025`, `delta_psnr=1.467603`, `restored_ssim=0.7265492499999999`, `delta_ssim=0.04139325`.

## Interpretation

`A_az03` answers the `A_ay03` failure cleanly. The over-invasive bridge was the reason for the large fidelity collapse: when the same demand attention was removed from the main backbone and kept only for control canvas/feedback, official PSNR recovered by about `+0.311 dB` over `A_ay03`.

But this is not the breakthrough. The run still trails `A_aw03` and `A_z03`, so control-only bridge gives the model a better interpretable demand signal but does not convert that signal into enough restoration gain. The current bottleneck is therefore no longer "bridge damages every path"; it is "structured control demand is present, but the reinjection/correction capacity or coupling is still too weak to beat the best calibrated-feedback baseline."
