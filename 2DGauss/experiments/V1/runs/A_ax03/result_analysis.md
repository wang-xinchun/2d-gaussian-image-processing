# Result Analysis: V1/T01/A_ax03

Status: completed and evaluated. Negative diagnostic, not a promotion candidate.

## Why This Run Exists

`A_aw03` almost caught the short leader and fixed the previous feedback-strength problem, but its mechanism panels still showed `attention_map std≈0.00017`. That means the model's restoration-demand logic is mostly living in the added `control_canvas/feedback_map`, while the primary Gaussian heatmap is still close to a constant field.

`A_ax03` is meant to break that specific bottleneck.

## Mechanism Changes From A_aw03

- Feedback demand source changes from the target/fallback path to the predicted canvas: `gaussian_control_canvas_feedback_demand_source = "canvas"`.
- A small attention alignment auxiliary term is added: `gaussian_attention_weight = 0.02`.
- The existing conservative fidelity and task-aware loss guardrails remain active.
- The geometry remains `batch_size=6`, `train_context_size=640`, supervision crop `512`.

## Preflight Evidence

- Static py_compile: passed
- CPU forward/backward: passed with `loss_control_canvas` and `loss_gaussian_attention`
- Formal CUDA preflight: passed with `total_params=374761`, `val_psnr=18.752572677892744`, `loss_total=0.21266134083271027`, `loss_control_canvas=0.19127671420574188`, `loss_gaussian_attention=0.19207045435905457`
- True smoke: passed with `max_memory_reserved_gb=91.947265625`, `max_memory_allocated_gb=88.66405534744263`, `avg_total_step_seconds=1.836618307357033`, `estimated_steps_target_wall_clock=1894`

## Launch Notes

- Tmux session: `2dgauss_a_ax03_uhddeblur_canvas_budget_attention_aligned_feedback_probe`
- Active log: `experiments/V1/runs/A_ax03/logs/train_20260520_151434.log`
- Startup metrics have reached at least `step=60`
- GPU was about `94861 MiB` and `100%` utilization during startup

## Training Endpoint

- Training finished cleanly at `step=2500`.
- Final validation: `val_loss=0.026070117484778166`, `val_psnr=28.688878115482986`.
- Final checkpoint: `experiments/V1/runs/A_ax03/checkpoints/final.pt`.

## Official Acceptance

- Official UHD_deblur/test300: `psnr_mean=25.233892769220702`, `loss_mean=0.03905327895656228`, `auxiliary_loss_mean=0.009553199315754076`, `total_loss_mean=0.0486064781807363`.
- This is far below `A_aw03=25.380666832181568` and `A_z03=25.385710820365173`.

## Task-Specific Diagnostics

- Test `limit=20`: `psnr_mean=23.12751232307337`, `ssim_mean=0.6937903940677643`.
- Compared with `A_aw03`, PSNR and SSIM both decline (`A_aw03 limit20: 23.18663806081702 / 0.6963050037622451`).
- Edge and local-contrast proxy errors improve slightly, but high-frequency and dark-channel errors worsen. This is not enough to offset the fidelity loss and must not be read as better visual quality.

## Mechanism Diagnosis

- Feedback is not the failure: `control_feedback_map mean≈0.443`, `std≈0.159`, so the calibrated non-saturated regime survives.
- The main heatmap is the failure: `attention_map std≈3.6e-05` to `4.2e-05`, even flatter than `A_aw03≈0.00017`.
- The predicted `control_canvas` remains mildly structured (`std≈0.0135` to `0.0138`), but the restoration-demand target is much stronger (`std≈0.1788` to `0.1815`).

## Final Judgment

`A_ax03` is on-topic but failed. It directly tested whether canvas-sourced feedback plus weak attention-demand L1 could make the primary Gaussian heatmap carry restoration demand. The answer is no: the effective main attention stayed near constant and official fidelity dropped.

The next step should not be a larger `gaussian_attention_weight`. The better hypothesis is a forward-path demand bridge that injects a learned HR demand prior into the effective `attention_map`, so downstream Gaussian-guided high-frequency modules actually receive structured control during inference.
