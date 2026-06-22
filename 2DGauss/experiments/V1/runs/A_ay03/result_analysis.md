# Result Analysis: V1/T01/A_ay03

Status: completed and evaluated. Not a promotion candidate.

## Why This Run Exists

`A_aw03` nearly matched the short leader and fixed the feedback-strength regime, but its main `attention_map` stayed nearly constant. `A_ax03` tried the simplest pressure: canvas-sourced feedback plus a small `gaussian_attention_weight=0.02` loss directly on the rendered attention. That failed. Official PSNR fell to `25.233892769220702`, task-specific PSNR fell to `23.12751232307337`, and mechanism panels showed the effective `attention_map std` was only `3.6e-05` to `4.2e-05`.

The hypothesis for `A_ay03` was that direct L1 supervision on the renderer output is too weak a control handle. `A_ay03` therefore added a forward-path demand bridge that predicts a structured `attention_bridge_map` from HR features, raw Gaussian attention, local contrast, and input structure, then blends it into the effective `attention_map`.

## Final Metrics

- Training endpoint: clean finish at `step=2500`.
- Final validation: `val_loss=0.029540793877094983`, `val_psnr=27.634088171325992`.
- Validation trend: `step=500 28.675151131465366`, `step=1000 28.72393012958687`, `step=1500 28.658803866275264`, `step=2000 28.432735276230538`, `step=2500 27.634088171325992`.
- Official final full-image test: `loss_mean=0.040016187620349225`, `psnr_mean=24.998189217540585`, `auxiliary_loss_mean=0.011719125838329394`, `total_loss_mean=0.05173531337330739`, `samples=300`.
- Task-specific limit20: `psnr_mean=22.975469413149693`, `ssim_mean=0.6872823059558868`, `edge_magnitude_l1_mean=0.13152226880192758`, `high_frequency_l1_mean=0.017945018922910096`, `local_contrast_l1_mean=0.021368088200688362`, `luminance_l1_mean=0.04324529198929668`, `chroma_l1_mean=0.005130211869254708`, `saturation_l1_mean=0.009870028588920832`, `dark_channel_l1_mean=0.039017872884869576`.
- Diagnostic `step_001000.pt` official test: `psnr_mean=25.062951734772415`, `loss_mean=0.039840590100114544`. This is not the accepted final checkpoint, and it still does not recover above `A_ax03`.

## Mechanism Diagnosis

The bridge did exactly what it was built to do mechanically:

- `raw_attention_map std≈0.000337-0.000341`, still near-flat.
- `attention_bridge_map std≈0.0738-0.0742`, strongly structured.
- Effective `attention_map std≈0.0221-0.0223`, far above `A_aw03≈0.00017` and `A_ax03≈4e-05`.
- Feedback stayed non-saturated: `control_feedback_map mean≈0.416-0.417`, `std≈0.214`.

But that is also the negative result. The attention path was activated, yet official and task-specific fidelity both declined. The failure is not simply that the heatmap remained flat; the failure is that the demand bridge became too invasive when the same bridged attention was fed into high-frequency split feedback, highpass modulation, decoder guidance, feature refinement, control canvas, terminal residual, and edge refinement.

## Judgment

`A_ay03` is an on-topic negative diagnostic. It breaks the heatmap bypass in a literal sense, but not in a useful restoration sense. A structured heatmap is not enough; it must be routed through a fidelity-protected control path.

The next probe should not increase `attention_bridge_weight`, increase bridge strength, or rely on early checkpoint selection. `step_001000.pt` is still weak. The next probe should keep the interpretable demand bridge but apply it only to the control canvas/feedback path, leaving raw Gaussian attention on the main fidelity-sensitive restoration backbone.
