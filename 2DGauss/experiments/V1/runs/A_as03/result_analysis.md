# Result Analysis: V1/T01/A_as03

Status: completed, evaluated, not promotion candidate.

`A_as03` closes the next question after `A_ar03`: can closed-loop target-conditioned feedback recover fidelity without flattening the control canvas again? The answer is partial but still negative for promotion. The official same-protocol UHD_deblur/test result is `psnr_mean=25.19097600105495`, with `loss_mean=0.03950027948866288` and `total_loss_mean=0.045946411602199075`, which is better than `A_ar03` but still below `A_aq03`, `A_z03`, and frozen `M04/A_y03`. The 20-sample task-specific diagnostics show a coherent but still weak restoration profile: `psnr_mean=22.916246050966077`, `ssim_mean=0.6880102306604385`, `edge_magnitude_l1_mean=0.13259193524718285`, `high_frequency_l1_mean=0.017471202369779348`, `local_contrast_l1_mean=0.02212119190953672`, `luminance_l1_mean=0.04328204309567809`, `chroma_l1_mean=0.005854859109967947`, `saturation_l1_mean=0.010486438125371932`, `dark_channel_l1_mean=0.04676322750747204`.

The mechanism panels show that the closed-loop control canvas is structured but still low-entropy: `attention_map mean≈0.50029`, `std≈0.00021`; `control_canvas mean≈0.44260`, `std≈0.01112`; `control_canvas_target std≈0.17664`; `control_feedback_map std≈0.00724`. That is a stronger structure signal than `A_aq03`, and the control canvas is more expressive than `A_ar03`, but the fidelity tradeoff remains too large to make this a promotion candidate. The key qualitative read is that feedback is doing real work, not noise, yet the loop still does not recover enough image quality to overtake the stronger frozen baseline.

The most useful comparison points are:

- `A_ar03`: stronger structure, weaker fidelity;
- `A_as03`: more useful closed-loop structure than `A_ar03`, but still not enough overall;
- `A_aq03`: slightly better official fidelity, but nearly flat control canvas;
- `A_z03` and frozen `M04/A_y03`: still ahead on official fidelity.

The next question is no longer whether Gaussian control can become non-flat. It clearly can. The remaining bottleneck is whether the closed-loop path can be made selective enough to preserve or improve official fidelity while keeping that structure alive. `A_at03` remains the next held candidate for that question, but `A_as03` itself is not the endpoint.
