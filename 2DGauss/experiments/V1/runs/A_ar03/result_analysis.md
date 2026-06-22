# Result Analysis: V1/T01/A_ar03

Status: completed, evaluated, not promotion candidate.

`A_ar03` answers the most immediate control-canvas question positively: the supervised canvas no longer stays near-constant. The final mechanism export shows `control_canvas mean≈0.44610` and `std≈0.00620`, while the target canvas has `std≈0.14722`. That is a real structural signal, not just noise.

The problem is that the fidelity cost is too large. The official same-protocol UHD_deblur/test result is `psnr_mean=25.104431332416773`, below `A_aq03`, `A_z03`, and the frozen `M04/A_y03` milestone. The task-specific diagnostics on the 20-sample stable subset also look weaker than the short leader on average, with `psnr_mean=22.933465197850488`, `ssim_mean=0.6846876949071884`, and visibly heavier edge/high-frequency/contrast/luminance residuals.

The takeaway is narrow but useful: supervision can make the Gaussian control canvas expressive, but `A_ar03` by itself does not recover enough restoration quality. The next question is therefore not "can the canvas move?" anymore, but "can closed-loop feedback restore fidelity without flattening the canvas again?" `A_as03` and `A_at03` remain gated pending that analysis rather than being launched by inertia.
