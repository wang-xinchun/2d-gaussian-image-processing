# Gate Check: V1/T01/A_aw03

## Branch Identity

- Run: `V1/T01/A_aw03`
- Config: `experiments/V1/runs/A_aw03/configs/a_aw03_short_uhddeblur_demand_calibrated_selective_feedback_probe.toml`
- Mechanism family: residual-demand Gaussian reinjection with demand-budget selectivity and target-mean feedback calibration

## Static / Preflight Gate

- [x] Config exists
- [x] Python compile/import check passed
- [x] Model forward/auxiliary check passed
- [x] Parameter count checked: `total_params=374761`
- [x] Formal preflight passed at `batch_size=6`, `train_context_size=640`, supervision crop `512`
- [x] True smoke calibration passed in the `85.0-93.0G` reserved-memory window: `max_memory_reserved_gb=91.947265625`

## Result Gate

- [x] Training finished cleanly at `step=2500`
- [x] Checkpoint endpoint verified: `final.pt`
- [x] Same-protocol official test completed
- [x] Task-specific diagnostics generated
- [x] Mechanism panels generated
- [x] Run-local and global logs synchronized

## Result Summary

Official UHD_deblur/test300 produced `psnr_mean=25.380666832181568`. This is better than `A_av03` and `A_au03`, but still slightly below `A_z03=25.385710820365173`, so the run is not accepted for promotion.

Mechanism evidence shows that calibration avoided both previous extremes: feedback is no longer `A_au03`-style saturated (`mean≈0.98474`), and no longer `A_av03`-style over-shrunk (`mean≈0.2415`). `A_aw03` lands at `control_feedback_map mean≈0.42420`, `std≈0.23441`.

The remaining failure is that the main Gaussian attention map remains almost flat (`attention_map std≈0.00017`), so the next gate should test whether the primary Gaussian heatmap can be aligned to restoration demand without damaging fidelity.
