# V1/T01/A_4kd29 Manifest

- Purpose: Replace A28's tiny route feature delta with hard high/low route decoder heads that own the final RGB projection.
- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd29/configs/a_4kd29_fast_4kdehaze_a24_route_decoder_heads.toml`
- Init: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/checkpoints/best_val.pt`, non-strict.
- Teacher A: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`.
- Teacher B: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/checkpoints/best_val.pt`.
- Main mechanism: `use_input_std_route_decoder_heads=true`; old residual, A27 terminal heads, and A28 feature expert are disabled.
- Route rule: high route if `input_std > 0.07026463374495506`, else low route.
- Decoder heads: `1` route feature block, strength `1.0`, learned gate blend `0.02`, detached gate.
- Initialization safety: high/low decoder heads copy the loaded base `decoder[2]` after init checkpoint loading, and route feature deltas start at zero.
- Branch weights: high/A08 `0.18`, low/A15 `0.04`; final router distillation remains `0.0`.
- Sampling: natural fast-probe sampling; no A25/A26 balanced sampler.
- Acceptance: must beat A24 PSNR `17.790202882742463`, strong target `17.82`, milestone target A18 router `17.84693437641316`, high-route gap target `>-0.10 dB`, low-route gap `>= +0.03 dB`.
- Training: GPU0 preflight completed, then formal training launched on GPU0. Formal run was stopped early after step `270` because validation degraded monotonically while route deltas grew; best-val selected step `50`.
- Evaluation: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd29/metrics/test_eval_best_val_gpu.json`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd29/metrics/test_task_metrics_best_val_gpu.json`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd29/metrics/a18_record_alignment_best_val_gpu.json`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd29/metrics/input_std_route_regime_best_val_gpu.json`.
- Result: not promoted. Fast-test PSNR `17.76127769536133` is below A24 `17.790202882742463`; high-route gap vs A18 router is `-0.236088142902285 dB`; low-route gap is `+0.027826703499809195 dB`.
- Analysis: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd29/result_analysis.md`.
- Status: completed negative mechanism diagnostic. Next mainline is conservative hard route decoder heads with route-delta trust-region or staged branch pressure.
