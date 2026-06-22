# V1/T01/A_4kd28 Manifest

- Purpose: Move the A27 route specialization earlier from terminal RGB heads into a route-conditioned `decoder_feature` expert.
- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd28/configs/a_4kd28_fast_4kdehaze_a24_route_feature_expert.toml`
- Init: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/checkpoints/best_val.pt`, non-strict.
- Teacher A: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`.
- Teacher B: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/checkpoints/best_val.pt`.
- Main mechanism: `use_input_std_route_feature_expert=true`, old `use_input_std_route_residual=false`, A27 `use_input_std_route_terminal_heads=false`.
- Route rule: high route if `input_std > 0.07026463374495506`, else low route.
- Feature expert: `2` blocks, strength `0.45`, learned gate blend `0.05`, detached gate.
- Branch weights: high/A08 `0.16`, low/A15 `0.05`; final router distillation remains `0.0`.
- Sampling: natural fast-probe sampling; no A25/A26 balanced sampler.
- Training: GPU3 formal run completed, log `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd28/logs/train_20260614_154636.log`, best-val step400 `val_psnr=18.44844816281949`.
- Evaluation: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd28/metrics/test_eval_best_val_gpu.json`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd28/metrics/test_task_metrics_best_val_gpu.json`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd28/metrics/a18_record_alignment_best_val_gpu.json`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd28/metrics/input_std_route_regime_best_val_gpu.json`.
- Result: not promoted. Fast-test PSNR `17.73849582669033` is below A24 `17.790202882742463`; high-route gap vs A18 router is `-0.3390800359165398 dB`.
- Analysis: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd28/result_analysis.md`.
- Status: completed negative mechanism diagnostic; next mainline is A29 hard route decoder heads.
