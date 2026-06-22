# V1/T01/A_4kd27 Manifest

- Purpose: Upgrade A24's late small route residual into higher-capacity high/low input-std terminal heads.
- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd27/configs/a_4kd27_fast_4kdehaze_a24_route_terminal_heads.toml`
- Init: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/checkpoints/best_val.pt`, non-strict.
- Teacher A: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`.
- Teacher B: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/checkpoints/best_val.pt`.
- Main mechanism: `use_input_std_route_terminal_heads=true`, old `use_input_std_route_residual=false`.
- Route rule: high route if `input_std > 0.07026463374495506`, else low route.
- Branch weights: high/A08 `0.14`, low/A15 `0.06`; final router distillation remains `0.0`.
- Sampling: natural fast-probe sampling; no A25/A26 balanced sampler.
- Status: completed on 2026-06-14; not promoted.
- Formal training: GPU3, 400 steps, log `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd27/logs/train_20260614_131135.log`, exit code `0`.
- Best validation: step `400`, `val_psnr=18.41533773497595`, `val_loss=0.11968508075922728`.
- Fast-test best-val: `psnr_mean=17.7513865188403`, `ssim_mean=0.700952832698822`, `loss_mean=0.1202681017667055`, `total_loss_mean=0.12349890783429146`.
- A18 alignment: overall gap vs A18 router `-0.09554785697412059 dB`; low-route gap `+0.06278010621493545 dB`; high-route gap `-0.3054244593410088 dB`.
- Conclusion: low route/proxy metrics remain healthy, but high-route A08-tail preservation worsened relative to A24; next mainline should move route conditioning earlier into feature/control/Gaussian branches.
