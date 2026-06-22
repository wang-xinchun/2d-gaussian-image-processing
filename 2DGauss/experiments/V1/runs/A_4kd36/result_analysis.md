# A_4kd36 Result Analysis

时间: 2026-06-14 22:08:00 CST

## 目的

A36 是单 checkpoint 双 backbone `input_std` router 上界验证。它把 A08 与 A15 两个 V1/T01 backbone 权重打包进同一个 `.pt` 文件，在 checkpoint 内部使用固定 `input_std` 阈值规则选择 high/low backbone。

这不是轻量单 backbone 压缩方案，而是机制上界: 如果完整 high route 和完整 low route 都保存在同一个 checkpoint 里，是否可以复现 A18 two-checkpoint router。

## 新增/改动路径

- `/home/ldj/wxc/2DGauss/scripts/create_dual_backbone_router_checkpoint_v1_t01.py`
- `/home/ldj/wxc/2DGauss/scripts/evaluate_dual_backbone_router_v1_t01.py`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd36/checkpoints/a08_a15_single_checkpoint_dual_backbone_input_std_router.pt`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd36/metrics/dual_backbone_checkpoint_manifest.json`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd36/metrics/test_eval_dual_backbone_router_gpu_smoke_limit2.json`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd36/metrics/test_eval_dual_backbone_router_gpu.json`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd36/logs/create_dual_backbone_checkpoint.stdout.log`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd36/logs/test_eval_dual_backbone_router_gpu_smoke_limit2.stdout.log`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd36/logs/test_eval_dual_backbone_router_gpu.stdout.log`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd36/logs/test_eval_dual_backbone_router_gpu.exit_code`

## Checkpoint

- High backbone: A08, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`, step `700`
- Low backbone: A15, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/checkpoints/best_val.pt`, step `300`
- Router: `input_std > 0.07026463374495506` selects high/A08; otherwise low/A15
- Packed checkpoint size: `1.4M`
- State key count per backbone: `236`

## Validation

- PASS: `py_compile` for create/evaluate dual-backbone scripts
- PASS: source checkpoint/config path checks
- PASS: checkpoint packer verifies high/low state keys match
- PASS: GPU3 smoke on 2 test samples, exit code `0`
- PASS: full GPU3 test on 100 samples, exit code `0`

## Full Test Result

Metric path: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd36/metrics/test_eval_dual_backbone_router_gpu.json`

- `psnr_mean=17.8470899392002`
- `loss_mean=0.11994637824594974`
- `ssim_mean=0.6992770892381668`
- `dark_channel_l1_mean=0.1449453890323639`
- `edge_magnitude_l1_mean=0.16096453994512558`
- `local_contrast_l1_mean=0.025455681346356868`
- `chroma_l1_mean=0.030336581747978927`
- `gap_router_mean=+0.00015556338577912498 dB`
- `gap_oracle_mean=-0.036465855886337196 dB`

Per-route:

- Low route: count `57`, `psnr_mean=16.55905670855848`, router gap `+0.00042343678412253124 dB`
- High route: count `43`, `psnr_mean=19.55448282632992`, router gap `-0.00019952460737376242 dB`
- A08-oracle group: count `46`, router gap `+0.0003500243837846099 dB`
- A15-oracle group: count `54`, router gap `-0.000010088575484806589 dB`

## Comparison

- A18 router record: `17.84693437581442`
- A36: `17.8470899392002`
- A36 vs A18: `+0.00015556338577912498 dB`
- A36 vs A33: `+0.024027627501937698 dB`
- A36 vs A35: `+0.02395780570655287 dB`
- A36 vs A24: `+0.05688705645773775 dB`

High-route bottleneck:

- A33 high-route gap: `-0.12677254958688497 dB`
- A35 high-route gap: `-0.12662823419234123 dB`
- A36 high-route gap: `-0.00019952460737376242 dB`

## Conclusion

A36 is a real milestone. It proves the fixed `input_std` rule and the A08/A15 complementarity are sufficient when the checkpoint contains full route-specific backbone capacity. The high-route bottleneck is essentially eliminated in this upper-bound form.

The remaining problem is no longer “whether the route signal works”; it is compression and sharing: how to distill this dual-backbone checkpoint into a more compact single-backbone or partially shared route-conditioned architecture without reopening the high-route gap.
