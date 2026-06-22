# Task-Aware Loss Adjustment Plan

日期: 2026-05-19

## 核心判断

后续提升性能不能再让 `PSNR/SSIM` 绝对主导训练和接受口径。`PSNR/SSIM` 仍然保留为 paired fidelity anchor，但训练目标要逐步引入能反映具体退化现象的 task-aware loss proxy，并在后验评测中配合任务专用指标和视觉 panel。

这一步的原则是: 先接入轻量、可微、稳定的代理损失；`FADE/NIQE/BRISQUE/LPIPS/DISTS` 这类更适合作为离线评测或感知报告的指标，暂不直接硬塞进主训练 loss。

## 已落地代码

新增/扩展文件:

- `/root/autodl-tmp/2DGauss/src/uhd_gaussian/training/losses.py`
- `/root/autodl-tmp/2DGauss/src/uhd_gaussian/training/__init__.py`
- `/root/autodl-tmp/2DGauss/scripts/train_v1_t01.py`
- `/root/autodl-tmp/2DGauss/scripts/evaluate_task_specific_metrics.py`

新增可微 loss proxy:

- `edge_magnitude_l1_loss`: 对齐 Sobel 边缘幅值，服务去模糊边缘清晰度。
- `high_frequency_l1_loss`: 对齐局部高频残差，服务细节恢复与过平滑诊断。
- `local_contrast_l1_loss`: 对齐局部亮度对比度，服务去雾/低光的真实观感。
- `luminance_l1_loss`: 对齐亮度，服务低光照度和去雾低频恢复。
- `chroma_l1_loss`: 对齐去亮度后的颜色分量，服务色偏控制。
- `saturation_l1_loss`: 对齐饱和度，服务去雾/低光中的过饱和或发灰问题。
- `dark_channel_l1_loss`: 对齐 dark channel proxy，服务去雾残雾/透射率相关诊断。

所有新训练项默认权重为 `0.0`，旧配置不写这些字段时行为保持兼容，不影响冻结 `M04/A_y03` 或既有 run 资产。

## 配置入口

在任意 TOML 的 `[loss]` 中可选加入:

```toml
edge_magnitude_weight = 0.0
high_frequency_weight = 0.0
high_frequency_kernel_size = 5
local_contrast_weight = 0.0
local_contrast_kernel_size = 7
luminance_weight = 0.0
chroma_weight = 0.0
saturation_weight = 0.0
dark_channel_weight = 0.0
dark_channel_kernel_size = 15
```

旧的 `charbonnier_weight`、`gradient_weight`、`laplacian_weight`、`use_edge_weighted_reconstruction` 等继续保留。

## 安全护栏

这些 proxy loss 只能作为主重建目标的辅助项，不能喧宾夺主。代码层已加入保守校验:

- 所有 loss 权重必须非负。
- `charbonnier_weight` 默认必须保持大于 `0`，除非显式设置 `allow_proxy_only_loss = true` 做诊断实验。
- 单个 task-aware proxy 权重默认不得超过 `0.10`。
- 所有 task-aware proxy 权重总和默认不得超过 `0.25`。
- `high_frequency_kernel_size`, `local_contrast_kernel_size`, `dark_channel_kernel_size` 必须为正，默认不得超过 `63`。
- 若要越过这些上限，必须显式设置 `allow_high_task_aware_loss_weights = true` 或 `allow_large_task_aware_kernels = true`，并且只能作为 diagnostic run，不得直接 promotion。

监督体系的基本原则:

- Fidelity anchor 不取消: `Charbonnier/PSNR/SSIM` 仍负责基本还原，不允许完全交给 proxy。
- Proxy 只纠偏: task-aware loss 只负责把训练从“像素平均正确”拉向“边缘、对比度、颜色、残雾、亮度更合理”。
- Evaluation 决定接受: proxy loss 降低不等于模型更好；必须同时看 official fidelity、task-specific metrics 和视觉 panel。
- Cross-task 不混用: deblur 权重组合不得直接当作 dehaze/low-light 配方；每个任务必须重新短探。

## 分任务建议

### Deblur

目标: 清晰边缘、结构稳定、细节不过平滑、不制造 ringing/halo。

建议从很小权重开始:

```toml
edge_magnitude_weight = 0.02
high_frequency_weight = 0.02
local_contrast_weight = 0.01
```

不要一开始把 high-frequency 权重加太大，否则可能用过锐化换 PSNR 或制造伪纹理。对 `A_aq03`，这组 loss 应与显式 low/high split 可视化一起看，重点检查 guided high-detail 是否更真实。

失败模式:

- `edge_magnitude_weight` 过大: 边缘变硬、halo/ringing 增加。
- `high_frequency_weight` 过大: 伪纹理、噪声增强、棋盘感。
- `local_contrast_weight` 过大: 局部对比过强，暗部或纹理区域发脏。

### Dehaze

目标: 去残雾、恢复对比度、保持颜色自然、避免 halo 和过增强。

建议起点:

```toml
local_contrast_weight = 0.02
luminance_weight = 0.02
chroma_weight = 0.02
saturation_weight = 0.01
dark_channel_weight = 0.02
```

`dark_channel_l1_loss` 是 dehaze proxy，不等同于 `FADE`。真正宣称去雾性能时，仍需离线加入 `FADE/NIQE/BRISQUE` 或可用替代项，以及人工视觉 panel。

失败模式:

- `dark_channel_weight` 过大: 画面可能被拉暗、局部颜色不自然。
- `local_contrast_weight` 过大: 过增强、halo、天空/平滑区域脏。
- `saturation_weight` 过大: 颜色过饱和，或者为了对齐饱和度压制真实恢复。

### Low-Light

目标: 亮度恢复、颜色稳定、不过曝、不噪声放大。

建议起点:

```toml
luminance_weight = 0.03
chroma_weight = 0.02
saturation_weight = 0.01
local_contrast_weight = 0.01
```

低光不要只追亮度差异，要同步看色偏、暗部噪声和局部对比度。

失败模式:

- `luminance_weight` 过大: 过曝或全局变亮但细节不真实。
- `local_contrast_weight` 过大: 暗部噪声被放大。
- `chroma/saturation` 过大: 色偏被“平均化”，颜色看似稳定但不鲜活。

## 评测脚本

新增离线脚本:

```bash
python scripts/evaluate_task_specific_metrics.py \
  --config experiments/V1/runs/A_y03/configs/a_y03_medium_uhddeblur_feature_refinement_plus_edge_weighted_lightedge_5h_confirmation.toml \
  --checkpoint experiments/V1/runs/A_y03/checkpoints/final.pt \
  --split test \
  --device cuda \
  --crop-size 0 \
  --output experiments/V1/runs/A_y03/metrics/task_specific_test_final.json
```

脚本输出:

- `psnr_mean`, `ssim_mean`
- `edge_magnitude_l1_mean`
- `high_frequency_l1_mean`
- `local_contrast_l1_mean`
- `luminance_l1_mean`
- `chroma_l1_mean`
- `saturation_l1_mean`
- `dark_channel_l1_mean`

这些是轻量诊断项，不替代 LPIPS/DISTS/FADE/NIQE/BRISQUE。下一步可以在这个脚本外层继续接入第三方指标库。

## 后续实验口径

1. `evaluate_v1_t01.py` 的 official PSNR 口径暂时保持不变，保证历史可比。
2. 所有新 loss 项都必须先做 short probe，不直接覆盖 medium milestone 配方。
3. `A_aq03` 若引入 task-aware loss，应先采用轻权重 deblur 组合，并同步导出 low/high band、Gaussian control canvas 和 task-specific metrics。
4. 去雾/低光不得复用 deblur checkpoint 直接宣称性能；必须有对应任务数据、loss 配方、指标和视觉 panel。
5. 如果任务 proxy loss 让 PSNR 小涨但视觉变差，判为失败；如果视觉变好但 PSNR 下降，记录为 perceptual tradeoff，不自动晋级。
6. 任何启用 task-aware proxy 的 run，都必须在 `result_analysis.md` 中报告 proxy 权重、proxy 指标变化、可视化观察和失败模式检查；不能只报 `loss_total`。
7. 若 proxy 指标改善但 official PSNR/SSIM 明显下降，或视觉 panel 出现过锐化、过增强、偏色、噪声放大、halo/ringing，该 run 不得 promotion。
