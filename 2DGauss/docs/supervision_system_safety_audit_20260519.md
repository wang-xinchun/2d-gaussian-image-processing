# Supervision System Safety Audit

日期: 2026-05-19

## 审计结论

当前监督体系可以继续使用，但必须按“主保真 + 小权重任务代理 + 离线任务指标 + 视觉 panel”的结构推进。不能把新引入的 task-aware proxies 当成新的单一主指标，否则会把模型带向过锐化、过增强、偏色或局部伪纹理。

本轮已在代码和文档中补上护栏，确保后续实验不会因为一组新 proxy loss 被误用而偏离主线。

## 为什么不能让 proxy 绝对主导

`PSNR/SSIM` 的问题是过于偏像素保真，可能鼓励过平滑；但 task-aware proxy 也有自己的偏差:

- edge/high-frequency proxy 可能鼓励锐化和伪纹理。
- local-contrast proxy 可能鼓励过增强和 halo。
- dark-channel proxy 可能误导颜色和亮度，尤其在非雾图或天空区域。
- luminance/chroma/saturation proxy 可能让低光结果看似稳定但缺少真实细节。

所以正确方向不是从“PSNR/SSIM 独裁”换成“task proxy 独裁”，而是让它们互相制衡。

## 代码护栏

已在 `/root/autodl-tmp/2DGauss/scripts/train_v1_t01.py` 中加入保守校验:

- 所有 loss 权重必须非负。
- `charbonnier_weight` 默认必须大于 `0`，防止只靠 proxy loss 训练。
- 单个 task-aware proxy 权重默认不得超过 `0.10`。
- task-aware proxy 总权重默认不得超过 `0.25`。
- task-aware kernel size 必须为正，默认不得超过 `63`。
- 只有显式设置 `allow_high_task_aware_loss_weights = true`, `allow_large_task_aware_kernels = true`, 或 `allow_proxy_only_loss = true` 时才能越界；这种情况只能记为 diagnostic run，不得直接 promotion。

这保证旧配置完全兼容，也保证新配置不会无意间让 proxy 抢走主训练目标。

## 分任务合理性

### Deblur

合理代理:

- `edge_magnitude_weight`: 检查边缘幅值是否恢复。
- `high_frequency_weight`: 检查细节是否被过平滑。
- `local_contrast_weight`: 辅助判断局部清晰度。

必须防的偏差:

- 过锐化。
- halo/ringing。
- 噪声增强。
- 伪纹理。

接受条件:

- official PSNR/SSIM 不能明显恶化。
- edge/high-frequency 诊断改善。
- crop panel 中边缘和纹理真实，不是硬锐化。

### Dehaze

合理代理:

- `local_contrast_weight`: 对比度恢复。
- `luminance_weight`: 亮度/低频恢复。
- `chroma_weight`: 色偏控制。
- `saturation_weight`: 防止发灰或过饱和。
- `dark_channel_weight`: 残雾/透射率相关近似。

必须防的偏差:

- 天空区域变脏。
- 暗部被压黑。
- 颜色过饱和。
- halo/过增强。
- 局部残雾指标好看但整体观感差。

接受条件:

- 有 GT 时 paired metrics 和 perceptual diagnostics 都要看。
- 无 GT/真实雾图时必须补 `FADE/NIQE/BRISQUE` 或可用替代项。
- 肉眼 panel 差时不得宣称去雾好。

### Low-Light

合理代理:

- `luminance_weight`: 亮度恢复。
- `chroma_weight`: 色彩稳定。
- `saturation_weight`: 饱和度自然。
- `local_contrast_weight`: 暗部层次。

必须防的偏差:

- 过曝。
- 暗部噪声放大。
- 色偏。
- 亮度看起来对齐但细节不真实。

接受条件:

- 亮度、颜色、暗部噪声和视觉 panel 同时过关。
- 不允许只因亮度接近 GT 就 promotion。

## 实验规则

1. 所有 task-aware proxy 必须从 short probe 开始。
2. 新 proxy 不得直接改 frozen `M04/A_y03` 或既有 run 配方。
3. `A_aq03` 若启用 proxy，建议只启用 deblur 小权重组合: `edge_magnitude_weight=0.02`, `high_frequency_weight=0.02`, `local_contrast_weight=0.01`。
4. result analysis 必须报告 proxy 权重、official PSNR/SSIM、task-specific diagnostics、视觉 panel 和失败模式检查。
5. proxy loss 改善但视觉变差，判失败。
6. 视觉改善但 official fidelity 下降，记为 perceptual tradeoff，不自动晋级。
7. 只有 fidelity、task-specific diagnostics、visual panel 同时支持，才可以说是真正性能提升。

## 下一步

监督体系现在具备可控入口，但还缺完整第三方评测指标。下一步应在离线评测层继续接入 `LPIPS/DISTS` 和 dehazing no-reference 指标，如 `FADE/NIQE/BRISQUE` 或可用替代项；训练层保持 proxy 小权重、短探优先。
