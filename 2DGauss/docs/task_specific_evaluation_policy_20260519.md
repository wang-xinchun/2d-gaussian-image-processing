# Task-Specific Evaluation Policy After Advisor Feedback

日期: 2026-05-19

## 结论

导师关于指标的提醒应被提升为项目级评价规则: `PSNR/SSIM` 仍然是有 GT 复原任务的基础保真度指标，但不能单独决定“效果好”。后续主线必须按任务建立评价闭环: 去模糊看结构清晰度、边缘/细节恢复和伪影；去雾看残余雾、颜色/对比度、自然度和去雾专用指标；所有跨任务样例都必须经过真实观感检查，不能只用传统像素指标解释。

## 当前实现审计

当前代码里的正式评价能力还比较窄:

- `/root/autodl-tmp/2DGauss/src/uhd_gaussian/training/metrics.py` 已实现 `psnr` 和 `ssim`。
- `/root/autodl-tmp/2DGauss/scripts/evaluate_v1_t01.py` 当前 official evaluation 只汇总 `psnr_mean` 和 `loss_mean`。
- `/root/autodl-tmp/2DGauss/scripts/export_a_y03_report_samples.py` 在样例包里计算 `PSNR/SSIM`，并导出可视化 panel、error heatmap、Gaussian attention。
- 当前没有接入 `LPIPS`、`DISTS`、`NIQE`、`BRISQUE`、`FADE` 等感知或去雾专用指标。

因此，现阶段不能把“指标体系已经完整”作为项目能力；它应成为下一步补齐的评测层。

## 指标分层

### 1. Deblur / UHD_deblur

`UHD_deblur` 是当前冻结里程碑 `M04/A_y03` 的正式任务，继续保留 same-protocol full-image PSNR 作为主锚点:

- 基础保真度: `PSNR`, `SSIM`, `loss_mean`。
- 感知相似度: 优先补 `LPIPS` 与 `DISTS`，用于发现 PSNR 上升但纹理发糊、过平滑或局部假纹理的问题。
- 结构/细节诊断: 梯度幅值一致性、Laplacian/edge residual、high-frequency error、边缘过冲/振铃检查。
- 可视化: input / output / GT / error heatmap / Gaussian control canvas / low-band / high-band / guided high-detail crop panel。

Deblur 的接受口径:

- official PSNR 仍是硬锚点，尤其用于同协议比较 `A_z03/A_ah03/A_ap03/A_aq03`。
- 若 PSNR 提升但视觉出现更强 ringing、halo、过锐化、色偏或结构破坏，不应直接宣称“更好”，只能称为 pixel metric positive。
- 若视觉明显改善但 PSNR 下降，应单独标记为 perceptual tradeoff，不能混作 official fidelity breakthrough。

### 2. Dehaze / UHD_haze / SOTS

当前 `A_y03` 是 `UHD_deblur` checkpoint，不是去雾模型。报告包中的 `UHD_haze/SOTS` 输出只能作为跨任务诊断或失败样例，不能宣称 dehazing performance。

如果后续正式做去雾，必须单独建立去雾评价:

- 有 GT 时: `PSNR`, `SSIM`, `LPIPS`, `DISTS`。
- 无 GT 或真实雾图时: `FADE` 作为雾密度/残雾诊断，`NIQE/BRISQUE` 作为自然图像质量参考；可选 `MUSIQ/CLIP-IQA` 做学习型无参考 IQA。
- 颜色/对比度: 对比度恢复、亮度直方图、饱和度、白平衡/色偏、CIEDE2000 或 Lab 空间颜色偏差。
- 去雾视觉 checklist: 残余雾、天空/白墙发灰、远景糊成一片、颜色漂移、过饱和、过增强、halo、暗部噪声、边缘断裂。

Dehaze 的接受口径:

- 不能因为 PSNR/SSIM 好就说去雾好。
- 不能因为某几张图局部变清楚就说去雾好。
- 只有当去雾专用指标、自然度/颜色指标、人工视觉 panel 同时支持，才可以说 dehazing 有真实提升。
- 如果视觉观感差，即使 PSNR/SSIM 提升，也应判为不合格或仅记为保真度指标改善。

### 3. General Restoration / Cross-Task Samples

跨任务样例必须使用“诊断”口径:

- 可以说模型在某类退化上显示了哪些行为，例如残雾、色偏、纹理丢失、过锐化。
- 不可以把未训练任务的样例 PSNR/SSIM 当作泛化能力证明。
- 所有跨任务结论必须带任务边界: trained-on deblur, tested visually on haze/SOTS/GoPro samples。

## 与性能提升的关系

指标不是汇报装饰，而是性能提升的方向盘。只看 PSNR/SSIM 时，模型很容易学到“像素平均更接近但观感更糊”的解；去雾任务尤其容易出现 PSNR 看起来能接受、但真实画面灰、脏、偏色、halo 明显的情况。

后续要用任务指标反推结构改进:

- 如果 deblur PSNR 提升但 LPIPS/DISTS 变差，说明高频细节或纹理自然度被破坏，`A_aq03` 的 high-detail feedback 需要更克制。
- 如果 edge/gradient 指标差，说明 Gaussian control canvas 没有正确回流到边缘/结构恢复。
- 如果 haze 的 FADE/残雾诊断差，说明当前 deblur-trained Gaussian prior 不能自动处理空气光/透射率类退化，需要独立去雾分支、数据或任务 adapter。
- 如果颜色/对比度差，说明不能只调高频结构，必须加入颜色/低频照度恢复约束。

## 对 `A_aq03` 的要求

`A_aq03` explicit frequency-split Gaussian feedback 不应只交一个 official PSNR。它至少需要同步导出:

- same-protocol `UHD_deblur/test300` official PSNR/loss。
- sample-level `PSNR/SSIM`。
- low/high band 可视化。
- Gaussian attention/control canvas 可视化。
- guided high-detail 或 feedback map 可视化。
- detail crop panels: 边缘、细线、纹理、文字/结构区域。
- artifact checklist: blur residue、ringing、halo、over-sharpen、color shift、checkerboard、edge break。

在评估口径上:

- Deblur promotion 仍必须先过同协议 official PSNR。
- 若 `A_aq03` 没有超过 short leader，但视觉/感知诊断显著改善，可记为机制正向，不作为性能突破。
- 若 `A_aq03` PSNR 小涨但视觉变差，应判为不成熟，不进入 medium promotion。

## 下一步落地

短期先做评测层，不急着把指标写进训练 loss:

1. 保持 `evaluate_v1_t01.py` 的 official PSNR 口径不变，避免破坏历史可比性。
2. 新增一个 offline evaluation/report 脚本，独立计算 SSIM、LPIPS/DISTS、NIQE/BRISQUE、FADE 或其可用替代项，并生成视觉 panel。
3. 对 `A_y03` 和后续 `A_aq03` 统一生成同模板评测包。
4. 只有当离线指标稳定揭示某类失真后，再考虑把对应约束加入训练，例如 edge consistency、perceptual loss、color/contrast loss 或 haze-specific loss。

这样可以避免过早把不稳定的无参考指标写进训练目标，同时又能保证后续实验不再只被 PSNR/SSIM 单线牵引。
