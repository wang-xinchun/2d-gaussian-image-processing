# Performance Hypothesis: Explicit Split and Gaussian Feedback

更新时间: 2026-05-19 21:56:41 CST

## 1. Core Agreement

本轮用户确认：导师反馈不只是为了把论文故事讲清楚，也能帮助提升性能。

这条判断成立。当前瓶颈不是单纯缺模型容量，而是职责混淆：

- Gaussian control 同时被期待表达结构、纹理、修复强度和模块分工；
- HR decoder 同时要保真细节、合成 residual、响应 Gaussian heatmap；
- router 试图后验分配角色，但输入端低/高频职责还不够清楚。

如果先把信息分流做清楚，再让 Gaussian 回流控制高频细节恢复，性能上有更强希望。原因不是“可解释性自然带来性能”，而是显式机制能减少错误干预、降低控制熵、让每个分支学更单纯的函数。

## 2. Why It Can Improve Performance

### 2.1 Reduce Role Confusion

当前 `A_y03` 已经有效，但 Gaussian heatmap 仍容易变成泛化 attention。显式分流后：

- low/structure band 交给 Gaussian 建模；
- high/detail band 交给 HR branch 和 decoder 保留/合成；
- Gaussian 只负责何处、何种方向/尺度/强度地干预。

这会让 Gaussian 参数更稳定，不必背负完整纹理生成任务。

### 2.2 Improve Selectivity

`A_am03` 和 `A_ap03` 表明：控制器可训练，但跨 split 选择性不足。显式分流可以让控制器看到更干净的输入：

- low band 提供结构与退化轮廓；
- high band 提供残余细节需求；
- feedback map 决定哪些 high-frequency 区域该加强、抑制或交给边缘分支。

这比让 router 在混合 feature 上直接竞争更稳。

### 2.3 Reduce Over-Restoration and Artifacts

去模糊和去雾都会遇到过增强、伪边、颜色偏移或局部纹理误修。闭环 Gaussian feedback 可以显式学习：

- where to restore;
- where to leave unchanged;
- where to suppress unreliable high-frequency hallucination;
- where edge/detail branch should act.

这会同时服务 PSNR/SSIM 和真实观感。

### 2.4 Make Evaluation More Useful

如果只看 PSNR，模型可能学到保守平均输出；如果只看观感，又可能引入 hallucination。下一阶段应该让机制和评价配套：

- official PSNR/SSIM 仍是 deblur milestone 接受口径；
- crop/panel/edge/contrast/color shift 用于定位真实失败；
- perceptual/no-reference 指标辅助判断观感；
- visualization maps 用于判断 Gaussian feedback 是否学到合理控制。

## 3. Next Experiment Priority

### Priority 1: `A_aq03` explicit split Gaussian feedback

目标：先做最小机制闭环，验证显式分流是否提升 deblur short-tier performance。

建议最小改动：

- 添加可选 explicit band split：
  - `low_band = blur/downsample/upsample(x)` 或 lightweight Laplacian low band；
  - `high_band = x - low_band`；
- LR Gaussian head 主要使用 low/structure band；
- HR stem 使用原 RGB 与 high/detail band；
- Gaussian heatmap 回流只调制 high-detail feature，而不是无差别调制全部 feature；
- 导出 `low_band`, `high_band`, `attention_map`, `guided_high_detail` 作为可视化。

验收：

- 先按 short-tier gate 过 static/preflight/smoke；
- official test 必须和 `A_z03/A_ah03/A_ap03` 同协议比较；
- 允许第一版只追平或小幅提升，但必须给出可解释的可视化。

### Priority 2: `A_ar03` Gaussian control canvas auxiliary supervision

目标：让 Gaussian heatmap 从“普通 attention”变成可学习、可解释的结构控制画板。

建议机制：

- 从 Gaussian params/attention/directional prior 输出 structure canvas；
- auxiliary target 使用 edge/gradient/Laplacian/high-pass，而不是 RGB；
- loss 权重必须轻，避免牺牲主恢复目标；
- 导出 canvas 与 target 对比。

验收：

- 看 canvas 是否对齐边缘、模糊方向、结构区域；
- 看是否改善 high-frequency restoration 但不引入过增强。

### Priority 3: `A_as03` closed-loop Gaussian feedback controller

目标：把高斯引导回流做成真正的闭环控制。

建议机制：

- Gaussian control canvas 提供结构先验；
- decoder detail state 提供当前恢复状态；
- controller 输出 `enhance/suppress/demand` maps；
- 回流到 decoder guidance、feature refinement 或 edge residual。

验收：

- feedback map 必须可视化；
- 不能退回 hard role competition；
- 与 `A_ap03` 对比，重点看跨 split official 与真实观感。

## 4. Haze Policy

去雾不能继续用 deblur checkpoint 做泛化宣称。短期要做 diagnostic：

- 建立 haze qualitative panel；
- 看颜色、对比度、残雾、halo、局部过增强；
- 辅助记录 LPIPS / NIQE / BRISQUE / CLIP-IQA / MUSIQ 中可用的指标；
- 如果要提升去雾，需要单独设计 haze-aware Gaussian control。

## 5. Immediate Decision

下一步不是 `A_an03/A_ao03`，也不是继续调 router temperature/delta。最合理的第一枪是 `A_aq03`，因为它同时回答导师的分流问题和性能瓶颈问题：

> 显式 low/high band split 能否让 Gaussian control 更干净地回流到 HR detail recovery，从而改善 short-tier official deblur performance 和真实观感？

如果 `A_aq03` 正向，再考虑把它晋级到更强的 canvas auxiliary 或 closed-loop feedback；如果 `A_aq03` 负向，也能明确“显式分流形式不对”，比继续盲调 router 更有信息量。
