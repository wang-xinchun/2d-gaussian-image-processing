# Bottleneck Root-Cause Analysis After A_as03

更新时间: 2026-05-20 10:36:51 CST

## 背景

从 `M04/A_y03` 之后，主线已经连续验证了三类增强路径：

- `A_aq03`: explicit low/high split + Gaussian feedback，官方 `psnr_mean=25.281006597978177`，但 Gaussian attention/control 仍接近常数，`attention_map std≈0.00017`。
- `A_ar03`: supervised Gaussian control canvas only，能把 `control_canvas` 拉出结构，`std≈0.00620`，但官方 `psnr_mean=25.104431332416773`，保真代价过大。
- `A_as03`: control-canvas supervision + target-conditioned closed-loop feedback，`control_canvas std≈0.01112`、`control_feedback_map std≈0.00724`，但官方 `psnr_mean=25.19097600105495`，仍低于 `A_aq03`、`A_z03` 和冻结的 `M04/A_y03`。

这说明当前瓶颈已经不是“画布能不能动”，而是“结构控制信号能不能变成有选择、有保护、有实际修复收益的回流”。

## 根因判断

### 1. 主 Gaussian attention 仍然低熵，结构画布没有替代核心控制热力图

`A_aq03` 与 `A_as03` 的 `attention_map std` 仍只有 `1e-4` 量级。`A_as03` 的 `control_canvas` 虽然结构更明显，但主干里大量 Gaussian-guided 模块仍主要使用近乎平坦的 `attention_map`。因此当前机制实际上变成了“一个结构化旁路画布 + 一个仍偏平的主 Gaussian heatmap”，论文故事和恢复收益没有完全合在一起。

### 2. canvas 辅助监督目标是通用结构，不是 restoration demand

训练中的 `control_canvas_weight` 使用 GT 图像生成结构目标，这能让 canvas 学到边缘/局部对比，但它监督的是“哪里有结构”，不是“哪里因为退化需要修复”。对去模糊任务而言，高结构区域不一定都需要强干预；已经能被主干恢复好的边缘如果再被强行调制，反而会损伤 PSNR/SSIM 和局部自然度。

### 3. 当前 feedback 是弱选择性的乘性门控，缺少保真保护

`GaussianControlCanvasModule` 当前通过 `decoder_feature * (1 + strength * gate * scale)` 做回流。这个形式有两个问题：

- 它主要是全通道乘性缩放，容易扰动已经正确的 decoder feature。
- `feedback_map` 是 gate 的通道均值，机制可解释性有了，但没有明确的 residual correction、identity preservation、error-demand gating 或 confidence normalization。

所以 `A_ar03/A_as03` 出现了很清楚的现象：结构信号变强，但保真度回不来。

### 4. `A_at03` 原样发车大概率只是组合旧问题

`A_at03` 当前组合的是 explicit split + structure hint + 同一套 closed-loop control canvas feedback。它有价值，但没有解决上述核心问题：canvas 监督仍偏“结构存在”，feedback 仍偏“乘性扰动”，不是“只在退化残差需求区域做受限修正”。因此 `A_at03` 应继续 held，不应作为惯性下一跑。

### 5. 显存几何下降是次要因素，不是主因

新机制家族从 `6x704/6x680` 降到 `6x640` 确实减少了训练上下文，但 `A_ar03/A_as03` 的机制证据已经足够明确：结构能被拉活，fidelity 不能恢复。主因仍是结构-保真度对齐机制不足，而不是单纯训练步数或 crop 几何。

## 对症方案

下一步不应继续调 router、temperature、hint strength，也不应原样启动 `A_at03`。应设计新的强机制短探，暂定为 `A_au03`：

### A_au03: Residual-Demand Aware, Fidelity-Protected Gaussian Reinjection

核心思想：Gaussian control canvas 不再只预测“结构热力图”，而要预测“退化残差需求图”；Gaussian 回流不再粗暴乘性扰动 decoder feature，而是通过受限 additive correction 只在高需求区域做小幅修正。

建议机制：

1. `restoration_demand_target`: 用 GT 和输入之间的结构差异构造辅助监督目标，例如 `structure(target)`、`structure(input)`、high-frequency residual 与梯度残差的组合，重点监督“哪里缺失/模糊/需要恢复”，而不是所有边缘。
2. `protected additive feedback`: 用 predicted demand canvas 生成空间 gate，输出 `tanh` bounded feature residual，采用 `decoder_feature + strength * demand_gate * residual_delta`，并保留 identity path。
3. `confidence / demand normalization`: 对 demand gate 做局部归一或均值约束，防止全图泛化成亮度/纹理增强。
4. `canvas-feedback decoupling`: canvas 可解释性监督和 feature correction 分开，避免为了画布好看而强行改坏恢复结果。
5. `evaluation`: promotion 仍必须同时满足 official fidelity、task-specific diagnostics、mechanism panels 和视觉 artifact 检查；proxy/canvas 指标单独变好不算突破。

## 结论

当前瓶颈根因是结构控制与保真恢复没有闭环对齐。我们已经证明了：

- 分流能带来小收益，但不能自动产生有效 Gaussian 控制；
- canvas 可以被监督拉活，但直接结构化会牺牲 fidelity；
- closed-loop 能增强反馈图，但当前反馈形式太弱、太粗、缺少残差需求约束。

因此下一步必须从“结构图监督”升级为“残差需求感知 + 保真保护回流”。这才是对导师所说“Gaussian 引导回流是核心创新”的真正推进方向。
