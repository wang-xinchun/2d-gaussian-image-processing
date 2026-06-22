# Advisor Feedback Mainline Analysis 2026-05-19

更新时间: 2026-05-19 21:47:47 CST

## 1. Feedback Original

本轮汇报后，导师给出的阶段性建议原文要作为后续主线依据保留：

```text
1. 对于分支入口，怎么分出来哪些是低频，哪些是高频，就是怎么把信息去分流
2. 具体技术细节。比如，上分支高频用的卷积还是什么技术，下面的LR用的 什么，下采样用的什么，每个模块具体的技术是什么。
3. 高斯的热力图到底是什么，物理意义是什么，能在画板上去重构细节。讲好故事。这个板块为什么能做好 。最后就是高斯引导回流，这个创新很核心，这个地方要深度研究，也是写论文的核心创新点。
4. 去雾效果不行。感观很差。不要局限于这些SSIM PSNR这些传统指标，而是要追究真实实际效果。
```

这不是简单的 presentation polish，而是对主线提出了四个硬要求：显式分流、模块级技术可解释、Gaussian heatmap 物理语义与可视化重构、真实观感/跨任务效果评估。

## 2. Current Achievements

当前可以保留为阶段性成果的部分：

- `M04/A_y03` 是当前冻结最强里程碑，`UHD_deblur/test` full-image official PSNR 为 `26.272695829299355`。
- 快速汇报包已经包含指标、推理图像、说明文档和轻量对比 PDF。
- 架构已经具备跨分辨率雏形：HR 分支保细节，LR 分支生成 Gaussian 控制先验，高斯 heatmap 回流调制 HR high-pass feature 和 decoder feature。
- 近期短训已经给出负面边界：hard competition 不稳，soft residual routing 有改善但不够；单 gate/cue 不能解决根本瓶颈。

当前还不能过度宣称的部分：

- 现有高低频分流更多是隐式工程结构，不是严格、可画、可验证的频带分解。
- `gaussian_attention.png` 当前主要是 renderer 输出的 attention/control map，可解释为结构控制热力图，但还不是直接可生成 RGB 细节的纹理画板。
- 去雾样例只是跨任务可视化，且观感和指标都差，不能作为 dehazing 性能。

## 3. Existing Technical Anatomy

当前 `A_y03` 的主要模块和技术落点如下。

| Module | Current technique | Meaning |
|---|---|---|
| HR branch | `3x3 Conv2d` + `StackedResidualBlocks` | 在原分辨率上保留空间细节和局部纹理底座 |
| LR entry | `F.pixel_unshuffle` with `lowres_scale=4` | 把局部 `4x4` 空间邻域折叠到通道，降低空间尺寸但保留局部采样信息 |
| LR projection | `1x1 Conv2d` + residual depthwise blocks | 把低分辨率高通道输入压成紧凑 feature |
| Texture/frequency extractor | grayscale + Sobel x/y + gradient magnitude + Laplacian, then `1x1 Conv2d` + residual depthwise blocks | 在 LR 上提取粗纹理/边缘/频率线索 |
| Gaussian head | `1x1 Conv2d` predicts atoms * 10 params and affine rank coeffs | 在 LR 网格预测可渲染 Gaussian 参数和 low-rank affine 系数 |
| Gaussian renderer | anisotropic Gaussian envelope + cosine texture carrier + spectral decay | 把 LR 参数渲染成 HR attention/control map |
| Directional prior | weighted direction/frequency/texture/coherence projection | 从 Gaussian 参数中汇总方向、频率、纹理强度和相干性 |
| HR high-pass path | `highres_feature - avg_pool3x3(highres_feature)` | 形成当前的高频残差候选 |
| Gaussian injection | `highres_feature + attention_map * highpass` | 用 Gaussian heatmap 选择性增强 HR high-pass feature |
| Low-rank affine | upsample coeffs + rank basis scale/bias | 用 LR Gaussian side information 调制 HR features |
| Decoder | `1x1 Conv2d` + residual depthwise blocks + `3x3 Conv2d` | 从融合 feature 输出 RGB residual |
| Gaussian decoder guidance | gate/scale/bias modulation | 让 Gaussian prior 调制 decoder feature |
| Feature refinement | feature-space gate/scale/bias | 让 Gaussian 保持结构控制角色，不直接生成 RGB |
| Edge refiner | Sobel/Laplacian/local high-pass descriptors + residual output | 给末端 residual 提供轻量边缘修正 |

这套实现已经能回答“用了什么模块”，但还没完全回答导师追问的“为什么这里就自然是高频、那里就自然是低频，以及信息怎么被分流”。

## 4. Core Gaps

### 4.1 Branch Entry Is Not Explicit Enough

当前分支入口是：

- HR branch: 原图直接过 `3x3 conv` 和 residual blocks；
- LR branch: 原图经 `pixel_unshuffle(scale=4)` 后走低分辨率表示；
- high-pass: 在 HR feature 上用 `feature - avg_pool3x3(feature)` 形成局部高通。

这能工作，但在论文叙事上仍偏弱。导师会继续追问：为什么 `pixel_unshuffle` 后就是低频？为什么 `avg_pool` 差分就是高频？哪些内容进入 Gaussian，哪些内容留给 HR decoder？目前答案是经验性的，不够机制化。

下一阶段应把分流做成显式、可视化、可消融的模块，例如：

- 用 `lowpass = blur/downsample/upsample` 与 `highpass = x - lowpass` 明确拆分；
- 用 Laplacian pyramid / Haar wavelet / DWT 作为可解释分解；
- 用 learned gate 在 low/high bands 上做 soft routing，但保留可视化 band maps；
- 把 Gaussian head 主要绑定到 low/structure band，把 HR detail branch 绑定到 high/detail band。

### 4.2 Gaussian Heatmap Needs a Stronger Physical Story

当前 heatmap 是 Gaussian renderer 输出：

```text
amplitude * anisotropic Gaussian envelope * texture carrier * spectral decay
```

再经过 sigmoid 得到高分辨率 attention/control map。它的合理语义应定义为：

> 每个像素处“结构修复需求/可信控制强度”的空间分布，携带位置、方向、尺度、局部频率倾向和置信强度。

它不是：

- GT error map；
- RGB texture map；
- 直接重构细节的图像；
- 普通 CNN attention。

导师提到“能在画板上去重构细节”，这里需要谨慎转化。更合理的论文故事是：

- Gaussian heatmap 本身是 control canvas，不是 RGB canvas；
- Gaussian atoms 在画板上重构的是结构响应场，例如边缘走向、模糊方向、纹理频率倾向和修复强度；
- 真正 RGB 细节由 HR decoder 在这个 control canvas 引导下合成；
- 因此创新不在“Gaussian 直接画出所有纹理”，而在“Gaussian 以低维可解释方式画出修复控制场，再回流驱动高分辨率细节恢复”。

这也解释了为什么 Gaussian 不能被要求承担所有高熵纹理生成，否则会失去结构控制器的清晰语义。

### 4.3 Gaussian-Guided Feedback Is the Paper Core

导师明确指出“高斯引导回流”是核心创新点。现有主线已经多次证明：

- Gaussian 只生成 heatmap 不够；
- 单独 feature refinement / decoder detail gate 不够；
- hard router 破坏协同；
- soft residual routing 稍好但还没形成突破。

因此下一阶段不该只是继续调 router，而应把“回流”定义成论文核心机制：

```text
LR structure Gaussian
-> interpretable control canvas
-> HR high-frequency/detail branch
-> decoder state feedback
-> Gaussian-guided selective reinjection
```

这里的关键是闭环：Gaussian 不只是前向给一个 mask，而是根据 HR detail/decoder state 决定哪里该增强、哪里该抑制、哪里需要边缘修正。

### 4.4 Dehazing Exposes Real-Effect Weakness

去雾样例差不是小问题。它说明当前 `A_y03` 是 deblur-specialized checkpoint，Gaussian control 对 haze degradation 没有学到正确物理变量。去雾需要处理：

- 全局/局部透射率；
- airlight / color cast；
- contrast recovery；
- depth-like smooth veil；
- halo 与过增强。

这些与 deblur 的局部 high-pass recovery 不同。如果继续用 deblur checkpoint 做 haze，只会出现观感差、颜色灰、局部伪影或过弱修复。后续必须把真实观感纳入评价，而不是只盯 PSNR/SSIM。

## 5. Mainline Pivot

导师反馈后的主线应调整为：

> 从“继续优化 shared-control router”升级为“显式高低频分流 + Gaussian control canvas + Gaussian-guided feedback/reinjection + perceptual/real-effect evaluation”。

这不是否定 `A_y03`，而是把 `A_y03` 的成功点讲清楚、做硬、做深。`A_y03` 证明 lightweight Gaussian-guided UHD deblur 可行；下一步要证明它为什么可行、哪里可解释、如何回流、如何改善真实观感。

## 6. Proposed Next Experiments

### A_aq03: explicit frequency split Gaussian feedback probe

目的：回答导师第 1 点“分支入口怎么分低频/高频”。

建议机制：

- 在输入端构造显式 low/high bands：
  - `low = blur/downsample/upsample(x)` 或 Laplacian pyramid low band；
  - `high = x - low`；
- LR Gaussian head 只吃 low/structure band + coarse frequency descriptors；
- HR branch 显式吃 high/detail band；
- Gaussian heatmap 回流时只调制 HR high-detail feature，而不是泛化地乘 feature high-pass。

验收重点：

- 可导出 `low_band.png`, `high_band.png`, `gaussian_control_canvas.png`, `guided_high_detail.png`；
- PSNR 不一定立刻突破，但分流图必须能讲清楚。

### A_ar03: Gaussian control canvas auxiliary reconstruction

目的：回答导师第 3 点“高斯热力图是什么，物理意义是什么，能否在画板上重构细节”。

建议机制：

- 从 Gaussian params/attention/directional prior 预测一个低容量 structure canvas；
- supervision 不直接用 RGB GT，而用 edge/gradient/Laplacian/high-pass target；
- 让 Gaussian canvas 对齐结构，而不是背负完整纹理生成任务。

验收重点：

- 导出 canvas vs edge/high-pass target；
- heatmap 解释从“黑箱 attention”变成“结构修复需求图”。

### A_as03: closed-loop Gaussian feedback controller

目的：把导师强调的“高斯引导回流”做成论文核心创新点。

建议机制：

- 先由 Gaussian 生成 control canvas；
- decoder 产生中间 detail state；
- controller 比较 Gaussian control 与 decoder detail/edge state；
- 输出 selective reinjection map，用于 feature refinement / edge residual / decoder modulation。

关键变化：

- 不再让三路角色硬竞争；
- 让 Gaussian control 和 decoder state 做闭环匹配；
- 输出可视化的 feedback map / demand map / suppression map。

### A_haze_diag: real-effect dehazing diagnostic

目的：正面处理导师第 4 点“去雾观感差，不要只看传统指标”。

建议先不直接训练大模型，而做诊断：

- 建立 haze qualitative set；
- 记录 input / output / GT / residual / contrast / color shift / saturation / edge artifact；
- 加入 LPIPS、NIQE/BRISQUE 或 CLIP-IQA/MUSIQ 一类 perceptual/no-reference 指标作为辅助；
- 明确当前 deblur checkpoint 对 haze 是 out-of-task failure，不能宣称泛化。

后续如果要做去雾，应单独设计 haze-aware Gaussian control：transmission-like smooth control + color/airlight correction branch + detail restoration branch。

## 7. Paper Story

建议论文核心叙事改为：

1. UHD restoration 的难点不是单纯卷积容量，而是全分辨率细节昂贵、全局结构低频、局部纹理高频混在一起。
2. 我们把 restoration 拆成 structure control 与 detail synthesis。
3. LR branch 用低成本方式估计 Gaussian control canvas，表达哪里需要修、方向/尺度/频率是什么。
4. HR branch 保留高频细节底座。
5. Gaussian-guided feedback/reinjection 把低维结构控制回流到高分辨率 decoder，选择性调制细节恢复。
6. Heatmap 是可解释的修复控制场，不是普通 attention，也不是 RGB 生成器。
7. 真实效果评估要包括观感、伪影、色彩、跨退化失败边界，而不仅是 PSNR/SSIM。

## 8. Immediate Policy

- 不启动 `A_an03/A_ao03` 作为惯性下一步。
- 不再把“调 router”当作唯一主线。
- 下一批工作先做设计和可视化/诊断，把显式分流与 Gaussian feedback 的故事做硬。
- `M04/A_y03` 继续冻结隔离，只作为 deblur milestone 和汇报 anchor。
- 去雾结果必须按 failure/diagnostic 处理，除非后续做 haze-specific 训练和评价。
