# 顶刊顶会相关工作背景综述

## 文档定位

这份文档用于给当前 `2DGauss` 项目的阶段汇报提供一份可直接使用的相关工作背景材料。目标不是逐篇精读所有已下载论文，而是从当前论文库中挑出最有代表性的顶刊顶会工作，概括它们的研究问题、核心思想和技术趋势，方便在汇报时介绍：

1. 当前主流图像复原方法的发展脉络。
2. 高分辨率友好 restoration 方法在解决什么问题。
3. 去雾、去模糊、统一复原、先验增强、生成式恢复等路线各自的特点。
4. 这些主流工作对我们当前 `V1` 项目的启发是什么。

本文主要基于项目目录中已经下载的代表论文进行整理，强调“汇报时可讲清楚的大体内容”，而不是论文逐段复述。

## 一、当前主流工作的总体脉络

从当前顶会顶刊相关工作来看，图像复原方向大致沿着以下几条主线在发展：

### 1.1 从 CNN 骨干走向更强的全局建模

早期和中期大量工作仍然以 CNN 为主，但已经逐步开始强调：

- 多阶段渐进恢复
- 多尺度特征融合
- 长程依赖建模
- 对高分辨率输入更加友好的计算结构

代表性的思路是：在保持局部建模能力的同时，让网络更有效地利用全图上下文，而不是只依赖局部卷积感受野。

### 1.2 从“单一任务专用模型”走向“多退化统一恢复”

传统方法通常一类退化训练一个模型，例如：

- 去噪一个模型
- 去模糊一个模型
- 去雾一个模型

但实际应用中，输入退化类型往往未知，而且退化强度和空间分布也不固定。因此越来越多工作开始研究：

- all-in-one restoration
- blind restoration
- 退化感知提示或条件控制

这条路线试图让模型不依赖人工指定退化类型，而是自己感知当前输入并做自适应恢复。

### 1.3 从纯判别式回归走向更强的先验与生成式恢复

随着 diffusion、预训练模型、离散 codebook、语义提示等方法的发展，越来越多工作开始引入更强的先验来弥补传统回归式网络在细节和纹理恢复上的不足。常见趋势包括：

- 利用生成式模型补充细节
- 利用大规模预训练先验提升复原质量
- 显式建模纹理先验、高质量先验或语义先验

这类方法的目标不只是降低像素误差，而是恢复更自然、更有质感的细节。

### 1.4 从“只关心网络结构”走向“兼顾训练-推理一致性和部署效率”

对于高分辨率 restoration，很多问题并不只来自模型结构本身，还来自：

- patch 训练和 full-image 推理的不一致
- 高分辨率图像下的显存与算力压力
- 全局操作在大图场景下的分布偏移

因此近年的主流工作不仅关注模型本身强不强，也越来越重视：

- 结构是否高分辨率友好
- 推理阶段是否稳定
- 是否能在实际大图上保持性能和效率

## 二、代表工作按方向梳理

下面按照当前最值得在汇报中介绍的几条方向来概括代表论文。

## 2.1 通用高分辨率友好 restoration backbone

这一类工作是当前图像复原领域的基础背景，主要回答的问题是：

> 如何设计一个既有较强恢复能力、又能适配高分辨率图像的通用复原骨干？

### 2.1.1 MPRNet

- 代表论文：`CVPR 2021, Multi-Stage Progressive Image Restoration`
- 文件位置：`papers/top_venues/conference/2021_CVPR_MPRNet_MultiStageProgressiveImageRestoration.pdf`

大体内容：

MPRNet 是比较典型的多阶段渐进式复原网络。它的核心思路是，不把图像恢复看成一次性完成的映射，而是拆成多个连续阶段，逐步修正退化结果。这样的设计更接近“先粗恢复、再细化”的过程，在去模糊、去雨、去噪等任务上都很有代表性。

汇报时可以强调：

- 这是 CNN 时代一类很重要的高性能 baseline。
- 它体现了“多阶段渐进恢复”这一主流思想。
- 虽然不是 Transformer，但对后续高性能 restoration 设计影响很大。

### 2.1.2 Uformer

- 代表论文：`CVPR 2022, Uformer: A General U-Shaped Transformer for Image Restoration`
- 文件位置：`papers/top_venues/conference/2022_CVPR_Uformer_ImageRestoration.pdf`

大体内容：

Uformer 将 U-Net 式层级编码器-解码器结构与 Transformer 结合起来，提出了适合 restoration 的 U 形 Transformer。它通过局部窗口 self-attention 降低高分辨率下的计算成本，并引入多尺度 restoration modulator 来增强细节恢复。

汇报时可以强调：

- 它代表了“Transformer 进入低层视觉 restoration”的重要阶段。
- 重点不只是用 Transformer，而是如何让 Transformer 更适合高分辨率 restoration。
- 它说明当前主流工作开始越来越重视“局部+全局”结合的结构。

### 2.1.3 Restormer

- 代表论文：`CVPR 2022, Restormer: Efficient Transformer for High-Resolution Image Restoration`
- 文件位置：`papers/top_venues/conference/2022_CVPR_Restormer_HighResolutionImageRestoration.pdf`

大体内容：

Restormer 是当前高分辨率 restoration 背景里非常重要的一篇。它的出发点很明确：普通 Transformer 在高分辨率图像上的复杂度太高，不适合实际 restoration。为此，它重新设计了注意力与前馈模块，使网络既能建模长程依赖，又能保持对大图像的适用性。

汇报时可以强调：

- 它是“高分辨率友好 Transformer restoration backbone”的代表论文之一。
- 它关注的核心问题和我们当前项目非常接近：不是只追求性能，而是关注高分辨率下的可计算性。
- 对我们来说，它是最值得作为通用 backbone 对比和参考的工作之一。

### 2.1.4 NAFNet

- 代表论文：`ECCV 2022, Simple Baselines for Image Restoration`
- 文件位置：`papers/top_venues/conference/2022_ECCV_NAFNet_SimpleBaselinesImageRestoration.pdf`

大体内容：

NAFNet 的亮点在于“简单但强”。它试图减少系统复杂度，甚至提出很多常见非线性激活并不是必须的，可以通过更简单的结构得到非常强的结果。其核心价值不是引入更复杂的新组件，而是证明简洁结构在 restoration 上也可以非常高效。

汇报时可以强调：

- 这篇论文代表“简单高效基线”的路线。
- 它提醒我们：复原方法不一定要越来越复杂，结构简洁、计算高效也可以很强。
- 对当前项目来说，它是一个很好的反思点：我们的创新应该来自问题建模，而不是纯粹堆复杂模块。

### 2.1.5 MAXIM

- 代表论文：`CVPR 2022, MAXIM: Multi-Axis MLP for Image Processing`
- 文件位置：`papers/top_venues/batch_04/conference/2022_CVPR_MAXIM_MultiAxisMLP_ImageProcessing.pdf`

大体内容：

MAXIM 不是走 CNN 或 Transformer，而是探索 MLP 结构在图像复原中的潜力。它通过 multi-axis gated MLP 和 cross-gating block 来同时处理局部和全局信息，并在多个 restoration 任务上表现出很强的通用性。

汇报时可以强调：

- 它表明主流工作不只在卷积和 Transformer 中选边站，而是在探索新的高效建模方式。
- 它的价值在于“如何兼顾高分辨率适配性和全局建模能力”。
- 对我们来说，它提供了一种思路：跨分辨率建模不一定要完全依赖经典 attention。

## 2.2 统一复原与复杂退化建模

这一类工作主要回答的问题是：

> 当退化类型未知、甚至混合存在时，模型如何在一个统一框架里做恢复？

### 2.2.1 AirNet

- 代表论文：`CVPR 2022, All-In-One Image Restoration for Unknown Corruption`
- 文件位置：`papers/top_venues/conference/2022_CVPR_AirNet_AllInOneImageRestoration.pdf`

大体内容：

AirNet 关注的是 all-in-one restoration，即希望一个模型同时处理多种未知退化。它通过退化编码器和退化引导恢复网络来感知当前退化，再据此执行恢复。其重点不在于单一任务做到极致，而在于通用性和输入退化未知条件下的适应能力。

汇报时可以强调：

- 这篇论文代表“统一复原”的经典方向。
- 它的重要性在于把“退化感知”引入恢复流程。
- 虽然这不是我们当前论文主线，但它说明主流领域非常重视复杂退化和未知退化场景。

### 2.2.2 PromptIR

- 代表论文：`NeurIPS 2023, PromptIR: Prompting for All-In-One Blind Image Restoration`
- 文件位置：`papers/top_venues/conference/2023_NeurIPS_PromptIR_AllInOneBlindImageRestoration.pdf`

大体内容：

PromptIR 将 prompt learning 的思想引入图像复原，通过轻量提示向量编码退化相关信息，动态引导恢复网络处理不同退化类型和强度。它的核心不只是“一个模型做多任务”，而是通过提示机制让模型具有更强的条件适应能力。

汇报时可以强调：

- 它代表统一复原方向从“显式退化编码”向“轻量条件提示”演化。
- 这种工作强调模型的可适应性，而不只是结构深度。
- 对我们来说，它提示了一个方向：如果未来要做多任务版本，可以考虑引入轻量条件控制机制。

### 2.2.3 TransWeather

- 代表论文：`CVPR 2022, TransWeather: Transformer-Based Restoration of Images Degraded by Adverse Weather Conditions`
- 文件位置：`papers/top_venues/batch_04/conference/2022_CVPR_TransWeather_AdverseWeatherRestoration.pdf`

大体内容：

TransWeather 面向 adverse weather 条件下的统一图像恢复，例如雨、雾、雪等。它采用单编码器、单解码器结构，并通过 Transformer 编码器和可学习天气类型嵌入来适配不同天气退化。相比多分支、多编码器方案，它更强调统一和高效。

汇报时可以强调：

- 这篇论文说明天气退化 restoration 也在走统一框架路线。
- 它与我们当前的去雾背景是有关联的，特别是“复杂天气退化不是单一物理模型能完全解释”的观念值得注意。

## 2.3 去雾方向代表工作

这一类工作主要回答的问题是：

> 真实世界去雾为什么比合成去雾更难，以及模型如何引入更强先验来应对真实雾化分布？

### 2.3.1 RIDCP

- 代表论文：`CVPR 2023, RIDCP: Revitalizing Real Image Dehazing via High-Quality Codebook Priors`
- 文件位置：`papers/top_venues/batch_04/conference/2023_CVPR_RIDCP_RealImageDehazing_CodebookPriors.pdf`

大体内容：

RIDCP 指出真实世界去雾的一大难点在于：真实雾图分布和传统合成雾图差异很大，仅靠物理散射模型和弱先验往往不够。因此它从两个角度改进去雾：

1. 更真实地模拟真实雾图退化；
2. 引入高质量 codebook priors，让网络在恢复过程中借助更高质量图像先验。

汇报时可以强调：

- 这篇论文非常适合作为我们当前 `UHD_haze` 线的背景代表工作。
- 它体现了当前真实去雾方向的一个重要趋势：单靠传统合成数据和简单物理模型不够，需要更强、更真实的先验。
- 对我们项目的启发是：如果未来继续推进去雾，真实数据分布与高质量先验会是关键问题。

## 2.4 去模糊方向代表工作

这一类工作主要回答的问题是：

> 面对复杂、方向不均匀、空间变化的模糊，模型如何做到高质量且高效率的恢复？

### 2.4.1 MIMO-UNet

- 代表论文：`ICCV 2021, Rethinking Coarse-To-Fine Approach in Single Image Deblurring`
- 文件位置：`papers/top_venues/batch_04/conference/2021_ICCV_MIMO_UNet_RethinkingCoarseToFine_Deblurring.pdf`

大体内容：

MIMO-UNet 重新审视了传统 coarse-to-fine 去模糊策略。以往很多方法通过堆叠多个子网络处理不同尺度，虽然有效，但计算代价较高。MIMO-UNet 的核心思想是用单个 U-Net 同时接收多尺度输入并输出多尺度结果，以更低成本模拟多级级联效果。

汇报时可以强调：

- 它代表了高效去模糊架构的一类经典设计。
- 重点不是单纯做大模型，而是重新组织多尺度信息流。
- 对我们的启发是：跨尺度处理不一定要用昂贵级联，也可以通过结构重组实现。

### 2.4.2 Stripformer

- 代表论文：`ECCV 2022, Stripformer: Strip Transformer for Fast Image Deblurring`
- 文件位置：`papers/top_venues/batch_04/conference/2022_ECCV_Stripformer_FastImageDeblurring.pdf`

大体内容：

Stripformer 观察到动态模糊通常具有方向性和非均匀性，因此提出 strip-based attention：沿水平和垂直方向构造条带 token，通过 intra-strip 和 inter-strip attention 建模不同方向和不同尺度的模糊模式。

汇报时可以强调：

- 它体现了当前主流方法开始把退化本身的几何结构融入网络设计。
- 对方向性退化的建模是这篇工作的关键。
- 对我们来说，它提示了一个重要想法：结构化退化往往值得用结构化建模方式去处理，而不是只用统一的 dense feature。

### 2.4.3 FFTformer

- 代表论文：`CVPR 2023, Efficient Frequency Domain-Based Transformers for High-Quality Image Deblurring`
- 文件位置：`papers/top_venues/batch_04/conference/2023_CVPR_FFTformer_FrequencyDomainTransformers_Deblurring.pdf`

大体内容：

FFTformer 将 attention 和前馈网络的部分计算放到频域中，利用频域乘法与卷积/相关之间的关系来实现更高效的全局建模。同时，它还引入频率选择机制，让模型区分哪些高低频信息应该保留。

汇报时可以强调：

- 它是频域 restoration 思路的代表工作。
- 它说明高质量去模糊并不一定只能在空间域做文章，频域可以更自然地处理细节与模糊关系。
- 对我们项目特别有启发，因为我们的正式研究方向本身就强调纹理/频率感知。

## 2.5 训练-推理一致性与高分辨率部署问题

### 2.5.1 TLC

- 代表论文：`ECCV 2022, Improving Image Restoration by Revisiting Global Information Aggregation`
- 文件位置：`papers/top_venues/batch_04/conference/2022_ECCV_TLC_GlobalInformationAggregation_ImageRestoration.pdf`

大体内容：

TLC 关注的是一个常被忽略、但在大图 restoration 中很实际的问题：训练时通常用 patch，推理时通常用 full-resolution image。这会导致依赖全局信息聚合的模块在训练和推理阶段看到的特征分布不同，从而带来性能损失。TLC 的做法是，在推理阶段把全局操作局部化，以减轻 patch-train / full-image-test 的不一致。

汇报时可以强调：

- 这篇论文对我们当前项目特别重要。
- 它不是在比谁的 backbone 更花哨，而是在指出“训练协议本身”会影响大图 restoration 的性能。
- 对我们当前 `V1/T01` 来说，这和 patch 训练、UHD 推理之间的关系非常相关，值得重点提。

## 2.6 先验增强与生成式恢复

这一类工作主要回答的问题是：

> 当传统判别式回归不足以恢复真实细节时，如何借助更强先验或生成式机制提升结果质量？

### 2.6.1 SR3

- 代表论文：`TPAMI 2022, Image Super-Resolution via Iterative Refinement`
- 文件位置：`papers/top_venues/journal/2022_TPAMI_SR3_ImageSuperResolutionViaIterativeRefinement.pdf`

大体内容：

SR3 是 diffusion 在图像恢复领域的代表性工作之一。它将超分辨率建模为一个从噪声反复去噪、逐步细化结果的过程。相比直接回归，diffusion 更适合处理一对多、不确定性较强的细节恢复问题，因此在高倍超分等场景中更容易生成更真实的细节。

汇报时可以强调：

- 这篇论文代表生成式恢复的大方向。
- 它的核心思想是“逐步细化而不是一步回归”。
- 对我们项目来说，它不是当前主线，但说明主流领域越来越重视细节真实性和先验表达能力。

### 2.6.2 SeeSR

- 代表论文：`CVPR 2024, SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution`
- 文件位置：`papers/top_venues/batch_02/conference/2024_CVPR_SeeSR_RealWorldImageSuperResolution.pdf`

大体内容：

SeeSR 利用预训练文本生成图像模型的生成先验来做真实世界超分，但它特别强调一个问题：低质量输入可能会让语义判断出错，从而生成“看起来合理但语义不对”的细节。为此，它引入 degradation-aware prompt extractor，让模型提取更可靠的软提示和硬提示，以保持语义一致性。

汇报时可以强调：

- 这篇论文代表“生成先验 + 语义约束”的路线。
- 它体现出生成式 restoration 不只是生成更清晰图像，还要避免语义偏差。

### 2.6.3 DTPM

- 代表论文：`CVPR 2024, Learning Diffusion Texture Priors for Image Restoration`
- 文件位置：`papers/top_venues/batch_04/conference/2024_CVPR_DTPM_LearningDiffusionTexturePriors_ImageRestoration.pdf`

大体内容：

DTPM 的核心不是把 diffusion 当成完整生成器直接端到端做 restoration，而是把 diffusion 用来学习高质量纹理先验，再通过适配模块用于下游 restoration 任务。它试图解决一个关键问题：传统回归模型的细粒度纹理往往恢复不足，而 diffusion 恰好擅长建模丰富纹理。

汇报时可以强调：

- 这篇论文和我们的“纹理/频率感知”方向有明显关联。
- 它说明当前主流工作也在尝试把“纹理先验”作为独立建模对象，而不仅仅是把它藏在 backbone 里。

### 2.6.4 PTG-RM

- 代表论文：`CVPR 2024, Boosting Image Restoration via Priors from Pre-trained Models`
- 文件位置：`papers/top_venues/batch_04/conference/2024_CVPR_PTG_RM_BoostingImageRestoration_PretrainedPriors.pdf`

大体内容：

PTG-RM 的想法是：高层预训练模型已经学到了很多有用先验，虽然这些先验不能直接拿来做 restoration，但可以通过一个轻量 refinement module 对已有恢复网络进行增强。也就是说，它不是完全替换原 restoration 网络，而是给它外挂一个“预训练先验增强器”。

汇报时可以强调：

- 这篇论文体现了当前一个非常实用的趋势：用预训练模型做 restoration 增强，而不是从头重训所有东西。
- 对我们项目来说，它提示我们未来的增强方向未必只能来自主干网络本身，也可以来自外接先验模块。

## 三、如何用这些论文讲“当前主流工作背景”

如果汇报时想把背景讲得更顺，可以按下面这条逻辑线组织：

### 3.1 第一层：通用 backbone 的演化

可以先讲：

1. 早期高性能 restoration 以 CNN 和多阶段网络为主，例如 `MPRNet`。
2. 后续工作逐步引入 Transformer 和其他新结构，例如 `Uformer`、`Restormer`、`MAXIM`。
3. 这条线的共同目标是：在提升全局建模能力的同时，让模型更适合高分辨率 restoration。

### 3.2 第二层：统一复原与复杂退化

然后讲：

1. 单一任务专用模型在现实中不够灵活。
2. 因此出现了 `AirNet`、`PromptIR`、`TransWeather` 这类 all-in-one 或 blind restoration 工作。
3. 它们关注的是：如何在不知道退化类型的情况下，也能完成恢复。

### 3.3 第三层：任务特化方向仍然很重要

再讲：

1. 虽然统一复原很热门，但具体任务的高性能设计仍然非常关键。
2. 去雾方向有 `RIDCP`，强调真实数据分布和高质量先验。
3. 去模糊方向有 `MIMO-UNet`、`Stripformer`、`FFTformer`，分别代表多尺度高效设计、方向结构建模和频域建模。

### 3.4 第四层：高分辨率部署与训练协议问题

最后补一层非常重要但常被忽视的问题：

1. 对大图 restoration 来说，不只是 backbone 要强。
2. patch 训练和 full-image 推理之间的不一致也会影响效果。
3. `TLC` 这类工作说明，训练协议和推理行为本身已经成为主流关注点。

### 3.5 第五层：先验增强与生成式趋势

最后再收束到：

1. 当直接回归不足以恢复真实细节时，研究者开始引入更强先验。
2. 这包括 diffusion、语义提示、纹理先验、预训练先验等。
3. 代表工作包括 `SR3`、`SeeSR`、`DTPM`、`PTG-RM`。

## 四、这些主流工作对我们项目的启发

结合我们当前的 `V1` 方向，可以把启发概括为以下几点：

### 4.1 高分辨率友好性必须成为方法设计的一等公民

`Restormer`、`Uformer`、`MAXIM`、`TLC` 都说明：高分辨率 restoration 不能只追求 backbone 性能，还必须考虑分辨率上升后的计算复杂度、训练推理一致性和大图部署行为。

这和我们当前项目把“低分辨率紧凑建模 + 高分辨率回注”作为主线是高度一致的。

### 4.2 纹理、频率和结构先验是当前重要趋势

`FFTformer`、`RIDCP`、`DTPM` 都在不同角度说明：

- 只做普通像素回归往往不够；
- 纹理、频率和高质量先验对于真实细节恢复非常关键。

这与我们当前强调“纹理/频率感知 Gaussian 参数化”的方向是相通的。

### 4.3 任务专用强 baseline 仍然必须认真对待

虽然我们的方法定位不是纯任务专用网络，但如果要做论文实验，对比时必须尊重各任务已有强 baseline：

- 去雾不能忽略 `RIDCP`、`TransWeather`
- 去模糊不能忽略 `MIMO-UNet`、`Stripformer`、`FFTformer`
- 通用 restoration backbone 不能忽略 `Restormer`、`NAFNet`

### 4.4 我们当前不宜直接把主线改成纯生成式恢复

生成式 restoration 很强，但从论文主线看，我们当前项目更适合把：

- 低分辨率紧凑建模
- 结构化引导
- 高频/纹理感知
- 高分辨率可部署性

作为主体贡献。

更现实的做法是：

- 把 `DTPM`、`PTG-RM`、`SeeSR` 这类工作作为“先验增强”参考
- 而不是把整个项目完全改造成 diffusion 主线

## 五、适合汇报时直接使用的背景总结

如果要在汇报中用一段相对完整的话概括当前主流背景，可以直接使用下面这段：

> 当前图像复原领域的主流工作大致沿着几条路线发展。第一条是通用高性能 restoration backbone 的演化，从 CNN 多阶段结构逐步发展到更适合高分辨率输入的 Transformer、MLP 和高效混合结构，代表工作包括 MPRNet、Uformer、Restormer、NAFNet 和 MAXIM。第二条是统一复原与复杂退化建模，代表工作如 AirNet、PromptIR 和 TransWeather，强调在未知退化条件下的统一恢复能力。第三条是任务特化方向，例如真实去雾中的 RIDCP、单幅去模糊中的 MIMO-UNet、Stripformer 和 FFTformer，这些方法分别从真实数据分布、结构方向性和频域建模角度提升恢复质量。第四条是高分辨率部署相关问题，TLC 说明 patch 训练与 full-image 推理之间的不一致已经成为影响 restoration 性能的重要因素。最后一条是先验增强和生成式恢复趋势，如 SR3、SeeSR、DTPM 和 PTG-RM，它们说明当前主流方法越来越重视利用生成模型、预训练模型和纹理先验来恢复更真实的细节。

## 六、如果只保留最值得讲的几篇

如果汇报时间有限，不需要展开所有论文，建议优先讲下面这些：

1. `Restormer`
   - 代表高分辨率友好 Transformer backbone
2. `NAFNet`
   - 代表简单高效 restoration baseline
3. `AirNet` 或 `PromptIR`
   - 代表统一复原方向
4. `RIDCP`
   - 代表真实去雾方向
5. `MIMO-UNet` + `FFTformer`
   - 代表高效去模糊和频域建模
6. `TLC`
   - 代表 patch-train / full-image-test 不一致问题
7. `DTPM` 或 `PTG-RM`
   - 代表先验增强和纹理先验方向

这样既能覆盖主流背景，也能和我们当前项目的真实问题形成直接联系。

## 七、当前阶段结论

综合这些顶刊顶会工作可以看到，当前主流研究已经不再只是“做一个更深的 restoration 网络”，而是在同时解决以下几个核心问题：

1. 如何让模型在高分辨率图像上仍然可计算、可部署。
2. 如何让模型适应复杂、未知和真实世界退化。
3. 如何通过结构、频率、纹理和生成先验恢复更真实的细节。
4. 如何减少训练协议和推理行为之间的不一致。

这也说明我们当前项目的主线是有现实意义的。我们的 `V1` 并不是在重复已有工作，而是在当前主流背景下，尝试围绕“低分辨率紧凑建模 + 高分辨率结构化回注 + 纹理/频率感知”这个矛盾点，探索一条更适合超高清图像复原的新路线。
