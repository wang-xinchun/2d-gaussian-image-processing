# Top-Venue Directions For The Current Root Bottleneck

## Record Time

- Time: `2026-05-18 00:55:00 CST`

## Why This Note Exists

当前 `2DGauss` 在 `UHD_deblur` 主线上的根本瓶颈，已经由 `A_aa03` 和 `A_ab03` 收敛为：

> **不是缺少新的局部 cue，而是 Gaussian 先验与高分辨率 decoder 之间缺少足够轻量、足够选择性的共享控制机制。**

这份文档不再泛泛回顾 restoration 历史，而是只筛选对这个瓶颈最有帮助的顶刊顶会方向，并回答三个问题：

1. 哪些机制和我们当前的卡点最相似。
2. 哪些想法可以直接迁移到现有轻量 Gaussian 主线。
3. 哪些方向虽然强，但当前不适合立刻引入。

## Ten-Line Conclusion

1. 顶会文献对当前判断是支持而不是推翻：真正关键的是 **selectivity** 和 **coordination**。
2. 强方法普遍不是“再加一个 cue”，而是让 cue 通过共享控制接口进入主干。
3. `confidence / uncertainty` 机制最贴合我们当前瓶颈，应该优先做成轻量 spatial gate。
4. `prompt / degradation-aware modulation` 的真正价值，不是 prompt 本身，而是提供统一的条件控制面。
5. `routing / expert / adaptive selection` 方向说明：区域和任务难度不同，计算和修正强度不应一刀切。
6. `UHD` 场景下，高频恢复更难，控制应优先面向 high-frequency / hard-region。
7. 这直接支持下一步优先级：`A_ac03` confidence-normalized refinement。
8. 次优先方向仍是 `A_ad03` shared-prior coupled decoder-guidance / feature-refinement。
9. 全量 prompt 分支、MoE、大生成式先验当前都偏重，不适合作为第一批 short probe。
10. 当前最好的策略不是“更复杂”，而是“更统一、更有选择性地用 Gaussian”。

## Most Relevant Papers And What They Mean For Us

### 1. Confidence / Uncertainty As A Control Signal

#### `CVPR 2019` UMRL

- Paper: `Uncertainty Guided Multi-Scale Residual Learning-Using a Cycle Spinning CNN for Single Image De-Raining`
- Source: `https://openaccess.thecvf.com/content_CVPR_2019/html/Yasarla_Uncertainty_Guided_Multi-Scale_Residual_Learning-Using_a_Cycle_Spinning_CNN_for_CVPR_2019_paper.html`

对我们最有价值的不是它的 deraining 任务，而是它把 **confidence measure** 放进恢复过程本身，而不是只在 loss 层面做权重。

可迁移结论：

- restoration 不同区域的“可恢复性”不同；
- 网络应当知道自己哪里更有把握，哪里不该强推细节；
- **修正强度应该被置信度调节，而不是全局固定。**

这和我们现在的痛点完全一致。`GaussianGuidedFeatureRefinement` 目前还是统一强度的 `gate * scale / bias`，没有显式回答：

- 当前 prior 在哪里可信；
- 当前 decoder 在哪里还需要额外推动；
- 哪些区域继续强化只会带来过修正。

因此，这篇工作的直接启发不是“照搬 uncertainty head”，而是：

> **给 Gaussian refinement 加一个轻量 confidence-normalizer，让 gate 不只是“能不能开”，而是“该开多大”。**

#### `CVPR 2023` Uncertainty-Aware Unsupervised Image Deblurring with Deep Residual Prior

- Paper: `Uncertainty-Aware Unsupervised Image Deblurring with Deep Residual Prior`
- Source: `https://openaccess.thecvf.com/content/CVPR2023/papers/Tang_Uncertainty-Aware_Unsupervised_Image_Deblurring_With_Deep_Residual_Prior_CVPR_2023_paper.pdf`

这篇是更贴近 deblurring 的证据。它的核心观点是：真实场景中的 blur / kernel estimation 天然有不确定性，若不显式建模 uncertainty，恢复过程容易引入 artifact。

对我们来说，可迁移的不是它的优化框架，而是一个更普适的原则：

> **当先验本身带有不稳定性时，控制模块不该把它当作刚性真值，而应当把它作为带置信度的 soft prior。**

这与当前 `A_ab03` 的失败也吻合。显式 edge-aware prior 不是没信息，而是它被直接灌入 feature refinement，却没有“可信度折扣”。

## 2. Shared Conditional Control Is More Important Than More Cues

#### `CVPR 2022` AirNet

- Paper: `All-in-One Image Restoration for Unknown Corruption`
- Source: `https://openaccess.thecvf.com/content/CVPR2022/html/Li_All-in-One_Image_Restoration_for_Unknown_Corruption_CVPR_2022_paper.html`

AirNet 的重要性在于，它把 degradation-aware 控制当成独立问题处理：先形成 degradation representation，再用它去引导 restoration。

对我们的启发是：

- 条件控制应该是一个明确对象，而不是散落在多个补丁模块里的副产品；
- 当一个先验要影响恢复主干时，最好先形成一个更稳定的中间表示，再去调制后续模块。

这正是我们当前缺少的东西。`decoder_guidance` 与 `feature_refinement` 都在看 Gaussian prior，但它们并没有共享一个统一的 prior-control surface。

#### `NeurIPS 2023` PromptIR

- Paper: `PromptIR: Prompting for All-in-One Image Restoration`
- Source: `https://proceedings.neurips.cc/paper_files/paper/2023/hash/e187897ed7780a579a0d76fd4a35d107-Abstract-Conference.html`

PromptIR 的关键价值不在于“prompt 很新”，而在于它证明了：

> **一个轻量条件分支，如果能稳定编码 degradation-specific information，就能更高效地调动主干恢复网络。**

这和我们当前最相关的迁移点是：

- 不必给 `decoder_guidance` 和 `feature_refinement` 分别塞更多 cue；
- 更应该构造一个共享的小控制向量或小控制图，再分别作用到这两个模块。

也就是说，PromptIR 支持的是 `A_ad03` 的总体哲学，而不是支持我们去做重 prompt 分支。

#### `NeurIPS 2023` PromptRestorer

- Paper: `PromptRestorer: A Prompting Image Restoration Method with Degradation Perception`
- Source: `https://proceedings.neurips.cc/paper_files/paper/2023/hash/1c364d98a5cdc426fd8c76fbb2c10e34-Abstract-Conference.html`

PromptRestorer 比 PromptIR 更贴合我们当前问题的一点是，它不仅做 degradation perception，还强调 **gated propagation**。

这个点非常关键，因为我们现在最缺的不是 perception，而是 propagation control。

可迁移结论：

- 恢复分支不应该被所有先验一视同仁地推动；
- 需要一个“传播阀门”决定哪些感知到的先验该进一步渗透到恢复分支。

这直接对应我们后续应该做的东西：

- 不再只是“prior encoder + gate”；
- 而是“prior confidence + propagation gate + shared modulation”。

## 3. Routing / Selection Beats Cue Accumulation

#### `CVPR 2019` Attention-Based Adaptive Selection of Operations for Image Restoration in the Presence of Unknown Combined Distortions

- Paper: `Attention-Based Adaptive Selection of Operations for Image Restoration in the Presence of Unknown Combined Distortions`
- Source: `https://openaccess.thecvf.com/content_CVPR_2019/html/Suganuma_Attention-Based_Adaptive_Selection_of_Operations_for_Image_Restoration_in_the_CVPR_2019_paper.html`

这篇工作的启发非常直接：图像恢复中的有效策略，往往不是固定一条处理路径，而是针对输入自适应选择不同 operation。

这对我们当前的意义是：

- `GaussianGuidedFeatureRefinement` 不一定要继续做成单一路径 `mix -> gate -> scale/bias`；
- 可以尝试非常轻量的多路校正思路，例如：
  - preserve 路径：尽量保持 decoder feature；
  - prior-boost 路径：强调 Gaussian 结构先验；
  - detail-boost 路径：强调高频补偿；
  - 再由一个小 gate 做 soft selection。

这里的重点不是上大分支，而是承认：

> **不同区域需要不同的修正模式。**

#### `CVPR 2023` Ingredient-Oriented Multi-Degradation Learning for Image Restoration

- Paper: `Ingredient-Oriented Multi-Degradation Learning for Image Restoration`
- Source: `https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Ingredient-Oriented_Multi-Degradation_Learning_for_Image_Restoration_CVPR_2023_paper.html`

这篇文章强调不要把多退化恢复理解成“多任务并列堆叠”，而要去挖共享 ingredients，再通过动态路由做重组。

对我们的启发是：

- 不要继续把 `edge cue`、`decoder detail cue`、`directional cue` 当成三个并列按钮；
- 更应该把它们压缩成少数几个更稳定的“控制 ingredient”，例如：
  - structure confidence
  - detail deficit
  - prior-consistency

这和我们现在要做 shared control surface 是一致的。

#### `CVPR 2025` Complexity Experts are Task-Discriminative Learners for Any Image Restoration

- Paper: `Complexity Experts are Task-Discriminative Learners for Any Image Restoration`
- Source: `https://openaccess.thecvf.com/content/CVPR2025/papers/Zamfir_Complexity_Experts_are_Task-Discriminative_Learners_for_Any_Image_Restoration_CVPR_2025_paper.pdf`

这篇工作说明，恢复中的不同输入确实需要不同复杂度与不同感受野的处理；关键是 **capacity allocation**。

它对我们不是“照着做一个 MoE”，因为那会偏重；真正重要的是它再次支持：

> **困难区域和容易区域不应接受相同的修正强度。**

因此，我们更适合提炼其轻量思想：

- 不是多 expert，而是多强度；
- 不是复杂路由器，而是 complexity-aware / confidence-aware spatial gate；
- 不是 task-level allocation，而是 region-level allocation。

这进一步支持 `A_ac03`。

## 4. Feature-Space Perturbation And Shared Modulation Are Better Than Direct Injection

#### `CVPR 2025` Degradation-Aware Feature Perturbation for All-in-One Image Restoration

- Paper: `Degradation-Aware Feature Perturbation for All-in-One Image Restoration`
- Source: `https://openaccess.thecvf.com/content/CVPR2025/papers/Tian_Degradation-Aware_Feature_Perturbation_for_All-in-One_Image_Restoration_CVPR_2025_paper.pdf`

这篇文献与我们当前结论最接近的一点，是它明确指出了：

- 多种退化/条件会发生 task interference；
- 单纯共享参数不够；
- 更合理的方法是在 **feature space** 中进行选择性调制或扰动，以对齐共享参数空间。

这和 `A_ab03` 的失败形成了一个很有价值的对照：

- 我们已经证明“直接把 edge-aware prior 塞进 refinement”不是正确路径；
- 文献也表明，真正有效的往往不是 raw cue injection，而是更谨慎的 feature-space perturbation / masking / selective modulation。

因此，后面即便继续使用 edge 信息，也更应把它放在：

- confidence estimation
- region selection
- modulation strength calibration

而不是继续当作并列先验直接拼接进去。

## 5. UHD Restoration Needs Frequency-Aware Control

#### `CVPR 2025` UHD-Processor

- Paper: `UHD-processer: Unified UHD Image Restoration with Progressive Frequency Learning and Degradation-aware Prompts`
- Source: `https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_UHD-processer_Unified_UHD_Image_Restoration_with_Progressive_Frequency_Learning_and_CVPR_2025_paper.pdf`

这篇与我们的 UHD 设定直接相关。它强调两个事实：

1. UHD 场景下，高频恢复更难；
2. degradation-aware 控制如果太重，会迅速拖垮资源效率。

对我们非常关键的迁移结论是：

> **UHD 下的选择性控制，应优先围绕 high-frequency / hard-region 展开，而不是对整幅图一视同仁。**

因此，我们后续的 confidence 机制最好不是只看 attention_map，而应至少同时考虑：

- local highpass / detail residual
- prior-decoder discrepancy
- directional / structural consistency

这会比“全图统一 strength”更符合 UHD 恢复的难点分布。

## 6. Strong But Not Immediate Directions

下面这些方向是值得知道的，但当前不建议作为第一优先级。

#### `ICCV 2023` AdaCode

- Paper: `Learning Image-Adaptive Codebooks for Class-Agnostic Image Restoration`
- Source: `https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Learning_Image-Adaptive_Codebooks_for_Class-Agnostic_Image_Restoration_ICCV_2023_paper.html`

优点：

- 说明强 prior 如果能做成 adaptive basis combination，会比固定 prior 更灵活。

不适合立刻引入的原因：

- generative prior 偏重；
- 设计重心会从 Gaussian-first 轻量控制，转向更强的外部先验表达。

它更适合作为中长期方向，而不是当前 short probe。

#### `CVPR 2024` Distilling Semantic Priors from SAM to Efficient Image Restoration Models

- Paper: `Distilling Semantic Priors from SAM to Efficient Image Restoration Models`
- Source: `https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Distilling_Semantic_Priors_from_SAM_to_Efficient_Image_Restoration_Models_CVPR_2024_paper.html`

优点：

- 它证明了“训练时用重 teacher，推理时不增重”是可行的；
- 对轻量模型来说，这是一个不破坏推理效率的增强思路。

当前不作为主线的原因：

- 这会把问题转向 teacher prior distillation；
- 与我们当前急需解决的 `Gaussian-decoder selectivity` 不是同一个最小闭环。

更适合作为后续补强，而不是下一条短训主线。

#### `CVPR 2024` Selective Hourglass Mapping for Universal Image Restoration Based on Diffusion Model

- Paper: `Selective Hourglass Mapping for Universal Image Restoration Based on Diffusion Model`
- Source: `https://openaccess.thecvf.com/content/CVPR2024/html/Zheng_Selective_Hourglass_Mapping_for_Universal_Image_Restoration_Based_on_Diffusion_CVPR_2024_paper.html`

优点：

- 它强调 shared distribution + strong selective guidance 的组合；
- 思想层面与“共享控制面”相通。

当前不建议直接跟进的原因：

- diffusion 主体太重；
- 与我们现在强调的轻量、显存受控、高斯主线不匹配。

它更像一个概念参考，而不是工程上下一步该做的版本。

## What This Means For Our Codebase

当前最关键的三个接口仍然是：

- `src/uhd_gaussian/models/decoder_guidance.py`
- `src/uhd_gaussian/models/feature_refinement.py`
- `src/uhd_gaussian/models/restorer.py`

### A. `A_ac03` Should Be The First Literature-Guided Probe

目标不是再添加一个 cue，而是让已有 cue 通过 confidence-normalized control 生效。

建议最小实现：

1. 在 refinement 内构造一个 `confidence_map`，输入来自：
   - `attention_map`
   - local contrast / highpass
   - decoder 当前能量或 residual deficit
   - 可选的 directional consistency
2. 用这个 `confidence_map` 去调节现有 `gate` 或 `strength`：
   - `effective_gate = gate * confidence`
   - 或 `effective_strength = base_strength * confidence`
3. 让 edge/high-frequency 信息只影响 confidence，而不是直接进入主融合。

这样做和当前证据最一致，也最轻量。

### B. `A_ad03` Should Stay As The Second Priority

目标是给 `decoder_guidance` 和 `feature_refinement` 一个共享的小控制干线，而不是各自独立看 prior。

建议最小实现：

1. 在 `restorer.py` 中先生成一次 shared prior control feature；
2. `decoder_guidance` 和 `feature_refinement` 共同消费它；
3. 两个模块保留各自输出头，但共享 control stem。

这会直接测试我们当前的 coordination 假设。

### C. A Small Optional Follow-Up

如果 `A_ac03` 有正信号，可以继续加一个非常轻的 frequency-aware variant：

- confidence 更偏向 high-frequency hard region；
- 或只在 top-k hard spatial regions 上增强 refinement。

但这应该放在 `A_ac03` 之后，而不是一开始同时引入。

## Priority Order After The Literature Review

1. `A_ac03`: confidence-normalized Gaussian feature refinement
2. `A_ad03`: shared-prior coupled decoder-guidance / feature-refinement
3. `A_ac03` 正信号后的 frequency-aware confidence variant
4. 训练期 teacher prior/distillation 补强
5. 更重的 prompt / MoE / diffusion / generative prior

## Final Judgment

如果把这轮文献分析压缩成一句对下一步最有指导意义的话，那就是：

> **顶会强方法给我们的最大启发，不是“换一个更强先验”，而是“让已有先验以更统一、更可置信、更有选择性的方式进入恢复主干”。**

因此，当前最值得做的不是继续堆新 cue，而是：

- 让 Gaussian 先验先学会“哪里该管、哪里不该管”；
- 再让 `decoder_guidance` 和 `feature_refinement` 学会“用同一套控制语言说话”。
