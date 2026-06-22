# 顶会范式驱动的瓶颈突破分析与策略沉淀

更新时间: 2026-05-10 CST

## 文档定位

这份文档用于把当前 `2DGauss` 项目真正卡住的地方、与顶刊顶会范式的对应关系、以及后续最值得投入的突破路线，系统地沉淀成一份可复用、可追溯、可直接指导后续实验设计的长期文档。

它不是新的实验设计包，也不是代码实现说明，而是：

1. 重新定义我们当前到底卡在哪些层级；
2. 从现有顶刊顶会论文中提取可借鉴的**范式**，而不是零散模块；
3. 判断哪些路线适合当前项目主线，哪些路线不适合；
4. 给出一条短中期可执行、同时保留中期升级空间的双轨策略。

本文默认服务于后续实验设计与论文路线收敛，不直接启动训练，也不修改代码。

---

## 一、问题重述：当前不是单一瓶颈，而是三层瓶颈叠加

如果只看最近几轮实验，很容易把问题简化成“模型还不够强”或者“显存还不够高”。但结合现有实验记录、阶段报告和 benchmark 对比表，当前真正卡住项目推进的不是一个点，而是三层瓶颈叠加：

### 1.1 第一层：`UHD_haze` 的 4K JPEG / DataLoader 吞吐瓶颈

这是当前最明确、最被实验证据支持的工程瓶颈。

- 项目阶段报告已经明确指出，`A_a02` 暴露出的真正问题不是 GPU 算力，而是数据获取与解码阶段，尤其是 `UHD_haze` 的 4K JPEG 解码和 DataLoader 吞吐限制 [docs/v1_architecture_and_progress_report_20260505.md:521](/root/autodl-tmp/2DGauss/docs/v1_architecture_and_progress_report_20260505.md:521)。
- 同一份报告还明确写出：`A_a02` 之所以选择低显存、较保守的配置，是为了换取有效的 train / validation / checkpoint 闭环，而不是因为模型本身必须这样训练 [docs/v1_architecture_and_progress_report_20260505.md:549](/root/autodl-tmp/2DGauss/docs/v1_architecture_and_progress_report_20260505.md:549)。
- `DEC-000034` 也已正式确认：未来 `UHD_haze` 的高显存训练，优先事项应该是 patch cache、LMDB、WebDataset 等数据管线优化，而不是直接继续堆 batch [logs/decisions.md](/root/autodl-tmp/2DGauss/logs/decisions.md:1)。

因此，`UHD_haze` 当前的第一瓶颈不是“模型不行”，而是“当前 IO 方式不允许我们用有效、可复现、可高利用率的训练配置去公平验证模型”。

### 1.2 第二层：`patch train / full-image test` 的协议不一致瓶颈

这是当前最容易被忽略、但其实已经被 `A03 -> A_b03 -> A_c03` 这条实验线反复暴露出来的问题。

- `A03` 建立了第一条 `UHD_deblur` official baseline，但 internal validation 与 official full-image test 的最优 checkpoint 不一致 [docs/next_experiment_design_20260507.md:13](/root/autodl-tmp/2DGauss/docs/next_experiment_design_20260507.md:13)。
- `A_b03` 试图通过把训练期 internal validation 改成 full-image `crop_size=0` 来缓解这个问题，但它后来被归档为 validation protocol probe，而不是正式高显存分支 [logs/current_state.md](/root/autodl-tmp/2DGauss/logs/current_state.md:14)。
- `A_c03` 进一步证明：即使满足了 `>=80G` 显存门槛，并用 full-image internal validation 选 checkpoint，也没有超过 `A03 final.pt` 的 official baseline [logs/current_state.md](/root/autodl-tmp/2DGauss/logs/current_state.md:20)。
- 本地论文背景综述已经把 `TLC` 标为与我们“特别重要”的相关工作，因为它直指大图 restoration 里的 patch-train / full-image-test mismatch [docs/top_venue_literature_background_20260505.md:281](/root/autodl-tmp/2DGauss/docs/top_venue_literature_background_20260505.md:281)。

因此，当前 `UHD_deblur` 线的核心问题，已经不能再简单理解为“只要把 validation 改成 full-image 就够了”。现有实验已经给出更强的信号：

> 训练协议和推理协议之间的分布差异，已经成为影响结果上限的结构性问题，而不是简单的 checkpoint 选择问题。

### 1.3 第三层：`V1` 当前在 `UHD_haze / UHD_deblur` 上的建模能力瓶颈

在工程和协议问题之外，我们还存在清晰的方法能力上限。

- `V1` 当前的正式主线是“低分辨率紧凑建模 + 高频/纹理感知 Gaussian 参数化 + 高分辨率结构化回注”，这条线在研究方向上是成立的 [docs/research_direction_baseline.md:13](/root/autodl-tmp/2DGauss/docs/research_direction_baseline.md:13)。
- 但统一 UHD benchmark 表显示，我们当前与本地整理的强方法仍有明显差距：`UHD-Haze` 强结果约在 `22.586 ~ 22.725 dB`，`UHD-Blur` 强结果约在 `28.821 ~ 29.527 dB`，而我们现有结果分别仅在 `19.x` 和 `25.x` 量级 [docs/ppt_assets/unified_benchmark_table.tex:27](/root/autodl-tmp/2DGauss/docs/ppt_assets/unified_benchmark_table.tex:27)。
- 当前代码的频率建模仍然以灰度、Sobel、Laplacian 等显式描述子为主，高分辨率侧调制也仍是标量 attention + low-rank affine modulation 的组合 [src/uhd_gaussian/models/restorer.py:59](/root/autodl-tmp/2DGauss/src/uhd_gaussian/models/restorer.py:59), [src/uhd_gaussian/models/frequency.py:10](/root/autodl-tmp/2DGauss/src/uhd_gaussian/models/frequency.py:10), [src/uhd_gaussian/models/gaussian.py:48](/root/autodl-tmp/2DGauss/src/uhd_gaussian/models/gaussian.py:48)。

这说明：

1. 当前主线不是错的；
2. 但它的表达力还不足以在真实 `UHD_haze / UHD_deblur` 上逼近本地整理的强方法；
3. 后续如果只继续“盲目扩容量”或“继续堆显存”，大概率不会形成论文级突破。

---

## 二、当前证据链：哪些结论已经被验证，哪些还只是猜测

为了避免后续讨论重复回到“也许是这个问题、也许是那个问题”的状态，下面先把当前证据链压实。

### 2.1 已被实验验证的结论

#### 2.1.1 `V1/T01` 主线本身是可训练的

- `A01` 已在 `UHD-LL_down4` 上证明端到端训练闭环成立，说明当前 `V1` 不是纸面方案，而是真实可执行的基线 [logs/current_state.md](/root/autodl-tmp/2DGauss/logs/current_state.md:10)。
- `A_a02` 已在 `UHD_haze` 上跑通有效学习信号，说明主线具备跨任务可训练性 [docs/v1_architecture_and_progress_report_20260505.md:551](/root/autodl-tmp/2DGauss/docs/v1_architecture_and_progress_report_20260505.md:551)。
- `A03` 又在 `UHD_deblur` 上建立了第一条 reproducible official baseline，进一步说明当前方向不是仅对低照有效 [logs/current_state.md](/root/autodl-tmp/2DGauss/logs/current_state.md:15)。

#### 2.1.2 `UHD_haze` 当前的主要问题是 IO，不是 trainability

- `A02` 原始高显存部署因为吞吐问题中止，而不是因为模型 forward/backward 崩溃 [logs/current_state.md](/root/autodl-tmp/2DGauss/logs/current_state.md:12)。
- `A_a02` 则证明，只要降低配置、换取吞吐平衡，训练本身可以稳定进行 [docs/v1_architecture_and_progress_report_20260505.md:549](/root/autodl-tmp/2DGauss/docs/v1_architecture_and_progress_report_20260505.md:549)。

#### 2.1.3 `full-image validation` 本身并不自动带来 official test 提升

- `A_b03` 和 `A_c03` 已经给出直接证据：把 internal validation 改成 full-image 协议，以及满足高显存门槛，并不会自动刷新 `A03` official baseline [logs/current_state.md](/root/autodl-tmp/2DGauss/logs/current_state.md:17), [logs/current_state.md](/root/autodl-tmp/2DGauss/logs/current_state.md:20)。

#### 2.1.4 仅通过高频细节增强模块，尚未证明能跨过 official baseline

- `A_d03` 加入 edge-guided detail refiner 与 Laplacian 高频一致性 loss 后，训练在机制上是可行的，但首次部署卡在 post-validation OOM [logs/current_state.md](/root/autodl-tmp/2DGauss/logs/current_state.md:21)。
- `A_e03` 通过 cache-safe 修正后顺利训练完，说明运行时问题可解，但它的 official full-image test 还未完成，因此还不能宣称这条强机制已经带来真实的 official 突破 [logs/current_state.md](/root/autodl-tmp/2DGauss/logs/current_state.md:23)。

### 2.2 目前仍只是强猜测、尚未正式验证的结论

#### 2.2.1 “低照线最可能率先打出主 SOTA 亮点”

这是当前最合理的战略判断，但还不是实验事实。

- 统一 UHD 表里，低照强方法在 `27.1 ~ 27.2 dB` 左右 [docs/ppt_assets/unified_benchmark_table.tex:18](/root/autodl-tmp/2DGauss/docs/ppt_assets/unified_benchmark_table.tex:18)。
- 我们现有最好的低照结果仍在 `UHD-LL_down4` 上，而不是统一表口径里的 `UHD-LL`，因此当前低照结果甚至没有被纳入 unified UHD 主比较 [docs/ppt_assets/unified_benchmark_table.tex:50](/root/autodl-tmp/2DGauss/docs/ppt_assets/unified_benchmark_table.tex:50)。

所以：

> “低照最有希望率先突破”是一个高质量战略假设，但它成立的前提，是我们必须先把结果从 `UHD-LL_down4` 迁移到严格统一口径的 `UHD-LL`。

#### 2.2.2 “`TLC` 式修正会明显改善 `UHD_deblur` official test`

从范式上看，这个推断非常合理，因为它正好对应当前暴露的问题；但目前仓库里还没有一个真正的 `TLC` 风格实验来验证。

#### 2.2.3 “外挂式纹理/预训练先验能比纯结构扩容更有效”

这与 `DTPM / PTG-RM` 的范式高度一致，也符合我们当前代码接口，但当前项目还没有真正做过这类先验增强分支。

### 2.3 当前问题的归类

结合上面的证据，当前瓶颈应明确分类如下：

| 问题 | 类型 | 当前状态 |
|---|---|---|
| `UHD_haze` 4K JPEG 读取过慢 | 工程 / IO 瓶颈 | 已验证 |
| `patch train / full-image test` 不一致 | 协议 / 分布瓶颈 | 已验证存在，但尚未被正式修正 |
| `V1` 频率/方向/真实先验表达不足 | 方法 / 建模瓶颈 | 高可信推断，尚未被针对性验证 |

---

## 三、顶刊顶会范式映射表：哪些论文真正对应我们的瓶颈

下面只保留与当前主线高度相关的论文。每篇论文按固定五个字段梳理：

1. 它解决的真实问题
2. 它为什么有效
3. 它对应我们哪个瓶颈
4. 我们应该借鉴“范式”还是“模块”
5. 为什么不能整篇照搬

### 3.1 `Restormer`

- 真实问题：高分辨率 restoration 下，普通 Transformer 计算代价太高，难以兼顾大图适配性与全局建模 [docs/top_venue_literature_background_20260505.md:80](/root/autodl-tmp/2DGauss/docs/top_venue_literature_background_20260505.md:80)。
- 为什么有效：它把“高分辨率友好”作为方法设计的一等公民，而不是先做大 backbone 再勉强上大图。
- 对应瓶颈：它主要对应我们的**第三层建模瓶颈**，同时为第二层协议问题提供背景约束。
- 借鉴对象：借鉴的是**高分辨率友好范式**，不是具体 backbone。
- 不能整篇照搬的原因：我们的正式主线不是“再做一个更强的 high-resolution dense backbone”，而是“低分辨率紧凑建模 + 高分辨率结构化回注” [docs/research_direction_baseline.md:52](/root/autodl-tmp/2DGauss/docs/research_direction_baseline.md:52)。

### 3.2 `NAFNet`

- 真实问题：很多 restoration 方法复杂度不断上升，但未必真正围绕问题本身建模 [docs/top_venue_literature_background_20260505.md:112](/root/autodl-tmp/2DGauss/docs/top_venue_literature_background_20260505.md:112)。
- 为什么有效：它强调简洁结构也能形成强基线，迫使方法贡献必须回到问题本身。
- 对应瓶颈：主要用于约束我们的**第三层建模瓶颈**讨论。
- 借鉴对象：借鉴的是**“不要为复杂而复杂”**的设计约束。
- 不能整篇照搬的原因：我们当前的挑战不是缺一个更简单的通用 backbone，而是缺一个更贴合 UHD 主线的结构化引导增强方式。

### 3.2.5 `MAXIM`

- 真实问题：如何在通用图像复原里兼顾高分辨率友好性、全局信息建模和结构效率，而不把方法叙事局限在 CNN 或 Transformer 二选一 [docs/top_venue_literature_background_20260505.md:139](/root/autodl-tmp/2DGauss/docs/top_venue_literature_background_20260505.md:139)。
- 为什么有效：它说明强方法也可以通过新的高效建模范式获得竞争力，而不是一味走更重的 backbone [docs/top_venue_literature_background_20260505.md:146](/root/autodl-tmp/2DGauss/docs/top_venue_literature_background_20260505.md:146)。
- 对应瓶颈：它主要用于约束我们的**第三层建模瓶颈**讨论，并补强“高分辨率友好性是一等问题”的判断 [docs/top_venue_literature_background_20260505.md:409](/root/autodl-tmp/2DGauss/docs/top_venue_literature_background_20260505.md:409)。
- 借鉴对象：借鉴的是**高分辨率友好与结构效率并重**的范式约束，而不是具体的 multi-axis MLP 结构。
- 不能整篇照搬的原因：我们的主线贡献位不是提出另一套通用 restoration backbone，而是解释并解决 UHD 场景下低分辨率紧凑建模与高分辨率结构化回注的矛盾。

### 3.3 `TLC`

- 真实问题：patch 训练与 full-image 推理之间存在分布不一致，导致大图 restoration 的训练协议本身影响最终性能 [docs/top_venue_literature_background_20260505.md:281](/root/autodl-tmp/2DGauss/docs/top_venue_literature_background_20260505.md:281)。
- 为什么有效：它不是简单“换 validation 协议”，而是承认并修正训练-推理行为的不一致。
- 对应瓶颈：它直接对应我们的**第二层协议瓶颈**。
- 借鉴对象：借鉴的是**范式**，即把 patch/full-image mismatch 当成一等问题，而不是只借某个局部算子。
- 不能整篇照搬的原因：我们的模型不是 TLC 的全局信息聚合 backbone，直接照搬其实现没有意义；我们要借鉴的是“协议修正思路要嵌回当前 lowres->gaussian->highres 主线”。

### 3.4 `RIDCP`

- 真实问题：真实去雾与合成去雾分布差异很大，仅靠传统物理建模和弱先验不够 [docs/top_venue_literature_background_20260505.md:210](/root/autodl-tmp/2DGauss/docs/top_venue_literature_background_20260505.md:210)。
- 为什么有效：它同时强调更真实的数据分布和更强的高质量先验。
- 对应瓶颈：它对应我们的**第一层 `UHD_haze` 工程 + 分布双重瓶颈**，以及第三层中的真实先验不足问题。
- 借鉴对象：借鉴的是**“真实分布 + 高质量先验”范式**。
- 不能整篇照搬的原因：当前 `UHD_haze` 的头号障碍仍是 IO 吞吐；在不先解决数据管线前，直接上更强去雾先验模块会把问题混在一起。

### 3.5 `MIMO-UNet`

- 真实问题：多尺度去模糊通常依赖高代价的 coarse-to-fine 级联，效率不理想 [docs/top_venue_literature_background_20260505.md:232](/root/autodl-tmp/2DGauss/docs/top_venue_literature_background_20260505.md:232)。
- 为什么有效：它通过单个 U-Net 重组多尺度信息流，而不是盲目堆叠更复杂网络。
- 对应瓶颈：主要对应我们的**第三层建模瓶颈**。
- 借鉴对象：借鉴的是**多尺度信息流重组范式**。
- 不能整篇照搬的原因：我们当前已有低分辨率分支与高分辨率回注接口，直接改成 MIMO 风格 backbone 会把论文主线改成“另一个 deblurring architecture”。

### 3.6 `Stripformer`

- 真实问题：动态模糊具有方向性和空间非均匀性，统一的 dense feature 处理对这种结构不够敏感 [docs/top_venue_literature_background_20260505.md:249](/root/autodl-tmp/2DGauss/docs/top_venue_literature_background_20260505.md:249)。
- 为什么有效：它把退化本身的几何结构引入模型设计，用条带方向建模来捕捉方向性模糊。
- 对应瓶颈：它主要对应我们的**第三层建模瓶颈**，特别是 `UHD_deblur`。
- 借鉴对象：借鉴的是**方向结构偏置范式**。
- 不能整篇照搬的原因：我们并不想把主线改成 strip-attention backbone；更合适的方式是把方向性先验嵌回当前 Gaussian 参数化或高分辨率调制接口。

### 3.7 `FFTformer`

- 真实问题：去模糊需要更有效地区分哪些高低频信息该保留、哪些该抑制 [docs/top_venue_literature_background_20260505.md:264](/root/autodl-tmp/2DGauss/docs/top_venue_literature_background_20260505.md:264)。
- 为什么有效：它不是只“多加一个频率模块”，而是把频率选择机制放到 restoration 的核心建模逻辑里。
- 对应瓶颈：它直接对应我们的**第三层建模瓶颈**，且与当前频率感知主线天然相邻。
- 借鉴对象：借鉴的是**频率选择范式**，必要时再抽象出局部模块。
- 不能整篇照搬的原因：我们的代码已经有显式频率描述子与 Gaussian 纹理参数，真正缺的是“更有效的频率选择机制”，不是切换成完整 FFTformer 架构。

### 3.8 `DTPM`

- 真实问题：传统回归型 restoration 对细粒度纹理的恢复能力不足 [docs/top_venue_literature_background_20260505.md:331](/root/autodl-tmp/2DGauss/docs/top_venue_literature_background_20260505.md:331)。
- 为什么有效：它把 diffusion 用作纹理先验学习器，而不是直接替换整个 restoration 模型。
- 对应瓶颈：它对应我们的**第三层建模瓶颈**，特别是“真实细节表达不足”。
- 借鉴对象：借鉴的是**外挂式纹理先验范式**。
- 不能整篇照搬的原因：当前项目边界已经明确，不把纯 diffusion / generative prior 路线作为论文主体 [docs/research_direction_baseline.md:94](/root/autodl-tmp/2DGauss/docs/research_direction_baseline.md:94)。

### 3.9 `PTG-RM`

- 真实问题：预训练模型已经学到了大量高质量先验，但 restoration 网络未必能直接利用 [docs/top_venue_literature_background_20260505.md:345](/root/autodl-tmp/2DGauss/docs/top_venue_literature_background_20260505.md:345)。
- 为什么有效：它不是换掉原 restoration 主体，而是外挂一个轻量 refinement / prior enhancer。
- 对应瓶颈：对应我们的**第三层建模瓶颈**，同时与中期升级路线高度契合。
- 借鉴对象：借鉴的是**外挂式预训练先验增强范式**。
- 不能整篇照搬的原因：当前短中期更需要先解决 IO 与协议问题；先验外挂更适合作为中期 `V2` 风格增强，而不是立刻替换当前主路径。

### 3.10 对照：为什么 `AirNet / PromptIR` 不是当前主线

- 它们解决的是 all-in-one / blind restoration 的统一退化建模问题 [docs/top_venue_literature_background_20260505.md:161](/root/autodl-tmp/2DGauss/docs/top_venue_literature_background_20260505.md:161)。
- 这条路线本身重要，但当前项目的正式研究方向并不是“统一未知退化恢复”，而是“超高清场景下的低分辨率紧凑建模 + 高分辨率结构化回注” [docs/research_direction_baseline.md:18](/root/autodl-tmp/2DGauss/docs/research_direction_baseline.md:18)。
- 因此，它们可以作为长期外延参考，但不应牵引当前主线偏航。

### 3.11 对照：为什么 `SR3 / SeeSR` 不是当前主线

- 它们代表的是生成式恢复和语义先验增强的大方向 [docs/top_venue_literature_background_20260505.md:302](/root/autodl-tmp/2DGauss/docs/top_venue_literature_background_20260505.md:302)。
- 但当前项目已经明确不把纯 diffusion / generative prior 路线作为论文主体 [docs/research_direction_baseline.md:98](/root/autodl-tmp/2DGauss/docs/research_direction_baseline.md:98)。
- 它们可作为“为什么后面要引入外挂式先验增强”的论据，但不应直接改写当前主贡献叙事。

---

## 四、适配性判断：哪些范式适合我们，哪些不适合

结合项目主线、当前代码接口和现有实验事实，下面给出明确判断。

### 4.1 明确适合当前项目的范式

#### 4.1.1 `TLC` 适合，因为它正中 `patch / full-image` 不一致

我们已经不缺“意识到 validation 口径不同”的认知，`A_b03/A_c03` 已经说明仅改 validation 协议本身不够。真正缺的是把训练-推理不一致作为范式级问题显式处理。

这条线适合当前项目，是因为它强化而不是破坏我们的正式主线：

- 仍然保留低分辨率紧凑建模；
- 仍然保留高分辨率结构化回注；
- 但承认并修正 patch/full-image 分布差异。

#### 4.1.2 `FFTformer` 适合，因为我们缺的是更有效的频率选择，而不是更频繁地堆频率描述子

当前 `V1` 已经显式使用 `gray / Sobel / Laplacian / grad_mag` 描述子 [src/uhd_gaussian/models/frequency.py:26](/root/autodl-tmp/2DGauss/src/uhd_gaussian/models/frequency.py:26)，也已经在 Gaussian 参数中引入 `texture_gain / frequency / phase / beta` [src/uhd_gaussian/models/gaussian.py:56](/root/autodl-tmp/2DGauss/src/uhd_gaussian/models/gaussian.py:56)。

因此我们并不是“没有频率主线”，而是：

> 频率信息已经进来了，但频率选择机制仍然太弱。

`FFTformer` 的价值恰好在于：它提醒我们应该把“哪些频率该被强调、哪些该被抑制”当成更核心的学习对象。

#### 4.1.3 `Stripformer` 适合借鉴方向结构偏置，但不适合整套 backbone 替换

`UHD_deblur` 的一个自然问题是模糊具有方向性和空间非均匀性。当前 `V1` 的高频增强虽然有 edge-guided refiner，但整体仍偏通用高频增强，还没有显式方向结构偏置。

因此它适合变成：

- Gaussian 参数的方向敏感扩展；
- 高频调制中的方向选择增强；
- edge-guided 分支里的方向性先验。

但不适合变成“把主体换成 Stripformer”。

#### 4.1.4 `RIDCP` 适合借鉴“真实分布 + 高质量先验”，尤其服务 `UHD_haze`

当前 `UHD_haze` 线最危险的错误路线，是一边保留 4K JPEG IO 瓶颈，一边又马上堆更强去雾模块。`RIDCP` 的真正启发不是“赶紧上 codebook”，而是：

1. 真实去雾与合成分布不同；
2. 更强先验很重要；
3. 但前提是实验闭环必须可信。

所以它告诉我们的不是“先上复杂先验”，而是：

> `UHD_haze` 必须先获得可信、稳定、可高利用率的训练/评估闭环，之后再讨论先验增强是否真有效。

#### 4.1.5 `DTPM / PTG-RM` 适合做外挂式增强，不适合改成主干替换

当前项目的正式研究方向已经明确，主贡献应该落在：

- UHD 建模矛盾
- 低分辨率紧凑参数化
- 高频/纹理感知 Gaussian
- 高分辨率结构化回注

因此最适合我们的借鉴方式，不是“把项目改成生成式 restoration”，而是：

- 保持当前主干；
- 在中期用轻量外挂先验增强真实细节表达；
- 把它写成增强版，而不是换题。

### 4.2 明确不适合当前项目的路线

#### 4.2.1 不适合把项目改成纯 diffusion 主线

原因不是 diffusion 不强，而是它会改写我们的论文问题定义。当前项目真正想回答的是：如何在 UHD 场景下避免 full-resolution dense 建模，又不丢掉细节 [docs/research_direction_baseline.md:20](/root/autodl-tmp/2DGauss/docs/research_direction_baseline.md:20)。

纯 diffusion 主线会把主叙事转向“生成更真实细节”，这不是当前项目最有辨识度的贡献位。

#### 4.2.2 不适合把项目转成 all-in-one restoration 主叙事

虽然 `AirNet / PromptIR / TransWeather` 很强，但当前 `T01` 仍是单任务、单数据集 paired restoration 模板；当前最有价值的是先把单任务 UHD 场景的建模矛盾讲深，而不是提前扩展到多退化统一恢复。

#### 4.2.3 不适合继续做“只换 backbone”的论文

这会直接削弱我们的主线辨识度，并使贡献滑向“又一个 restoration backbone”，与当前研究方向基线冲突 [docs/research_direction_baseline.md:52](/root/autodl-tmp/2DGauss/docs/research_direction_baseline.md:52)。

这里不仅包括继续追 `Restormer` 或 `NAFNet` 风格的强主干替换，也包括把路线滑向 `MAXIM` 这类“换一种高性能通用 backbone”叙事。对当前项目而言，这些论文更适合作为约束参照，提醒我们高分辨率友好与结构效率很重要，但不应把论文主贡献改写成 backbone 竞赛 [docs/top_venue_literature_background_20260505.md:409](/root/autodl-tmp/2DGauss/docs/top_venue_literature_background_20260505.md:409)。

#### 4.2.4 不适合在 `UHD_haze` 上先做模型升级、后修 IO

这是当前最该避免的顺序错误。因为如果训练本身的吞吐和显存利用率不可信，就无法公平判断先验模块、频率模块或更大模型到底有没有意义。

---

## 五、最终选择的双轨解决方案

基于上面的证据链与范式映射，当前最合理的不是单一路线，而是**双轨方案**：

1. **短中期桥接方案：V1.5 风格**
   - 不脱离当前 `V1` 主线
   - 目标是解决最妨碍结果上限和论文说服力的现实瓶颈
2. **中期范式升级方案：V2 风格**
   - 保留当前论文问题定义
   - 增强当前主线对真实高频细节的表达能力

### 5.1 短中期桥接方案：V1.5 风格

#### 主线 1：低照 benchmark 口径闭环

当前低照最关键的问题，不是 “A01 不够好”，而是：

> `A01` 的结果在 `UHD-LL_down4` 上，而 unified UHD 主比较表使用的是 `UHD-LL`。

这点已经在本地数据文档与 benchmark 表中写得很清楚：

- `UHD-LL` 才是论文 `Setting 2` 的核心 UHD benchmark [docs/dataset_inventory_from_2d_dataset_paper.md:32](/root/autodl-tmp/2DGauss/docs/dataset_inventory_from_2d_dataset_paper.md:32)。
- `UHD-LL_down4` 本来就是项目早期用来快速跑通 dataloader、loss 和 validation 的可行性入口，而不是最终统一 benchmark [docs/dataset_organization_status_20260424.md:224](/root/autodl-tmp/2DGauss/docs/dataset_organization_status_20260424.md:224)。
- unified UHD 表也明确写了：当前低照结果因为在 `UHD-LL-down4` 上，所以没有被纳入统一主表 [docs/ppt_assets/unified_benchmark_table.tex:50](/root/autodl-tmp/2DGauss/docs/ppt_assets/unified_benchmark_table.tex:50)。

因此，低照线若要承担“最可能率先形成主 SOTA 亮点”的角色，第一任务不是继续在 `UHD-LL_down4` 上涨点，而是：

1. 迁移到严格统一口径的 `UHD-LL`；
2. 补齐 `PSNR + SSIM`；
3. 让低照结果真正进入统一 UHD 主表比较。

#### 主线 2：`TLC` 式训练-推理一致性修正

`A_b03/A_c03` 已经证明：只改 validation 协议不够，必须把 patch-train / full-image-test mismatch 作为范式级问题处理。

当前最应该推进的不是继续盲目试：

- 更大 batch
- 更大容量
- 更多 full-image val 采样

而是：

1. 明确哪些操作在 patch 训练时依赖了局部统计、而在 full-image test 时发生分布漂移；
2. 识别 `V1` 里最可能受这个问题影响的部位：
   - lowres feature -> gaussian parameter prediction
   - gaussian rendered attention -> highres modulation
   - edge-guided detail refiner
3. 设计一条 `TLC` 风格的“一致性修正”分支，而不是只改 evaluation schedule。

#### 主线 3：方向/频率增强，而非继续盲目扩容量

当前 `A_a03` 做了 modest capacity / Gaussian-expression 增强，`A_c03` 又做了高显存 full-image validation，结果都表明：

> 单纯扩容量、扩 batch 或改 validation，不足以把 `UHD_deblur` 推到论文主表级别。

因此，针对 `UHD_deblur`，后续更值得投入的是：

1. `Stripformer` 风格的方向结构偏置
2. `FFTformer` 风格的频率选择增强
3. 但实现上仍嵌入当前：
   - lowres compact modeling
   - gaussian parameter prediction
   - highres modulation

换句话说，正确方向不是“变成 Stripformer/FFTformer”，而是让当前 Gaussian-guided 线具备：

- 更强的方向敏感性
- 更有效的频率选择能力

#### 主线 4：`UHD_haze` 先修 IO，再谈模型

这条线没有模糊空间。当前项目已经有足够多证据说明：

1. `UHD_haze` 的首要障碍是 IO；
2. 没有可信 IO 闭环，就没有可信模型结论；
3. 因此 `LMDB / patch cache / WebDataset` 是前置条件，不是附属优化。

`UHD_haze` 线的正确节奏应当是：

1. 先把数据管线改造到能支撑高显存、有效 validation/checkpoint 闭环；
2. 再重建可信 official baseline；
3. 之后再讨论先验增强。

### 5.2 中期范式升级方案：V2 风格

在不改论文主线的前提下，中期最有价值的升级方向不是继续 backbone 替换，而是：

> 增加轻量的纹理先验 / 预训练先验增强分支，作为外挂 enhancer，补齐当前 `Sobel/Laplacian + scalar attention` 对真实细节表达仍然偏弱的问题。

这条线最贴近 `DTPM / PTG-RM` 的共同启发：

1. 先保留现有 restoration 主干；
2. 让更强的纹理或预训练先验进入恢复过程；
3. 把它们当增强器，而不是主体。

对当前项目来说，这条中期路线的最大价值在于：

- 它不会改写当前论文问题定义；
- 它能强化“高频/纹理感知 Gaussian + 高分辨率结构化回注”这条主线；
- 它比“直接转 diffusion 主线”更可控，也更符合项目边界。

---

## 六、优先级排序后的实验路线建议

下面给出的不是正式 run 分配，而是后续实验设计回合应采用的固定优先级顺序。

### 6.1 路线 1：低照统一口径闭环

#### 它验证什么假设

验证“当前主线在正式 `UHD-LL` benchmark 上也有机会形成论文级亮点，而不是只在 `down4` 口径上成立”。

#### 它若失败，说明什么

说明当前低照正向信号对正式 UHD benchmark 的迁移不够顺利，那么“低照主突破”这个战略判断就需要降级。

#### 它若成功，下一步该接什么

成功后，应优先补齐 `SSIM`，并把低照线正式纳入统一主表；随后再决定是否把它作为第一主结果线。

### 6.2 路线 2：`UHD_deblur` 的 `TLC` 式一致性修正

#### 它验证什么假设

验证“当前 `UHD_deblur` 结果受限于 patch/full-image mismatch，而不只是受限于容量或 validation 协议”。

#### 它若失败，说明什么

说明协议不一致虽然存在，但并不是当前最主要的结果上限来源，届时应把注意力更多转向建模增强。

#### 它若成功，下一步该接什么

成功后应立即进入方向/频率增强分支，因为这意味着协议瓶颈已被部分打开。

### 6.3 路线 3：`UHD_deblur` 的方向/频率增强分支

#### 它验证什么假设

验证“当前去模糊上限主要来自对方向结构与频率选择的表达不足”。

#### 它若失败，说明什么

说明 `UHD_deblur` 的主瓶颈可能不在方向/频率建模，而在更上游的训练协议或更下游的真实先验表达。

#### 它若成功，下一步该接什么

成功后可以继续推进 `UHD_deblur` 主结果线，并视结果决定是否让它从机制验证线升级为论文主表线。

### 6.4 路线 4：`UHD_haze` 的 IO 管线改造

#### 它验证什么假设

验证“当前 `UHD_haze` 结果上限首先受制于数据吞吐，而不是结构本身”。

#### 它若失败，说明什么

若 IO 改造后仍无法显著改善有效训练包络，则说明 `UHD_haze` 线可能还存在更深层的训练配置或数据分布问题。

#### 它若成功，下一步该接什么

成功后应先重建可信 official baseline，再决定是否进入先验增强。

### 6.5 路线 5：`UHD_haze` 的先验增强分支

#### 它验证什么假设

验证“在真实去雾上，引入更强真实先验比单纯 backbone 调整更有效”。

#### 它若失败，说明什么

说明当前主瓶颈可能仍是数据分布、评估口径或训练协议，而不是先验本身。

#### 它若成功，下一步该接什么

成功后可将其作为第二亮点线或跨任务泛化强证据。

### 6.6 路线 6：中期 `V2` 先验外挂路线

#### 它验证什么假设

验证“当前主线的核心建模框架是对的，但真实高频细节需要更强外挂先验补充”。

#### 它若失败，说明什么

说明当前主线在表达接口设计上可能本身还不足以吸收外挂先验，届时需要回头重审 `gaussian -> modulation` 的中间接口。

#### 它若成功，下一步该接什么

成功后可把它发展成 `V2` 风格的增强版主结果线。

---

## 七、明确拒绝的路线

为避免后续反复摇摆，下面这些路线在当前阶段应明确拒绝。

### 7.1 不把项目改成纯 diffusion 主线

原因：这会改写论文问题定义，使主贡献从“UHD 建模矛盾与结构化回注”漂移到“生成更真实细节”。

### 7.2 不把项目转成 all-in-one restoration 主叙事

原因：当前 `T01` 仍是单任务 paired restoration 模板；而统一退化恢复不是当前最有辨识度的问题位。

### 7.3 不做“只换更强 backbone”的论文

原因：这会削弱当前主线辨识度，并与研究方向基线冲突 [docs/research_direction_baseline.md:52](/root/autodl-tmp/2DGauss/docs/research_direction_baseline.md:52)。

### 7.4 不把 `UHD_haze` 的模型升级排在 IO 修复之前

原因：没有可信吞吐闭环，就没有可信结论。

### 7.5 不把 `UHD_deblur` 当前 `25.x dB` 的微增幅误当成论文级主突破

原因：相对 `A03` 的 `+0.3 / +0.5 dB` 对内部研发很重要，但与本地统一 UHD 表中的强方法相比，仍然只是“机制有用”的信号，而不是“论文主表已形成竞争力”的信号。

---

## 八、最终结论

综合当前实验、代码接口、benchmark 表以及本地顶刊顶会论文背景，最适合我们当前项目的突破路线不是“继续盲目扩模型”，也不是“换成另一个强 backbone”，而是：

1. 先把真正的工程瓶颈和协议瓶颈拆开；
2. 在短中期用 `UHD-LL` 统一口径闭环、`TLC` 式一致性修正、`Stripformer/FFTformer` 风格的方向/频率增强，以及 `UHD_haze` 的 IO 优先策略，构成一条不脱离当前主线的 `V1.5` 路线；
3. 在中期再用 `DTPM / PTG-RM` 风格的外挂式纹理/预训练先验增强，构成 `V2` 风格升级路线；
4. 始终保持论文主线不变：**低分辨率紧凑建模 + 高频/纹理感知 2D Gaussian 参数化 + 高分辨率结构化回注**。

换句话说，当前最值得投入的，不是“再找一个更强网络替换 `V1`”，而是：

> 让当前这条跨分辨率 Gaussian-guided 主线，先在正式 benchmark 口径上站稳，再用训练-推理一致性修正和方向/频率/先验增强，把它真正推向论文级结果。
