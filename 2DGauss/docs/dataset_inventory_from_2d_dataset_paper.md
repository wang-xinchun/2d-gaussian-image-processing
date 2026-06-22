# 数据集清单：来自《2D数据集论文.pdf》的数据源梳理

## 文档目的

这份文档用于梳理 [2D数据集论文.pdf](/root/autodl-tmp/2DGauss/2D数据集论文.pdf) 中涉及到的全部核心数据集，并明确：

- 数据集名称
- 对应任务
- 在论文中的角色
- 论文内给出的来源线索
- 后续我们寻找下载链接时应该优先追哪篇参考文献

## 总体结论

这篇论文围绕 3 个 UHD 任务组织实验：

1. `UHD low-light image enhancement`
2. `UHD image dehazing`
3. `UHD image deblurring`

对应地，文中实际涉及到的数据集可以分成两类：

- `A 类：直接使用的 UHD 基准数据集`
- `B 类：Setting 1 用到的通用训练/测试数据集`

## A 类：直接使用的 UHD 基准数据集

这些是论文真正围绕的核心 UHD benchmark。

| 数据集 | 任务 | 在文中的角色 | 数据规模 | 分辨率 | 来源线索 |
|---|---|---|---:|---|---|
| `UHD-LL` | 低光照增强 | UHD benchmark；Setting 2 直接训练/测试；也用于消融 | train `2000` / test `150` | `3840×2160` | 文中明确指向 `Li et al., 2023` |
| `UHD-Haze` | 去雾 | UHD benchmark；Setting 2 直接训练/测试 | train `2290` / test `230` | `3840×2160` | 文中说明使用 `Wang et al., 2024b` 提供的重组版本 |
| `UHD-Blur` | 去模糊 | UHD benchmark；Setting 2 直接训练/测试 | train `1964` / test `300` | `3840×2160` | 文中说明使用 `Wang et al., 2024b` 提供的重组版本 |

## B 类：Setting 1 使用的通用数据集

这些不是 UHD benchmark 本体，而是 Setting 1 中用于训练的一般分辨率数据集，随后拿去测试 UHD benchmark。

| 数据集 | 任务 | 在文中的角色 | 来源线索 | 备注 |
|---|---|---|---|---|
| `LOL` | 低光照增强 | Setting 1 训练集；也作为后续 general benchmark 对比集 | `Wei et al., 2018` | 文中多次明确提到 |
| `SOTS-ITS` | 去雾 | Setting 1 中写作训练用数据 | `Li et al., 2019a` | 论文写法略混乱，和后文 `SOTS-Indoor` 需要一起看 |
| `SOTS-Indoor` | 去雾 | 后文 general benchmark 对比集 | `Li et al., 2019a` | 很可能与上面的 `SOTS-ITS` 属于同一 RESIDE 室内去雾数据线 |
| `GoPro` | 去模糊 | Setting 1 训练集；也作为后续 general benchmark 对比集 | `Nah et al., 2017` | 文中多次明确提到 |

## 数据源追踪关系

下面按“我们后续找下载链接时应该先追谁”来整理。

### 1. UHD-LL

- 数据集名：`UHD-LL`
- 任务：UHD 低光照增强
- 文中出处：`UHD-LL (Li et al., 2023)`
- 对应参考文献标题：
  `Li C, Guo CL, Zhou M, Liang Z, Zhou S, Feng R, Loy CC (2023) Embedding fourier for ultra-high-definition low-light image enhancement. In: ICLR`
- 说明：
  这是最直接的数据源入口。后续如果要找 `UHD-LL` 的下载链接，优先看这篇 `Li et al., 2023`。

### 2. UHD-Haze

- 数据集名：`UHD-Haze`
- 任务：UHD 去雾
- 文中说明：
  采用 `Wang et al., 2024b` 提供的重组版本。
- 对应重组版本参考文献标题：
  `Wang C, Pan J, Wang W, Fu G, Liang S, Wang M, Wu X, Liu J (2024b) Correlation matching transformation transformers for UHD image restoration. In: AAAI`
- 文中进一步说明：
  该重组版本是从原始数据来源 `Zheng et al., 2021` 重建的。
- 对应原始来源标题：
  `Zheng Z, Ren W, Cao X, Hu X, Wang T, Song F, Jia X (2021) Ultra-high-definition image dehazing via multi-guided bilateral learning. In: CVPR, pp 16185–16194`
- 说明：
  如果只想要作者在这篇论文里真正使用的版本，优先追 `Wang et al., 2024b`。
  如果想要追溯最原始 UHD 去雾数据来源，再看 `Zheng et al., 2021`。

### 3. UHD-Blur

- 数据集名：`UHD-Blur`
- 任务：UHD 去模糊
- 文中说明：
  采用 `Wang et al., 2024b` 提供的重组版本。
- 对应重组版本参考文献标题：
  `Wang C, Pan J, Wang W, Fu G, Liang S, Wang M, Wu X, Liu J (2024b) Correlation matching transformation transformers for UHD image restoration. In: AAAI`
- 文中进一步说明：
  该重组版本由原始来源 `Deng et al., 2021b` 重建而来。
- 对应原始来源标题：
  `Deng S, Ren W, Yan Y, Wang T, Song F, Cao X (2021b) Multi-scale separable network for ultra-high-definition video deblurring. In: ICCV, pp 14030–14039`
- 说明：
  后续如果要找作者论文实际使用版本，优先追 `Wang et al., 2024b`；如果要追最原始 UHD 去模糊数据来源，再看 `Deng et al., 2021b`。

### 4. LOL

- 数据集名：`LOL`
- 任务：一般分辨率低光照增强
- 文中出处：`Wei et al., 2018`
- 对应参考文献标题：
  `Wei C, Wang W, Yang W, Liu J (2018) Deep retinex decomposition for low-light enhancement. In: BMVC`
- 说明：
  这是 Setting 1 的通用训练来源，也是后文 general benchmark 对比集。

### 5. SOTS-ITS / SOTS-Indoor

- 数据集名：`SOTS-ITS` / `SOTS-Indoor`
- 任务：一般分辨率去雾
- 文中出处：`Li et al., 2019a`
- 对应参考文献标题：
  `Li B, Ren W, Fu D, Tao D, Feng D, Zeng W, Wang Z (2019a) Benchmarking single-image dehazing and beyond. TIP 28(1):492–505`
- 说明：
  文中在 Setting 1 写的是 `SOTS-ITS`，而在后续 general benchmark 对比表里写的是 `SOTS-Indoor`。
  这大概率指向同一条数据线，即 `RESIDE` 室内去雾 benchmark：`ITS` 常作为训练集，`SOTS-Indoor` 常作为测试集。
  后续找下载链接时，优先通过 `Li et al., 2019a` 这篇 benchmark 论文回溯。

### 6. GoPro

- 数据集名：`GoPro`
- 任务：一般分辨率动态去模糊
- 文中出处：`Nah et al., 2017`
- 对应参考文献标题：
  `Nah S, Hyun Kim T, Mu Lee K (2017) Deep multi-scale convolutional neural network for dynamic scene deblurring. In: CVPR`
- 说明：
  这是 Setting 1 使用的通用去模糊训练源，也是后文的通用 benchmark 对比集。

## 按任务汇总

### 低光照增强

- `UHD-LL`
- `LOL`

### 去雾

- `UHD-Haze`
- `SOTS-ITS`
- `SOTS-Indoor`

### 去模糊

- `UHD-Blur`
- `GoPro`

## 当前最重要的结论

如果我们后续的目标是“所有数据集都通过这篇论文追到下载入口”，那么最关键的上游文献入口一共是下面 6 条：

1. `Li et al., 2023` for `UHD-LL`
2. `Wang et al., 2024b` for reorganized `UHD-Haze`
3. `Wang et al., 2024b` for reorganized `UHD-Blur`
4. `Zheng et al., 2021` for original UHD dehazing source
5. `Deng et al., 2021b` for original UHD deblurring source
6. `Wei et al., 2018` / `Li et al., 2019a` / `Nah et al., 2017` for general benchmark datasets `LOL / SOTS / GoPro`

## 当前阶段未做的事

本轮只完成了**数据集梳理**，尚未继续做下面两步：

- 逐篇打开上游数据集论文并提取官方下载链接
- 验证这些下载链接当前是否仍然可用

如果继续推进，下一步最自然的工作就是：
**逐篇打开这 6 条上游文献，把每个数据集的下载链接、官网/仓库地址、许可证或使用说明整理成正式数据源表。**
