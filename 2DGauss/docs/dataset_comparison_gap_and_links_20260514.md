# 数据集横向比较缺口与下载入口（2026-05-14）

## 目的

这份文档用于回答当前两个问题：

1. 我们现在是否已经具备与同方向顶刊顶会论文做直接横向比较的数据集基础；
2. 若还不完整，缺哪些标准数据集，以及这些数据集的相对官方获取入口在哪里。

本轮只整理链接与比较口径，不执行任何下载。

## 结论

当前不是“完全没法横向比较”，而是：

- 已经可以覆盖 `UHD` 主线比较；
- 但还不能完整覆盖这篇 `2D数据集论文.pdf` 以及同方向论文常见的通用 benchmark 横向表。

当前本地已经具备：

- `UHD-LL`
- `UHD-LL_down4`
- `UHD_haze`
- `UHD_deblur`
- `LOL-v1`
- `LOL-v2`

当前本地仍缺、会阻碍更完整横向比较的标准数据集：

- `GoPro`
- `RESIDE` 数据线，至少包括 `ITS` 与 `SOTS-Indoor / SOTS`

因此：

- 如果只看 UHD 低光照 / 去雾 / 去模糊主线，我们已经可以做一部分 paper-style 对比；
- 如果要更完整对齐论文里 `Setting 1` 或通用 benchmark 表格，还需要补 `GoPro` 与 `RESIDE` 这条数据线。

## 论文里是否直接给了下载链接

对 `/root/autodl-tmp/2DGauss/2D数据集论文.pdf` 做定点抽取后，可以确认：

- 论文给出了数据集名称、参考文献线索，以及一个代码入口 `https://github.com/supersupercong/`；
- 论文没有把 `UHD-LL / UHD-Haze / UHD-Blur / LOL / SOTS-ITS / GoPro` 的直接下载 URL 全部逐个写出来；
- `UHD-Haze` 和 `UHD-Blur` 在论文中明确写的是采用 `Wang et al., 2024b` 提供的重组版本，因此更实用的下载入口要去对应项目页或代码仓库追。

## 推荐下载入口

下面优先列相对官方、作者页、项目页或论文代码仓库里公开给出的入口。

### 1. UHD-LL

- 官方项目页：`https://li-chongyi.github.io/UHDFour/`
- 官方直链（项目页 Dataset 按钮）：`https://drive.google.com/drive/folders/1IneTwBsSiSSVXGoXQ9_hE1cO2d4Fd4DN`

### 2. UHD-Haze

- 官方代码仓库：`https://github.com/supersupercong/UHDformer`
- 官方直链（README 的 Datasets Download）：`https://drive.google.com/drive/folders/1PVCPkhqU_voPVFZj3FzAtUkJnQnF9lSa`

### 3. UHD-Blur

- 官方代码仓库：`https://github.com/supersupercong/UHDformer`
- 官方直链（README 的 Datasets Download）：`https://drive.google.com/drive/folders/1O6JYkOELLhpEkirAnxUB2JGWMqgwVvmX`

### 4. LOL

- 官方项目页：`https://daooshee.github.io/BMVC2018website/`
- 官方直链（LOL paired dataset, Google Drive）：`https://drive.google.com/open?id=157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB`

### 5. RESIDE / ITS / SOTS-Indoor

- 官方数据页：`https://sites.google.com/view/reside-dehaze-datasets/reside-standard`
- `ITS (Indoor Training Set)` 官方页给出的 Dropbox 跳转：`https://bit.ly/3iwHmh0`
- `SOTS (Synthetic Objective Testing Set)` 官方页给出的 Dropbox 跳转：`https://bit.ly/2XZH498`

说明：

- 在 `2D数据集论文.pdf` 的语境里，`SOTS-ITS` 与后文 `SOTS-Indoor` 基本可以视作同一条 `RESIDE` 室内去雾 benchmark 线索；
- 实际使用时通常是 `ITS` 作为训练集，`SOTS-Indoor` 作为测试集。

### 6. GoPro

- 官方数据页：`https://seungjunnah.github.io/Datasets/gopro.html`
- 官方 Google Drive 直链：`https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view?usp=sharing`
- 官方 Hugging Face 直链：`https://huggingface.co/datasets/snah/GOPRO_Large/resolve/main/GOPRO_Large.zip`

## 可选辅助入口

这些入口不是必须，但有时能帮助你核对版本来源：

- `UHDformer` 预训练与相关数据总入口：`https://drive.google.com/drive/folders/1XwIWOjOepUA-kXoaeOlzMDijVK4LMj-1?usp=sharing`
- `UHDformer` 仓库 README：`https://github.com/supersupercong/UHDformer`

## 当前建议

如果你的目标是“尽快补齐能和同方向论文直接横向对照的最小缺口”，优先级建议是：

1. 先补 `GoPro`
2. 再补 `RESIDE` 的 `ITS + SOTS`

原因很直接：

- `LOL` 和三条 UHD 主线我们已经有；
- 现在真正妨碍复现更完整 benchmark 横表的，就是去模糊通用集 `GoPro` 和去雾通用集 `RESIDE`。

## 备注

本轮我没有替你下载任何数据，只整理了可追溯的公开入口，供你手动判断和获取。
