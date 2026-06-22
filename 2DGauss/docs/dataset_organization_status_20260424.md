# 数据集整理情况总览

## 记录时间

- 北京时间：`2026-04-24`
- 数据根目录：`/root/autodl-tmp/2D_Gaussian_Dataset`
- 项目记录目录：`/root/autodl-tmp/2DGauss/docs`

## 当前磁盘状态

- 数据目录总占用：约 `316G`
- `/root/autodl-tmp` 当前可用空间：约 `484G`

### 2026-04-25 更新

- 新增解包并验收：`UHD_haze`、`UHD_deblur`、`4K`
- 解包后 `/root/autodl-tmp` 当前可用空间：约 `223G`
- 删除已验收原始压缩包后 `/root/autodl-tmp` 当前可用空间：约 `343G`
- 本轮新增验收报告：`/root/autodl-tmp/2DGauss/docs/unpacked_dataset_acceptance_20260425.md`
- 本轮原始验收 JSON：`/root/autodl-tmp/2DGauss/datasets/unpacked_acceptance_20260425.json`

## 当前已整理数据集

| 数据集 | 当前路径 | 当前状态 | 是否可直接训练 | 任务/用途 |
|---|---|---|---|---|
| `UHD-LL` | `/root/autodl-tmp/2D_Gaussian_Dataset/UHD-LL` | 已下载、已验收、已成对整理 | 是 | UHD 低光照增强 |
| `UHD-LL_down4` | `/root/autodl-tmp/2D_Gaussian_Dataset/UHD-LL_down4` | 已解包、已验收、已成对整理 | 是 | `UHD-LL` 下采样训练/调试 |
| `LOL-v1` | `/root/autodl-tmp/2D_Gaussian_Dataset/LOL-v1` | 已下载、已验收、已成对整理 | 是 | 低光照增强，普通分辨率 |
| `LOL-v2` | `/root/autodl-tmp/2D_Gaussian_Dataset/LOL-v2` | 已下载、已验收、已成对整理 | 是 | 低光照增强，普通分辨率 |
| `4K-Rain13k` | `/root/autodl-tmp/2D_Gaussian_Dataset/4K-Rain13k` | 已解包、已验收、已成对整理 | 是 | 4K 去雨 |
| `8K` | `/root/autodl-tmp/2D_Gaussian_Dataset/8K/extracted/8K` | split ZIP 已处理、已解包、已验收 | 是 | 8K 超分辨率 |
| `4KDehaze` | `/root/autodl-tmp/2D_Gaussian_Dataset/4KDehaze/usable/groundtrues` | 已解包并建立 clean usable 视图 | 部分可用 | clean GT 图像集合，不是成对去雾数据 |
| `UHD-Haze` | `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze` | 已解包、已验收、已成对整理 | 是 | UHD 去雾 |
| `UHD-Blur` | `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur` | 已解包、已验收、已成对整理 | 是 | UHD 去模糊 |
| `4K` | `/root/autodl-tmp/2D_Gaussian_Dataset/4K` | split ZIP 已处理、已解包、已验收 | 是 | 4K 超分辨率 |

## 详细数量

### UHD-LL

- 训练 input：`2000`
- 训练 gt：`2000`
- 测试 input：`150`
- 测试 gt：`150`
- 分辨率：`3840x2160`
- 状态：可直接作为首个 UHD 监督训练闭环数据集。

### UHD-LL_down4

- 训练 input：`2000`
- 训练 gt：`2000`
- 测试 input：`150`
- 测试 gt：`150`
- 分辨率：`960x540`
- 状态：可用于快速调试、低成本训练验证、数据管线冒烟测试。

### LOL-v1

- `our485/input`：`485`
- `our485/gt`：`485`
- `eval15/input`：`15`
- `eval15/gt`：`15`
- 分辨率：`600x400`
- 状态：可用，适合普通分辨率低光照预实验。

### LOL-v2

- `Train/input`：`689`
- `Train/gt`：`689`
- `Test/input`：`100`
- `Test/gt`：`100`
- 分辨率：`600x400`
- 状态：可用，适合普通分辨率低光照预实验。

### 4K-Rain13k

- `train/input`：`12500`
- `train/target`：`12500`
- `test/input`：`500`
- `test/target`：`500`
- 分辨率：`3840x2160`
- 状态：可直接作为 UHD/4K 监督训练数据集。

### 8K

- 解包路径：`/root/autodl-tmp/2D_Gaussian_Dataset/8K/extracted/8K`
- `train/HR`：`2029`
- `test/HR`：`937`
- 每个 split 均包含：
  - `HR`
  - `LR_X2`
  - `LR_X3`
  - `LR_X3_BD`
  - `LR_X3_DN`
  - `LR_X4`
  - `LR_X8`
  - `LR_X16`
- 各 LR 目录与 HR 文件名集合完全对齐。
- 分辨率：
  - `HR`: `7680x4320`
  - `LR_X2`: `3840x2160`
  - `LR_X3` / `LR_X3_BD` / `LR_X3_DN`: `2560x1440`
  - `LR_X4`: `1920x1080`
  - `LR_X8`: `960x540`
  - `LR_X16`: `480x270`
- 状态：可直接作为 8K 超分辨率训练/测试数据集。

### 4KDehaze

- usable clean groundtruth：`2040`
- 路径：`/root/autodl-tmp/2D_Gaussian_Dataset/4KDehaze/usable/groundtrues`
- 已排除坏图：`002040.jpg`
- 状态：可作为 clean UHD 图像源、GT 图像库或合成退化源；当前没有 hazy/input 配对，因此不是完整监督去雾训练集。

### UHD-Haze

- 解包路径：`/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze`
- `train/input`：`2290`
- `train/gt`：`2290`
- `test/input`：`231`
- `test/gt`：`231`
- 配对情况：train/test input-gt 文件名集合完全一致。
- 分辨率：`3840x2160`
- 状态：可作为配对 UHD 去雾训练/测试数据集。
- 口径备注：本地 `test` split 为 `231` 对，比论文清单记录的 `230` 多 1 对；后续训练/评估 manifest 必须显式记录采用口径。

### UHD-Blur

- 解包路径：`/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur`
- `train/input_new`：`1964`
- `train/gt_new`：`1964`
- `test/input300`：`300`
- `test/gt300`：`300`
- 配对情况：train/test input-gt 文件名集合完全一致。
- 分辨率：`3840x2160`
- 状态：可作为配对 UHD 去模糊训练/测试数据集。
- 备注：目录内有少量 `.DS_Store` 和 `.baiduyun.uploading.cfg` 元数据文件，后续 dataloader 必须按图片扩展名过滤。

### 4K

- 解包路径：`/root/autodl-tmp/2D_Gaussian_Dataset/4K`
- `train/HR`：`5999`
- `test/HR`：`2100`
- 每个 split 均包含：
  - `HR`
  - `LR_X2`
  - `LR_X3`
  - `LR_X3_BD`
  - `LR_X3_DN`
  - `LR_X4`
  - `LR_X8`
  - `LR_X16`
- 各 LR 目录与 HR 文件名集合完全对齐。
- 分辨率：
  - `HR`: `3840x2160`
  - `LR_X2`: `1920x1080`
  - `LR_X3` / `LR_X3_BD` / `LR_X3_DN`: `1280x720`
  - `LR_X4`: `960x540`
  - `LR_X8`: `480x270`
  - `LR_X16`: `240x135`
- 状态：可直接作为 4K 超分辨率训练/测试数据集。

## 已删除的中间/冗余压缩包

下面这些压缩包已经解包并验收通过，因此按用户要求删除以释放空间：

- `/root/autodl-tmp/2D_Gaussian_Dataset/UHD-LL_down4.zip`
- `/root/autodl-tmp/2D_Gaussian_Dataset/4K-Rain13k.zip`

8K 处理过程中生成过临时单体包：

- `/root/autodl-tmp/2D_Gaussian_Dataset/8K/8K_unsplit_tmp.zip`

该临时包已在 8K 解包验收完成后删除。

## 仍保留的原始归档

8K 原始 split ZIP 分卷仍保留：

- `/root/autodl-tmp/2D_Gaussian_Dataset/8K/8K.z01`
- `/root/autodl-tmp/2D_Gaussian_Dataset/8K/8K.z02`
- `/root/autodl-tmp/2D_Gaussian_Dataset/8K/8K.z03`
- `/root/autodl-tmp/2D_Gaussian_Dataset/8K/8K.z04`
- `/root/autodl-tmp/2D_Gaussian_Dataset/8K/8K.z05`
- `/root/autodl-tmp/2D_Gaussian_Dataset/8K/8K.z06`
- `/root/autodl-tmp/2D_Gaussian_Dataset/8K/8K.zip`

4KDehaze 原始分卷归档仍保留：

- `/root/autodl-tmp/2D_Gaussian_Dataset/4KDehaze/4KDehazing.part01.rar`
- ...
- `/root/autodl-tmp/2D_Gaussian_Dataset/4KDehaze/4KDehazing.part12.rar`
- `/root/autodl-tmp/2D_Gaussian_Dataset/4KDehaze/4KDehazing.pth`

2026-04-25 新增下载归档已删除：

这些归档对应的数据集已经解包并验收通过。按用户“只保留解压缩后的数据集”的要求，下面这些原始压缩包已在 `2026-04-25 12:10:05 CST` 后删除：

- `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze-20260424T115742Z-3-001.zip`
- `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze-20260424T115742Z-3-002.zip`
- `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze-20260424T115742Z-3-003.zip`
- `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze-20260424T115742Z-3-004.zip`
- `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze-20260424T115742Z-3-005.zip`
- `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur-20260424T115900Z-3-001.zip`
- `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur-20260424T115900Z-3-002.zip`
- `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur-20260424T115900Z-3-003.zip`
- `/root/autodl-tmp/2D_Gaussian_Dataset/4K.z01`
- `/root/autodl-tmp/2D_Gaussian_Dataset/4K.z02`
- `/root/autodl-tmp/2D_Gaussian_Dataset/4K.z03`
- `/root/autodl-tmp/2D_Gaussian_Dataset/4K.z04`
- `/root/autodl-tmp/2D_Gaussian_Dataset/4K.z05`
- `/root/autodl-tmp/2D_Gaussian_Dataset/4K.zip`

保留的解包目录：

- `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze`
- `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur`
- `/root/autodl-tmp/2D_Gaussian_Dataset/4K`

## 训练建议

当前项目若进入可行性验证，建议优先顺序：

1. `UHD-LL_down4`：快速跑通 dataloader、loss、validation 和可视化。
2. `UHD-LL`：首个正式 UHD 低光照训练闭环。
3. `4K-Rain13k`：第二个 UHD/4K 成对复原任务，用于验证方法泛化。
4. `UHD-Haze`：UHD 去雾主线数据集，可用于论文清单的去雾 Setting 2。
5. `UHD-Blur`：UHD 去模糊主线数据集，可用于论文清单的去模糊 Setting 2。
6. `4K`：用于 4K 超分辨率方向和多倍率 HR/LR 管线验证。
7. `8K`：用于超分辨率方向和极高分辨率推理/显存验证。
8. `LOL-v1` / `LOL-v2`：低成本普通分辨率预实验。
9. `4KDehaze`：暂作为 clean 图像源，不作为成对去雾训练集。

## 相关验收文档

- `docs/dataset_acceptance_20260424.md`
- `docs/unpacked_dataset_acceptance_20260424.md`
- `docs/unpacked_dataset_acceptance_20260425.md`
- `docs/dataset_inventory_from_2d_dataset_paper.md`
