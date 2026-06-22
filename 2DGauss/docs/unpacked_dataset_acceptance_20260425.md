# 新增解包数据集验收报告

## 验收时间

- 北京时间：`2026-04-25 11:59:18 CST`
- 数据目录：`/root/autodl-tmp/2D_Gaussian_Dataset`
- 项目目录：`/root/autodl-tmp/2DGauss`
- 原始验收 JSON：`/root/autodl-tmp/2DGauss/datasets/unpacked_acceptance_20260425.json`

## 本次对象

用户已完成压缩包文件完整性校验。本次只负责解压、目录结构验收、图片数量验收、配对关系验收和抽样图片可读性检查。

| 数据集 | 原始压缩包 | 解包路径 | 处理方式 |
|---|---|---|---|
| `UHD-Haze` | `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze-20260424T115742Z-3-001.zip` ... `-005.zip` | `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze` | 普通 zip 分包，逐包 `unzip -q -n` |
| `UHD-Blur` | `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur-20260424T115900Z-3-001.zip` ... `-003.zip` | `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur` | 普通 zip 分包，逐包 `unzip -q -n` |
| `4K` | `/root/autodl-tmp/2D_Gaussian_Dataset/4K.z01` ... `4K.z05` and `4K.zip` | `/root/autodl-tmp/2D_Gaussian_Dataset/4K` | split ZIP，先 `zip -s 0` 合并临时包，再 `unzip -q -n` |

4K 解包过程中生成过临时合并包：

- `/root/autodl-tmp/2D_Gaussian_Dataset/4K_unsplit_tmp_20260425.zip`

该临时包已在解包完成后删除。

### 2026-04-25 12:10:05 CST 归档删除更新

用户要求只保留解压后的数据集，因此本轮验收通过后，以下原始压缩包已删除：

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

删除后再次确认解包目录仍存在，图片数量不变：

- `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze`: `5042` 张图片
- `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur`: `4528` 张图片
- `/root/autodl-tmp/2D_Gaussian_Dataset/4K`: `64792` 张图片

## 验收方法

本次验收使用脚本：

- `/root/autodl-tmp/2DGauss/scripts/validate_unpacked_datasets_20260425.py`

脚本检查内容：

- 每个关键目录的文件数、图片数、非图片文件数。
- `input`/`gt` 或 `HR`/`LR` 文件名集合是否完全对齐。
- 每个关键目录抽样打开图片，读取格式、分辨率、色彩模式。
- 输出原始 JSON 到 `/root/autodl-tmp/2DGauss/datasets/unpacked_acceptance_20260425.json`。

Pillow 可用版本：`11.3.0`。

## UHD-Haze

- 路径：`/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze`
- 任务：UHD 去雾
- 论文清单期望：train `2290` / test `230`

| 子集 | input | gt | 成对数 | 缺 input | 缺 gt | 抽样分辨率 | 结论 |
|---|---:|---:|---:|---:|---:|---|---|
| `train` | 2290 | 2290 | 2290 | 0 | 0 | `3840x2160` RGB JPEG | 符合论文清单 |
| `test` | 231 | 231 | 231 | 0 | 0 | `3840x2160` RGB JPEG | 本地为 231 对，比论文清单 230 多 1 对 |

验收结论：

> `UHD-Haze` 已解包，训练集数量符合论文清单，训练/测试 input-gt 文件名集合均完全对齐，抽样图片可读。
> 注意：本地 `test` split 是 `231` 对，不是论文清单记录的 `230` 对。后续任何训练、验证、评估 manifest 必须显式写明采用本地 `231` 对口径，或者另行核对官方清单后固定一个过滤列表。

## UHD-Blur

- 路径：`/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur`
- 任务：UHD 去模糊
- 论文清单期望：train `1964` / test `300`

| 子集 | input | gt | 成对数 | 缺 input | 缺 gt | 抽样分辨率 | 结论 |
|---|---:|---:|---:|---:|---:|---|---|
| `train` | 1964 | 1964 | 1964 | 0 | 0 | `3840x2160` RGB JPEG | 符合论文清单 |
| `test` | 300 | 300 | 300 | 0 | 0 | `3840x2160` RGB JPEG | 符合论文清单 |

`UHD_deblur/test/gt300` 中存在 4 个非图片元数据文件：

- `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/test/gt300/.VID_002_00274.jpg.baiduyun.uploading.cfg`
- `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/test/gt300/.VID_008_00059.jpg.baiduyun.uploading.cfg`
- `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/test/gt300/.VID_012_00173.jpg.baiduyun.uploading.cfg`
- `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/test/gt300/.VID_013_00226.jpg.baiduyun.uploading.cfg`

另外存在 `.DS_Store` 元数据文件：

- `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/.DS_Store`
- `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/test/.DS_Store`

这些元数据文件未删除；后续 dataloader 必须按图片扩展名过滤。

验收结论：

> `UHD-Blur` 已解包并通过验收，可作为配对 UHD 去模糊训练/测试数据集使用。

## 4K

- 路径：`/root/autodl-tmp/2D_Gaussian_Dataset/4K`
- 任务：超分辨率
- split：`train` / `test`
- 每个 split 均包含：`HR`、`LR_X2`、`LR_X3`、`LR_X3_BD`、`LR_X3_DN`、`LR_X4`、`LR_X8`、`LR_X16`

| 子集 | HR | 每个 LR 目录 | 配对情况 |
|---|---:|---:|---|
| `train` | 5999 | 5999 | 每个 LR 目录与 HR 文件名集合完全一致 |
| `test` | 2100 | 2100 | 每个 LR 目录与 HR 文件名集合完全一致 |

4K 总图片数：

- train：`5999 * 8 = 47992`
- test：`2100 * 8 = 16800`
- total：`64792`

抽样分辨率：

| 目录 | 分辨率 | 格式 |
|---|---|---|
| `HR` | `3840x2160` | RGB PNG |
| `LR_X2` | `1920x1080` | RGB PNG |
| `LR_X3` | `1280x720` | RGB PNG |
| `LR_X3_BD` | `1280x720` | RGB PNG |
| `LR_X3_DN` | `1280x720` | RGB PNG |
| `LR_X4` | `960x540` | RGB PNG |
| `LR_X8` | `480x270` | RGB PNG |
| `LR_X16` | `240x135` | RGB PNG |

验收结论：

> `4K` 已解包并通过验收，可作为 4K 超分辨率训练/测试数据集使用。HR 与所有 LR 多倍率目录的文件名集合完全对齐。

## 磁盘与归档状态

- 解包验收完成时 `/root/autodl-tmp` 可用空间约 `223G`。
- 按用户后续要求删除已无用原始压缩包后，`/root/autodl-tmp` 可用空间约 `343G`。
- 已删除本轮生成的 4K 临时合并包 `/root/autodl-tmp/2D_Gaussian_Dataset/4K_unsplit_tmp_20260425.zip`。
- 已删除本轮新解包并验收通过的 `UHD-Haze`、`UHD-Blur`、`4K` 原始压缩包。

## 总结论

| 数据集 | 验收状态 | 是否可直接训练 | 口径备注 |
|---|---|---|---|
| `UHD-Haze` | 通过，带 test 数量口径备注 | 是 | train `2290`；本地 test `231`，比论文清单 `230` 多 1 对 |
| `UHD-Blur` | 通过 | 是 | train `1964` / test `300` |
| `4K` | 通过 | 是 | train HR `5999` / test HR `2100`，各 LR 完全对齐 |

本轮之后，论文清单里最高优先级的 `UHD-Haze` 和 `UHD-Blur` 已从“缺失”推进为“本地可训练”。`4K` 也已成为可用的 4K 超分辨率数据集。
