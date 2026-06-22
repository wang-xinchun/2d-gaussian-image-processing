# 数据集验收报告：2D Gaussian Dataset

## 验收时间

- 北京时间：`2026-04-24 08:34:05 CST`
- 数据目录：`/root/autodl-tmp/2D_Gaussian_Dataset`
- 百度网盘工具区：`/root/autodl-tmp/Baiduwangpan`

## 本次验收依据

本次先只读查看了百度网盘工具区的使用说明和脚本口径，重点确认：

- `review/QUICKSTART.md`
- `review/ACCESS_PLAN.md`
- `manifests/README.md`
- `scripts/README.md`
- `scripts/run_baidu_download_list.sh`
- `scripts/start_baidu_download_list_bg.sh`

本次真正用于完整性验收的清单是：

- Manifest：
  `/root/autodl-tmp/Baiduwangpan/manifests/download_2d_gaussian_dataset_20260423.csv`
- Results：
  `/root/autodl-tmp/Baiduwangpan/manifests/download_2d_gaussian_dataset_20260423.results.csv`

清单字段含义：

- Manifest 字段：`remote_path`, `local_path`, `expected_size`
- Results 字段：`remote_path`, `local_path`, `status`, `finished_at_utc`, `notes`
- 下载脚本中，`downloaded` 或 `already_present` 且 `notes=size_match_*` 表示该目标曾按期望大小成功落地。

## 验收方法

本次采用两层验收：

1. **下载完整性验收**
   - 逐条读取 manifest 的 `local_path` 和 `expected_size`
   - 检查本地文件是否存在
   - 检查本地文件大小是否等于 `expected_size`
   - 检查 results 中是否至少有一次 `downloaded/already_present + size_match_*` 成功记录

2. **已解包成对数据可用性验收**
   - 检查 input/gt 是否成对
   - 检查 `data_list.txt` 是否与实际文件名一致
   - 对已解包图像执行 PIL 可读性检查
   - 对未解包大归档仅做 manifest 级验收，不擅自解压

## Manifest 完整性结论

总体结论：

> **Manifest 中的 `6906 / 6906` 个目标文件全部存在，本地大小全部等于 `expected_size`。**

进一步检查 results：

> **`6906 / 6906` 个目标在 results 历史记录中均至少出现过一次 `downloaded` 或 `already_present`，并带有 `size_match_*` 记录。**

需要注意：

- results 的“最后一条状态”并不全部干净。
- 有 `513` 条 UHD-LL 记录的最后状态是 `failed/download_command_failed`。
- 但这 `513` 条在更早的 results 历史中已经有 `downloaded/already_present + size_match_*` 记录，并且当前本地文件也存在且大小完全匹配。
- 因此本次判定为：**results 尾部存在并发/重试追加造成的滞后失败记录，不构成本地下载完整性失败。**

## 按数据集汇总

| 数据项 | Manifest 条目 | 期望总字节数 | 本地大小匹配 | Results 历史成功 | 验收结论 |
|---|---:|---:|---:|---:|---|
| `4KDehaze` | 13 | 50,955,842,396 | 13/13 | 13/13 | 通过，归档完整；usable 视图见下文 |
| `8K` | 7 | 132,041,868,189 | 7/7 | 7/7 | 通过，归档完整；尚未解包验收内容语义 |
| `LOL-v1` | 1002 | 348,062,896 | 1002/1002 | 1002/1002 | 通过，成对图像可用 |
| `LOL-v2` | 1580 | 553,982,416 | 1580/1580 | 1580/1580 | 通过，成对图像可用 |
| `UHD-LL` | 4302 | 2,802,299,212 | 4302/4302 | 4302/4302 | 通过，成对 UHD 图像可用 |
| `4K-Rain13k.zip` | 1 | 14,153,963,710 | 1/1 | 1/1 | 通过，归档完整；未解包 |
| `UHD-LL_down4.zip` | 1 | 571,350,122 | 1/1 | 1/1 | 通过，归档完整；未解包 |
| **总计** | **6906** | **201,427,368,941** | **6906/6906** | **6906/6906** | **下载完整性通过** |

当前数据目录总占用约：

- `192G`

## 已解包成对数据验收

### LOL-v1

| 子集 | input | gt | 成对数 | data_list | 缺失配对 | 图像坏文件 | 分辨率 |
|---|---:|---:|---:|---:|---:|---:|---|
| `our485` | 485 | 485 | 485 | 485 | 0 | 0 | 600x400 |
| `eval15` | 15 | 15 | 15 | 15 | 0 | 0 | 600x400 |

结论：`LOL-v1` 可直接作为成对低光照增强数据使用。

### LOL-v2

| 子集 | input | gt | 成对数 | data_list | 缺失配对 | 图像坏文件 | 分辨率 |
|---|---:|---:|---:|---:|---:|---:|---|
| `Train` | 689 | 689 | 689 | 689 | 0 | 0 | 600x400 |
| `Test` | 100 | 100 | 100 | 100 | 0 | 0 | 600x400 |

结论：`LOL-v2` 可直接作为成对低光照增强数据使用。

### UHD-LL

| 子集 | input | gt | 成对数 | data_list | 缺失配对 | 图像坏文件 | 分辨率 |
|---|---:|---:|---:|---:|---:|---:|---|
| `training_set` | 2000 | 2000 | 2000 | 2000 | 0 | 0 | 3840x2160 |
| `testing_set` | 150 | 150 | 150 | 150 | 0 | 0 | 3840x2160 |

结论：`UHD-LL` 是当前最适合作为首个训练闭环的数据集。它已经具备标准 input/gt 成对结构，且训练/测试规模与前期数据集清单一致。

## 4KDehaze usable 视图

`4KDehaze` 的原始分卷归档已经通过 manifest 完整性验收。

此前已解压并建立可用视图：

- 可用清晰图：`2040`
- 路径：`/root/autodl-tmp/2D_Gaussian_Dataset/4KDehaze/usable/groundtrues`
- 已知坏图：`1`
- 坏图：`002040.jpg`
- 坏图记录：
  `/root/autodl-tmp/2D_Gaussian_Dataset/4KDehaze/usable/corrupt_images.csv`

结论：

- `4KDehaze` 下载归档完整。
- 当前可用视图是 `2040` 张 UHD clean ground truth 图像。
- 它还不能被判定为完整成对去雾训练集，因为当前 usable 视图只有 `groundtrues`，没有对应 hazy/input 配对。

## 归档类数据状态

下面这些项目已经通过 manifest 文件大小验收，但本次没有擅自解包，因此只给出“归档完整”结论，不给出内容语义验收：

- `8K`
- `4K-Rain13k.zip`
- `UHD-LL_down4.zip`

如后续需要使用它们，应单独执行解包、目录结构审计和图像可读性检查。

## 最终验收结论

本次可以正式给出两条结论：

1. **下载完整性层面：通过。**
   - Manifest 全部 `6906` 个目标文件本地存在。
   - 所有本地文件大小均与 `expected_size` 完全一致。
   - 所有目标在 results 历史中均至少有一次成功 size-match 记录。

2. **当前可直接用于训练的成对数据：通过。**
   - `LOL-v1` 可用。
   - `LOL-v2` 可用。
   - `UHD-LL` 可用，且是当前最适合启动首个 UHD 训练闭环的数据集。

需要保留的边界：

- `4KDehaze` 当前可用的是清晰图集合，不是完整成对去雾数据。
- `8K`、`4K-Rain13k.zip`、`UHD-LL_down4.zip` 目前只完成归档级验收，尚未进行解包后内容验收。
- results 文件存在 `513` 条尾部 failed 记录，但本地文件与历史成功记录均支持判定为完整；后续如果要让 results 末尾状态也干净，可以单独做一次非下载式状态回写或生成 cleaned acceptance 表。
