# A_4kd35 Result Analysis

时间: 2026-06-14 22:01:30 CST

## 目的

A35 针对 A33/A34 剩余瓶颈新增 default-off `input_std_route_post_adapter`，位置放在 A33 `input_std_route_decoder_heads` 之后，目标是在不破坏 A33 surgery 输出的前提下，只给 high route 增加零初始化后处理容量。

## 改动路径

- `/home/ldj/wxc/2DGauss/src/uhd_gaussian/config.py`
- `/home/ldj/wxc/2DGauss/src/uhd_gaussian/__init__.py`
- `/home/ldj/wxc/2DGauss/src/uhd_gaussian/models/restorer.py`
- `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/configs/a_4kd35_fast_4kdehaze_a33_post_high_adapter.toml`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/launch_tmux.sh`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/wait_and_launch_a_4kd35_memory_gated_gpu_safe.sh`

## 验证

- `py_compile` 通过: `config.py`, `restorer.py`, `train_v1_t01.py`
- TOML/path 检查通过
- launcher `bash -n` 通过
- 旧 A34 配置默认关闭 post adapter，不影响旧路径
- A35 aux keys 存在，zero-init 时 `post_delta_max=0.0`
- 从 A33 surgery checkpoint 加载时 `max_restored_diff=0.0`，`post_delta_max=0.0`，missing keys 仅为 `input_std_route_post_adapter.*`
- 训练参数策略仅开放 `input_std_route_post_adapter`，`12995/181401` 参数可训练

## 训练结果

- GPU0 preflight: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/logs/train_20260614_214051.log`, exit code `0`, `val_psnr=18.319243528848677`
- GPU0 formal: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/logs/train_20260614_214218.log`, exit code `0`
- Best-val: step50, `val_psnr=18.319709866984056`, `val_loss=0.12191925317049027`
- Final step100: `val_psnr=18.31929441448564`

## Test Metrics

- Standard fast-test: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/metrics/test_eval_best_val_gpu.json`
  - `psnr_mean=17.823132133493647`
  - `loss_mean=0.1202590474113822`
  - `total_loss_mean=0.12349013719707727`
- Task metrics: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/metrics/test_task_metrics_best_val_gpu.json`
  - `ssim_mean=0.7052112752199173`
  - dark `0.14492145068943502`
  - edge `0.1583659380674362`
  - local `0.024761935714632272`
  - chroma `0.02959282923489809`
- A18 alignment: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/metrics/a18_record_alignment_best_val_gpu.json`
  - overall gap `-0.02380224232077156 dB`
  - low-route gap `+0.05376824277532487 dB`
  - high-route gap `-0.12662823419234123 dB`
- Route-regime: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd35/metrics/input_std_route_regime_best_val_gpu.json`
  - selected delta `0.008263196442276239`
  - high selected delta `0.019216735912270324`
  - low selected delta `0.0`
  - high branch L1 vs A08 `0.3066226146941961`
  - low branch L1 vs A15 `0.3575197092273779`

## 结论

A35 只比 A33 高 `+0.00006982179538539413 dB`，high-route gap 只从 A33 的 `-0.12677254958688497 dB` 改到 `-0.12662823419234123 dB`，改善 `+0.00014431539454373699 dB`。这不是有效突破。

科学解释: A35 证实在 A33 输出之后再叠一个小型 additive high adapter，已经触碰不到主要短板。剩余差距来自 route-specific internal representation/full high-route path，而不是末端再加一点 post correction。A35 不应替代 A33 作为主线 promoted candidate；下一步需要做 A36 full dual-backbone single-checkpoint upper-bound 验证，直接回答“单 checkpoint 内保存完整 A08 high path + A15 low path 是否能复现 A18 router”。
