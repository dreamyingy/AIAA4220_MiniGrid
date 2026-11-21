# AIAA4220 Final Simulation Project — MiniGrid DoorKey

本项目研究 MiniGrid DoorKey 中的局部可观测稀疏奖励导航，通过官方 Baseline 复现、一项受控主要改进（**Memory: 4-Frame Stacking**）与严格配对评测，分析样本效率、失败机制及从 `6×6` 到 `8×8` 的零样本泛化表现。Baseline 沿用官方 MiniGrid PPO 教程的自定义 CNN 特征提取器与 Stable-Baselines3 PPO。

## 项目结构

```text
AIAA4220_MiniGrid
├── README.md
├── requirements.txt                # 锁定依赖版本（含 Linux CUDA 平台标记以支持跨平台安装）
├── configs/
│   ├── baseline.json               # Baseline 配置（单帧 7×7×3，frame_stack=1）
│   └── frame_stack.json            # 主要改进配置（4 帧堆叠 12×7×7，其余超参与预算完全一致）
├── common.py                       # 官方 CNN 特征提取器、帧堆叠包装器、环境构造与元数据推断
├── train.py                        # 训练、每 50k 周期评测、checkpoint 与 final 模型保存
├── evaluate.py                     # 独立确定性评测（自动识别并校验模型配置）
├── analyze.py                      # 单方法汇总及 Baseline vs. 改进方法受控对比与绘图
├── record_video.py                 # 单回合视频、JSON 轨迹与事件关键帧导出
├── compare_trajectories.py         # 双方法同回合对照筛选、周期死循环检测与 1080p 52s 分屏演示合成
└── results/                        # 正式模型、元数据、原始逐回合 CSV、汇总表与证据
    ├── baseline/                   # Baseline 三个训练种子（seed_1, seed_2, seed_3）
    ├── frame_stack/                # Frame Stack 改进实验输出目录（结构与 baseline 相同）
    ├── summary/                    # Baseline 单方法汇总 CSV 与图表
    ├── comparison/                 # Baseline vs. Frame Stack 受控对比表、协议校验与对比图
    └── videos/                     # Baseline 及双方法对比视频、关键帧与轨迹 JSON
```

以下命令均在仓库根目录运行。

---

## 环境配置

Linux x86_64（CUDA GPU 服务器）：

```bash
conda create -n Minigrid python=3.11 -y
conda activate Minigrid
python -m pip install -r requirements.txt
python -m pip check
```

macOS Apple Silicon（本地验证与冒烟测试）：

```bash
CONDA_SUBDIR=osx-arm64 conda create -n Minigrid python=3.11 -y
conda activate Minigrid
python -m pip install -r requirements.txt
python -m pip check
```

正式实验环境锁定为 Python `3.11.18`、MiniGrid `3.1.0`、Gymnasium `1.4.0`、Stable-Baselines3 `2.9.2`、PyTorch `2.15.0`、NumPy `2.4.6`；`requirements.txt` 通过 PEP 508 平台标记保留 Linux x86_64 上的精确 CUDA 13 依赖，同时支持非 CUDA 平台安装核心环境。

---

## Baseline 复现

### 1. Baseline 6×6 训练

输入为 `configs/baseline.json` 和训练 seed（`1, 2, 3`）。该配置使用 `MiniGrid-DoorKey-6x6-v0`、局部 `7×7×3` 编码、`ImgObsWrapper`、官方自定义 CNN（`features_dim=128`）、`normalize_images=False` 和 PPO `CnnPolicy`，不改变原始任务奖励。

```bash
for seed in 1 2 3; do
  python train.py \
    --env-id MiniGrid-DoorKey-6x6-v0 \
    --algo ppo \
    --total-steps 1000000 \
    --seed "$seed" \
    --device cuda:0
done
```

- 每个 seed 使用单环境（`n_envs=1`）、`n_steps=2048` 和一致的网络与 PPO 超参数。名义预算为 `1,000,000` 步，保留完整 PPO rollout 更新后实际完成 `1,001,472` 步（`489 × 2048`）。
- 训练脚本在 step `0` 及 `50k, 100k, …, 1M` 共 21 个时点各评测 100 episodes，固定使用 episode seeds `10000–10099`，逐回合结果写入 `periodic_episodes.csv`。
- **周期 1M checkpoint 与 final 模型说明**：`checkpoints/step_1000000.zip` 在交互步数刚达 1M 时保存（第 489 个 rollout 尚未完成参数更新）；`learn()` 返回并完成最后一次参数更新后保存 `final.zip`（`1,001,472` 步）。
- **可追溯性说明**：`results/baseline/seed_{1,2,3}/run.log` 在仓库首次提交（`b597ed8`）前启动运行，因此 `Working tree status` 显示未跟踪文件，其代码与网络结构即对应初始提交 `b597ed8`。运行后续实验前应先提交代码（保持 `git status` 干净），以确保 `run.log` 记录精确 commit hash。

### 2. Baseline 6×6 与 8×8 零样本最终评测

在固定测试集上独立评估三个 seeds 的 `final.zip`（`6×6` 使用 episode seeds `20000–20099`；`8×8` 零样本迁移使用 episode seeds `30000–30099`，与周期评测 `10000–10099` 完全隔离）：

```bash
for seed in 1 2 3; do
  python evaluate.py \
    --model "results/baseline/seed_${seed}/final.zip" \
    --env-id MiniGrid-DoorKey-6x6-v0 \
    --episodes 100 \
    --episode-seed-start 20000 \
    --output "results/baseline/seed_${seed}/final_6x6.csv"

  python evaluate.py \
    --model "results/baseline/seed_${seed}/final.zip" \
    --env-id MiniGrid-DoorKey-8x8-v0 \
    --episodes 100 \
    --episode-seed-start 30000 \
    --output "results/baseline/seed_${seed}/final_8x8.csv"
done
```

### 3. Baseline 结果统计与图表生成

```bash
python analyze.py --input results/baseline --output results/summary
```

输出写入 `results/summary/`，包含 `periodic_by_seed.csv`、`periodic_summary.csv`、`seed_metrics.csv`、`final_summary.csv`（含失败阶段分解）、`training_summary.csv` 以及 `figures/learning_curve.{png,pdf}` 和 `figures/final_success.{png,pdf}`。

### 4. 单回合轨迹与视频导出

```bash
python record_video.py \
  --model results/baseline/seed_1/final.zip \
  --env-id MiniGrid-DoorKey-6x6-v0 \
  --episode-seed 20000 \
  --output results/videos/seed1_6x6_episode20000.mp4 \
  --fps 10
```

Baseline 四条代表性轨迹（2 成功 + 2 失败）及 `44s` 演示视频说明见 `results/videos/baseline/README.md`。

---

## 改进方法（One Primary Improvement）：Memory — 4-Frame Stacking

### 1. 研究动机与可检验假设（Hypothesis）

对 Baseline 最终 600 条评测轨迹的阶段拆解表明：
- 在 `6×6` 上 Baseline 成功率为 `91.67% ± 6.81%`（25 次失败中 24 次发生于拾取钥匙前）；
- 在 `8×8` 零样本迁移上成功率降至 `49.33% ± 22.28%`，且 **152 次失败中有 89 次（58.6%）发生在拿到钥匙之后**（44 次拿钥匙后未开门，如 `Episode 30007` 的 6 步往返循环；45 次开门穿过后未到终点，如 `Episode 30009` 在 `(4,1)` 原地右转/左转的 2 步循环）。

根本机制之一在于：单帧前馈策略 $\pi(a_t \mid o_t)$ 仅依赖当前 `7×7×3` 局部观测，在 `deterministic=True` 评测下只要智能体回到相同 `(位置, 朝向, 门状态)`，观测张量即完全重现（如 `Episode 30009` 中第 12 步与第 14 步状态完全一致），必然锁死在周期循环中直至超时。

- **受控单一改动（Bounded Change）**：遵循 Handbook §3.1 **Memory** 改进项，在 `configs/frame_stack.json` 中仅将局部观测由单帧（`frame_stack=1`，`3×7×7`）改为堆叠最近 4 帧（`frame_stack=4`，`12×7×7`，episode reset 时历史帧置零，与 Stable-Baselines3 `VecFrameStack(..., n_stack=4, channels_order="first")` 逐比特一致）。仅第一层 `Conv2d` 输入通道由 `3` 增至 `12`，可训练策略参数量仅由 `167,096` 增至 `167,672`（仅增加 **576** 个参数，`+0.34%`）；**训练环境、原始稀疏奖励、PPO 全部超参数、1,000,000 名义步数（1,001,472 实际步数）、训练种子 `[1, 2, 3]` 及全部测试集 episode seeds 均保持严格不变**。
- **可证伪假设（Falsifiable Hypothesis）**：引入 4 帧短期观测历史能够打破短周期朝向/局部观测混淆（例如区分“刚穿门朝东”与“朝南看到目标后左转回东”），从而降低 **`8×8` 拾取钥匙后的死循环超时率（`final_8x8_post_key_fail_rate`）** 并提升 **`8×8` 零样本 Episode Success Rate（`final_8x8_success_rate`）**，同时保持 `6×6` 成功率与训练 AUC 不退化。若短周期转向循环减少但 `8×8` 整体成功率仍未显著提升，则证伪“短期记忆缺失是唯一主瓶颈”，表明 `6×6` 训练下全连接层对更大房间远距离目标相对坐标的空间尺度过拟合才是主导限制。

### 2. Frame Stack 训练（保持工作树干净后运行）

```bash
git status --short   # 确认无未提交改动，保证 run.log 记录干净 commit

# CPU 单线程运行（与 configs/frame_stack.json 默认一致；支持三 seed 并行）
for seed in 1 2 3; do
  python train.py \
    --config configs/frame_stack.json \
    --seed "$seed" \
    --device cpu
done
```

> **硬件与耗时说明**：两套实验软件版本完全一致（Python `3.11.18`、MiniGrid `3.1.0`、Gymnasium `1.4.0`、SB3 `2.9.2`、PyTorch `2.15.0`、NumPy `2.4.6`）。Baseline 训练运行于 NVIDIA A40（`cuda:0`），而 `frame_stack` 运行于 Intel Xeon Platinum 8369B（`cpu`, `torch_threads=1`，对应 commit `c49eb51`）；由于单环境（`n_envs=1`）下局部 `7×7` 张量极小，CPU 免去了逐步跨 PCIe 传输与单样本内核启动开销，训练耗时更短，因此跨方法耗时对比时应注明硬件差异（最终独立评测均在单线程 CPU 上执行）。

运行结果自动写入 `results/frame_stack/seed_1/`、`seed_2/`、`seed_3/`。

### 3. Frame Stack 6×6 与 8×8 零样本最终评测

`evaluate.py` 会自动读取同目录下的 `config.json` 并校验模型通道数（`frame_stack=4`）：

```bash
for seed in 1 2 3; do
  python evaluate.py \
    --model "results/frame_stack/seed_${seed}/final.zip" \
    --env-id MiniGrid-DoorKey-6x6-v0 \
    --episodes 100 \
    --episode-seed-start 20000 \
    --output "results/frame_stack/seed_${seed}/final_6x6.csv"

  python evaluate.py \
    --model "results/frame_stack/seed_${seed}/final.zip" \
    --env-id MiniGrid-DoorKey-8x8-v0 \
    --episodes 100 \
    --episode-seed-start 30000 \
    --output "results/frame_stack/seed_${seed}/final_8x8.csv"
done
```

### 4. 单方法汇总与 Baseline vs. Frame Stack 受控对比统计

```bash
# 可选：单独生成 Frame Stack 的三 seed 汇总与图表
python analyze.py \
  --input results/frame_stack \
  --output results/summary_frame_stack

# 核心：验证公平协议并生成 Baseline vs. Frame Stack 对比表与三张对比图
python analyze.py \
  --input results/baseline \
  --compare results/frame_stack \
  --output results/comparison
```

`results/comparison/` 输出文件说明：

| 文件 | 内容和用途 |
|---|---|
| `protocol_check.json` | 自动核验双方法预算、超参数、训练种子与测试集种子完全一致，仅隔离 `frame_stack` 变更 |
| `comparison_table.csv` | 主次指标及失败阶段分解的 `Mean ± SD (ddof=1)`、效应量 `Δ`、配对 seed 差值标准差及逐 seed 原始值 |
| `final_summary.csv` / `training_summary.csv` | 合并后的 `6×6`、`8×8` 最终表现与训练效率（AUC、T80、分阶段耗时）汇总 |
| `seed_metrics.csv` / `periodic_summary.csv` | 逐 seed 明细与 21 个周期评测点汇总 |
| `figures/learning_curve_comparison.{png,pdf}` | `6×6` 三 seed 学习曲线与 `± SD` 阴影同图对比 |
| `figures/final_comparison.{png,pdf}` | `6×6` 与 `8×8` 零样本成功率柱状图 + 配对单 seed 散点连线 |
| `figures/failure_breakdown_comparison.{png,pdf}` | 成功与三类失败阶段（未拿钥匙 / 拿钥匙未开门 / 开门未到终点）比例分解对比 |

### 5. 双方法同回合对照与 1080p 52 秒演示视频生成

```bash
python compare_trajectories.py \
  --baseline-dir results/baseline \
  --modified-dir results/frame_stack \
  --comparison-csv results/comparison/comparison_table.csv \
  --seed 1 \
  --output-dir results/videos/comparison
```

如需基于已有录屏和轨迹 JSON 快速重渲染视频（无需重新加载模型推理），可加 `--recompose`：

```bash
python compare_trajectories.py --recompose \
  --output-dir results/videos/comparison \
  --comparison-csv results/comparison/comparison_table.csv
```

该脚本自动筛选同回合对照案例（含 2 成功、2 失败及针对 `30007`/`30009` 等死循环回合的对照），检测确定性状态循环周期，导出单回合 MP4、JSON、关键帧、`cases_summary.json` 以及符合 **30–60 秒**展示要求的 `1920×1080` 高清演示视频 `comparison_demo_52s.mp4`（含开场协议页、全局地图轨迹尾迹、实时 `7×7` 单帧 vs `4×7×7` 帧堆叠真实局部观测面板、`KEY→DOOR→GOAL` 里程碑、动作历史丝带与结尾定量总结页）。详细案例分析见 `results/videos/comparison/README.md`。

### 6. 核心受控对比结果（3 Seeds Mean ± Sample SD, `ddof=1`）

完整数据见 `results/comparison/comparison_table.csv`：

| 指标 | Baseline (`k=1`) | Frame Stack (`k=4`) | 差值 `Δ (Mod − Base)` |
|---|---|---|---|
| **6×6 成功率（主指标）** | `91.67% ± 6.81%` (`84%, 94%, 97%`) | **`97.00% ± 1.00%`** (`98%, 97%, 96%`) | **`+5.33 pp`**（跨 seed 标准差缩小 `6.8×`） |
| **8×8 零样本成功率（迁移主指标）** | `49.33% ± 22.28%` (`38%, 75%, 35%`) | **`58.33% ± 16.26%`** (`40%, 64%, 71%`) | **`+9.00 pp`** |
| 6×6 平均原始回报 | `0.888 ± 0.066` | `0.936 ± 0.009` | `+0.048` |
| 8×8 平均原始回报 | `0.482 ± 0.217` | `0.564 ± 0.158` | `+0.082` |
| 6×6 成功回合中位步数 | `12.33 ± 0.58` | `13.33 ± 0.58` | `+1.00` 步 |
| 8×8 成功回合中位步数 | `15.33 ± 0.58` | `21.67 ± 1.15` | `+6.33` 步 |
| 6×6 训练归一化 AUC | **`0.829 ± 0.026`** | `0.672 ± 0.092` | `-0.158` |
| 首次达到 80% 成功率步数 `T80` | **`150k ± 0`** (`3/3`) | `383k ± 144k` (`3/3`) | `+233k` 步 |
| 6×6 未拾取钥匙失败率 | `8.00% ± 6.24%` | **`1.00% ± 1.00%`** | **`-7.00 pp`** |
| 8×8 未拾取钥匙失败率 | `21.00% ± 12.17%` | **`13.67% ± 3.06%`** | **`-7.33 pp`** |
| 8×8 拾取钥匙后超时率 | `29.67% ± 19.14%` | `28.00% ± 13.23%` | `-1.67 pp` |
| 训练耗时 / 最终评测耗时 | `6061.9 ± 1110.4 s` (A40) / `30.5 ± 11.7 s` (CPU) | `1873.8 ± 39.5 s` (Xeon CPU) / `19.4 ± 6.3 s` (CPU) | 见硬件说明 |

---

## 统计约定

- 先逐训练 seed（独立实验单位，`n=3`）计算 100 个测试 episodes 上的指标，再汇总三个 seeds 的均值 ± 样本标准差（`ddof=1`）；绘图中的成功率不确定度阴影裁剪至物理有效区间 `[0%, 100%]`。
- 某 seed 无成功 episode 时，成功步数中位数记为 `NaN`；仅当 `3/3` seeds 均有效（或均达到 `T80`）时才报告三 seed 均值与标准差。
- 归一化训练 AUC 使用 `6×6` 上 `0–1M` 共 21 个周期评测点按梯形法则积分并除以 `1,000,000`；独立最终评测（`final_6x6.csv` / `final_8x8.csv`）不混入周期曲线、AUC 或 `T80`。

## 参考资料

- [MiniGrid Stable-Baselines3 训练教程](https://minigrid.farama.org/content/training/)
- [MiniGrid DoorKey 环境文档](https://minigrid.farama.org/main/environments/minigrid/DoorKeyEnv/)
- [Stable-Baselines3 PPO 与 VecFrameStack 文档](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
