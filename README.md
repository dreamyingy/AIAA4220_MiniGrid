# AIAA4220 Final Simulation Project — MiniGrid DoorKey

本项目研究 MiniGrid DoorKey 中的局部可观测导航，通过 baseline 复现、一项主要改进和受控实验，分析学习表现及从 6×6 到 8×8 的零样本泛化。Baseline 沿用官方 MiniGrid PPO 教程的自定义 CNN 和 Stable-Baselines3 PPO。

## 项目结构

```text
AIAA4220_MiniGrid
├── README.md                   
├── requirements.txt           
├── configs/
│   └── baseline.json           # 网络、PPO 和评估协议配置
├── common.py                   # 官方特征提取器、环境构造、配置读取
├── train.py                    # 训练、周期评测、checkpoint 和 final 保存
├── evaluate.py                 # 评测逻辑
├── analyze.py                  # 指标汇总与绘图
├── record_video.py             # 视频、轨迹和关键帧导出
└── results/                    # 正式模型、元数据与结果纳入 Git
    ├── baseline/
    │   └── seed_1/             # seed_2、seed_3 结构相同
    │       ├── config.json     # 本次运行的生效配置与预算记录
    │       ├── run.log         # 命令、版本、设备与运行信息
    │       ├── progress.csv    # PPO 训练日志
    │       ├── monitor.csv     # 训练 episode 回报与长度
    │       ├── checkpoints/   # 每 50k 模型，默认不纳入 Git
    │       ├── final.zip       # 最后一次参数更新完成后的模型
    │       ├── periodic_episodes.csv  # 周期评测的逐 episode 结果
    │       ├── final_6x6.csv    # 手动执行 6×6 final 评测后生成
    │       └── final_8x8.csv    # 手动执行 8×8 final 评测后生成
    ├── summary/               # 汇总 CSV 与 figures/ 下的图表
    └── videos/                # 视频、轨迹 JSON 和关键帧
```

以下命令均在仓库根目录运行。

## Baseline 复现

### 1. 环境配置

```bash
conda create -n Minigrid python=3.11 -y
conda activate Minigrid
python -m pip install -r requirements.txt
python -m pip check
```

环境配置为 Python 3.11.18、MiniGrid 3.1.0、Gymnasium 1.4.0、Stable-Baselines3 2.9.2、PyTorch 2.15.0、NumPy 2.4.6；完整依赖见 `requirements.txt`。

### 2. Baseline 6×6 训练

输入为 `configs/baseline.json` 和训练 seed。该配置使用 `MiniGrid-DoorKey-6x6-v0`、局部 `7×7×3` 编码、`ImgObsWrapper`、官方自定义 CNN、`features_dim=128`、`normalize_images=False` 和 PPO `CnnPolicy`。不改变原始任务奖励。

下面的正式训练命令使用 GPU。默认配置中的 CPU 设置会被命令行覆盖。

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

`--algo` 当前只支持 `ppo`。环境、算法、名义预算、训练 seed 和设备由命令行指定，其他设置读取默认配置。运行结果分别写入 `results/baseline/seed_1/`、`seed_2/`、`seed_3/`。

每个 seed 使用单环境、`n_steps=2048` 和一致的网络与 PPO 超参数。名义预算为 1,000,000 步，保留完整 rollout 后实际完成 1,001,472 步。已有 seed 输出目录不会被覆盖。

训练脚本在 step 0，以及 50k、100k、…、1M 上各评测 100 episodes，共 21 个时点，固定使用 episode seeds `10000–10099`。每 50k 保存模型，逐 episode 结果写入 `periodic_episodes.csv`。

**周期 1M checkpoint 与 final 不是同一个模型。** `checkpoints/step_1000000.zip` 在交互步数达到 1M 时保存，此时最后一个 rollout 的参数更新尚未完成。`learn()` 返回后才保存 `final.zip`，实际步数为 1,001,472。

### 3. 6×6 评测

输入为三个 seeds 的 `final.zip`，在 `MiniGrid-DoorKey-6x6-v0` 上分别评估 100 episodes；固定 episode seeds 为 `20000–20099`。

```bash
for seed in 1 2 3; do
  python evaluate.py \
    --model "results/baseline/seed_${seed}/final.zip" \
    --env-id MiniGrid-DoorKey-6x6-v0 \
    --episodes 100 \
    --episode-seed-start 20000 \
    --output "results/baseline/seed_${seed}/final_6x6.csv"
done
```

每次运行生成对应 seed 目录下的 `final_6x6.csv`，每行记录一个 episode 的成功状态、原始回报、步数和行为事件。`--episode-seed-start` 指的是评估环境的随机种子起点。

### 4. 8×8 评测 （零样本迁移）


直接在 `MiniGrid-DoorKey-8x8-v0` 上评估同一批 `final.zip`，不进行训练、微调或其他适配。每个 seed 使用固定 episode seeds `30000–30099`，各评估 100 episodes。

```bash
for seed in 1 2 3; do
  python evaluate.py \
    --model "results/baseline/seed_${seed}/final.zip" \
    --env-id MiniGrid-DoorKey-8x8-v0 \
    --episodes 100 \
    --episode-seed-start 30000 \
    --output "results/baseline/seed_${seed}/final_8x8.csv"
done
```

输出为各 seed 的 `final_8x8.csv`。

### 5. 结果统计

输入必须包含三个 seeds 的完整周期结果和两种 final 结果。每个正式运行应有 2,100 条周期 episode 记录，以及 6×6、8×8 各 100 条 final 记录。

```bash
python analyze.py --input results/baseline --output results/summary
```

输出写入 `results/summary/`，将生成的 CSV 和 PNG/PDF 一起纳入仓库：

| 文件 | 内容和用途 |
|---|---|
| `periodic_by_seed.csv` | 各 seed、各周期时点的指标，支持原始结果核对 |
| `periodic_summary.csv` | 各周期时点跨三个 seeds 的均值、样本标准差和有效数量 |
| `seed_metrics.csv` | 逐 seed 的最终表现与学习效率指标 |
| `final_summary.csv` | 6×6、8×8 的跨 seed 最终比较 |
| `training_summary.csv` | AUC、T80、达标数、时间与预算汇总 |
| `figures/learning_curve.*`、`figures/final_success.*` | 周期学习曲线与 final Success 比较图，各提供 PNG 和 PDF |

统计遵循以下约定：

- 先逐 seed 计算 Success、mean original return 和成功 episodes 的 median steps，再报告三个 seeds 的均值 ± 样本标准差（`ddof=1`），不使用置信区间。
- 某 seed 没有成功 episode 时，其成功步数中位数为 `NaN`，不会用 0 代替，也不会将不完整的有效 seeds 伪装成三 seed 汇总。
- 归一化 AUC 只使用 6×6 的 21 个周期成功率点，在 0–1M 上按梯形法积分，再除以 1,000,000。成功率用 0–1 表示。
- T80 是同一周期网格上首次达到 80% Success 的步数。未达到时记为 `NaN` 和 `not reached (>1M)`；始终报告达标 seed 数，只有 3/3 达标才汇总 T80 的均值和标准差。
- Final 独立汇报，不进入周期学习曲线、AUC 或 T80。计时需区分包含周期评测的训练耗时、周期评测耗时和独立 final 评测耗时。

### 6. 视频生成

从评估 CSV 中选择需要解释的 episode，输入对应模型、环境和 episode seed。以下示例重放 seed 1 模型在 6×6 的 episode seed 20000；应根据实际结果选择成功和失败案例。

```bash
python record_video.py \
  --model results/baseline/seed_1/final.zip \
  --env-id MiniGrid-DoorKey-6x6-v0 \
  --episode-seed 20000 \
  --output results/videos/seed1_6x6_episode20000.mp4 \
  --fps 10
```

输出包括同名 MP4 和 JSON，以及 `seed1_6x6_episode20000_step_*.png` 关键帧，用于行为分析和演示。关键帧覆盖初始状态、拾钥匙、开门、到达目标和结束状态中实际发生的事件。轨迹中记录动作、位置、方向及关键事件；渲染画面仅用于展示，策略输入仍是局部观测。

`results/videos/` 下的视频、图片和轨迹 JSON 纳入 Git；CSV 仅保留在本地，由 `.gitignore` 排除。

## 改进方法（One Primary Improvement）


## 参考资料

- [MiniGrid Stable-Baselines3 训练教程](https://minigrid.farama.org/content/training/)
- [MiniGrid DoorKey 环境](https://minigrid.farama.org/main/environments/minigrid/DoorKeyEnv/)
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
