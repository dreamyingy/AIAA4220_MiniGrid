# Baseline Final 轨迹证据

本目录保存训练 seed 1 的 final 模型在四个固定回合上的两条成功、两条失败轨迹。模型为 `../../baseline/seed_1/final.zip`。全局地图渲染用于解释行为，策略输入仍为局部 `7×7×3` 编码。

MP4、图片（包括关键帧、失败循环图和预览图）、说明和合成脚本纳入 Git；本目录的轨迹 JSON 和 CSV 由 `.gitignore` 排除，但仍保留在本地。下文指向 JSON 和 CSV 的相对链接供本地查阅，克隆仓库后需重新生成或取得相应文件。

## 四条回合

步骤从初始状态 0 开始计数。JSON 保存每一步的动作、位置、方向、携带物、门状态与局部观测；原始视频均为 10 fps，包含初始帧及全部动作后的帧。

| 回合 | 结果与关键事件 | 完整证据 | 事件关键帧 |
|---|---|---|---|
| 6×6 / 20000 | 成功；第 2 步拿钥匙，第 7 步开门，第 11 步到终点；回报 0.9725 | [视频](seed1_6x6_episode20000.mp4)、[JSON](seed1_6x6_episode20000.json) | [初始](seed1_6x6_episode20000_step_0000.png)、[钥匙](seed1_6x6_episode20000_step_0002.png)、[门](seed1_6x6_episode20000_step_0007.png)、[终点](seed1_6x6_episode20000_step_0011.png) |
| 8×8 / 30000 | 成功；第 2 步拿钥匙，第 10 步开门，第 15 步到终点；回报 0.97890625 | [视频](seed1_8x8_episode30000.mp4)、[JSON](seed1_8x8_episode30000.json) | [初始](seed1_8x8_episode30000_step_0000.png)、[钥匙](seed1_8x8_episode30000_step_0002.png)、[门](seed1_8x8_episode30000_step_0010.png)、[终点](seed1_8x8_episode30000_step_0015.png) |
| 8×8 / 30007 | 失败；第 4 步拿钥匙，未开门，640 步超时；回报 0 | [视频](seed1_8x8_episode30007.mp4)、[JSON](seed1_8x8_episode30007.json) | [初始](seed1_8x8_episode30007_step_0000.png)、[钥匙](seed1_8x8_episode30007_step_0004.png)、[超时](seed1_8x8_episode30007_step_0640.png) |
| 8×8 / 30009 | 失败；第 4 步拿钥匙，第 10 步开门，未到终点，640 步超时；回报 0 | [视频](seed1_8x8_episode30009.mp4)、[JSON](seed1_8x8_episode30009.json) | [初始](seed1_8x8_episode30009_step_0000.png)、[钥匙](seed1_8x8_episode30009_step_0004.png)、[门](seed1_8x8_episode30009_step_0010.png)、[超时](seed1_8x8_episode30009_step_0640.png) |

## 一致性核对

核对来源为 [6×6 原始评测 CSV](../../baseline/seed_1/final_6x6.csv) 与 [8×8 原始评测 CSV](../../baseline/seed_1/final_8x8.csv)。四条回放的 `train_seed`、`model_step`、`env_id`、`episode_seed`、`success`、`return`、`steps`、`terminated`、`truncated`、`key_step`、`door_step`、`goal_step` 与对应评测行一致。成功回合正常终止；失败回合均为超时截断。

另保存 10 张循环关键帧：30007 的 steps 6–12，30009 的 steps 12–14。加上事件关键帧，共 25 张独立 PNG；[失败循环图](failure_cycles.png) 汇集关键状态。

## 两种可观察的失败行为

坐标为 `(x, y)`，原点在地图左上角；方向编码为东 0、南 1、西 2、北 3。JSON 中某步的动作是从前一步状态到该步状态所执行的动作。动作编码为左转 0、右转 1、前进 2、拾取 3、放下 4、开关门 5、完成 6。

### 30007：拿到钥匙后，在两个格子之间往返

第 4 步拿钥匙后，代理到达 `(3,4)`，从 step 6 的状态起进入严格的六步循环：

```text
step 6:  (3,4) 南
step 7:  (3,5) 南  -- 前进
step 8:  (3,5) 西  -- 右转
step 9:  (3,5) 北  -- 右转
step 10: (3,4) 北  -- 前进
step 11: (3,4) 西  -- 左转
step 12: (3,4) 南  -- 左转，回到 step 6 状态
```

动作序列 `前进 → 右转 → 右转 → 前进 → 左转 → 左转` 重复到 640 步超时，包含 105 个完整循环及末尾 4 个动作。坐标、方向、携带物、门状态和局部观测都按六步周期重复。门位于 `(5,5)`，始终保持锁定，整个回合未执行开门动作。

直接失败原因是确定性策略陷入往返循环，始终没有接近并打开门。循环内前进的目标格为空，不是碰墙导致移动失败；step 6、7、9 的局部观测已包含门编码，也不能解释成始终看不到门。

### 30009：开门并穿过后，在原地交替转向

第 10 步打开 `(3,1)` 的门，第 11 步进入门格，第 12 步穿过门到达 `(4,1)`，随后进入严格的两步循环：

```text
step 12: (4,1) 东
step 13: (4,1) 南  -- 右转
step 14: (4,1) 东  -- 左转，回到 step 12 状态
```

从第 13 步至第 640 步的 628 个动作全部为交替右转、左转，共 314 个完整循环。代理不再前进，位置始终为 `(4,1)`；两个朝向的前方格 `(5,1)` 和 `(4,2)` 均为空。step 13 向南时，局部观测中包含位于全局 `(6,6)` 的目标，但下一步仍转回东侧。门一直打开，最后超时。

直接失败原因是穿门后的原地转向循环，已完成开门并没有转化为持续向目标移动；不是门未打开或通道被障碍物封锁。

两条案例的循环状态见 [失败循环图](failure_cycles.png)。它们说明同一采用确定性动作的前馈 CNN 策略在这些布局上无法脱离重复的观测—动作循环。是否主要源于记忆不足、训练布局覆盖或其他因素，需要受控对照才能判断；不能据此认定某种改进一定能解决。两条选择性案例也不能代表全部 152 个 8×8 失败回合，完整三 seed 指标见 [Final 汇总](../../summary/final_summary.csv)。

## 44 秒演示视频

[baseline_demo_44s.mp4](baseline_demo_44s.mp4) 为 1280×720、30 fps、无声英文注释演示，仅展示四条轨迹，不包含开头介绍和结尾总结页面。保留实际步骤编号，使用减速展示与明确标注的跳转。完整回合仍保存在上表的原始视频中，尤其两条失败原视频均包含全部 640 个动作。

成片规格为 1,320 帧、44.00 秒。完整四条回放分别为 1.2、1.6、64.1、64.1 秒。

| 视频时间 | 内容 |
|---|---|
| 0–8 秒 | 6×6 / 20000 成功回合 |
| 8–16 秒 | 8×8 / 30000 成功回合 |
| 16–30 秒 | 8×8 / 30007，展示 step 0–18，再明确跳到 step 640 |
| 30–44 秒 | 8×8 / 30009，展示 step 0–24，再明确跳到 step 640 |

## 复现命令

在仓库根目录激活 `Minigrid` 环境后，以下命令可重放相同回合。这些是本次产物的生成命令；脚本拒绝覆盖已有视频或 JSON，重复操作时应将 `--output` 改为新的输出目录。

```bash
python record_video.py --model results/baseline/seed_1/final.zip --env-id MiniGrid-DoorKey-6x6-v0 --episode-seed 20000 --device cpu --fps 10 --output results/videos/baseline/seed1_6x6_episode20000.mp4
python record_video.py --model results/baseline/seed_1/final.zip --env-id MiniGrid-DoorKey-8x8-v0 --episode-seed 30000 --device cpu --fps 10 --output results/videos/baseline/seed1_8x8_episode30000.mp4
python record_video.py --model results/baseline/seed_1/final.zip --env-id MiniGrid-DoorKey-8x8-v0 --episode-seed 30007 --device cpu --fps 10 --output results/videos/baseline/seed1_8x8_episode30007.mp4
python record_video.py --model results/baseline/seed_1/final.zip --env-id MiniGrid-DoorKey-8x8-v0 --episode-seed 30009 --device cpu --fps 10 --output results/videos/baseline/seed1_8x8_episode30009.mp4

python results/videos/baseline/make_demo.py
```

演示合成脚本读取已有 MP4、轨迹 JSON 和上述循环关键帧，不重新推理或训练；使用现有环境中的 ImageIO、Pillow 和 Matplotlib 字体，不需要安装新依赖。由于轨迹 JSON 不纳入 Git，合成前应确认本地输入齐全。合成脚本也拒绝覆盖已有演示视频，重建时可将脚本与输入文件复制到新的目录并排除旧的演示 MP4。核心训练、评测、视频导出脚本和模型权重保持不变。
