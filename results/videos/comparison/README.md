# Baseline vs. Frame Stack (k=4) 同回合对照轨迹证据

本目录保存训练 `seed 1` 下 **Baseline (`frame_stack=1`)** 与 **Frame Stack (`frame_stack=4`)** 在四个固定测试回合上的同回合对照录像、轨迹 JSON、事件关键帧、周期循环检测摘要 [cases_summary.json](cases_summary.json) 以及 **1080p 52 秒高清演示视频** [comparison_demo_52s.mp4](comparison_demo_52s.mp4)。

视频同时展示全局地图（含实时运动轨迹尾迹）与策略真实接收的局部观测（Baseline 为单帧 `7×7` 视角，Frame Stack 为 `t−3 → t` 实时 4 帧滑动缓冲），直观呈现局部观测混淆、短周期死循环与有限窗口饱和机制。

## 四组同回合对照（包含 Frame Stack 的 2 条成功 + 2 条失败轨迹）

| 环境 / Episode Seed | Baseline (`k=1`) 行为结果 | Frame Stack (`k=4`) 行为结果 | 机制解释 |
|---|---|---|---|
| `6×6` / `20000` | **成功**（11 步，回报 `0.9725`） | **成功**（11 步，回报 `0.9725`） | 标准任务上保持最优 11 步路径（第 2 步拿钥匙 → 第 7 步开门 → 第 11 步到终点） |
| `8×8` / `30008` | **失败**（640 步超时；从第 0 步起在原地重复 `pick up` 单步死循环 640 次） | **成功**（17 步零样本通关；第 2 步拿钥匙 → 第 6 步开门 → 第 17 步到达终点） | 短期历史消除了初始无钥匙视野下的单步原地动作锁死，驱动智能体搜索房间并完成迁移 |
| `8×8` / `30009` | **失败**（第 10 步开门穿过至 `(4,1)` 后陷入 `右转↔左转` **2 步周期转向循环**，重复 314 次至超时） | **失败**（第 12 步开门穿过后**成功打破 `(4,1)` 的 2 步转向死循环**并推进至 `(5,1)`，随后回退至门格面墙重复 `forward`，4 帧窗口饱和为 `[o,o,o,o]`） | 直接验证了假设与证伪边界：4 帧记忆消除了短周期转向混淆（`obs[12] != obs[14]`），但 `6×6 → 8×8` 远端目标空间尺度过拟合仍是主导限制 |
| `8×8` / `30007` | **失败**（第 4 步拿钥匙后陷入 `前进-右-右-前进-左-左` **6 步周期往返循环**，共重复 105 次至 640 步超时） | **失败**（未陷入 6 步往返循环，但在第 12 步后连续 4 次无状态变化动作使 4 帧窗口饱和为 `[o,o,o,o]`，退化为单步 `pick up` 自锁） | 揭示有限窗口 `k=4` 的边界：一旦连续 4 步状态不变，历史帧被相同观测填满，重新产生固定点 |

## 1080p 52 秒高清演示视频

- 文件：[comparison_demo_52s.mp4](comparison_demo_52s.mp4)（`1920×1080` Full HD、`30 fps`、`1,560` 帧 = `52.0s`，符合 Handbook **30–60 秒**展示要求）。
- 内容结构（含平滑淡入淡出转场与底部 6 段进度条）：
  - `0.0–4.0s`：开场标题与受控实验协议页（Baseline vs. 4-Frame Stacking 单变量改动说明）
  - `4.0–12.5s`：`Case 1` — `6×6 / Episode 20000`（标准任务最优解 + 单帧与 4 帧真实观测窗口对照）
  - `12.5–24.5s`：`Case 2` — `8×8 / Episode 30008`（零样本救回案例：Baseline 第 0 步自锁 vs. Frame Stack 17 步通关）
  - `24.5–36.5s`：`Case 3` — `8×8 / Episode 30009`（破除 2 步转向循环与剩余 8×8 空间尺度瓶颈）
  - `36.5–46.5s`：`Case 4` — `8×8 / Episode 30007`（6 步往返循环 vs. 4 帧窗口饱和 `[o,o,o,o]`）
  - `46.5–52.0s`：三 seed 600 回合定量指标汇总与校准结论页（数据直读 `results/comparison/comparison_table.csv`）

## 复现命令

```bash
# 完整从模型重放并生成 1080p 52 秒演示视频
python compare_trajectories.py \
  --baseline-dir results/baseline \
  --modified-dir results/frame_stack \
  --comparison-csv results/comparison/comparison_table.csv \
  --seed 1 \
  --device cpu \
  --output-dir results/videos/comparison

# 或直接基于已有 MP4/JSON 快速重渲染视频
python compare_trajectories.py --recompose \
  --output-dir results/videos/comparison \
  --comparison-csv results/comparison/comparison_table.csv
```
