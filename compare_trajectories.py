"""Select matched episodes, detect deterministic cycles, and render side-by-side demo videos."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
from matplotlib import font_manager
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
import torch

from common import infer_model_metadata, make_env
from evaluate import run_episode


ENVS = {"6x6": "MiniGrid-DoorKey-6x6-v0", "8x8": "MiniGrid-DoorKey-8x8-v0"}
FPS = 30
SIZE = (1280, 720)
BACKGROUND = "#0f172a"
PANEL_BG = "#1e293b"
WHITE = "#f8fafc"
MUTED = "#cbd5e1"
ACCENT = "#38bdf8"
SUCCESS_COLOR = "#4ade80"
FAIL_COLOR = "#f87171"
WARN_COLOR = "#fbbf24"
ACTIONS = {
    None: "reset",
    0: "turn left",
    1: "turn right",
    2: "forward",
    3: "pick up",
    4: "drop",
    5: "toggle door",
    6: "done",
}
DIRECTIONS = ("east", "south", "west", "north")


def load_fonts():
    font_path = font_manager.findfont("DejaVu Sans")
    return {size: ImageFont.truetype(font_path, size) for size in (16, 18, 20, 22, 26, 32)}


def detect_state_cycle(trajectory, min_repeats=3):
    """Detect whether the physical state (pos, dir, carrying, doors) enters a periodic loop."""
    signatures = [
        (
            tuple(step["position"]),
            step["direction"],
            step["carrying"],
            tuple((d["open"], d["locked"]) for d in step["doors"]),
        )
        for step in trajectory
    ]
    total = len(signatures)
    for start in range(total):
        first_sig = signatures[start]
        for period in range(1, (total - start) // min_repeats + 1):
            remaining = total - start
            if remaining < period * min_repeats:
                break
            if all(signatures[idx] == signatures[start + (idx - start) % period]
                   for idx in range(start, total)):
                actions = [
                    ACTIONS[trajectory[start + k]["action"]]
                    for k in range(1, period + 1)
                ]
                return {
                    "has_cycle": True,
                    "start_step": int(start),
                    "period": int(period),
                    "repeats": int((total - 1 - start) // period),
                    "action_pattern": actions,
                }
    return {"has_cycle": False, "start_step": None, "period": None, "repeats": 0, "action_pattern": []}


def replay_and_export(model, meta, env_id, episode_seed, out_mp4, fps=10):
    env = make_env(env_id, render_mode="rgb_array", frame_stack=meta["frame_stack"])
    try:
        row, frames, trajectory = run_episode(model, env, episode_seed, capture=True)
    finally:
        env.close()

    cycle = detect_state_cycle(trajectory)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_mp4, fps=fps, codec="libx264", macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)

    key_steps = sorted({0, row["steps"], *[
        row[event] for event in ("key_step", "door_step", "goal_step") if row[event] is not None
    ]})
    keyframes = []
    for step in key_steps:
        path = out_mp4.with_name(f"{out_mp4.stem}_step_{step:04d}.png")
        imageio.imwrite(path, frames[step])
        keyframes.append(path.name)

    trace = {
        "model": str(meta["model_path"]),
        "method": meta["method"],
        "frame_stack": meta["frame_stack"],
        "train_seed": model.seed,
        "model_step": model.num_timesteps,
        "env_id": env_id,
        "deterministic": True,
        "fps": fps,
        **row,
        "cycle": cycle,
        "keyframes": keyframes,
        "trajectory": trajectory,
    }
    out_mp4.with_suffix(".json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return trace, frames


def select_representative_cases(baseline_dir, modified_dir, seed):
    """Choose 4 informative episodes (2 modified successes, 2 modified failures when available)."""
    b_6 = pd.read_csv(baseline_dir / f"seed_{seed}" / "final_6x6.csv").set_index("episode_seed")
    m_6 = pd.read_csv(modified_dir / f"seed_{seed}" / "final_6x6.csv").set_index("episode_seed")
    b_8 = pd.read_csv(baseline_dir / f"seed_{seed}" / "final_8x8.csv").set_index("episode_seed")
    m_8 = pd.read_csv(modified_dir / f"seed_{seed}" / "final_8x8.csv").set_index("episode_seed")

    cases = []
    # Case 1: 6x6 success (prefer 20000 if modified succeeded, else first modified 6x6 success)
    succ_6 = m_6.index[m_6["success"].eq(1)].tolist()
    ep_6 = 20000 if 20000 in succ_6 else (succ_6[0] if succ_6 else int(m_6.index[0]))
    cases.append(("6x6", ENVS["6x6"], int(ep_6), "6x6 Task Evaluation"))

    # Case 2: 8x8 rescued episode (baseline failed, modified succeeded), preferring 30007/30009 if rescued
    rescued_8 = [
        int(ep) for ep in m_8.index
        if int(b_8.loc[ep, "success"]) == 0 and int(m_8.loc[ep, "success"]) == 1
    ]
    preferred_rescued = [ep for ep in (30007, 30009) if ep in rescued_8]
    if preferred_rescued:
        ep_8_succ = preferred_rescued[0]
        label_2 = "8x8 Zero-Shot — Rescued Baseline Cycle"
    elif rescued_8:
        ep_8_succ = rescued_8[0]
        label_2 = "8x8 Zero-Shot — Rescued Failure"
    else:
        succ_8 = m_8.index[m_8["success"].eq(1)].tolist()
        ep_8_succ = 30000 if 30000 in succ_8 else (int(succ_8[0]) if succ_8 else int(m_8.index[0]))
        label_2 = "8x8 Zero-Shot Transfer"
    cases.append(("8x8", ENVS["8x8"], int(ep_8_succ), label_2))

    # Cases 3 & 4: Two failure cases on 8x8 (or 6x6 fallback), preferring 30007/30009 when still failing
    used = {(c[0], c[2]) for c in cases}
    fail_8 = [int(ep) for ep in m_8.index if int(m_8.loc[ep, "success"]) == 0 and ("8x8", int(ep)) not in used]
    ordered_fail_8 = [ep for ep in (30007, 30009) if ep in fail_8] + [
        ep for ep in fail_8 if ep not in (30007, 30009)
    ]
    for ep in ordered_fail_8[:2]:
        cases.append(("8x8", ENVS["8x8"], int(ep), "8x8 Zero-Shot Failure Analysis"))
        used.add(("8x8", int(ep)))

    if len(cases) < 4:
        fail_6 = [int(ep) for ep in m_6.index if int(m_6.loc[ep, "success"]) == 0 and ("6x6", int(ep)) not in used]
        for ep in fail_6[: 4 - len(cases)]:
            cases.append(("6x6", ENVS["6x6"], int(ep), "6x6 Failure Analysis"))
            used.add(("6x6", int(ep)))

    # Fallback if policy achieved 100% everywhere
    for ep in m_8.index:
        if len(cases) >= 4:
            break
        if ("8x8", int(ep)) not in used:
            cases.append(("8x8", ENVS["8x8"], int(ep), "8x8 Zero-Shot Episode"))
            used.add(("8x8", int(ep)))
    return cases[:4]


def draw_side_by_side_frame(fonts, header, subtitle, playback_tag, left_pack, right_pack):
    canvas = Image.new("RGB", SIZE, BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((36, 18), header, font=fonts[32], fill=WHITE)
    draw.text((36, 62), subtitle, font=fonts[20], fill=MUTED)
    draw.text((920, 62), playback_tag, font=fonts[20], fill=WARN_COLOR)

    for idx, (title, trace, frames, step_idx) in enumerate((left_pack, right_pack)):
        x0 = 36 + idx * 612
        y0 = 102
        draw.rounded_rectangle((x0, y0, x0 + 596, y0 + 572), radius=12, fill=PANEL_BG)
        status_ok = bool(trace["success"])
        badge_text = "SUCCESS" if status_ok else "TIMEOUT / FAIL"
        badge_color = SUCCESS_COLOR if status_ok else FAIL_COLOR
        draw.text((x0 + 20, y0 + 14), title, font=fonts[22], fill=ACCENT)
        draw.text((x0 + 410, y0 + 16), badge_text, font=fonts[20], fill=badge_color)

        clamped = min(step_idx, trace["steps"])
        img = Image.fromarray(frames[clamped]).resize((360, 360), Image.Resampling.NEAREST)
        canvas.paste(img, (x0 + 118, y0 + 52))

        rec = trace["trajectory"][clamped]
        door = rec["doors"][0]
        door_state = "open" if door["open"] else ("locked" if door["locked"] else "closed")
        cyc = trace["cycle"]
        cyc_line = (
            f"Cycle: period {cyc['period']} from step {cyc['start_step']} ({cyc['repeats']}x)"
            if cyc["has_cycle"]
            else f"Return: {trace['return']:.4f} | Total steps: {trace['steps']}"
        )
        info_lines = [
            f"Step {clamped} / {trace['steps']}   |   Action: {ACTIONS[rec['action']]}",
            f"Pos: {tuple(rec['position'])} ({DIRECTIONS[rec['direction']]})   |   Carry: {rec['carrying'] or 'none'}   |   Door: {door_state}",
            cyc_line,
        ]
        for line_i, line in enumerate(info_lines):
            draw.text((x0 + 20, y0 + 426 + line_i * 34), line, font=fonts[18],
                      fill=WARN_COLOR if (line_i == 2 and cyc["has_cycle"]) else WHITE)

    draw.text(
        (36, 688),
        "Full grid rendered for analysis only; policies act deterministically from local 7x7 observations.",
        font=fonts[16],
        fill=MUTED,
    )
    return np.asarray(canvas)


def append_comparison_segment(writer, fonts, label, size_tag, episode_seed,
                              b_trace, b_frames, m_trace, m_frames, duration_sec):
    b_title = f"Baseline (k={b_trace['frame_stack']})"
    m_title = f"Modified ({m_trace['method']}, k={m_trace['frame_stack']})"
    header = f"{label}  |  DoorKey {size_tag} (episode {episode_seed})"
    subtitle = f"Seed {m_trace['train_seed']} final models ({m_trace['model_step']:,} training steps)"

    # Play active window up to max meaningful step (cap at 28 steps before jump to terminal)
    focus_steps = []
    for tr in (b_trace, m_trace):
        if tr["success"]:
            focus_steps.append(tr["steps"])
        elif tr["cycle"]["has_cycle"]:
            focus_steps.append(min(tr["steps"], tr["cycle"]["start_step"] + 2 * tr["cycle"]["period"]))
        else:
            focus_steps.append(min(tr["steps"], 24))
    max_play_step = max(1, min(32, max(focus_steps)))

    play_frames = int((duration_sec - 2) * FPS)
    for idx in range(play_frames):
        step = min(max_play_step, idx * (max_play_step + 1) // play_frames)
        frame = draw_side_by_side_frame(
            fonts, header, subtitle, f"REPLAY steps 0–{max_play_step}",
            (b_title, b_trace, b_frames, step),
            (m_title, m_trace, m_frames, step),
        )
        writer.append_data(frame)

    end_frame = draw_side_by_side_frame(
        fonts, header, subtitle, "FINAL EPISODE OUTCOME",
        (b_title, b_trace, b_frames, b_trace["steps"]),
        (m_title, m_trace, m_frames, m_trace["steps"]),
    )
    for _ in range(2 * FPS):
        writer.append_data(end_frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=Path("results/baseline"))
    parser.add_argument("--modified-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1, help="Training seed to replay (default: 1)")
    parser.add_argument("--output-dir", type=Path, default=Path("results/videos/comparison"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    torch.set_num_threads(1)
    b_model_path = args.baseline_dir / f"seed_{args.seed}" / "final.zip"
    m_model_path = args.modified_dir / f"seed_{args.seed}" / "final.zip"
    b_model = PPO.load(b_model_path, device=args.device)
    m_model = PPO.load(m_model_path, device=args.device)
    b_meta = {**infer_model_metadata(b_model_path, b_model), "model_path": b_model_path}
    m_meta = {**infer_model_metadata(m_model_path, m_model), "model_path": m_model_path}

    cases = select_representative_cases(args.baseline_dir, args.modified_dir, args.seed)
    durations = (9, 11, 12, 12)  # 44 seconds total (within required 30-60s window)
    fonts = load_fonts()
    demo_path = args.output_dir / "comparison_demo_44s.mp4"
    summary_records = []

    with imageio.get_writer(
        demo_path, fps=FPS, codec="libx264", quality=8,
        macro_block_size=1, ffmpeg_params=["-movflags", "+faststart"],
    ) as writer:
        for (size_tag, env_id, ep_seed, label), dur in zip(cases, durations):
            b_mp4 = args.output_dir / f"{b_meta['method']}_seed{args.seed}_{size_tag}_episode{ep_seed}.mp4"
            m_mp4 = args.output_dir / f"{m_meta['method']}_seed{args.seed}_{size_tag}_episode{ep_seed}.mp4"
            b_trace, b_frames = replay_and_export(b_model, b_meta, env_id, ep_seed, b_mp4)
            m_trace, m_frames = replay_and_export(m_model, m_meta, env_id, ep_seed, m_mp4)
            append_comparison_segment(
                writer, fonts, label, size_tag, ep_seed,
                b_trace, b_frames, m_trace, m_frames, dur,
            )
            summary_records.append({
                "size": size_tag,
                "env_id": env_id,
                "episode_seed": ep_seed,
                "label": label,
                "baseline_success": b_trace["success"],
                "baseline_steps": b_trace["steps"],
                "baseline_cycle": b_trace["cycle"],
                "modified_success": m_trace["success"],
                "modified_steps": m_trace["steps"],
                "modified_cycle": m_trace["cycle"],
            })

    (args.output_dir / "cases_summary.json").write_text(
        json.dumps(summary_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved 44s side-by-side comparison video: {demo_path}")
    print(f"Saved per-episode MP4s, JSON traces, keyframes, and {args.output_dir / 'cases_summary.json'}")


if __name__ == "__main__":
    main()
