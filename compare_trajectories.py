"""Select matched episodes, detect deterministic cycles, and render a 1080p presentation demo."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
from matplotlib import font_manager
from minigrid.core.grid import Grid
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
import torch

from common import infer_model_metadata, make_env
from evaluate import run_episode


ENVS = {"6x6": "MiniGrid-DoorKey-6x6-v0", "8x8": "MiniGrid-DoorKey-8x8-v0"}
FPS = 30
SIZE = (1920, 1080)
FADE_FRAMES = 10  # ~0.33s smooth cross-fade between sections

# Deep-slate high-contrast palette aligned with matplotlib comparison plots
BG_TOP = (9, 14, 26)
BG_BOT = (16, 25, 44)
CARD_BG = (21, 31, 50)
CARD_BORDER = (41, 56, 82)
WELL_BG = (12, 19, 34)
WELL_BORDER = (32, 46, 70)
DIVIDER = (38, 52, 76)

WHITE = (248, 250, 252)
MUTED = (168, 184, 206)
SUBTLE = (118, 136, 162)

BASE_COLOR = (56, 189, 248)      # Sky blue — matches baseline plot series
MOD_COLOR = (251, 146, 60)       # Warm orange — matches frame_stack plot series
SUCCESS_FG = (74, 222, 128)
SUCCESS_BG = (17, 54, 36)
FAIL_FG = (248, 113, 113)
FAIL_BG = (62, 22, 29)
WARN_FG = (251, 191, 36)
WARN_BG = (61, 44, 15)
RUN_FG = (125, 211, 252)
RUN_BG = (14, 45, 68)
GOLD = (250, 204, 21)

ACTIONS = {
    None: "reset",
    0: "turn left",
    1: "turn right",
    2: "forward",
    3: "pick up",
    4: "drop",
    5: "toggle",
    6: "done",
}
ACTION_SHORT = {
    None: "START",
    0: "LEFT",
    1: "RIGHT",
    2: "FWD",
    3: "PICK",
    4: "DROP",
    5: "DOOR",
    6: "DONE",
}
DIRECTIONS = ("East", "South", "West", "North")


def load_fonts():
    sans_reg = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="normal"))
    sans_bold = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    mono_reg = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans Mono", weight="normal"))
    mono_bold = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans Mono", weight="bold"))
    fonts = {}
    for size in (14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 32, 34, 38, 44, 52):
        fonts[("sans", size)] = ImageFont.truetype(sans_reg, size)
        fonts[("bold", size)] = ImageFont.truetype(sans_bold, size)
        fonts[("mono", size)] = ImageFont.truetype(mono_reg, size)
        fonts[("mbold", size)] = ImageFont.truetype(mono_bold, size)
    return fonts


def make_gradient_bg():
    w, h = SIZE
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    top = np.array(BG_TOP, dtype=np.float32)[None, None, :]
    bot = np.array(BG_BOT, dtype=np.float32)[None, None, :]
    arr = np.broadcast_to((top * (1.0 - t) + bot * t).astype(np.uint8), (h, w, 3)).copy()
    return Image.fromarray(arr, mode="RGB")


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
        for period in range(1, (total - start) // min_repeats + 1):
            remaining = total - start
            if remaining < period * min_repeats:
                break
            if all(
                signatures[idx] == signatures[start + (idx - start) % period]
                for idx in range(start, total)
            ):
                actions = [ACTIONS[trajectory[start + k]["action"]] for k in range(1, period + 1)]
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

    key_steps = sorted({
        0, row["steps"],
        *[row[event] for event in ("key_step", "door_step", "goal_step") if row[event] is not None],
    })
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
    """Choose 4 informative episodes: 6x6 success, 8x8 rescue, and 2 informative 8x8 failures."""
    b_6 = pd.read_csv(baseline_dir / f"seed_{seed}" / "final_6x6.csv").set_index("episode_seed")
    m_6 = pd.read_csv(modified_dir / f"seed_{seed}" / "final_6x6.csv").set_index("episode_seed")
    b_8 = pd.read_csv(baseline_dir / f"seed_{seed}" / "final_8x8.csv").set_index("episode_seed")
    m_8 = pd.read_csv(modified_dir / f"seed_{seed}" / "final_8x8.csv").set_index("episode_seed")

    cases = []
    succ_6 = m_6.index[m_6["success"].eq(1)].tolist()
    ep_6 = 20000 if 20000 in succ_6 else (succ_6[0] if succ_6 else int(m_6.index[0]))
    cases.append(("6x6", ENVS["6x6"], int(ep_6), "6x6 Task Evaluation"))

    rescued_8 = [
        int(ep) for ep in m_8.index
        if int(b_8.loc[ep, "success"]) == 0 and int(m_8.loc[ep, "success"]) == 1
    ]
    preferred_rescued = [ep for ep in (30007, 30009, 30008) if ep in rescued_8]
    if preferred_rescued:
        ep_8_succ = preferred_rescued[0]
        label_2 = "8x8 Zero-Shot — Rescued Failure"
    elif rescued_8:
        ep_8_succ = rescued_8[0]
        label_2 = "8x8 Zero-Shot — Rescued Failure"
    else:
        succ_8 = m_8.index[m_8["success"].eq(1)].tolist()
        ep_8_succ = 30000 if 30000 in succ_8 else (int(succ_8[0]) if succ_8 else int(m_8.index[0]))
        label_2 = "8x8 Zero-Shot Transfer"
    cases.append(("8x8", ENVS["8x8"], int(ep_8_succ), label_2))

    used = {(c[0], c[2]) for c in cases}
    fail_8 = [int(ep) for ep in m_8.index if int(m_8.loc[ep, "success"]) == 0 and ("8x8", int(ep)) not in used]
    # Show 30009 before 30007 so the short-cycle escape narrative follows directly after the rescue
    ordered_fail_8 = [ep for ep in (30009, 30007) if ep in fail_8] + [
        ep for ep in fail_8 if ep not in (30009, 30007)
    ]
    for ep in ordered_fail_8[:2]:
        cases.append(("8x8", ENVS["8x8"], int(ep), "8x8 Zero-Shot Failure Analysis"))
        used.add(("8x8", int(ep)))

    if len(cases) < 4:
        fail_6 = [int(ep) for ep in m_6.index if int(m_6.loc[ep, "success"]) == 0 and ("6x6", int(ep)) not in used]
        for ep in fail_6[: 4 - len(cases)]:
            cases.append(("6x6", ENVS["6x6"], int(ep), "6x6 Failure Analysis"))
            used.add(("6x6", int(ep)))
    return cases[:4]


def text_size(draw, text, font):
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_pill(draw, x, y, text, font, fg, bg, border=None, pad_x=14, pad_y=6, radius=10):
    tw, th = text_size(draw, text, font)
    w = tw + 2 * pad_x
    h = th + 2 * pad_y
    draw.rounded_rectangle(
        (x, y, x + w, y + h),
        radius=radius,
        fill=bg,
        outline=border or fg,
        width=1 if border else 0,
    )
    draw.text((x + pad_x, y + pad_y - 1), text, font=font, fill=fg)
    return w, h


def draw_pill_right(draw, right_x, y, text, font, fg, bg, border=None, pad_x=14, pad_y=6, radius=10):
    tw, th = text_size(draw, text, font)
    w = tw + 2 * pad_x
    draw_pill(draw, right_x - w, y, text, font, fg, bg, border=border, pad_x=pad_x, pad_y=pad_y, radius=radius)
    return w


def render_obs_tile(obs_hwc, size_px=112):
    """Render an authentic 7x7 MiniGrid egocentric observation via Grid.decode."""
    if not np.any(obs_hwc):
        return None
    grid, vis_mask = Grid.decode(obs_hwc.astype(np.uint8))
    img = grid.render(tile_size=16, agent_pos=(3, 6), agent_dir=3, highlight_mask=vis_mask)
    return Image.fromarray(img).resize((size_px, size_px), Image.Resampling.NEAREST)


def unpack_observation_frames(obs_raw, frame_stack):
    """Return list of `frame_stack` (7, 7, 3) uint8 arrays from oldest to newest."""
    arr = np.asarray(obs_raw, dtype=np.uint8)
    if frame_stack == 1:
        return [arr]
    frames = []
    for i in range(frame_stack):
        frames.append(np.transpose(arr[3 * i: 3 * (i + 1)], (1, 2, 0)))
    return frames


def render_map_with_trail(raw_frame, trace, step_idx, grid_n, map_size, accent_rgb):
    """Upscale global MiniGrid frame and overlay a crisp fading trajectory trail."""
    base_img = Image.fromarray(raw_frame).resize((map_size, map_size), Image.Resampling.NEAREST).convert("RGBA")
    overlay = Image.new("RGBA", (map_size, map_size), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    cell = map_size / float(grid_n)
    clamped = min(step_idx, trace["steps"])
    # Limit trail window on terminal 640 jump so we show the active loop cleanly
    start_i = max(0, clamped - 32)
    pts = [
        (
            (trace["trajectory"][i]["position"][0] + 0.5) * cell,
            (trace["trajectory"][i]["position"][1] + 0.5) * cell,
        )
        for i in range(start_i, clamped + 1)
    ]

    if len(pts) >= 2:
        n_seg = len(pts) - 1
        for i in range(n_seg):
            frac = (i + 1) / float(n_seg)
            alpha = int(55 + 165 * frac)
            width = max(3, int(round(3 + 3 * frac)))
            odraw.line((pts[i], pts[i + 1]), fill=(*accent_rgb, alpha), width=width)
        for i, (px, py) in enumerate(pts[:-1]):
            frac = (i + 1) / float(len(pts))
            r = 4 if i == 0 else 3
            alpha = int(70 + 140 * frac)
            odraw.ellipse((px - r, py - r, px + r, py + r), fill=(*accent_rgb, alpha))

    # Subtle start-cell ring
    sx, sy = (trace["trajectory"][0]["position"][0] + 0.5) * cell, (trace["trajectory"][0]["position"][1] + 0.5) * cell
    odraw.ellipse((sx - 7, sy - 7, sx + 7, sy + 7), outline=(255, 255, 255, 175), width=2)

    # Pulse ring around current agent cell if stuck in a loop
    cyc = trace["cycle"]
    if cyc["has_cycle"] and clamped >= cyc["start_step"]:
        cx, cy = pts[-1]
        r_ring = int(cell * 0.44)
        odraw.ellipse(
            (cx - r_ring, cy - r_ring, cx + r_ring, cy + r_ring),
            outline=(*WARN_FG, 225),
            width=4,
        )

    composed = Image.alpha_composite(base_img, overlay).convert("RGB")
    return composed


def case_narrative(size_tag, ep_seed, b_trace, m_trace):
    """Return crisp, factually grounded title & one-line takeaway for each case."""
    if size_tag == "6x6" and b_trace["success"] and m_trace["success"]:
        return (
            f"6×6 Training Task  ·  Episode {ep_seed}",
            "Sanity check: both policies solve optimally in 11 steps (Key → Door → Goal); compare k=1 vs k=4 inputs below.",
        )
    if not b_trace["success"] and m_trace["success"]:
        b_cyc = b_trace["cycle"]
        if b_cyc["has_cycle"] and b_cyc["period"] == 1:
            return (
                f"8×8 Zero-Shot Rescue  ·  Episode {ep_seed}",
                f"Baseline locks in a 1-step '{b_cyc['action_pattern'][0]}' stall at step 0; 4-frame history explores & solves in {m_trace['steps']} steps.",
            )
        return (
            f"8×8 Zero-Shot Rescue  ·  Episode {ep_seed}",
            f"Baseline times out at 640 steps; Frame Stack (k=4) transfers zero-shot and reaches the goal in {m_trace['steps']} steps.",
        )
    if ep_seed == 30009 and b_trace["cycle"]["has_cycle"] and b_trace["cycle"]["period"] == 2:
        return (
            f"8×8 Short-Cycle Escape & Remaining OOD Limit  ·  Episode {ep_seed}",
            "Baseline traps in a 2-step turn loop after crossing the door; k=4 breaks the turn loop, then stalls facing wall (buffer saturates).",
        )
    if ep_seed == 30007 and b_trace["cycle"]["has_cycle"] and b_trace["cycle"]["period"] == 6:
        return (
            f"8×8 Failure Modes: Multi-Step Loop vs. Window Saturation  ·  Episode {ep_seed}",
            "Baseline enters a 6-step back-and-forth cycle after key pickup; k=4 stalls when 4 identical frames fill [o, o, o, o].",
        )
    return (
        f"DoorKey {size_tag} Comparison  ·  Episode {ep_seed}",
        f"Deterministic replay (Seed {m_trace['train_seed']}): Baseline {'solved' if b_trace['success'] else 'timed out'} vs. Frame Stack {'solved' if m_trace['success'] else 'timed out'}.",
    )


def draw_timeline_bar(draw, fonts, segments, active_idx, seg_frac):
    """Draw a sleek bottom segmented timeline bar across the full 1920px canvas."""
    bar_y = 1022
    left = 48
    right = 1872
    total_w = right - left
    gap = 8
    total_dur = sum(d for _, d in segments)
    usable_w = total_w - gap * (len(segments) - 1)

    cur_x = float(left)
    for idx, (label, dur) in enumerate(segments):
        seg_w = usable_w * (dur / total_dur)
        x0, x1 = int(round(cur_x)), int(round(cur_x + seg_w))
        draw.rounded_rectangle((x0, bar_y, x1, bar_y + 8), radius=4, fill=(30, 42, 64))
        if idx < active_idx:
            draw.rounded_rectangle((x0, bar_y, x1, bar_y + 8), radius=4, fill=BASE_COLOR)
        elif idx == active_idx:
            fill_x = x0 + max(8, int(round((x1 - x0) * seg_frac)))
            draw.rounded_rectangle((x0, bar_y, min(x1, fill_x), bar_y + 8), radius=4, fill=MOD_COLOR)
        lbl_color = WHITE if idx == active_idx else (MUTED if idx < active_idx else SUBTLE)
        draw.text((x0, bar_y + 14), label, font=fonts[("bold" if idx == active_idx else "sans", 14)], fill=lbl_color)
        cur_x += seg_w + gap


def draw_title_card(bg, fonts, segments, seg_frac, comp_stats):
    canvas = bg.copy()
    draw = ImageDraw.Draw(canvas)

    # Top category badge
    draw_pill(draw, 96, 128, "AIAA 4220  ·  EMBODIED AI  ·  TRACK 1: MINIGRID DOORKEY", fonts[("bold", 17)], BASE_COLOR, RUN_BG, border=BASE_COLOR, pad_x=18, pad_y=8, radius=12)

    # Main title
    draw.text((96, 198), "Does Short-Term Frame Stacking Reduce", font=fonts[("bold", 52)], fill=WHITE)
    draw.text((96, 266), "Partial-Observability Failure Loops?", font=fonts[("bold", 52)], fill=MOD_COLOR)

    draw.text(
        (96, 352),
        "A controlled PPO comparison on MiniGrid-DoorKey-6x6-v0 and zero-shot transfer to 8x8.",
        font=fonts[("sans", 26)],
        fill=MUTED,
    )

    # Three protocol cards
    cards = [
        ("FIXED BASELINE", BASE_COLOR, [
            "Official MiniGrid CNN + SB3 PPO",
            "Single partial 7×7×3 egocentric view",
            "1,000,000 nominal steps × 3 seeds",
            "167,096 trainable parameters",
        ]),
        ("ISOLATED MODIFICATION", MOD_COLOR, [
            "Memory: 4-Frame Stacking (k = 4)",
            "Stacked 12×7×7 input (t−3 … t)",
            "Identical PPO, rewards & test seeds",
            "167,672 params (+576 / +0.34%)",
        ]),
        ("WHAT THIS DEMO SHOWS", SUCCESS_FG, [
            "Side-by-side deterministic rollouts",
            "Live global map + trajectory trail",
            "True 7×7 egocentric policy inputs",
            "2 successes & 2 failure diagnostics",
        ]),
    ]
    card_w, card_h = 548, 300
    for i, (title, accent, lines) in enumerate(cards):
        x0 = 96 + i * (card_w + 42)
        y0 = 446
        draw.rounded_rectangle((x0, y0, x0 + card_w, y0 + card_h), radius=16, fill=CARD_BG, outline=CARD_BORDER, width=2)
        draw.rounded_rectangle((x0 + 2, y0 + 2, x0 + card_w - 2, y0 + 8), radius=3, fill=accent)
        draw.text((x0 + 28, y0 + 28), title, font=fonts[("bold", 20)], fill=accent)
        for j, line in enumerate(lines):
            draw.ellipse((x0 + 30, y0 + 88 + j * 48, x0 + 38, y0 + 96 + j * 48), fill=accent)
            draw.text((x0 + 52, y0 + 78 + j * 48), line, font=fonts[("sans", 20)], fill=WHITE)

    # Bottom footer callout
    draw.rounded_rectangle((96, 792, 1824, 882), radius=14, fill=WELL_BG, outline=WELL_BORDER, width=2)
    draw.text(
        (126, 818),
        "Evaluation Protocol:  Matched final models (1,001,472 steps)  ·  Held-out test seeds 20000–20099 (6×6) & 30000–30099 (8×8 zero-shot)",
        font=fonts[("mono", 18)],
        fill=MUTED,
    )

    draw_timeline_bar(draw, fonts, segments, 0, seg_frac)
    return np.asarray(canvas)


def draw_summary_card(bg, fonts, segments, seg_frac, comp_stats):
    canvas = bg.copy()
    draw = ImageDraw.Draw(canvas)

    draw_pill(draw, 96, 72, "CONTROLLED QUANTITATIVE SUMMARY  ·  3 SEEDS (MEAN ± SAMPLE SD, DDOF=1)", fonts[("bold", 17)], MOD_COLOR, WARN_BG, border=MOD_COLOR, pad_x=18, pad_y=8, radius=12)
    draw.text((96, 132), "Key Findings Across All 600 Final Test Episodes", font=fonts[("bold", 44)], fill=WHITE)

    def fmt_pct(row, col_m, col_s):
        return f"{row[col_m] * 100:.1f}% ± {row[col_s] * 100:.1f} pp"

    sr6 = comp_stats["final_6x6_success_rate"]
    sr8 = comp_stats["final_8x8_success_rate"]
    nk8 = comp_stats["final_8x8_fail_no_key_rate"]
    pk8 = comp_stats["final_8x8_post_key_fail_rate"]
    auc = comp_stats["normalized_auc"]
    t80 = comp_stats["t80"]

    kpis = [
        ("6×6 Training Task SR", fmt_pct(sr6, "baseline_mean", "baseline_std"), fmt_pct(sr6, "modified_mean", "modified_std"), f"+{sr6['delta_mean'] * 100:.1f} pp  (6.8× lower SD)", SUCCESS_FG),
        ("8×8 Zero-Shot Transfer SR", fmt_pct(sr8, "baseline_mean", "baseline_std"), fmt_pct(sr8, "modified_mean", "modified_std"), f"+{sr8['delta_mean'] * 100:.1f} pp  (high seed var.)", SUCCESS_FG),
        ("8×8 Pre-Key Stall Fail Rate", fmt_pct(nk8, "baseline_mean", "baseline_std"), fmt_pct(nk8, "modified_mean", "modified_std"), f"{nk8['delta_mean'] * 100:+.1f} pp  (fewer stalls)", SUCCESS_FG),
        ("6×6 Sample Efficiency (T80 / AUC)", f"{int(round(t80['baseline_mean'] / 1000))}k  /  {auc['baseline_mean']:.3f}", f"{int(round(t80['modified_mean'] / 1000))}k  /  {auc['modified_mean']:.3f}", "Slower early learning trade-off", WARN_FG),
    ]

    kw, kh = 840, 176
    for idx, (label, b_val, m_val, delta_txt, delta_col) in enumerate(kpis):
        col = idx % 2
        row = idx // 2
        x0 = 96 + col * (kw + 48)
        y0 = 216 + row * (kh + 28)
        draw.rounded_rectangle((x0, y0, x0 + kw, y0 + kh), radius=16, fill=CARD_BG, outline=CARD_BORDER, width=2)
        draw.text((x0 + 28, y0 + 22), label, font=fonts[("bold", 22)], fill=WHITE)
        draw_pill_right(draw, x0 + kw - 24, y0 + 18, delta_txt, fonts[("bold", 17)], delta_col, WELL_BG, border=delta_col, pad_x=14, pad_y=6, radius=10)

        draw.text((x0 + 28, y0 + 78), "Baseline (k=1)", font=fonts[("bold", 16)], fill=BASE_COLOR)
        draw.text((x0 + 28, y0 + 106), b_val, font=fonts[("mbold", 26)], fill=MUTED)

        draw.text((x0 + 440, y0 + 78), "Frame Stack (k=4)", font=fonts[("bold", 16)], fill=MOD_COLOR)
        draw.text((x0 + 440, y0 + 106), m_val, font=fonts[("mbold", 26)], fill=WHITE)

    # Bottom calibrated conclusion box
    draw.rounded_rectangle((96, 622, 1824, 964), radius=16, fill=WELL_BG, outline=CARD_BORDER, width=2)
    draw.text((130, 648), "CALIBRATED MECHANISM CONCLUSION & NEXT DIAGNOSTIC", font=fonts[("bold", 20)], fill=BASE_COLOR)
    bullets = [
        ("What Improved:", "Asymptotic 6×6 reliability rises to 97.0% ± 1.0% and pre-key freezes drop on both 6×6 (−7.0 pp) and 8×8 (−7.3 pp).", SUCCESS_FG),
        ("What Persisted:", f"Post-key 8×8 timeout rate remains high ({pk8['modified_mean']*100:.1f}% vs. {pk8['baseline_mean']*100:.1f}%): 4-frame stacks saturate [o,o,o,o] after 4 no-op steps.", WARN_FG),
        ("Root Limit:", "Fixed 6×6 training exposes at most 4-tile spans; CNN Flatten→Linear overfits to 6×6 coordinates under 8×8 zero-shot transfer.", MOD_COLOR),
        ("Next Diagnostic:", "Test recurrent PPO (LSTM/GRU) for unbounded memory, or room-size domain randomization for spatial generalization.", BASE_COLOR),
    ]
    for i, (head, body, col) in enumerate(bullets):
        y = 700 + i * 62
        draw.rounded_rectangle((130, y + 6, 138, y + 32), radius=3, fill=col)
        draw.text((156, y + 4), head, font=fonts[("bold", 20)], fill=col)
        draw.text((374, y + 4), body, font=fonts[("sans", 19)], fill=WHITE)

    draw_timeline_bar(draw, fonts, segments, len(segments) - 1, seg_frac)
    return np.asarray(canvas)


def draw_panel(canvas, draw, fonts, x0, y0, card_w, card_h, accent_rgb, method_title, sub_tag,
               trace, frames, step_idx, is_final_hold, grid_n, obs_cache):
    """Draw one method's complete 894x876 dashboard card."""
    # Card frame & top accent strip
    draw.rounded_rectangle((x0, y0, x0 + card_w, y0 + card_h), radius=18, fill=CARD_BG, outline=CARD_BORDER, width=2)
    draw.rounded_rectangle((x0 + 2, y0 + 2, x0 + card_w - 2, y0 + 8), radius=3, fill=accent_rgb)

    clamped = min(step_idx, trace["steps"])
    rec = trace["trajectory"][clamped]
    cyc = trace["cycle"]
    in_cycle = bool(cyc["has_cycle"] and clamped >= cyc["start_step"])
    reached_goal = bool(trace["success"] and clamped >= trace["steps"])
    timed_out = bool((not trace["success"]) and (is_final_hold or clamped >= trace["steps"]))

    # 1. Card header
    draw.text((x0 + 24, y0 + 22), method_title, font=fonts[("bold", 26)], fill=WHITE)
    tw_title, _ = text_size(draw, method_title, fonts[("bold", 26)])
    draw.text((x0 + 38 + tw_title, y0 + 29), sub_tag, font=fonts[("mono", 17)], fill=MUTED)

    # Dynamic non-spoiling status pill
    if reached_goal:
        draw_pill_right(draw, x0 + card_w - 24, y0 + 20, f"✓ SOLVED  ·  {trace['steps']} STEPS", fonts[("bold", 17)], SUCCESS_FG, SUCCESS_BG, border=SUCCESS_FG)
    elif timed_out:
        draw_pill_right(draw, x0 + card_w - 24, y0 + 20, f"✗ TIMEOUT  ·  {trace['steps']} STEPS", fonts[("bold", 17)], FAIL_FG, FAIL_BG, border=FAIL_FG)
    elif in_cycle:
        draw_pill_right(draw, x0 + card_w - 24, y0 + 20, f"⟳ LOOP LOCKED (P={cyc['period']})", fonts[("bold", 17)], WARN_FG, WARN_BG, border=WARN_FG)
    else:
        draw_pill_right(draw, x0 + card_w - 24, y0 + 20, "● RUNNING", fonts[("bold", 17)], RUN_FG, RUN_BG, border=RUN_FG)

    # 2. Milestone stepper (KEY -> DOOR -> GOAL) + Step counter
    sy = y0 + 74
    milestones = [
        ("KEY", trace["key_step"], GOLD),
        ("DOOR", trace["door_step"], BASE_COLOR),
        ("GOAL", trace["goal_step"], SUCCESS_FG),
    ]
    mx = x0 + 24
    for idx, (m_name, m_step, m_col) in enumerate(milestones):
        done = m_step is not None and clamped >= m_step
        lbl = f"✓ {m_name} @{m_step}" if done else f"○ {m_name}"
        fg = m_col if done else SUBTLE
        bg = (24, 38, 56) if done else WELL_BG
        w_p, _ = draw_pill(draw, mx, sy, lbl, fonts[("bold", 15)], fg, bg, border=m_col if done else WELL_BORDER, pad_x=12, pad_y=6, radius=9)
        mx += w_p
        if idx < 2:
            draw.line((mx + 6, sy + 16, mx + 22, sy + 16), fill=m_col if done else DIVIDER, width=2)
            mx += 28

    step_str = f"STEP {clamped:03d} / {trace['steps']:03d}"
    sw, _ = text_size(draw, step_str, fonts[("mbold", 19)])
    draw.text((x0 + card_w - 24 - sw, sy + 5), step_str, font=fonts[("mbold", 19)], fill=WHITE)

    # 3. Full global map with trajectory trail
    map_size = 416
    map_x = x0 + (card_w - map_size) // 2
    map_y = y0 + 124
    map_img = render_map_with_trail(frames[clamped], trace, clamped, grid_n, map_size, accent_rgb)
    draw.rounded_rectangle((map_x - 3, map_y - 3, map_x + map_size + 3, map_y + map_size + 3), radius=10, outline=CARD_BORDER, width=2)
    canvas.paste(map_img, (map_x, map_y))

    # Subtle side caption on left of map
    draw.text((x0 + 24, map_y + 4), "GLOBAL MAP", font=fonts[("bold", 14)], fill=SUBTLE)
    draw.text((x0 + 24, map_y + 24), "(for analysis)", font=fonts[("sans", 14)], fill=SUBTLE)
    door_info = rec["doors"][0]
    door_st = "OPEN" if door_info["open"] else ("LOCKED" if door_info["locked"] else "CLOSED")
    state_lines = [
        ("POS", f"{tuple(rec['position'])}"),
        ("DIR", DIRECTIONS[rec["direction"]]),
        ("CARRY", rec["carrying"].upper() if rec["carrying"] else "NONE"),
        ("DOOR", door_st),
        ("RETURN", f"{trace['return']:.3f}" if (reached_goal or timed_out) else "0.000"),
    ]
    for i, (k_lbl, v_lbl) in enumerate(state_lines):
        ly = map_y + 72 + i * 64
        draw.text((x0 + 24, ly), k_lbl, font=fonts[("bold", 14)], fill=SUBTLE)
        val_col = SUCCESS_FG if (k_lbl == "DOOR" and door_st == "OPEN") or (k_lbl == "CARRY" and v_lbl == "KEY") else WHITE
        draw.text((x0 + 24, ly + 20), v_lbl, font=fonts[("mbold", 18)], fill=val_col)

    # 4. Policy Observation Well (TRUE 7x7 egocentric input!)
    well_x = x0 + 24
    well_y = y0 + 556
    well_w = card_w - 48
    well_h = 178
    draw.rounded_rectangle((well_x, well_y, well_x + well_w, well_y + well_h), radius=14, fill=WELL_BG, outline=WELL_BORDER, width=2)

    k_stack = int(trace["frame_stack"])
    obs_frames = unpack_observation_frames(rec["observation"], k_stack)
    tile_px = 112

    if k_stack == 1:
        draw.text((well_x + 18, well_y + 12), "POLICY INPUT: SINGLE 7×7 EGOCENTRIC OBSERVATION  (t)", font=fonts[("bold", 15)], fill=accent_rgb)
        cache_key = (trace["method"], trace["episode_seed"], clamped, 0)
        if cache_key not in obs_cache:
            obs_cache[cache_key] = render_obs_tile(obs_frames[0], size_px=tile_px)
        tile_img = obs_cache[cache_key]
        tx, ty = well_x + 22, well_y + 44
        border_col = WARN_FG if in_cycle else accent_rgb
        draw.rounded_rectangle((tx - 3, ty - 3, tx + tile_px + 3, ty + tile_px + 3), radius=6, outline=border_col, width=3)
        if tile_img is not None:
            canvas.paste(tile_img, (tx, ty))

        # Mechanism annotation beside single obs
        ix = tx + tile_px + 28
        draw.text((ix, ty + 4), "Memory Window:  1 frame (stateless)", font=fonts[("bold", 17)], fill=WHITE)
        if in_cycle:
            short_map = {v: ACTION_SHORT[k] for k, v in ACTIONS.items()}
            pat = " → ".join(short_map.get(a, a) for a in cyc["action_pattern"])
            draw.text((ix, ty + 36), f"Observational Aliasing Locked (P={cyc['period']}):", font=fonts[("bold", 16)], fill=WARN_FG)
            draw.text((ix, ty + 62), "Same 7×7 view repeats → π(a|o) repeats", font=fonts[("sans", 16)], fill=MUTED)
            draw.text((ix, ty + 88), f"Cycle: [{pat}] ({cyc['repeats']}×)", font=fonts[("mono", 15)], fill=WARN_FG)
        else:
            draw.text((ix, ty + 36), "Policy maps current 7×7 grid → action.", font=fonts[("sans", 16)], fill=MUTED)
            draw.text((ix, ty + 64), "No memory of prior turns or visited tiles.", font=fonts[("sans", 16)], fill=SUBTLE)
            if reached_goal:
                draw.text((ix, ty + 90), "Direct line-of-sight cues sufficed.", font=fonts[("sans", 16)], fill=SUCCESS_FG)
    else:
        non_zero = [f for f in obs_frames if np.any(f)]
        unique_cnt = len({f.tobytes() for f in non_zero})
        saturated = (len(non_zero) == k_stack) and (unique_cnt == 1)
        draw.text((well_x + 18, well_y + 12), "POLICY INPUT: 4-FRAME STACKED BUFFER  (t−3 → t)", font=fonts[("bold", 15)], fill=accent_rgb)
        if saturated:
            draw_pill_right(draw, well_x + well_w - 16, well_y + 8, "BUFFER SATURATED  [o, o, o, o]", fonts[("bold", 14)], WARN_FG, WARN_BG, border=WARN_FG, pad_x=10, pad_y=4, radius=8)
        else:
            draw_pill_right(draw, well_x + well_w - 16, well_y + 8, f"DISTINCT FRAMES: {unique_cnt} / {k_stack}", fonts[("mono", 14)], SUCCESS_FG if unique_cnt > 1 else MUTED, CARD_BG, border=WELL_BORDER, pad_x=10, pad_y=4, radius=8)

        slot_labels = ("t−3", "t−2", "t−1", "t (now)")
        spacing = 24
        start_tx = well_x + 22
        ty = well_y + 42
        for slot_i in range(k_stack):
            sx_t = start_tx + slot_i * (tile_px + spacing)
            cache_key = (trace["method"], trace["episode_seed"], clamped, slot_i)
            if cache_key not in obs_cache:
                obs_cache[cache_key] = render_obs_tile(obs_frames[slot_i], size_px=tile_px)
            tile_img = obs_cache[cache_key]
            is_latest = slot_i == k_stack - 1
            b_col = WARN_FG if saturated else (accent_rgb if is_latest else CARD_BORDER)
            draw.rounded_rectangle((sx_t - 2, ty - 2, sx_t + tile_px + 2, ty + tile_px + 2), radius=6, outline=b_col, width=3 if (is_latest or saturated) else 2)
            if tile_img is not None:
                canvas.paste(tile_img, (sx_t, ty))
            else:
                draw.rectangle((sx_t, ty, sx_t + tile_px, ty + tile_px), fill=(15, 22, 36))
                pw, ph = text_size(draw, "0-PAD", fonts[("mono", 14)])
                draw.text((sx_t + (tile_px - pw) // 2, ty + (tile_px - ph) // 2), "0-PAD", font=fonts[("mono", 14)], fill=SUBTLE)
            # Slot caption
            lw, _ = text_size(draw, slot_labels[slot_i], fonts[("bold" if is_latest else "mono", 14)])
            draw.text((sx_t + (tile_px - lw) // 2, ty + tile_px + 4), slot_labels[slot_i], font=fonts[("bold" if is_latest else "mono", 14)], fill=accent_rgb if is_latest else MUTED)

        # Mini summary on right side of the 4 tiles
        rx = start_tx + k_stack * (tile_px + spacing) + 6
        draw.text((rx, ty + 6), "12×7×7", font=fonts[("mbold", 20)], fill=WHITE)
        draw.text((rx, ty + 32), "stacked tensor", font=fonts[("sans", 15)], fill=MUTED)
        if saturated:
            draw.text((rx, ty + 64), "4 no-op steps", font=fonts[("bold", 15)], fill=WARN_FG)
            draw.text((rx, ty + 84), "erase history!", font=fonts[("bold", 15)], fill=WARN_FG)
        elif unique_cnt > 1:
            draw.text((rx, ty + 64), "Motion & turn", font=fonts[("sans", 15)], fill=SUCCESS_FG)
            draw.text((rx, ty + 84), "context active", font=fonts[("sans", 15)], fill=SUCCESS_FG)
        else:
            draw.text((rx, ty + 64), "Episode start", font=fonts[("sans", 15)], fill=SUBTLE)
            draw.text((rx, ty + 84), "(zero-padded)", font=fonts[("sans", 15)], fill=SUBTLE)

    # 5. Live Action History Ribbon
    ry = y0 + 752
    draw.text((x0 + 24, ry + 6), "RECENT ACTIONS", font=fonts[("bold", 15)], fill=SUBTLE)
    hist_start = max(0, clamped - 6)
    ax = x0 + 178
    for s_i in range(hist_start, clamped + 1):
        act_id = trace["trajectory"][s_i]["action"]
        act_txt = ACTION_SHORT[act_id]
        is_cur = s_i == clamped
        is_cyc_step = bool(cyc["has_cycle"] and s_i >= cyc["start_step"] and s_i > 0)
        if is_cur:
            fg = WARN_FG if is_cyc_step else WHITE
            bg = WARN_BG if is_cyc_step else (37, 63, 94)
            bd = WARN_FG if is_cyc_step else accent_rgb
        elif is_cyc_step:
            fg, bg, bd = WARN_FG, (45, 34, 18), (110, 82, 26)
        else:
            fg, bg, bd = MUTED, WELL_BG, WELL_BORDER
        w_p, _ = draw_pill(draw, ax, ry, act_txt, fonts[("mbold" if is_cur else "mono", 15)], fg, bg, border=bd, pad_x=11, pad_y=6, radius=8)
        ax += w_p + 10

    # Bottom status summary line inside card
    fy = y0 + 812
    draw.line((x0 + 24, fy - 10, x0 + card_w - 24, fy - 10), fill=DIVIDER, width=1)
    if reached_goal:
        foot = f"Outcome: Solved in {trace['steps']} steps  ·  Original Return = {trace['return']:.4f}"
        f_col = SUCCESS_FG
    elif cyc["has_cycle"] and (in_cycle or is_final_hold):
        short_map = {v: ACTION_SHORT[k] for k, v in ACTIONS.items()}
        pat = "→".join(short_map.get(a, a) for a in cyc["action_pattern"])
        foot = f"Cycle (P={cyc['period']} from step {cyc['start_step']}): [{pat}] × {cyc['repeats']}  →  timeout at 640"
        f_col = WARN_FG
    else:
        foot = f"Current Action: {ACTIONS[rec['action']]}  ·  Deterministic policy π(a|o)"
        f_col = MUTED
    draw.text((x0 + 24, fy + 4), foot, font=fonts[("sans", 17)], fill=f_col)


def draw_comparison_frame(bg, fonts, case_idx, total_cases, title, takeaway, playback_tag, is_jump_tag,
                          left_pack, right_pack, grid_n, segments, active_seg_idx, seg_frac, obs_cache):
    canvas = bg.copy()
    draw = ImageDraw.Draw(canvas)

    # Top header
    case_pill = f"CASE {case_idx} / {total_cases}"
    pw, _ = draw_pill(draw, 48, 22, case_pill, fonts[("bold", 16)], WHITE, (35, 52, 82), border=BASE_COLOR, pad_x=14, pad_y=6, radius=10)
    draw.text((48 + pw + 18, 16), title, font=fonts[("bold", 32)], fill=WHITE)
    draw.text((48, 66), takeaway, font=fonts[("sans", 20)], fill=MUTED)

    tag_fg = WARN_FG if is_jump_tag else BASE_COLOR
    tag_bg = WARN_BG if is_jump_tag else RUN_BG
    draw_pill_right(draw, 1872, 20, playback_tag, fonts[("mbold", 17)], tag_fg, tag_bg, border=tag_fg, pad_x=16, pad_y=7, radius=10)
    meta_txt = f"Seed {right_pack[1]['train_seed']} · 1,001,472 steps"
    mw, _ = text_size(draw, meta_txt, fonts[("mono", 15)])
    draw.text((1872 - mw, 68), meta_txt, font=fonts[("mono", 15)], fill=SUBTLE)

    # Two side-by-side cards
    card_w, card_h = 894, 876
    draw_panel(
        canvas, draw, fonts, 48, 118, card_w, card_h, BASE_COLOR,
        "Baseline  (k = 1)", "3×7×7 obs",
        left_pack[1], left_pack[2], left_pack[3], left_pack[4], grid_n, obs_cache,
    )
    draw_panel(
        canvas, draw, fonts, 978, 118, card_w, card_h, MOD_COLOR,
        "Frame Stack  (k = 4)", "12×7×7 obs",
        right_pack[1], right_pack[2], right_pack[3], right_pack[4], grid_n, obs_cache,
    )

    draw_timeline_bar(draw, fonts, segments, active_seg_idx, seg_frac)
    return np.asarray(canvas)


def blend_frames(frame_a, frame_b, alpha):
    return (frame_a.astype(np.float32) * (1.0 - alpha) + frame_b.astype(np.float32) * alpha).astype(np.uint8)


def write_with_crossfade(writer, prev_tail_frame, segment_frames):
    if prev_tail_frame is not None and len(segment_frames) >= FADE_FRAMES:
        for i in range(FADE_FRAMES):
            # Smoothstep easing for crisp transitions
            t = (i + 1) / float(FADE_FRAMES + 1)
            alpha = t * t * (3.0 - 2.0 * t)
            writer.append_data(blend_frames(prev_tail_frame, segment_frames[i], alpha))
        for frame in segment_frames[FADE_FRAMES:]:
            writer.append_data(frame)
    else:
        for frame in segment_frames:
            writer.append_data(frame)
    return segment_frames[-1]


def build_case_frames(bg, fonts, case_idx, total_cases, size_tag, ep_seed, b_trace, b_frames,
                      m_trace, m_frames, duration_sec, segments, seg_idx, obs_cache):
    grid_n = 6 if size_tag == "6x6" else 8
    title, takeaway = case_narrative(size_tag, ep_seed, b_trace, m_trace)

    # Determine informative active window
    focus_steps = []
    for tr in (b_trace, m_trace):
        if tr["success"]:
            focus_steps.append(tr["steps"])
        elif tr["cycle"]["has_cycle"]:
            focus_steps.append(min(tr["steps"], tr["cycle"]["start_step"] + max(6, 2 * tr["cycle"]["period"])))
        else:
            focus_steps.append(min(tr["steps"], 24))
    max_play_step = max(1, min(30, max(focus_steps)))

    total_frames = int(round(duration_sec * FPS))
    hold_frames = int(round(2.4 * FPS))
    play_frames = total_frames - hold_frames
    out_frames = []

    for idx in range(play_frames):
        step = min(max_play_step, idx * (max_play_step + 1) // play_frames)
        seg_frac = idx / float(max(1, total_frames - 1))
        out_frames.append(
            draw_comparison_frame(
                bg, fonts, case_idx, total_cases, title, takeaway,
                f"REPLAY  STEPS 0 → {max_play_step}", False,
                ("baseline", b_trace, b_frames, step, False),
                ("frame_stack", m_trace, m_frames, step, False),
                grid_n, segments, seg_idx, seg_frac, obs_cache,
            )
        )

    has_timeout = (not b_trace["success"]) or (not m_trace["success"])
    end_tag = "JUMP → STEP 640 (FINAL)" if has_timeout else "FINAL EPISODE OUTCOME"
    for idx in range(hold_frames):
        seg_frac = (play_frames + idx) / float(max(1, total_frames - 1))
        out_frames.append(
            draw_comparison_frame(
                bg, fonts, case_idx, total_cases, title, takeaway,
                end_tag, has_timeout,
                ("baseline", b_trace, b_frames, b_trace["steps"], True),
                ("frame_stack", m_trace, m_frames, m_trace["steps"], True),
                grid_n, segments, seg_idx, seg_frac, obs_cache,
            )
        )
    return out_frames


def load_comparison_stats(comparison_csv):
    if comparison_csv is None or not Path(comparison_csv).is_file():
        return None
    df = pd.read_csv(comparison_csv).set_index("metric")
    return {metric: df.loc[metric].to_dict() for metric in df.index}


def load_saved_episode(json_path):
    trace = json.loads(json_path.read_text(encoding="utf-8"))
    mp4_path = json_path.with_suffix(".mp4")
    with imageio.get_reader(mp4_path) as reader:
        frames = [reader.get_data(i) for i in range(trace["steps"] + 1)]
    return trace, frames


def compose_demo_video(output_dir, case_packs, comp_stats):
    bg = make_gradient_bg()
    fonts = load_fonts()
    obs_cache = {}

    has_summary = comp_stats is not None
    intro_dur = 4.0
    case_durs = (8.5, 12.0, 12.0, 10.0)
    summary_dur = 5.5 if has_summary else 0.0
    total_sec = int(round(intro_dur + sum(case_durs[:len(case_packs)]) + summary_dur))

    seg_labels = [
        ("Intro", intro_dur),
        *[
            (f"{i + 1}: {sz} #{ep}", case_durs[i])
            for i, (sz, _, ep, _, _, _, _) in enumerate(case_packs)
        ],
    ]
    if has_summary:
        seg_labels.append(("Summary", summary_dur))

    demo_path = output_dir / f"comparison_demo_{total_sec}s.mp4"
    prev_tail = None
    with imageio.get_writer(
        demo_path, fps=FPS, codec="libx264", quality=8,
        macro_block_size=1, ffmpeg_params=["-movflags", "+faststart"],
    ) as writer:
        # 0. Title card with smooth fade-in from black
        intro_n = int(round(intro_dur * FPS))
        black = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
        intro_frames = []
        for i in range(intro_n):
            frame = draw_title_card(bg, fonts, seg_labels, i / float(max(1, intro_n - 1)), comp_stats)
            if i < FADE_FRAMES:
                t = (i + 1) / float(FADE_FRAMES + 1)
                frame = blend_frames(black, frame, t * t * (3.0 - 2.0 * t))
            intro_frames.append(frame)
        prev_tail = write_with_crossfade(writer, None, intro_frames)

        # 1..4. Side-by-side comparison cases
        for idx, (size_tag, _, ep_seed, b_trace, b_frames, m_trace, m_frames) in enumerate(case_packs):
            seg_frames = build_case_frames(
                bg, fonts, idx + 1, len(case_packs), size_tag, ep_seed,
                b_trace, b_frames, m_trace, m_frames,
                case_durs[idx], seg_labels, idx + 1, obs_cache,
            )
            prev_tail = write_with_crossfade(writer, prev_tail, seg_frames)

        # 5. Closing quantitative summary card
        if has_summary:
            sum_n = int(round(summary_dur * FPS))
            sum_frames = [
                draw_summary_card(bg, fonts, seg_labels, i / float(max(1, sum_n - 1)), comp_stats)
                for i in range(sum_n)
            ]
            write_with_crossfade(writer, prev_tail, sum_frames)

    return demo_path, total_sec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=Path("results/baseline"))
    parser.add_argument("--modified-dir", type=Path, default=Path("results/frame_stack"))
    parser.add_argument("--comparison-csv", type=Path, default=Path("results/comparison/comparison_table.csv"))
    parser.add_argument("--seed", type=int, default=1, help="Training seed to replay (default: 1)")
    parser.add_argument("--output-dir", type=Path, default=Path("results/videos/comparison"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--recompose",
        action="store_true",
        help="Re-render the 1080p demo video directly from existing MP4/JSON files in --output-dir",
    )
    args = parser.parse_args()

    comp_stats = load_comparison_stats(args.comparison_csv)

    if args.recompose:
        summary_file = args.output_dir / "cases_summary.json"
        if not summary_file.is_file():
            raise FileNotFoundError(f"Cannot --recompose without {summary_file}")
        records = json.loads(summary_file.read_text(encoding="utf-8"))
        # Order: 6x6 (20000), 8x8 rescue (30008), 8x8 short-cycle escape (30009), 8x8 limit (30007)
        order_pref = {20000: 0, 30008: 1, 30009: 2, 30007: 3}
        records.sort(key=lambda r: order_pref.get(int(r["episode_seed"]), 99))
        case_packs = []
        for rec in records:
            sz = rec["size"]
            ep = int(rec["episode_seed"])
            b_json = args.output_dir / f"baseline_seed{args.seed}_{sz}_episode{ep}.json"
            m_json = args.output_dir / f"frame_stack_seed{args.seed}_{sz}_episode{ep}.json"
            b_trace, b_frames = load_saved_episode(b_json)
            m_trace, m_frames = load_saved_episode(m_json)
            case_packs.append((sz, rec["env_id"], ep, b_trace, b_frames, m_trace, m_frames))
        demo_path, total_sec = compose_demo_video(args.output_dir, case_packs, comp_stats)
        summary_file.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Recomposed {total_sec}s 1080p comparison demo video: {demo_path}")
        return

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
    case_packs = []
    summary_records = []
    for size_tag, env_id, ep_seed, label in cases:
        b_mp4 = args.output_dir / f"{b_meta['method']}_seed{args.seed}_{size_tag}_episode{ep_seed}.mp4"
        m_mp4 = args.output_dir / f"{m_meta['method']}_seed{args.seed}_{size_tag}_episode{ep_seed}.mp4"
        b_trace, b_frames = replay_and_export(b_model, b_meta, env_id, ep_seed, b_mp4)
        m_trace, m_frames = replay_and_export(m_model, m_meta, env_id, ep_seed, m_mp4)
        case_packs.append((size_tag, env_id, ep_seed, b_trace, b_frames, m_trace, m_frames))
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

    demo_path, total_sec = compose_demo_video(args.output_dir, case_packs, comp_stats)
    (args.output_dir / "cases_summary.json").write_text(
        json.dumps(summary_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {total_sec}s 1080p comparison demo video: {demo_path}")
    print(f"Saved per-episode MP4s, JSON traces, keyframes, and {args.output_dir / 'cases_summary.json'}")


if __name__ == "__main__":
    main()
