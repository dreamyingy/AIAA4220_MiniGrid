"""Compose only the four recorded trajectories into a 44-second demonstration."""

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
FPS = 30
SIZE = (1280, 720)
BACKGROUND = "#101827"
WHITE = "#f1f5f9"
MUTED = "#bcc9dc"
ACCENT = "#67e8f9"
FONT_PATH = font_manager.findfont("DejaVu Sans")
FONTS = {size: ImageFont.truetype(FONT_PATH, size) for size in (18, 22, 26, 30, 40)}
ACTIONS = {None: "reset", 0: "turn left", 1: "turn right", 2: "forward",
           3: "pick up", 4: "drop", 5: "toggle door", 6: "done"}
DIRECTIONS = ("east", "south", "west", "north")


def text(draw, xy, value, size=26, color=WHITE):
    draw.text(xy, value, font=FONTS[size], fill=color)


def panel(frame, trace, step, title, notes, playback):
    canvas = Image.new("RGB", SIZE, BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    text(draw, (40, 26), title, 40)
    text(draw, (40, 84), f"Final model: seed 1 | 1,001,472 training steps | episode {trace['episode_seed']}", 22, MUTED)
    canvas.paste(Image.fromarray(frame).resize((512, 512), Image.Resampling.NEAREST), (40, 140))
    record = trace["trajectory"][step]
    door = record["doors"][0]
    door_status = "open" if door["open"] else "locked" if door["locked"] else "closed"
    lines = [
        f"Step {step} / {trace['steps']}",
        f"Action: {ACTIONS[record['action']]}",
        f"Position: {tuple(record['position'])} | {DIRECTIONS[record['direction']]}",
        f"Carrying: {record['carrying'] or 'nothing'} | door: {door_status}",
    ]
    for i, line in enumerate(lines):
        text(draw, (590, 150 + i * 46), line, 26)
    draw.line((590, 350, 1236, 350), fill="#334155", width=2)
    for i, line in enumerate(notes):
        text(draw, (590, 376 + i * 44), line, 26, ACCENT if i == 0 else MUTED)
    text(draw, (590, 594), playback, 22, "#fbbf24")
    text(draw, (40, 680), "Full map is for visualization; the policy receives only its local 7 x 7 observation.", 18, MUTED)
    return np.asarray(canvas)


def case(writer, stem, title, notes, last_step, duration):
    trace = json.loads((HERE / f"{stem}.json").read_text())
    with imageio.get_reader(HERE / f"{stem}.mp4") as reader:
        frames = {step: reader.get_data(step) for step in [*range(last_step + 1), trace["steps"]]}
    # Show every selected source frame in order, followed by a two-second end hold.
    play_frames = int((duration - 2) * FPS)
    for i in range(play_frames):
        step = min(last_step, i * (last_step + 1) // play_frames)
        writer.append_data(panel(frames[step], trace, step, title, notes,
                                 f"SLOWED REPLAY | steps 0-{last_step}"))
    ending = "END FRAME | held for 2 seconds" if trace["success"] else "JUMP TO TIME LIMIT | step 640"
    final = panel(frames[trace["steps"]], trace, trace["steps"], title, notes, ending)
    for _ in range(2 * FPS):
        writer.append_data(final)


def cycle_sheet():
    canvas = Image.new("RGB", SIZE, BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    text(draw, (30, 20), "Two observed failure cycles | final seed 1, DoorKey 8 x 8", 30)
    text(draw, (30, 74), "Episode 30007: six-step cycle after key pickup; door stays locked", 26, ACCENT)
    for i, step in enumerate(range(6, 12)):
        frame = Image.open(HERE / f"seed1_8x8_episode30007_step_{step:04d}.png")
        x = 30 + i * 208
        canvas.paste(frame.resize((184, 184), Image.Resampling.NEAREST), (x, 122))
        text(draw, (x, 316), f"Step {step}", 22)
    text(draw, (30, 357), "Steps 7-12: forward, right, right, forward, left, left. State 12 equals state 6.", 22, MUTED)
    text(draw, (30, 408), "Episode 30009: two-step turn cycle after passing the door", 26, ACCENT)
    for i, step in enumerate((12, 13, 14)):
        frame = Image.open(HERE / f"seed1_8x8_episode30009_step_{step:04d}.png")
        x = 30 + i * 208
        canvas.paste(frame.resize((184, 184), Image.Resampling.NEAREST), (x, 455))
        text(draw, (x, 648), f"Step {step}", 22)
    for i, line in enumerate(("Position stays at (4, 1).", "Actions: right, left, repeat.",
                              "State 14 equals state 12.", "Both episodes time out at step 640.")):
        text(draw, (674, 464 + i * 48), line, 22, MUTED)
    canvas.save(HERE / "failure_cycles.png")


def main():
    destination = HERE / "baseline_demo_44s.mp4"
    if destination.exists():
        raise FileExistsError(destination)
    with imageio.get_writer(destination, fps=FPS, codec="libx264", quality=8,
                            macro_block_size=1, ffmpeg_params=["-movflags", "+faststart"]) as writer:
        case(writer, "seed1_6x6_episode20000", "SUCCESS  /  6 x 6", [
            "Key -> door -> goal", "Key: step 2", "Door: step 7", "Goal: step 11",
            "Return: 0.972500"], 11, 8)
        case(writer, "seed1_8x8_episode30000", "SUCCESS  /  8 x 8 zero-shot", [
            "Successful transfer in this episode", "Key: step 2", "Door: step 10", "Goal: step 15",
            "Return: 0.978906"], 15, 8)
        case(writer, "seed1_8x8_episode30007", "FAILURE  /  key collected, door still locked", [
            "Six-step cycle from state 6", "Moves between (3, 4) and (3, 5).",
            "Forward, right, right,", "forward, left, left; repeat.", "No door-toggle action; timeout at 640."], 18, 14)
        case(writer, "seed1_8x8_episode30009", "FAILURE  /  door crossed, goal not reached", [
            "Two-step cycle from state 12", "Stays at (4, 1), after crossing the door.",
            "Turn right, turn left; repeat.", "No further forward action.", "Timeout at step 640."], 24, 14)
    cycle_sheet()
    with imageio.get_reader(destination) as reader:
        assert reader.count_frames() == 44 * FPS
        meta = reader.get_meta_data()
        assert meta["size"] == SIZE and meta["fps"] == FPS and meta["duration"] == 44.0
    print(f"Saved {destination.name}: 44 s, 1280 x 720, 30 fps, 1320 frames.")
    print("Saved failure_cycles.png; source MP4, JSON and event keyframes preserved.")


if __name__ == "__main__":
    main()
