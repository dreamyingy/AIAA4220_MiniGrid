"""Replay one fixed episode and export a video, trajectory, and keyframes."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import torch

from common import infer_model_metadata, make_env
from evaluate import load_ppo_model, run_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--episode-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output .mp4 path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".mp4" or args.fps < 1:
        parser.error("--output must end in .mp4 and --fps must be positive")
    trace_path = args.output.with_suffix(".json")
    if args.output.exists() or trace_path.exists():
        raise FileExistsError(f"Video or trajectory already exists: {args.output}")

    torch.set_num_threads(1)
    model = load_ppo_model(args.model, device=args.device)
    meta = infer_model_metadata(args.model, model=model)
    env = make_env(args.env_id, render_mode="rgb_array", frame_stack=meta["frame_stack"])
    try:
        row, frames, trajectory = run_episode(model, env, args.episode_seed, capture=True)
    finally:
        env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(args.output, fps=args.fps, codec="libx264", macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)

    key_steps = sorted({0, row["steps"], *[
        row[event] for event in ("key_step", "door_step", "goal_step") if row[event] is not None
    ]})
    keyframes = []
    for step in key_steps:
        path = args.output.with_name(f"{args.output.stem}_step_{step:04d}.png")
        imageio.imwrite(path, frames[step])
        keyframes.append(path.name)

    trace = {
        "model": str(args.model),
        "method": meta["method"],
        "frame_stack": meta["frame_stack"],
        "train_seed": model.seed,
        "model_step": model.num_timesteps,
        "env_id": args.env_id,
        "deterministic": True,
        "fps": args.fps,
        **row,
        "keyframes": keyframes,
        "trajectory": trajectory,
    }
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}, {trace_path}, and {len(keyframes)} keyframes "
          f"({meta['method']}, frame_stack={meta['frame_stack']}); "
          f"success={row['success']}, episode_steps={row['steps']}, "
          f"video_seconds={len(frames) / args.fps:.1f}")


if __name__ == "__main__":
    main()
