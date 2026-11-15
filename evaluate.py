"""Evaluate a saved PPO policy on fixed DoorKey episode seeds."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import torch
from stable_baselines3 import PPO

from common import make_env


def run_episode(model, env, episode_seed, capture=False):
    """Use the same deterministic action loop for evaluation and video replay."""
    started = perf_counter()
    observation, _ = env.reset(seed=int(episode_seed))
    base = env.unwrapped
    doors = [cell for cell in base.grid.grid if cell is not None and cell.type == "door"]
    key_step = door_step = goal_step = None
    episode_return = 0.0
    steps = 0
    frames, trajectory = [], []

    def capture_step(action, reward):
        frames.append(env.render())
        trajectory.append({
            "step": steps,
            "action": action,
            "reward": float(reward),
            "position": [int(v) for v in base.agent_pos],
            "direction": int(base.agent_dir),
            "carrying": None if base.carrying is None else base.carrying.type,
            "doors": [{"open": bool(door.is_open), "locked": bool(door.is_locked)} for door in doors],
            "observation": observation.tolist(),
        })

    if capture:
        capture_step(None, 0.0)
    while True:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, _ = env.step(int(action))
        steps += 1
        episode_return += float(reward)
        if key_step is None and base.carrying is not None and base.carrying.type == "key":
            key_step = steps
        if door_step is None and any(door.is_open for door in doors):
            door_step = steps
        # In the unmodified DoorKey task, positive reward means reaching the goal.
        # A goal reached on the time limit may also have truncated=True.
        if goal_step is None and reward > 0:
            goal_step = steps
        if capture:
            capture_step(int(action), reward)
        if terminated or truncated:
            break

    row = {
        "episode_seed": int(episode_seed),
        "success": int(goal_step is not None),
        "return": episode_return,
        "steps": steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "key_step": key_step,
        "door_step": door_step,
        "goal_step": goal_step,
        "elapsed_seconds": perf_counter() - started,
    }
    return row, frames, trajectory


def evaluate_episodes(model, env_id, episode_seeds, method="baseline", phase="periodic"):
    """Return one raw row per episode; training RNG and environment are untouched."""
    env = make_env(env_id)
    rows = []
    try:
        for episode_seed in episode_seeds:
            row, _, _ = run_episode(model, env, episode_seed)
            rows.append({
                "method": method,
                "train_seed": model.seed,
                "model_step": model.num_timesteps,
                "phase": phase,
                "env_id": env_id,
                **row,
            })
    finally:
        env.close()
    return rows


def write_results(rows, path, append=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not append or not path.exists()
    with path.open("a" if append else "x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--episode-seed-start", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--method", default="baseline")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.output.exists():
        raise FileExistsError(f"Result already exists; choose another --output: {args.output}")

    torch.set_num_threads(1)
    model = PPO.load(args.model, device=args.device)
    seeds = range(args.episode_seed_start, args.episode_seed_start + args.episodes)
    rows = evaluate_episodes(model, args.env_id, seeds, method=args.method, phase="final")
    write_results(rows, args.output)
    success_rate = sum(row["success"] for row in rows) / len(rows)
    elapsed = sum(row["elapsed_seconds"] for row in rows)
    print(f"{args.env_id}: {len(rows)} episodes, success={success_rate:.1%}, "
          f"model_step={model.num_timesteps}, evaluation_seconds={elapsed:.2f}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
