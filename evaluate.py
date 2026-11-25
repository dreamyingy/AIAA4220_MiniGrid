"""Evaluate a saved PPO policy on fixed DoorKey episode seeds."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from stable_baselines3 import PPO

from common import infer_model_metadata, make_env


def ppo_class(algo):
    if algo == "ppo":
        return PPO
    if algo == "recurrent_ppo":
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO
    raise ValueError(f"Unsupported PPO algorithm: {algo}")


def load_ppo_model(model_path, device="cpu"):
    """Load the PPO variant identified by the training configuration."""
    config = infer_model_metadata(model_path)["config"] or {}
    return ppo_class(config.get("algo", "ppo")).load(model_path, device=device)


def run_episode(model, env, episode_seed, capture=False, *, record_actions=False):
    """Use the same deterministic action loop for evaluation and video replay."""
    started = perf_counter()
    observation, _ = env.reset(seed=int(episode_seed))
    base = env.unwrapped
    doors = [cell for cell in base.grid.grid if cell is not None and cell.type == "door"]
    key_step = door_step = goal_step = None
    episode_return = 0.0
    steps = 0
    frames, trajectory = [], []
    actions = [] if record_actions else None
    recurrent = getattr(getattr(model, "policy", None), "lstm_actor", None) is not None
    recurrent_state = None
    episode_start = np.ones(1, dtype=bool) if recurrent else None

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
        if recurrent:
            action, recurrent_state = model.predict(
                observation, state=recurrent_state, episode_start=episode_start, deterministic=True,
            )
            episode_start.fill(False)
        else:
            action, _ = model.predict(observation, deterministic=True)
        action = int(action)
        if actions is not None:
            actions.append(str(action))
        observation, reward, terminated, truncated, _ = env.step(action)
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
            capture_step(action, reward)
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
    if actions is not None:
        if row["success"]:
            outcome = "success"
        elif key_step is None:
            outcome = "no_key"
        elif door_step is None:
            outcome = "key_no_door"
        else:
            outcome = "door_no_goal"
        row["outcome"] = outcome
        row["actions"] = " ".join(actions)
    return row, frames, trajectory


def evaluate_episodes(
    model, env_id, episode_seeds, method="baseline", phase="periodic", frame_stack=1,
    *, record_actions=False,
):
    """Return one raw row per episode; training RNG and environment are untouched."""
    env = make_env(env_id, frame_stack=frame_stack)
    rows = []
    try:
        for episode_seed in episode_seeds:
            row, _, _ = run_episode(model, env, episode_seed, record_actions=record_actions)
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
    parser.add_argument("--phase", choices=("dev", "final"), default="final")
    parser.add_argument("--record-actions", action="store_true")
    parser.add_argument("--method", default=None, help="Defaults to method in adjacent config.json")
    parser.add_argument("--frame-stack", type=int, default=None, help="Defaults to frame_stack in adjacent config.json or model channels")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.output.exists():
        raise FileExistsError(f"Result already exists; choose another --output: {args.output}")

    torch.set_num_threads(1)
    model = load_ppo_model(args.model, device=args.device)
    meta = infer_model_metadata(args.model, model=model)
    method = args.method if args.method is not None else meta["method"]
    frame_stack = args.frame_stack if args.frame_stack is not None else meta["frame_stack"]
    if frame_stack != meta["frame_stack"]:
        raise ValueError(
            f"--frame-stack={frame_stack} conflicts with model metadata frame_stack={meta['frame_stack']}"
        )

    seeds = range(args.episode_seed_start, args.episode_seed_start + args.episodes)
    rows = evaluate_episodes(
        model, args.env_id, seeds, method=method, phase=args.phase, frame_stack=frame_stack,
        record_actions=args.record_actions,
    )
    write_results(rows, args.output)
    success_rate = sum(row["success"] for row in rows) / len(rows)
    elapsed = sum(row["elapsed_seconds"] for row in rows)
    print(f"{args.env_id} ({method}, frame_stack={frame_stack}): {len(rows)} episodes, "
          f"success={success_rate:.1%}, model_step={model.num_timesteps}, "
          f"evaluation_seconds={elapsed:.2f}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
