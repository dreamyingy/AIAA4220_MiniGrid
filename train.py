"""Train the tutorial PPO baseline or controlled modification; final evaluation is separate."""

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from common import (
    get_features_extractor_class,
    get_frame_stack,
    get_representation,
    get_size_schedule,
    get_train_shapes,
    get_train_sizes,
    load_config,
    make_train_env,
)
from evaluate import evaluate_episodes, ppo_class, write_results
from exploration import get_count_exploration


class PeriodicEvaluation(BaseCallback):
    def __init__(self, config, output_dir):
        super().__init__()
        self.config = config
        self.output_dir = output_dir
        self.eval_seconds = 0.0

    def evaluate(self):
        start = time.perf_counter()
        cfg = self.config
        seeds = range(cfg["periodic_seed_start"],
                      cfg["periodic_seed_start"] + cfg["eval_episodes"])
        rows = evaluate_episodes(
            self.model,
            cfg["env_id"],
            seeds,
            method=cfg["method"],
            phase="periodic",
            frame_stack=get_frame_stack(cfg),
            record_actions=cfg.get("record_actions", False),
        )
        write_results(rows, self.output_dir / "periodic_episodes.csv", append=True)
        self.eval_seconds += time.perf_counter() - start
        success = sum(row["success"] for row in rows) / len(rows)
        print(f"Periodic step={self.num_timesteps}: success={success:.3f}", flush=True)

    def _on_training_start(self):
        self.evaluate()

    def _on_step(self):
        if (self.num_timesteps <= self.config["total_steps"]
                and self.num_timesteps % self.config["eval_freq"] == 0):
            self.model.save(self.output_dir / "checkpoints" / f"step_{self.num_timesteps}.zip")
            self.evaluate()
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--algo", choices=["ppo", "recurrent_ppo"], default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/baseline.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--frame-stack", type=int, default=None)
    parser.add_argument(
        "--record-actions", action="store_true",
        help="Record action sequences for periodic evaluations and randomized training episodes.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    for key in ("env_id", "algo", "total_steps", "device", "frame_stack"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    cfg["seed"] = args.seed
    if args.record_actions:
        cfg["record_actions"] = True
    frame_stack = get_frame_stack(cfg)
    representation = get_representation(cfg)
    train_sizes = get_train_sizes(cfg)
    train_shapes = get_train_shapes(cfg)
    size_schedule = get_size_schedule(cfg)
    randomize_goal = bool(cfg.get("randomize_goal", False))
    count_exploration = get_count_exploration(cfg)
    progress_potential = cfg.get("progress_potential")
    features_extractor_class = get_features_extractor_class(cfg)
    if cfg["n_envs"] != 1:
        raise ValueError("This protocol uses PPO and exactly one training environment.")
    model_class = ppo_class(cfg["algo"])
    recurrent = cfg["algo"] == "recurrent_ppo"
    if cfg["total_steps"] <= 0 or cfg["eval_freq"] <= 0 or cfg["eval_episodes"] <= 0:
        raise ValueError("Training and evaluation budgets must be positive.")
    if cfg["total_steps"] % cfg["eval_freq"]:
        raise ValueError("total_steps must be a multiple of eval_freq to include the budget endpoint.")

    output = args.output_dir or Path("results") / cfg["method"] / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    torch.set_num_threads(cfg["torch_threads"])
    env = make_train_env(
        cfg["env_id"],
        monitor_path=output / "monitor.csv",
        frame_stack=frame_stack,
        train_sizes=train_sizes,
        randomize_goal=randomize_goal,
        size_schedule=size_schedule,
        train_shapes=train_shapes,
        record_actions=cfg.get("record_actions", False),
        count_exploration=count_exploration,
        progress_potential=progress_potential,
    )
    model = model_class(
        "CnnLstmPolicy" if recurrent else "CnnPolicy",
        env, seed=args.seed, device=cfg["device"], verbose=1,
        policy_kwargs=dict(
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=dict(features_dim=cfg["features_dim"]),
            normalize_images=False,
            **(cfg.get("recurrent", {}) if recurrent else {}),
        ),
        **cfg["ppo"],
    )
    model.set_logger(configure(str(output), ["stdout", "csv"]))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", ":!results/"],
        capture_output=True,
        text=True,
    )
    packages = ("minigrid", "gymnasium", "stable-baselines3", "torch", "numpy")
    if recurrent:
        packages += ("sb3-contrib",)
    total_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    with (output / "run.log").open("w", encoding="utf-8") as log:
        log.write(f"Command: {shlex.join([sys.executable, *sys.argv])}\n")
        log.write(f"Started: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        log.write(f"Method: {cfg['method']}\nFrame stack: {frame_stack}\n")
        log.write(f"Representation: {representation}\nTrain sizes: {train_sizes}\nSize schedule: {size_schedule}\nRandomize goal: {randomize_goal}\n")
        if train_shapes is not None:
            log.write(f"Train shapes (width, height): {train_shapes}\n")
        if count_exploration is not None:
            log.write(f"Count exploration: {count_exploration}\n")
        if progress_potential is not None:
            log.write(f"Progress potential: {progress_potential}\n")
        log.write(f"Trainable policy parameters: {total_params}\n")
        log.write(f"Python: {platform.python_version()}\nPlatform: {platform.platform()}\n")
        log.write(f"CPU: {platform.processor()}\nTorch threads: {torch.get_num_threads()}\n")
        log.write(f"Device: {model.device}\nCUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', 'unset')}\n")
        if model.device.type == "cuda":
            log.write(f"GPU: {torch.cuda.get_device_name(model.device)}\n")
        log.write(f"Versions: { {name: version(name) for name in packages} }\n")
        log.write(f"Code commit: {commit.stdout.strip() or 'unavailable'}\n")
        log.write(f"Working tree status:\n{dirty.stdout}\nNetwork:\n{model.policy}\n")

    callback = PeriodicEvaluation(cfg, output)
    start = time.perf_counter()
    try:
        model.learn(total_timesteps=cfg["total_steps"], callback=callback)
        learn_seconds = time.perf_counter() - start
        # This is deliberately after learn() and the final PPO parameter update.
        model.save(output / "final.zip")
        cfg["actual_steps"] = model.num_timesteps
        cfg["timing"] = {"learn_seconds": learn_seconds,
                         "periodic_eval_seconds": callback.eval_seconds}
        (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        with (output / "run.log").open("a", encoding="utf-8") as log:
            log.write(f"Nominal steps: {cfg['total_steps']}\nActual steps: {model.num_timesteps}\n")
            log.write(f"Timing (seconds): {cfg['timing']}\nFinal model: {output / 'final.zip'}\n")
        print(f"Saved {output / 'final.zip'} at {model.num_timesteps} steps. "
              "Run evaluate.py separately for final 6x6 and 8x8 tests.")
    finally:
        env.close()
        model.logger.close()


if __name__ == "__main__":
    main()
