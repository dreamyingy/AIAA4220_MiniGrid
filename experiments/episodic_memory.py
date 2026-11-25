"""Screen frozen PPO policies with deterministic, observation-only episodic memory."""

import argparse
from collections import Counter, deque
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import torch
from minigrid.core.actions import Actions
from minigrid.core.constants import OBJECT_TO_IDX, STATE_TO_IDX


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import make_env
from evaluate import load_ppo_model, write_results


EMPTY, FLOOR, GOAL, KEY, DOOR = (OBJECT_TO_IDX[name] for name in ("empty", "floor", "goal", "key", "door"))
OPEN, CLOSED = (STATE_TO_IDX[name] for name in ("open", "closed"))
LEFT, RIGHT, FORWARD, PICKUP, TOGGLE = (int(action) for action in (
    Actions.left, Actions.right, Actions.forward, Actions.pickup, Actions.toggle,
))
DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))
SPLITS = {"screening": (45000, 55000), "confirmation": (46000, 56000)}
GRID_IDS = {side: f"MiniGrid-DoorKey-{side}x{side}-v0" for side in (6, 8)}


@dataclass(frozen=True)
class Settings:
    mode: str
    repeat_visits: int = 3
    count_penalty: float = 2.0


SETTINGS = {
    "ppo": Settings("ppo"),
    "count": Settings("count"),
    "frontier": Settings("frontier"),
    "visible_frontier": Settings("visible_frontier", repeat_visits=2),
}


def is_open(tile):
    return tile[0] == DOOR and tile[2] == OPEN


def can_enter(tile):
    return tile[0] in (EMPTY, FLOOR, GOAL) or is_open(tile)


def can_open(tile, carrying):
    return tile[0] == DOOR and (tile[2] == CLOSED or tile[2] != OPEN and carrying)


def visible_actions(observation):
    center = observation.shape[0] // 2
    front = observation[center, -2]
    carrying = observation[center, -1, 0] == KEY
    actions = [LEFT, RIGHT]
    if can_enter(front):
        actions.append(FORWARD)
    if can_open(front, carrying):
        actions.append(TOGGLE)
    if front[0] == KEY and not carrying:
        actions.append(PICKUP)
    return tuple(actions)


def route_to_visible_target(observation):
    """Choose the first move of a shortest route in the current visible 7×7 grid."""
    width, height = observation.shape[:2]
    start = (width // 2, height - 1, 0)
    carrying = observation[start[0], start[1], 0] == KEY
    visible = set(observation[:, :, 0].flat)
    if GOAL not in visible and (DOOR if carrying else KEY) not in visible:
        return None
    queue = deque([(start, None)])
    visited = {start}
    interaction_route = None
    while queue:
        (x, y, direction), first = queue.popleft()
        dx, dy = DIRECTIONS[direction]
        next_x, next_y = x + dx, y + dy
        options = [(LEFT, (x, y, (direction - 1) % 4)), (RIGHT, (x, y, (direction + 1) % 4))]
        if 0 <= next_x < width and 0 <= next_y < height:
            tile = observation[next_x, next_y]
            if tile[0] == GOAL:
                return FORWARD if first is None else first
            if interaction_route is None:
                if carrying and can_open(tile, carrying):
                    interaction_route = TOGGLE if first is None else first
                if not carrying and tile[0] == KEY:
                    interaction_route = PICKUP if first is None else first
            if (next_x, next_y) == start[:2] or can_enter(tile):
                options.append((FORWARD, (next_x, next_y, direction)))
        for action, state in options:
            if state not in visited:
                visited.add(state)
                queue.append((state, action if first is None else first))
    return interaction_route


class EpisodicController:
    """Maintain visits and transitions over observations, with no environment access."""

    def __init__(self, settings):
        self.settings = settings
        self.visits = Counter()
        self.moves = {}
        self.available = {}
        self.counts = Counter()
        self.routes = {}
        self.activated_at = None
        self.overrides = 0

    def frontier_action(self, current, logits):
        queue = deque([(current, None)])
        seen = {current}
        while queue:
            state, first = queue.popleft()
            actions = self.available[state]
            unexplored = [action for action in actions if (state, action) not in self.moves]
            if unexplored:
                return first if first is not None else max(unexplored, key=lambda a: logits[a])
            for action in actions:
                destination = self.moves[state, action]
                if destination not in seen:
                    seen.add(destination)
                    queue.append((destination, action if first is None else first))
        return None

    def choose(self, observation, logits, step):
        baseline = int(np.argmax(logits))
        mode = self.settings.mode
        if mode == "ppo":
            return baseline
        key = observation.tobytes()
        self.visits[key] += 1
        if key not in self.available:
            self.available[key] = visible_actions(observation)
        action = baseline
        if self.activated_at is not None or self.visits[key] >= self.settings.repeat_visits:
            if self.activated_at is None:
                self.activated_at = step
            chosen = None
            if mode == "visible_frontier":
                if key not in self.routes:
                    self.routes[key] = route_to_visible_target(observation)
                target_action = self.routes[key]
                if target_action is not None and self.counts[key, target_action] < 2:
                    chosen = target_action
            if chosen is None and mode in ("frontier", "visible_frontier"):
                chosen = self.frontier_action(key, logits)
            if chosen is None:
                chosen = max(
                    self.available[key],
                    key=lambda a: logits[a] - self.settings.count_penalty * self.counts[key, a],
                )
            action = int(chosen)
            self.overrides += action != baseline
        self.counts[key, action] += 1
        return action

    def observe(self, current, action, following):
        if self.settings.mode in ("frontier", "visible_frontier"):
            destination = following.tobytes()
            if destination not in self.available:
                self.available[destination] = visible_actions(following)
            self.moves[current.tobytes(), action] = destination


@dataclass
class Case:
    episode_seed: int
    env: object
    controller: EpisodicController
    observation: np.ndarray
    actions: list
    return_: float = 0.0
    key_step: int | None = None
    door_step: int | None = None

    def advance(self, action):
        following, reward, terminated, truncated, _ = self.env.step(action)
        self.actions.append(str(action))
        self.return_ += float(reward)
        step = len(self.actions)
        center = following.shape[0] // 2
        if self.key_step is None and following[center, -1, 0] == KEY:
            self.key_step = step
        if self.door_step is None and action == TOGGLE and is_open(following[center, -2]):
            self.door_step = step
        self.controller.observe(self.observation, action, following)
        self.observation = following
        if not (terminated or truncated):
            return None
        success = int(reward > 0)
        if success:
            outcome = "success"
        elif self.key_step is None:
            outcome = "no_key"
        elif self.door_step is None:
            outcome = "key_no_door"
        else:
            outcome = "door_no_goal"
        return {
            "episode_seed": self.episode_seed, "success": success, "return": self.return_,
            "steps": step, "terminated": terminated, "truncated": truncated,
            "key_step": self.key_step, "door_step": self.door_step,
            "goal_step": step if success else None, "outcome": outcome,
            "memory_activated_at": self.controller.activated_at,
            "non_greedy_actions": self.controller.overrides, "actions": " ".join(self.actions),
        }


def evaluate_batch(policy, settings, env_id, first_seed, episodes, cache):
    active, rows, all_envs = [], [], []
    try:
        for seed in range(first_seed, first_seed + episodes):
            env = make_env(env_id)
            all_envs.append(env)
            observation, _ = env.reset(seed=seed)
            active.append(Case(seed, env, EpisodicController(settings), observation, []))
        while active:
            observations = {}
            for case in active:
                key = case.observation.tobytes()
                if key not in cache:
                    observations[key] = case.observation
            if observations:
                tensor, _ = policy.obs_to_tensor(np.stack(list(observations.values())))
                with torch.inference_mode():
                    logits = policy.get_distribution(tensor).distribution.logits.cpu().numpy()
                cache.update(zip(observations, logits))
            continuing = []
            for case in active:
                logits = cache[case.observation.tobytes()]
                action = case.controller.choose(case.observation, logits, len(case.actions) + 1)
                row = case.advance(action)
                if row is None:
                    continuing.append(case)
                else:
                    rows.append(row)
            active = continuing
    finally:
        for env in all_envs:
            env.close()
    return sorted(rows, key=lambda row: row["episode_seed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, default="screening")
    parser.add_argument("--methods", nargs="+", choices=("baseline", "rand_size"), default=["rand_size", "baseline"])
    parser.add_argument("--controllers", nargs="+", choices=SETTINGS, default=list(SETTINGS))
    parser.add_argument("--train-seeds", nargs="+", type=int, choices=(1, 2, 3), default=[1, 2, 3])
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    if not 0 < args.episodes <= 100:
        parser.error("Use between 1 and 100 episodes per grid.")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = args.output / args.split
    out.mkdir(parents=True, exist_ok=False)
    starts = dict(zip(GRID_IDS, SPLITS[args.split]))
    protocol = {
        "controllers": {name: asdict(SETTINGS[name]) for name in args.controllers},
        "split": args.split, "first_episode_seed": starts, "episodes": args.episodes,
        "training_seeds": args.train_seeds, "zero_further_training": True,
        "deterministic_inference": True, "controller_inputs": ["7x7 symbolic observation", "frozen PPO logits", "episodic observation and action history"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "frozen_models": {
            f"{method}/{seed}": hashlib.sha256(
                (ROOT / "results" / method / f"seed_{seed}" / "final.zip").read_bytes()
            ).hexdigest()
            for method in args.methods for seed in args.train_seeds
        },
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    by_seed = []
    for method in args.methods:
        for train_seed in args.train_seeds:
            path = ROOT / "results" / method / f"seed_{train_seed}" / "final.zip"
            model = load_ppo_model(path)
            model.policy.set_training_mode(False)
            cache = {}
            for controller in args.controllers:
                for side, env_id in GRID_IDS.items():
                    prefix = {
                        "method": method, "controller": controller, "train_seed": train_seed,
                        "model_step": model.num_timesteps, "phase": args.split, "env_id": env_id,
                    }
                    rows = evaluate_batch(model.policy, SETTINGS[controller], env_id, starts[side], args.episodes, cache)
                    write_results(
                        [{**prefix, **row} for row in rows],
                        out / method / controller / f"seed_{train_seed}_{side}x{side}.csv",
                    )
                    score = {**prefix, "success_rate": statistics.mean(row["success"] for row in rows)}
                    by_seed.append(score)
                    print(f"{method:9} {controller:17} seed={train_seed} {side}x{side} {score['success_rate']:.1%}", flush=True)
    write_results(by_seed, out / "by_seed.csv")
    summary = []
    for method in args.methods:
        for controller in args.controllers:
            for side, env_id in GRID_IDS.items():
                values = [r["success_rate"] for r in by_seed if
                          (r["method"], r["controller"], r["env_id"]) == (method, controller, env_id)]
                summary.append({
                    "method": method, "controller": controller, "env_id": env_id,
                    "mean": statistics.mean(values), "sample_sd": statistics.stdev(values) if len(values) > 1 else "",
                    "n_train_seeds": len(values), "n_episodes_per_seed": args.episodes,
                })
    write_results(summary, out / "summary.csv")


if __name__ == "__main__":
    main()
