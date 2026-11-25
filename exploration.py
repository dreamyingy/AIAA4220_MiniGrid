"""Training-only episodic visitation bonuses for the DoorKey PPO experiments."""

from dataclasses import dataclass
import math

import gymnasium as gym


@dataclass(frozen=True)
class CountExplorationConfig:
    episode_budget: float
    anneal_steps: int
    include_direction: bool = False

    def __post_init__(self):
        if (isinstance(self.episode_budget, bool)
                or not isinstance(self.episode_budget, (int, float))
                or not math.isfinite(self.episode_budget) or self.episode_budget <= 0):
            raise ValueError("count_exploration.episode_budget must be positive and finite.")
        if type(self.anneal_steps) is not int or self.anneal_steps < 0:
            raise ValueError("count_exploration.anneal_steps must be a nonnegative integer.")
        if type(self.include_direction) is not bool:
            raise ValueError("count_exploration.include_direction must be a boolean.")


def get_count_exploration(config):
    settings = config.get("count_exploration")
    if settings is None:
        return None
    if not isinstance(settings, dict):
        raise ValueError("count_exploration must be an object.")
    return CountExplorationConfig(**settings)


class EpisodicCountExploration(gym.Wrapper):
    """Add a budgeted novelty bonus outside an original-reward Monitor.

    The bonus uses the first-visit condition from NovelD (Zhang et al., NeurIPS
    2021). Tabular position keys replace learned observation novelty. A
    monotone phase allows a cell to become novel again after first picking up
    the key or opening the door. The wrapper uses these grid-state details
    solely to compute the training bonus.
    """

    def __init__(self, env, settings):
        super().__init__(env)
        self.settings = settings
        self._steps = 0
        self._phase = 0
        self._visited = set()

    def _state_key(self):
        x, y = self._board.agent_pos
        key = (int(x), int(y), self._phase)
        if self.settings.include_direction:
            key += (int(self._board.agent_dir),)
        return key

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        base = self.env.unwrapped
        self._board = getattr(base, "_active_env", base).unwrapped
        tiles = self._board.grid.grid
        self._door = next(tile for tile in tiles if tile is not None and tile.type == "door")
        usable = sum(tile is None or tile.type != "wall" for tile in tiles)
        possible_keys = usable * 3 * (4 if self.settings.include_direction else 1)
        self._visit_bonus = self.settings.episode_budget / possible_keys
        self._phase = 0
        self._visited = {self._state_key()}
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1
        horizon = self.settings.anneal_steps
        if horizon and self._steps >= horizon:
            return observation, reward, terminated, truncated, info

        if self._door.is_open:
            self._phase = 2
        elif self._phase == 0 and getattr(self._board.carrying, "type", None) == "key":
            self._phase = 1
        state = self._state_key()
        if state not in self._visited:
            self._visited.add(state)
            remaining = 1.0 - self._steps / horizon if horizon else 1.0
            reward += self._visit_bonus * remaining
        return observation, reward, terminated, truncated, info


def with_count_exploration(monitored_env, settings):
    """Keep the Monitor inside the reward wrapper so episode logs stay comparable."""
    if settings is None:
        return monitored_env
    return EpisodicCountExploration(monitored_env, settings)
