"""A training-only DoorKey potential based on the next unfinished subgoal."""

import math

import gymnasium as gym


class DoorKeyProgressPotential(gym.Wrapper):
    """Redistribute the discounted reward using a privileged training signal."""

    def __init__(self, env, *, scale, gamma):
        super().__init__(env)
        if not math.isfinite(scale) or scale <= 0 or not 0 < gamma <= 1:
            raise ValueError("The shaping scale must be positive and gamma in (0, 1].")
        self.scale = scale
        self.gamma = gamma

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        base = self.env.unwrapped
        self.board = getattr(base, "_active_env", base).unwrapped
        objects = {
            tile.type: tile for tile in self.board.grid.grid
            if tile is not None and tile.type in ("key", "door", "goal")
        }
        self.key, self.door, self.goal = (objects[name] for name in ("key", "door", "goal"))
        self.previous = self.potential()
        return observation, info

    def potential(self):
        if not self.door.is_locked:
            phase, target = 2, self.goal
        elif getattr(self.board.carrying, "type", None) == "key":
            phase, target = 1, self.door
        else:
            phase, target = 0, self.key
        distance = sum(abs(int(a) - int(b)) for a, b in zip(self.board.agent_pos, target.cur_pos))
        board_diameter = self.board.width + self.board.height - 4
        proximity = 1 - distance / board_diameter
        return self.scale * (phase + proximity) / 3

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        potential = 0.0 if terminated or truncated else self.potential()
        shaped_reward = reward + self.gamma * potential - self.previous
        self.previous = potential
        return observation, shaped_reward, terminated, truncated, info
