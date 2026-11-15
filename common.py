"""Shared environment and the official MiniGrid tutorial's feature extractor."""

import json
from pathlib import Path

import gymnasium as gym
import minigrid  # Registers the MiniGrid environments.
import torch
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


def load_config(path=Path(__file__).parent / "configs" / "baseline.json"):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def make_env(env_id, render_mode=None, monitor_path=None):
    env = ImgObsWrapper(gym.make(env_id, render_mode=render_mode))
    return Monitor(env, filename=str(monitor_path) if monitor_path else None)


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    """From https://minigrid.farama.org/content/training/.

    Keep this class in an importable module so saved PPO models load after cloning
    the repository, rather than depending on a training script's __main__.
    """

    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))
