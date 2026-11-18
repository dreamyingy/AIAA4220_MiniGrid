"""Shared environment builders, frame-stacking wrapper, and feature extractor."""

import json
from pathlib import Path

import gymnasium as gym
import minigrid  # Registers the MiniGrid environments.
import numpy as np
import torch
from gymnasium import spaces
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from torch import nn


def load_config(path=Path(__file__).parent / "configs" / "baseline.json"):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def get_frame_stack(config):
    frame_stack = int(config.get("frame_stack", 1))
    if frame_stack < 1:
        raise ValueError(f"frame_stack must be >= 1, got {frame_stack}")
    return frame_stack


def infer_model_metadata(model_path, model=None):
    """Infer method name and frame_stack from adjacent config.json and/or loaded model."""
    model_path = Path(model_path)
    cfg = None
    for candidate in (model_path.parent / "config.json", model_path.parent.parent / "config.json"):
        if candidate.is_file():
            cfg = load_config(candidate)
            break

    inferred_stack = None
    if model is not None and hasattr(model, "observation_space"):
        obs_shape = model.observation_space.shape
        if obs_shape is not None and len(obs_shape) == 3 and obs_shape[0] % 3 == 0:
            inferred_stack = int(obs_shape[0] // 3)

    if cfg is not None:
        cfg_stack = get_frame_stack(cfg)
        if inferred_stack is not None and cfg_stack != inferred_stack:
            raise ValueError(
                f"Mismatch between config.json frame_stack={cfg_stack} and "
                f"model observation channels={model.observation_space.shape} for {model_path}"
            )
        return {
            "method": cfg["method"],
            "frame_stack": cfg_stack,
            "config": cfg,
        }

    frame_stack = inferred_stack if inferred_stack is not None else 1
    default_method = "baseline" if frame_stack == 1 else f"frame_stack_{frame_stack}"
    return {
        "method": default_method,
        "frame_stack": frame_stack,
        "config": None,
    }


class ChannelFrameStackWrapper(gym.Wrapper):
    """Stack the last `n_stack` (H, W, C) frames into channels-first (C * n_stack, H, W).

    Matches Stable-Baselines3's `VecFrameStack(VecTransposeImage(...), n_stack, "first")`
    bit-for-bit (zero-padded history on reset, oldest-to-newest channel order) while
    preserving single-environment step/reset semantics for deterministic evaluation and
    keyframe/trajectory recording.
    """

    def __init__(self, env, n_stack):
        super().__init__(env)
        if n_stack < 2:
            raise ValueError("ChannelFrameStackWrapper requires n_stack >= 2.")
        obs_space = env.observation_space
        if not isinstance(obs_space, spaces.Box) or len(obs_space.shape) != 3:
            raise TypeError(f"Expected 3D Box observation space, got {obs_space!r}")
        self.n_stack = int(n_stack)
        height, width, channels = obs_space.shape
        self._height = height
        self._width = width
        self._channels = channels
        self._frames = np.zeros((self.n_stack, channels, height, width), dtype=obs_space.dtype)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(channels * self.n_stack, height, width),
            dtype=obs_space.dtype,
        )

    def _stacked_obs(self):
        return self._frames.reshape(self.n_stack * self._channels, self._height, self._width).copy()

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._frames.fill(0)
        self._frames[-1] = np.transpose(obs, (2, 0, 1))
        return self._stacked_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames[:-1] = self._frames[1:]
        self._frames[-1] = np.transpose(obs, (2, 0, 1))
        return self._stacked_obs(), reward, terminated, truncated, info


def make_env(env_id, render_mode=None, monitor_path=None, frame_stack=1):
    env = ImgObsWrapper(gym.make(env_id, render_mode=render_mode))
    if frame_stack > 1:
        env = ChannelFrameStackWrapper(env, n_stack=frame_stack)
    return Monitor(env, filename=str(monitor_path) if monitor_path else None)


def make_train_env(env_id, monitor_path=None, frame_stack=1):
    """Build training environment; preserves unmodified baseline path when frame_stack=1."""
    if frame_stack == 1:
        return make_env(env_id, monitor_path=monitor_path, frame_stack=1)
    vec_env = DummyVecEnv([lambda: make_env(env_id, monitor_path=monitor_path, frame_stack=1)])
    vec_env = VecTransposeImage(vec_env)
    return VecFrameStack(vec_env, n_stack=frame_stack, channels_order="first")


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
