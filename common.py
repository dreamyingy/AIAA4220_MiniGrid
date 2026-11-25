"""Shared environment builders, frame-stacking wrapper, domain randomization, and feature extractors."""

import json
from pathlib import Path

import gymnasium as gym
import minigrid  # Registers the MiniGrid environments.
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key
from minigrid.envs.doorkey import DoorKeyEnv
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
import torch
from torch import nn
from torch.nn import functional as F

from exploration import with_count_exploration
from progress_potential import DoorKeyProgressPotential


def load_config(path=Path(__file__).parent / "configs" / "baseline.json"):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def get_frame_stack(config):
    frame_stack = int(config.get("frame_stack", 1))
    if frame_stack < 1:
        raise ValueError(f"frame_stack must be >= 1, got {frame_stack}")
    return frame_stack


def get_representation(config):
    rep = str(config.get("representation", "raw"))
    if rep not in ("raw", "one_hot"):
        raise ValueError(f"Unsupported representation: {rep!r}")
    return rep


def get_train_sizes(config):
    sizes = config.get("train_sizes", None)
    if sizes is None:
        return None
    sizes = [int(s) for s in sizes]
    if not sizes or any(s < 5 for s in sizes):
        raise ValueError(f"train_sizes must be a non-empty list of integers >= 5, got {sizes}")
    return sizes


def get_train_shapes(config):
    shapes = config.get("train_shapes")
    if shapes is None:
        return None
    if config.get("train_sizes") is not None or config.get("size_schedule") is not None:
        raise ValueError("Use train_shapes without train_sizes or size_schedule.")
    if not isinstance(shapes, list) or not shapes or any(
        not isinstance(shape, (list, tuple))
        or len(shape) != 2
        or any(type(side) is not int or side < 5 for side in shape)
        for shape in shapes
    ):
        raise ValueError("train_shapes must contain [width, height] pairs of integers >= 5.")
    shapes = [tuple(shape) for shape in shapes]
    if len(set(shapes)) != len(shapes):
        raise ValueError("train_shapes must be unique; each shape is sampled uniformly.")
    return shapes


def get_size_schedule(config):
    schedule = config.get("size_schedule", None)
    if schedule is None:
        return None
    parsed = []
    for step_thresh, sizes in schedule:
        shapes = tuple(_training_shape(size) for size in sizes)
        if not shapes:
            raise ValueError("Each size_schedule stage requires at least one shape.")
        parsed.append((int(step_thresh), shapes))
    parsed.sort(key=lambda item: item[0])
    if not parsed or parsed[0][0] != 0:
        raise ValueError("size_schedule must start at step threshold 0.")
    return parsed


def _training_shape(size):
    shape = tuple(map(int, size)) if isinstance(size, (tuple, list)) else (int(size), int(size))
    if len(shape) != 2 or min(shape) < 5:
        raise ValueError(f"Training dimensions need two sides >= 5, got {shape}")
    return shape


def get_features_extractor_class(config):
    rep = get_representation(config)
    if rep == "one_hot":
        return OneHotMinigridFeaturesExtractor
    return MinigridFeaturesExtractor


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


class CustomGoalDoorKeyEnv(DoorKeyEnv):
    """DoorKey with optional rectangular rooms and randomized right-room goals."""

    def __init__(self, size=6, randomize_goal=False, max_steps=None, *, shape=None, **kwargs):
        self.randomize_goal = bool(randomize_goal)
        if shape is None:
            super().__init__(size=size, max_steps=max_steps, **kwargs)
            return
        width, height = shape
        MiniGridEnv.__init__(
            self,
            mission_space=MissionSpace(mission_func=self._gen_mission),
            width=width,
            height=height,
            max_steps=10 * width * height if max_steps is None else max_steps,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        if not self.randomize_goal:
            super()._gen_grid(width, height)
            return
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        split_idx = self._rand_int(2, width - 2)
        self.grid.vert_wall(split_idx, 0)
        self.place_agent(size=(split_idx, height))
        door_idx = self._rand_int(1, height - 2)
        self.put_obj(Door("yellow", is_locked=True), split_idx, door_idx)
        self.place_obj(obj=Key("yellow"), top=(0, 0), size=(split_idx, height))
        self.place_obj(
            obj=Goal(),
            top=(split_idx + 1, 1),
            size=(width - split_idx - 2, height - 2),
        )
        self.mission = "use the key to open the door and then get to the goal"


class RandomizedDoorKeyEnv(gym.Env):
    """Sample a DoorKey shape on each reset, optionally scheduling square sizes."""

    metadata = {"render_modes": ["rgb_array"]}
    CASE_FIELDS = ("layout_seed", "train_width", "train_height", "success", "actions")

    def __init__(
        self, sizes=(5, 6, 7), randomize_goal=False, size_schedule=None,
        render_mode=None, *, shapes=None, record_actions=False,
    ):
        super().__init__()
        if shapes is not None and size_schedule is not None:
            raise ValueError("Use shapes without a square size schedule.")
        self.size_schedule = (
            [(step, tuple(_training_shape(size) for size in pool))
             for step, pool in size_schedule]
            if size_schedule is not None else None
        )
        if shapes is None:
            if self.size_schedule is not None:
                shapes = sorted({shape for _, group in self.size_schedule for shape in group})
            else:
                shapes = [_training_shape(size) for size in sizes]
        self.shapes = tuple(tuple(shape) for shape in shapes)
        if not self.shapes or any(
            len(shape) != 2 or any(side < 5 for side in shape) for shape in self.shapes
        ):
            raise ValueError(f"All DoorKey shapes need two sides >= 5, got {self.shapes}")
        self.randomize_goal = bool(randomize_goal)
        self.render_mode = render_mode
        self._total_steps = 0
        self._episode_actions = [] if record_actions else None
        self._layout_seed = None
        self._sub_envs = {
            shape: ImgObsWrapper(
                CustomGoalDoorKeyEnv(
                    shape=shape,
                    randomize_goal=self.randomize_goal,
                    render_mode=render_mode,
                )
            )
            for shape in self.shapes
        }
        self._active_shape = self.shapes[0]
        self._active_env = self._sub_envs[self._active_shape]
        self.observation_space = self._active_env.observation_space
        self.action_space = self._active_env.action_space

    def _current_pool(self):
        if self.size_schedule is None:
            return self.shapes
        pool = self.size_schedule[0][1]
        for thresh, sizes in self.size_schedule:
            if self._total_steps >= thresh:
                pool = sizes
            else:
                break
        return pool

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        pool = self._current_pool()
        self._active_shape = pool[int(self.np_random.integers(0, len(pool)))]
        self._active_env = self._sub_envs[self._active_shape]
        self._layout_seed = int(self.np_random.integers(0, 2**31 - 1))
        if self._episode_actions is not None:
            self._episode_actions.clear()
        obs, info = self._active_env.reset(seed=self._layout_seed, options=options)
        width, height = self._active_shape
        info["train_grid_shape"] = self._active_shape
        if width == height:
            info["train_grid_size"] = width
        return obs, info

    def step(self, action):
        self._total_steps += 1
        observation, reward, terminated, truncated, info = self._active_env.step(action)
        if self._episode_actions is not None:
            self._episode_actions.append(str(int(action)))
            if terminated or truncated:
                width, height = self._active_shape
                info.update(
                    layout_seed=self._layout_seed, train_width=width, train_height=height,
                    success=int(reward > 0), actions=" ".join(self._episode_actions),
                )
        return observation, reward, terminated, truncated, info

    def render(self):
        return self._active_env.render()

    def close(self):
        for env in self._sub_envs.values():
            env.close()


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


class FixedDoorKeyActionRecorder(gym.Wrapper):
    """Record the original fixed-distribution episode with replay metadata."""

    CASE_FIELDS = RandomizedDoorKeyEnv.CASE_FIELDS + ("layout_rng_state",)

    def reset(self, *, seed=None, options=None):
        self._layout_seed = seed if seed is not None else ""
        self._layout_rng_state = (
            json.dumps(self.env.unwrapped.np_random.bit_generator.state, separators=(",", ":"))
            if seed is None else ""
        )
        self._actions = []
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._actions.append(str(int(action)))
        if terminated or truncated:
            board = self.env.unwrapped
            info.update(
                layout_seed=self._layout_seed, layout_rng_state=self._layout_rng_state,
                train_width=board.width, train_height=board.height,
                success=int(reward > 0), actions=" ".join(self._actions),
            )
        return observation, reward, terminated, truncated, info


def make_train_env(
    env_id,
    monitor_path=None,
    frame_stack=1,
    train_sizes=None,
    randomize_goal=False,
    size_schedule=None,
    train_shapes=None,
    record_actions=False,
    count_exploration=None,
    progress_potential=None,
):
    """Build training environment; preserves unmodified baseline path when defaults are used."""
    def _build_single():
        if (train_sizes is not None or randomize_goal
                or size_schedule is not None or train_shapes is not None):
            sizes = train_sizes if train_sizes is not None else [6]
            base = RandomizedDoorKeyEnv(
                sizes=sizes,
                randomize_goal=randomize_goal,
                size_schedule=size_schedule,
                shapes=train_shapes,
                record_actions=record_actions,
            )
            monitored = Monitor(
                base, filename=str(monitor_path) if monitor_path else None,
                info_keywords=base.CASE_FIELDS if record_actions else (),
            )
        elif record_actions:
            base = FixedDoorKeyActionRecorder(ImgObsWrapper(gym.make(env_id)))
            monitored = Monitor(
                base, filename=str(monitor_path) if monitor_path else None,
                info_keywords=base.CASE_FIELDS,
            )
        else:
            monitored = make_env(env_id, monitor_path=monitor_path, frame_stack=1)
        if progress_potential is not None:
            return DoorKeyProgressPotential(monitored, **progress_potential)
        return with_count_exploration(monitored, count_exploration)

    if count_exploration is not None and progress_potential is not None:
        raise ValueError("Choose either count exploration or progress potential.")
    if frame_stack == 1:
        return _build_single()
    vec_env = DummyVecEnv([_build_single])
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


class OneHotMinigridFeaturesExtractor(BaseFeaturesExtractor):
    """Tutorial CNN preceded by exact semantic one-hot encoding of (object, color, state).

    Converts each 3-channel MiniGrid symbolic frame into 11 (object) + 6 (color) + 3 (state)
    = 20 binary channels while keeping the identical 3-layer 2x2 CNN + 128-d linear head.
    """

    N_OBJECTS = 11
    N_COLORS = 6
    N_STATES = 3
    CHANNELS_PER_FRAME = N_OBJECTS + N_COLORS + N_STATES  # 20

    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        raw_channels, height, width = observation_space.shape
        if raw_channels % 3 != 0:
            raise ValueError(f"Expected channel count divisible by 3, got {observation_space.shape}")
        self.n_frames = raw_channels // 3
        in_channels = self.n_frames * self.CHANNELS_PER_FRAME
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros((1, in_channels, height, width), dtype=torch.float32)
            n_flatten = self.cnn(dummy).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def _encode(self, observations):
        obs_int = observations.long()
        encoded_frames = []
        for idx in range(self.n_frames):
            base = 3 * idx
            obj = F.one_hot(obs_int[:, base].clamp(0, self.N_OBJECTS - 1), num_classes=self.N_OBJECTS)
            col = F.one_hot(obs_int[:, base + 1].clamp(0, self.N_COLORS - 1), num_classes=self.N_COLORS)
            state = F.one_hot(obs_int[:, base + 2].clamp(0, self.N_STATES - 1), num_classes=self.N_STATES)
            encoded_frames.append(torch.cat([obj, col, state], dim=-1))
        stacked = torch.cat(encoded_frames, dim=-1)  # (B, H, W, 20 * n_frames)
        return stacked.permute(0, 3, 1, 2).float()

    def forward(self, observations):
        return self.linear(self.cnn(self._encode(observations)))
