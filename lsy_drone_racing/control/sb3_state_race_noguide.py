"""Guide-free Stable-Baselines3 PPO controller and training entrypoint.

This is a deliberately simplified sibling of ``sb3_state_race.py``. The policy here has
**no guide path**: it emits absolute pos/vel/acc/yaw setpoints directly, navigating purely
off the relative positions of the upcoming gates. A gate spline still exists, but only to
shape the training reward (dense progress signal) -- the policy never sees it.

Three pieces, each with one job:
    - ``RaceCodec``: framework-free translator. dict obs -> flat vector, 10-D action -> 13-D command.
    - ``DroneRaceSB3Env``: Gymnasium training env. Owns a codec, adds reward + reset/step.
    - ``SB3StateRaceController``: deployable controller. Owns a codec, runs the saved policy.

Example usage:
    python lsy_drone_racing/control/sb3_state_race_noguide.py train --config=level2.toml
    python lsy_drone_racing/control/sb3_state_race_noguide.py evaluate --config=level2.toml --render=True
    python scripts/sim.py --config level2.toml --controller sb3_state_race_noguide.py

Training metrics are logged to Weights & Biases (no TensorBoard).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import os
import sys

os.environ.setdefault("SCIPY_ARRAY_API", "1")

import fire
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.utils import load_config

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
    from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
except ImportError:
    PPO = None
    BaseCallback = None
    CheckpointCallback = None
    EvalCallback = None
    HumanOutputFormat = None
    KVWriter = object  # so the output-format subclass below still imports without SB3
    Logger = None
    Monitor = None
    DummyVecEnv = None
    SubprocVecEnv = None

try:
    import wandb
except ImportError:
    wandb = None

if TYPE_CHECKING:
    from numpy.typing import NDArray


PROJECT_ROOT = Path(__file__).parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
MODEL_PATH = Path(__file__).with_suffix(".zip")
RUNS_DIR = PROJECT_ROOT / "runs" / Path(__file__).stem
CHECKPOINT_DIR = RUNS_DIR / "checkpoints"
BEST_MODEL_DIR = RUNS_DIR / "best_model"
EVAL_LOG_DIR = RUNS_DIR / "eval"
BEST_MODEL_PATH = BEST_MODEL_DIR / "best_model.zip"
OBS_CLIP = 20.0


@dataclass
class EnvSpec:
    """Action scaling, observation layout, and reward parameters."""

    # How many upcoming gates the policy sees (starting from the current target).
    gate_count: int = 3
    # How many of the nearest obstacles the policy sees. Fixed window (zero-padded) so the
    # observation size -- and thus the trained model -- is independent of the track's obstacle count.
    obstacle_count: int = 4

    # Action scaling: the 10-D action in [-1, 1] maps to these physical ranges.
    pos_scale_xy: float = 0.55   # position offset from the drone, metres
    pos_scale_z: float = 0.35
    vel_scale_xy: float = 1.2    # absolute velocity setpoint, m/s
    vel_scale_z: float = 0.8
    acc_scale_xy: float = 2.0    # absolute acceleration setpoint, m/s^2
    acc_scale_z: float = 1.6
    yaw_scale: float = np.pi     # absolute yaw setpoint, radians

    # Commanded height is always clamped into this band.
    min_height: float = 0.15
    max_height: float = 2.00

    # Reward spline (training only): a Hermite path threaded through the true gate centres.
    gate_axis_offset: float = 0.35
    path_samples: int = 600
    path_progress_window: int = 80
    path_progress_bins: int = 100
    path_bin_reward: float = 1.0
    path_reward_radius: float = 0.30

    # Sparse event rewards.
    gate_pass_bonus: float = 8.0
    finish_bonus: float = 40.0
    crash_penalty: float = 20.0
    timeout_penalty: float = 6.0


def _require_sb3() -> None:
    """Raise a clear error when SB3 is not installed in the selected interpreter."""
    if PPO is None or DummyVecEnv is None or Monitor is None:
        raise ImportError(
            "stable-baselines3 is not installed in this interpreter. "
            "Install it into the project venv first."
        )


def _require_wandb() -> None:
    """Raise a clear error when wandb is not installed in the selected interpreter."""
    if wandb is None:
        raise ImportError(
            "wandb is not installed in this interpreter. Install it into the project venv "
            "(`pip install wandb`) or run training with --wandb_mode=disabled."
        )


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _target_gate(obs: dict[str, NDArray[np.floating]]) -> int:
    """Index of the next gate to fly through (-1 once the course is finished)."""
    return int(np.asarray(obs["target_gate"]).item())


def _resolve_model_path(model_path: str | Path | None = None) -> Path:
    """Resolve the newest available model unless an explicit path is provided."""
    if model_path is not None:
        return Path(model_path)

    candidates: list[Path] = []
    if BEST_MODEL_PATH.exists():
        candidates.append(BEST_MODEL_PATH)
    if MODEL_PATH.exists():
        candidates.append(MODEL_PATH)
    if CHECKPOINT_DIR.exists():
        candidates.extend(CHECKPOINT_DIR.glob("*.zip"))
    if not candidates:
        return BEST_MODEL_PATH if BEST_MODEL_DIR.exists() else MODEL_PATH
    return max(candidates, key=lambda p: p.stat().st_mtime)


class RaceCodec:
    """Framework-free translator between the sim's dict world and the policy's vector world.

    This is the *single* place that defines the observation layout and the action decoding,
    so training and deployment are guaranteed to agree. It holds no episode state and never
    touches simulator internals -- it is pure ``dict -> vector`` and ``action -> command``.

    Observation vector (all positions relative to the drone; absolute drone position omitted):
        [ quat(4), vel(3), ang_vel(3),
          nearest ``obstacle_count`` obstacles: [known, rel_x, rel_y, rel_z],
          next ``gate_count`` gates: [exists, rel_x, rel_y, rel_z, quat_xyzw] ]

    Both the obstacle and gate sections are fixed-size, zero-padded windows, so the observation
    size is independent of how many obstacles/gates the track actually has.

    Action (10-D, each in [-1, 1]) decodes to a *direct, guide-free* setpoint:
        [0:3] -> position offset from the drone     [6:9] -> absolute acceleration
        [3:6] -> absolute velocity                  [9]   -> absolute yaw
    """

    def __init__(self, spec: EnvSpec):
        self.spec = spec

    @property
    def obs_dim(self) -> int:
        """Length of the flat observation vector."""
        return 4 + 3 + 3 + 4 * self.spec.obstacle_count + self.spec.gate_count * (1 + 3 + 4)

    @property
    def action_dim(self) -> int:
        """Length of the policy action vector."""
        return 10

    def encode(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.float32]:
        """Assemble the flat observation vector consumed by the policy."""
        pos = np.asarray(obs["pos"], dtype=np.float32)
        pieces: list[NDArray[np.float32]] = [
            np.asarray(obs["quat"], dtype=np.float32),
            np.asarray(obs["vel"], dtype=np.float32),
            np.asarray(obs["ang_vel"], dtype=np.float32),
        ]

        # Nearest `obstacle_count` obstacles, sorted by distance and zero-padded to fixed size.
        obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=np.float32)
        obstacles_known = np.asarray(obs["obstacles_visited"], dtype=np.float32)
        obstacle_feats = np.zeros((self.spec.obstacle_count, 4), dtype=np.float32)
        if obstacles_pos.size:
            rel = obstacles_pos - pos[None, :]
            nearest = np.argsort(np.linalg.norm(rel, axis=1))[: self.spec.obstacle_count]
            for slot, idx in enumerate(nearest):
                obstacle_feats[slot] = (obstacles_known[idx], rel[idx, 0], rel[idx, 1], rel[idx, 2])
        pieces.append(obstacle_feats.reshape(-1))

        gates_pos = np.asarray(obs["gates_pos"], dtype=np.float32)
        gates_quat = np.asarray(obs["gates_quat"], dtype=np.float32)
        target = _target_gate(obs)
        for offset in range(self.spec.gate_count):
            gate_idx = target + offset
            if target >= 0 and gate_idx < gates_pos.shape[0]:
                rel = gates_pos[gate_idx] - pos
                pieces.append(
                    np.concatenate(
                        (np.array([1.0], dtype=np.float32), rel, gates_quat[gate_idx])
                    ).astype(np.float32)
                )
            else:
                pieces.append(np.zeros(8, dtype=np.float32))

        obs_vec = np.concatenate(pieces, dtype=np.float32)
        return np.clip(obs_vec, -OBS_CLIP, OBS_CLIP)

    def decode(
        self, obs: dict[str, NDArray[np.floating]], action: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Turn a bounded policy action into a 13-D absolute state command."""
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        pos = np.asarray(obs["pos"], dtype=np.float32)
        spec = self.spec

        pos_sp = pos + np.array(
            [action[0] * spec.pos_scale_xy, action[1] * spec.pos_scale_xy, action[2] * spec.pos_scale_z],
            dtype=np.float32,
        )
        pos_sp[2] = np.clip(pos_sp[2], spec.min_height, spec.max_height)

        vel_sp = np.array(
            [action[3] * spec.vel_scale_xy, action[4] * spec.vel_scale_xy, action[5] * spec.vel_scale_z],
            dtype=np.float32,
        )
        acc_sp = np.array(
            [action[6] * spec.acc_scale_xy, action[7] * spec.acc_scale_xy, action[8] * spec.acc_scale_z],
            dtype=np.float32,
        )
        yaw = _wrap_angle(float(action[9]) * spec.yaw_scale)

        return np.concatenate(
            (pos_sp, vel_sp, acc_sp, np.array([yaw, 0.0, 0.0, 0.0], dtype=np.float32))
        ).astype(np.float32)


class EpisodeStatsCallback(BaseCallback if BaseCallback is not None else object):
    """Log gates passed and success rate to TensorBoard."""

    def __init__(self):
        super().__init__()
        self._gates_passed: list[float] = []
        self._successes: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones, strict=False):
            if not done:
                continue
            self._gates_passed.append(float(info.get("gates_passed", 0.0)))
            self._successes.append(float(info.get("success", 0.0)))
        if self._gates_passed:
            self.logger.record("rollout/gates_passed_mean", float(np.mean(self._gates_passed[-100:])))
            self.logger.record("rollout/success_rate", float(np.mean(self._successes[-100:])))
        return True


class WandbOutputFormat(KVWriter):
    """SB3 logger sink that forwards scalar metrics straight to wandb.

    This replaces TensorBoard entirely: every value SB3 records (PPO internals plus the custom
    stats above) is pushed to ``wandb.log`` as it is written, so no ``.tfevents`` files are created.
    """

    def write(self, key_values: dict, key_excluded: dict, step: int = 0) -> None:
        scalars = {
            key: value
            for key, value in key_values.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        }
        if scalars:
            wandb.log(scalars, step=step)

    def close(self) -> None:
        pass


class DroneRaceSB3Env(gym.Env):
    """Single-environment training wrapper for the guide-free policy.

    The codec handles obs/action translation. This class adds the parts that only matter
    during training: resetting the wrapped sim, computing a dense progress reward from a
    gate spline (built once per episode from the sim's true gate poses), and the event
    bonuses/penalties for passing gates, finishing, crashing, and timing out.
    """

    metadata = {"render_modes": []}

    def __init__(self, config_name: str = "level2.toml", render: bool = False):
        super().__init__()
        self.config_name = config_name
        self.config = load_config(CONFIG_DIR / config_name)
        self.config.sim.render = render
        self.spec = EnvSpec()
        self._base_env = self._make_base_env(self.config)

        self.codec = RaceCodec(self.spec)
        self.observation_space = spaces.Box(
            low=-OBS_CLIP, high=OBS_CLIP, shape=(self.codec.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.codec.action_dim,), dtype=np.float32
        )

        self._obs: dict[str, NDArray[np.floating]] | None = None
        self._path_points: NDArray[np.floating] | None = None
        self._path_arclength: NDArray[np.floating] | None = None
        self._prev_path_progress = 0.0
        self._best_progress_bin = 0

    @staticmethod
    def _make_base_env(config: Any):
        """Build the original environment and convert JAX arrays to NumPy arrays."""
        env = gym.make(
            config.env.id,
            freq=config.env.freq,
            sim_config=config.sim,
            sensor_range=config.env.sensor_range,
            control_mode="state",
            track=config.env.track,
            disturbances=config.env.get("disturbances"),
            randomizations=config.env.get("randomizations"),
            seed=config.env.seed,
        )
        return JaxToNumpy(env)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset the wrapped environment and rebuild the reward spline."""
        obs, info = self._base_env.reset(seed=seed, options=options)
        self._obs = self._np_obs(obs)
        self._path_points, self._path_arclength = self._build_reward_path(self._obs["pos"])
        self._prev_path_progress, _ = self._path_progress(self._obs["pos"])
        self._best_progress_bin = self._progress_bin(self._prev_path_progress)
        return self.codec.encode(self._obs), info

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Apply one policy action and return the shaped reward."""
        if self._obs is None:
            raise RuntimeError("Environment must be reset before stepping.")
        env_action = self.codec.decode(self._obs, action)
        next_obs, _, terminated, truncated, info = self._base_env.step(env_action)
        next_obs = self._np_obs(next_obs)
        if self.config.sim.render:  # SB3's learn() never calls render(), so do it here
            self._base_env.render()

        next_progress, path_distance = self._path_progress(next_obs["pos"], self._prev_path_progress)
        reward = self._reward(self._obs, next_obs, next_progress, path_distance, terminated, truncated)

        info = dict(info)
        info["gates_passed"] = self._gates_passed(next_obs)
        info["success"] = float(_target_gate(next_obs) == -1)

        self._obs = next_obs
        self._prev_path_progress = next_progress
        if path_distance <= self.spec.path_reward_radius:
            self._best_progress_bin = max(self._best_progress_bin, self._progress_bin(next_progress))
        return self.codec.encode(next_obs), float(reward), terminated, truncated, info

    def render(self):
        """Forward rendering to the wrapped simulation."""
        return self._base_env.render()

    def close(self):
        """Close the wrapped simulation."""
        self._base_env.close()

    @staticmethod
    def _np_obs(obs: dict[str, Any]) -> dict[str, NDArray[np.floating]]:
        """Convert environment observations into NumPy arrays."""
        return {k: np.asarray(v, dtype=np.float32) for k, v in obs.items()}

    def _gates_passed(self, obs: dict[str, NDArray[np.floating]]) -> int:
        """Number of gates already passed in the current episode."""
        target = _target_gate(obs)
        n_gates = len(self.config.env.track.gates)
        if target == -1:
            return n_gates
        return int(np.clip(target, 0, n_gates))

    # --- Reward -----------------------------------------------------------------

    def _reward(
        self,
        prev_obs: dict[str, NDArray[np.floating]],
        next_obs: dict[str, NDArray[np.floating]],
        next_progress: float,
        path_distance: float,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """Dense progress-bin reward plus sparse gate/finish/crash/timeout events."""
        prev_target = _target_gate(prev_obs)
        next_target = _target_gate(next_obs)

        reward_active = path_distance <= self.spec.path_reward_radius
        current_bin = self._progress_bin(next_progress)
        newly_reached = max(0, current_bin - self._best_progress_bin) if reward_active else 0
        reward = self.spec.path_bin_reward * float(newly_reached)

        if next_target != -1 and prev_target != -1 and next_target > prev_target:
            reward += self.spec.gate_pass_bonus * float(next_target - prev_target)
        if prev_target >= 0 and next_target == -1:
            reward += self.spec.finish_bonus
        if terminated and next_target != -1:
            reward -= self.spec.crash_penalty
        if truncated and next_target != -1:
            reward -= self.spec.timeout_penalty

        return reward

    # --- Reward spline (training only; never seen by the policy) -----------------

    def _build_reward_path(
        self, start_pos: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Build a dense Hermite spline through the gate holes using true gate poses."""
        gates_pos, gates_quat = self._true_gates()
        start_pos = np.asarray(start_pos, dtype=np.float64)
        if gates_pos.size == 0:
            path = np.repeat(start_pos[None, :], 2, axis=0).astype(np.float32)
            return path, np.array([0.0, 1e-3], dtype=np.float32)

        offset = float(self.spec.gate_axis_offset)
        points: list[NDArray[np.floating]] = [start_pos]
        preferred_dirs: list[NDArray[np.floating] | None] = [None]
        prev_point = start_pos
        for gate_pos, gate_quat in zip(gates_pos, gates_quat, strict=False):
            gate_pos = np.asarray(gate_pos, dtype=np.float64)
            gate_axis = self._gate_axis(gate_quat, gate_pos - prev_point).astype(np.float64)
            points.extend([gate_pos - offset * gate_axis, gate_pos, gate_pos + offset * gate_axis])
            preferred_dirs.extend([gate_axis, gate_axis, gate_axis])
            prev_point = gate_pos + offset * gate_axis

        knot_points = np.asarray(points, dtype=np.float64)
        tangents = np.zeros_like(knot_points)
        for idx in range(len(knot_points)):
            if idx == 0:
                tangent = knot_points[1] - knot_points[0]
            elif idx == len(knot_points) - 1:
                tangent = knot_points[-1] - knot_points[-2]
            else:
                tangent = 0.5 * (knot_points[idx + 1] - knot_points[idx - 1])
            tangent_scale = max(0.25, float(np.linalg.norm(tangent)))
            preferred = preferred_dirs[idx]
            if preferred is not None:
                preferred = np.asarray(preferred, dtype=np.float64)
                preferred_norm = float(np.linalg.norm(preferred))
                if preferred_norm > 1e-6:
                    tangent = tangent_scale * preferred / preferred_norm
            tangents[idx] = tangent

        knots = np.arange(len(knot_points), dtype=np.float64)
        spline = CubicHermiteSpline(knots, knot_points, tangents, axis=0)
        sample_count = max(int(self.spec.path_samples), 50 * len(knot_points))
        sample_t = np.linspace(knots[0], knots[-1], sample_count)
        path_points = np.asarray(spline(sample_t), dtype=np.float32)
        segment_lengths = np.linalg.norm(np.diff(path_points, axis=0), axis=1)
        path_arclength = np.concatenate(([0.0], np.cumsum(segment_lengths))).astype(np.float32)
        return path_points, path_arclength

    def _path_progress(
        self, pos: NDArray[np.floating], reference_progress: float | None = None
    ) -> tuple[float, float]:
        """Project onto the reward spline; return (arclength progress, distance to path)."""
        if self._path_points is None or self._path_arclength is None:
            return 0.0, np.inf
        pos = np.asarray(pos, dtype=np.float32)
        if reference_progress is None:
            distances = np.linalg.norm(self._path_points - pos[None, :], axis=1)
            best_idx = int(np.argmin(distances))
            return float(self._path_arclength[best_idx]), float(distances[best_idx])

        ref_idx = int(np.searchsorted(self._path_arclength, reference_progress, side="left"))
        start_idx = max(0, ref_idx - 3)
        end_idx = min(len(self._path_points), ref_idx + self.spec.path_progress_window)
        distances = np.linalg.norm(self._path_points[start_idx:end_idx] - pos[None, :], axis=1)
        best_local_idx = int(np.argmin(distances))
        best_idx = start_idx + best_local_idx
        return (
            max(float(reference_progress), float(self._path_arclength[best_idx])),
            float(distances[best_local_idx]),
        )

    def _progress_bin(self, progress: float) -> int:
        """Map spline arclength progress to a monotonic discrete bin index."""
        if self._path_arclength is None or len(self._path_arclength) == 0:
            return 0
        total_length = float(self._path_arclength[-1])
        if total_length <= 1e-6:
            return 0
        normalized = float(np.clip(progress / total_length, 0.0, 1.0))
        return int(np.floor(normalized * self.spec.path_progress_bins + 1e-6))

    def _true_gates(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Read the exact gate poses for the current episode from the wrapped env."""
        env = self._base_env.unwrapped
        gates_pos = np.asarray(env.data.gates_pos, dtype=np.float32)
        gates_quat = np.asarray(env.data.gates_quat, dtype=np.float32)
        if gates_pos.ndim == 3:
            gates_pos = gates_pos[0]
        if gates_quat.ndim == 3:
            gates_quat = gates_quat[0]
        return gates_pos, gates_quat

    @staticmethod
    def _gate_axis(
        gate_quat: NDArray[np.floating], fallback: NDArray[np.floating]
    ) -> NDArray[np.float32]:
        """Gate-forward axis used to thread the spline through the square opening."""
        gate_quat = np.asarray(gate_quat, dtype=np.float32)
        if np.linalg.norm(gate_quat) > 0.5:
            axis = R.from_quat(gate_quat / np.linalg.norm(gate_quat)).apply(
                np.array([1.0, 0.0, 0.0], dtype=np.float32)
            )
            fallback = np.asarray(fallback, dtype=np.float32)
            if np.dot(axis, fallback) < 0.0:
                axis = -axis
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm > 1e-6:
                return (axis / axis_norm).astype(np.float32)
        fallback = np.asarray(fallback, dtype=np.float32)
        fallback_norm = float(np.linalg.norm(fallback))
        if fallback_norm > 1e-6:
            return (fallback / fallback_norm).astype(np.float32)
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def _make_monitored_env(config_name: str, render: bool = False):
    """Factory used by DummyVecEnv."""

    def thunk():
        return Monitor(DroneRaceSB3Env(config_name=config_name, render=render))

    return thunk


class RenderEvalCallback(BaseCallback if BaseCallback is not None else object):
    """Continuously fly the *current* policy in a live viewer window.

    A separate render-enabled env lives in the main process (so the MuJoCo window actually opens).
    It is advanced **one step per training step** -- not in bursts -- and rendered every time, which
    keeps the viewer's event loop pumped so the window stays responsive (you can pan it and close
    it). The episode auto-resets when the drone crashes or finishes, always using the latest
    weights, with stochastic actions (``deterministic=False``) so the window shows the same
    exploration the training rollouts experience.

    ``render_every`` throttles how often the live env is stepped/rendered (in training steps). 1 is
    smoothest but adds the most overhead; a few keeps the window fluid while barely touching
    training throughput.
    """

    def __init__(self, config_name: str, render_every: int = 4):
        super().__init__()
        self.config_name = config_name
        self.render_every = max(1, render_every)
        self._env: DroneRaceSB3Env | None = None
        self._obs: NDArray[np.float32] | None = None

    def _on_training_start(self) -> None:
        self._env = DroneRaceSB3Env(config_name=self.config_name, render=True)
        self._obs, _ = self._env.reset()

    def _on_step(self) -> bool:
        if self._env is None or self.n_calls % self.render_every != 0:
            return True
        action, _ = self.model.predict(self._obs, deterministic=False)  # show real exploration
        self._obs, _, terminated, truncated, _ = self._env.step(action)  # step() renders itself
        if terminated or truncated:
            self._obs, _ = self._env.reset()
        return True

    def _on_training_end(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


def train(
    config: str = "level2.toml",
    total_timesteps: int = 10_000_000,
    n_envs: int = 8,
    seed: int = 7,
    device: str = "auto",
    learning_rate: float = 3e-4,
    batch_size: int = 256,
    n_steps: int = 4096,
    gamma: float = 0.99,
    eval_freq: int = 20_000,
    save_freq: int = 50_000,
    render: bool = False,
    live_view: bool = True,
    wandb_project: str = "lsy-drone-racing",
    wandb_entity: str | None = None,
    wandb_mode: str = "online",
) -> Path:
    """Train a PPO policy and save it next to this file.

    All metrics (PPO's own logs plus the custom gates/success stats) flow to Weights & Biases;
    no TensorBoard files are written. Set ``--wandb_mode=offline`` to log locally without a
    network connection, or ``--wandb_mode=disabled`` to skip wandb entirely.

    Two different render modes:
      * ``--live_view`` (default on): a *showcase* window that continuously flies the latest policy
        in a separate env, one step per training step, with stochastic actions so you see the same
        exploration the training rollouts experience. It auto-resets on crash/finish and stays
        responsive (pannable, closable). The parallel training envs keep running at full speed.
        This is the "watch how training is going" view. Disable with ``--live_view=False``.
      * ``--render=True``: renders every *training* rollout env (noisy, exploratory). Forces
        ``--n_envs=1`` to be useful and is mainly for debugging the raw env, not for watching progress.

    Hardware notes (Apple Silicon): keep ``--device=auto`` (resolves to CPU). The policy is a small
    MLP, so MPS would only add host<->GPU transfer overhead -- it does not speed PPO up here. The
    throughput bottleneck is the CPU-bound JAX sim, so the lever that matters is ``--n_envs``: with
    more than one env (and no render) the envs are stepped in parallel across cores via SubprocVecEnv.
    Set ``--n_envs`` around the core count (8 on this machine).
    """
    _require_sb3()
    use_wandb = wandb_mode != "disabled"
    if use_wandb:
        _require_wandb()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_LOG_DIR.mkdir(parents=True, exist_ok=True)

    n_envs = max(1, int(n_envs))
    env_fns = [_make_monitored_env(config, render=render) for _ in range(n_envs)]
    if render or n_envs == 1:
        # Rendering needs the env in-process; a single env gains nothing from subprocesses.
        train_env = DummyVecEnv(env_fns)
    else:
        # Step the envs in parallel across CPU cores -- the real bottleneck is the CPU-bound
        # JAX sim, not the tiny MLP, so this is where the M1's cores actually help. "spawn"
        # avoids fork-related deadlocks with JAX/MuJoCo.
        train_env = SubprocVecEnv(env_fns, start_method="spawn")
    eval_env = DummyVecEnv([_make_monitored_env(config, render=False)])
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

    run_config = {
        "config": config,
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
        "seed": seed,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "gamma": gamma,
    }
    run = None
    if use_wandb:
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config=run_config,
            save_code=True,
            mode=wandb_mode,
        )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=None,  # logging goes to wandb via the custom output format below
        verbose=1,
        device=device,
        seed=seed,
    )
    if run is not None:
        model.set_logger(
            Logger(folder=None, output_formats=[HumanOutputFormat(sys.stdout), WandbOutputFormat()])
        )

    callbacks: list[BaseCallback] = [
        EpisodeStatsCallback(),
        CheckpointCallback(
            save_freq=max(1, save_freq // max(1, int(n_envs))),
            save_path=str(CHECKPOINT_DIR),
            name_prefix=Path(__file__).stem,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(BEST_MODEL_DIR),
            log_path=str(EVAL_LOG_DIR),
            eval_freq=max(1, eval_freq // max(1, int(n_envs))),
            deterministic=True,
            render=False,
        ),
    ]
    # Live showcase window of the latest policy. Skipped when --render already renders every
    # training env (no point in two windows) or when disabled with --live_view=False.
    if live_view and not render:
        callbacks.append(RenderEvalCallback(config))

    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks, tb_log_name=run_name)
        model.save(MODEL_PATH)
        return MODEL_PATH
    finally:
        train_env.close()
        eval_env.close()
        if run is not None:
            run.finish()


def evaluate(
    config: str = "level2.toml",
    model_path: str | None = None,
    n_episodes: int = 5,
    render: bool = False,
) -> dict[str, float]:
    """Run deterministic evaluation episodes for a trained policy."""
    _require_sb3()
    model = PPO.load(str(_resolve_model_path(model_path)), device="cpu")
    env = DroneRaceSB3Env(config_name=config, render=render)

    rewards: list[float] = []
    gates_passed: list[float] = []
    gate_successes = 0
    try:
        for episode in range(n_episodes):
            obs, _ = env.reset(seed=episode + 1)
            done = False
            truncated = False
            episode_reward = 0.0
            while not done and not truncated:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                episode_reward += float(reward)
                if render:
                    env.render()
            rewards.append(episode_reward)
            final_obs = env._obs
            gates_passed.append(float(env._gates_passed(final_obs)) if final_obs is not None else 0.0)
            gate_successes += int(final_obs is not None and _target_gate(final_obs) == -1)
    finally:
        env.close()

    summary = {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "mean_gates_passed": float(np.mean(gates_passed)) if gates_passed else 0.0,
        "success_rate": gate_successes / max(1, len(rewards)),
    }
    print(summary)
    return summary


class SB3StateRaceController(Controller):
    """Deployable controller: encode obs -> run policy -> decode to a state command.

    No guide, no spline, no simulator internals -- the controller just owns a codec and the
    trained policy, and chains them together every control step.
    """

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        _require_sb3()
        self._policy = PPO.load(str(_resolve_model_path()), device="cpu")
        self._codec = RaceCodec(EnvSpec())

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.float32]:
        """Convert the live race observation into a deterministic policy command."""
        obs_np = DroneRaceSB3Env._np_obs(obs)
        flat_obs = self._codec.encode(obs_np)
        action, _ = self._policy.predict(flat_obs, deterministic=True)
        return self._codec.decode(obs_np, np.asarray(action))

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Keep running until the environment terminates."""
        return False


def main():
    """Expose training and evaluation commands via Fire."""
    fire.Fire({"train": train, "evaluate": evaluate})


if __name__ == "__main__":
    main()
