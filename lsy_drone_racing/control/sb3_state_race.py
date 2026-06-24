"""Stable-Baselines3 PPO controller and training entrypoint for state-control racing.

This file intentionally keeps everything in one place:
    - `DroneRaceSB3Env`: a Gymnasium env wrapper with the requested observation layout
    - `train` / `evaluate`: SB3 utilities with TensorBoard logging
    - `SB3StateRaceController`: the deployable controller class used by `scripts/sim.py`

Example usage:
    python lsy_drone_racing/control/sb3_state_race.py train --config=level2.toml
    python lsy_drone_racing/control/sb3_state_race.py evaluate --config=level2.toml --render=True
    python scripts/sim.py --config level2.toml --controller sb3_state_race.py
    tensorboard --logdir runs/sb3_state_race
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import os

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
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    PPO = None
    BaseCallback = None
    CheckpointCallback = None
    EvalCallback = None
    Monitor = None
    DummyVecEnv = None

if TYPE_CHECKING:
    from numpy.typing import NDArray


PROJECT_ROOT = Path(__file__).parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
MODEL_PATH = Path(__file__).with_suffix(".zip")
RUNS_DIR = PROJECT_ROOT / "runs" / Path(__file__).stem
CHECKPOINT_DIR = RUNS_DIR / "checkpoints"
BEST_MODEL_DIR = RUNS_DIR / "best_model"
EVAL_LOG_DIR = RUNS_DIR / "eval"
OBS_CLIP = 20.0
BEST_MODEL_PATH = BEST_MODEL_DIR / "best_model.zip"


@dataclass
class EnvSpec:
    """State-control action scaling and reward parameters."""

    sensor_range: float = 0.7
    gate_count: int = 3
    pos_scale_xy: float = 0.55
    pos_scale_z: float = 0.35
    vel_scale_xy: float = 1.2
    vel_scale_z: float = 0.8
    acc_scale_xy: float = 2.0
    acc_scale_z: float = 1.6
    yaw_offset_scale: float = 0.7
    guide_step: float = 0.35
    guide_speed: float = 0.75
    rod_radius: float = 0.08
    near_obstacle_margin: float = 0.30
    min_height: float = 0.15
    max_height: float = 2.00
    path_progress_bins: int = 100
    path_bin_reward: float = 1.0
    path_reward_radius: float = 0.30
    gate_pass_bonus: float = 8.0
    finish_bonus: float = 40.0
    crash_penalty: float = 20.0
    obstacle_hit_penalty: float = 25.0
    timeout_penalty: float = 6.0
    gate_axis_offset: float = 0.35
    path_samples: int = 600
    path_lookahead: float = 0.55
    path_progress_window: int = 80
    path_waypoint_count: int = 4
    path_waypoint_spacing: float = 0.40


def _require_sb3() -> None:
    """Raise a clear error when SB3 is not installed in the selected interpreter."""
    if PPO is None or DummyVecEnv is None or Monitor is None:
        raise ImportError(
            "stable-baselines3 is not installed in this interpreter. "
            "Install it into the project venv first."
        )


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


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
        return BEST_MODEL_PATH if BEST_MODEL_PATH.parent.exists() else MODEL_PATH
    return max(candidates, key=lambda p: p.stat().st_mtime)

class EpisodeStatsCallback(BaseCallback):
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

class DroneRaceSB3Env(gym.Env):
    """Single-environment wrapper that exposes the requested RL observation layout.

    Observation layout:
        - Drone absolute quaternion, linear velocity, angular velocity
        - All obstacles as `[exact_known, rel_x, rel_y, rel_z]`
        - Next 3 gates as `[exists, rel_x, rel_y, rel_z, abs_quat_xyzw]`

    All positions are translated into the agent-centered frame but remain aligned with world axes.
    That keeps gravity consistent while still making the policy position-relative.
    """

    metadata = {"render_modes": []}

    def __init__(self, config_name: str = "level2.toml", render: bool = False):
        super().__init__()
        self.config_name = config_name
        self.config = load_config(CONFIG_DIR / config_name)
        self.config.sim.render = render
        self.spec_cfg = EnvSpec(sensor_range=float(self.config.env.sensor_range))
        self._base_env = self._make_base_env(self.config)
        self._obs: dict[str, NDArray[np.floating]] | None = None
        self._prev_target_gate = 0
        self._prev_gate_distance = 0.0
        self._path_points: NDArray[np.floating] | None = None
        self._path_arclength: NDArray[np.floating] | None = None
        self._prev_path_progress = 0.0
        self._best_progress_bin = 0

        n_obstacles = len(self.config.env.track.obstacles)
        obs_dim = (
            4
            + 3
            + 3
            + 4 * n_obstacles
            + self.spec_cfg.gate_count * (1 + 3 + 4)
            + 3 * self.spec_cfg.path_waypoint_count
        )
        self.observation_space = spaces.Box(
            low=-OBS_CLIP, high=OBS_CLIP, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)

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
        """Reset the wrapped environment."""
        obs, info = self._base_env.reset(seed=seed, options=options)
        self._obs = self._np_obs(obs)
        self._prev_target_gate = self._target_gate(self._obs)
        self._prev_gate_distance = self._gate_distance(self._obs, self._prev_target_gate)
        self._path_points, self._path_arclength = self._build_reward_path(self._obs["pos"])
        self._prev_path_progress, _ = self._path_progress_and_distance(self._obs["pos"])
        self._best_progress_bin = self._progress_bin(self._prev_path_progress)
        return self._build_observation(self._obs), info

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Apply one normalized policy action."""
        if self._obs is None:
            raise RuntimeError("Environment must be reset before stepping.")
        clipped_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        env_action = self._action_to_state_command(self._obs, clipped_action)
        next_obs, _, terminated, truncated, info = self._base_env.step(env_action)
        next_obs = self._np_obs(next_obs)
        next_path_progress, path_distance = self._path_progress_and_distance(
            next_obs["pos"], self._prev_path_progress
        )
        reward = self._reward(
            self._obs,
            next_obs,
            clipped_action,
            next_path_progress,
            path_distance,
            terminated,
            truncated,
        )
        info = dict(info)
        info["gates_passed"] = self._gates_passed(next_obs)
        info["success"] = float(self._target_gate(next_obs) == -1)
        self._obs = next_obs
        self._prev_path_progress = next_path_progress
        if path_distance <= self.spec_cfg.path_reward_radius:
            self._best_progress_bin = max(self._best_progress_bin, self._progress_bin(next_path_progress))
        self._prev_target_gate = self._target_gate(next_obs)
        self._prev_gate_distance = self._gate_distance(next_obs, self._prev_target_gate)
        return self._build_observation(next_obs), float(reward), terminated, truncated, info

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

    @staticmethod
    def _target_gate(obs: dict[str, NDArray[np.floating]]) -> int:
        """Extract the target gate index as a Python int."""
        return int(np.asarray(obs["target_gate"]).item())

    def _gate_distance(self, obs: dict[str, NDArray[np.floating]], target_gate: int) -> float:
        """Euclidean distance to the current target gate."""
        if target_gate < 0:
            return 0.0
        gates_pos = np.asarray(obs["gates_pos"], dtype=np.float32)
        target_gate = int(np.clip(target_gate, 0, len(gates_pos) - 1))
        return float(np.linalg.norm(gates_pos[target_gate] - obs["pos"]))

    def _build_observation(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.float32]:
        """Assemble the flat observation vector used by SB3."""
        pos = obs["pos"]
        quat = obs["quat"]
        vel = obs["vel"]
        ang_vel = obs["ang_vel"]

        pieces: list[NDArray[np.float32]] = [
            quat.astype(np.float32),
            vel.astype(np.float32),
            ang_vel.astype(np.float32),
        ]

        obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=np.float32)
        obstacles_known = np.asarray(obs["obstacles_visited"], dtype=np.float32)
        for idx in range(obstacles_pos.shape[0]):
            rel = obstacles_pos[idx] - pos
            pieces.append(
                np.array(
                    [obstacles_known[idx], rel[0], rel[1], rel[2]],
                    dtype=np.float32,
                )
            )

        gates_pos = np.asarray(obs["gates_pos"], dtype=np.float32)
        gates_quat = np.asarray(obs["gates_quat"], dtype=np.float32)
        target_gate = self._target_gate(obs)
        for offset in range(self.spec_cfg.gate_count):
            gate_idx = target_gate + offset
            if target_gate >= 0 and gate_idx < gates_pos.shape[0]:
                rel = gates_pos[gate_idx] - pos
                pieces.append(
                    np.concatenate(
                        (
                            np.array([1.0], dtype=np.float32),
                            rel.astype(np.float32),
                            gates_quat[gate_idx].astype(np.float32),
                        )
                    )
                )
            else:
                pieces.append(np.zeros(8, dtype=np.float32))

        path_progress, _ = self._path_progress_and_distance(pos, self._prev_path_progress)
        for waypoint_idx in range(self.spec_cfg.path_waypoint_count):
            waypoint_progress = path_progress + (waypoint_idx + 1) * self.spec_cfg.path_waypoint_spacing
            waypoint = self._path_lookahead_point(waypoint_progress)
            pieces.append((waypoint - pos).astype(np.float32))

        obs_vec = np.concatenate(pieces, dtype=np.float32)
        return np.clip(obs_vec, -OBS_CLIP, OBS_CLIP)

    def _reward(
        self,
        prev_obs: dict[str, NDArray[np.floating]],
        next_obs: dict[str, NDArray[np.floating]],
        action: NDArray[np.float32],
        next_progress: float,
        path_distance: float,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """Discrete reward for newly reached path bins plus explicit collision penalties."""
        del action
        prev_target = self._target_gate(prev_obs)
        next_target = self._target_gate(next_obs)
        current_bin = self._progress_bin(next_progress)
        reward_active = path_distance <= self.spec_cfg.path_reward_radius
        newly_reached_bins = max(0, current_bin - self._best_progress_bin) if reward_active else 0
        reward = self.spec_cfg.path_bin_reward * float(newly_reached_bins)

        if next_target != -1 and prev_target != -1 and next_target > prev_target:
            reward += self.spec_cfg.gate_pass_bonus * float(next_target - prev_target)
        if prev_target >= 0 and next_target == -1:
            reward += self.spec_cfg.finish_bonus
        if terminated and next_target != -1:
            reward -= self.spec_cfg.crash_penalty
            if self._hit_obstacle(next_obs["pos"]):
                reward -= self.spec_cfg.obstacle_hit_penalty
        if truncated and next_target != -1:
            reward -= self.spec_cfg.timeout_penalty

        return reward

    def _build_reward_path(
        self, start_pos: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Build a dense Hermite spline through the gate holes using true gate poses."""
        gates_pos, gates_quat = self._true_gates()
        start_pos = np.asarray(start_pos, dtype=np.float64)
        if gates_pos.size == 0:
            path = np.repeat(start_pos[None, :], 2, axis=0).astype(np.float32)
            return path, np.array([0.0, 1e-3], dtype=np.float32)

        offset = float(self.spec_cfg.gate_axis_offset)
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
        sample_count = max(int(self.spec_cfg.path_samples), 50 * len(knot_points))
        sample_t = np.linspace(knots[0], knots[-1], sample_count)
        path_points = np.asarray(spline(sample_t), dtype=np.float32)
        segment_lengths = np.linalg.norm(np.diff(path_points, axis=0), axis=1)
        path_arclength = np.concatenate(([0.0], np.cumsum(segment_lengths))).astype(np.float32)
        return path_points, path_arclength

    def _path_progress_and_distance(
        self, pos: NDArray[np.floating], reference_progress: float | None = None
    ) -> tuple[float, float]:
        """Project onto the sampled reward spline and return both progress and distance."""
        if self._path_points is None or self._path_arclength is None:
            return 0.0, np.inf
        pos = np.asarray(pos, dtype=np.float32)
        if reference_progress is None:
            distances = np.linalg.norm(self._path_points - pos[None, :], axis=1)
            best_idx = int(np.argmin(distances))
            return float(self._path_arclength[best_idx]), float(distances[best_idx])

        ref_idx = int(np.searchsorted(self._path_arclength, reference_progress, side="left"))
        start_idx = max(0, ref_idx - 3)
        end_idx = min(len(self._path_points), ref_idx + self.spec_cfg.path_progress_window)
        distances = np.linalg.norm(self._path_points[start_idx:end_idx] - pos[None, :], axis=1)
        best_local_idx = int(np.argmin(distances))
        best_idx = start_idx + best_local_idx
        return (
            max(float(reference_progress), float(self._path_arclength[best_idx])),
            float(distances[best_local_idx]),
        )

    def _path_progress(
        self, pos: NDArray[np.floating], reference_progress: float | None = None
    ) -> float:
        """Project the drone position onto the sampled reward spline."""
        progress, _ = self._path_progress_and_distance(pos, reference_progress)
        return progress

    def _progress_bin(self, progress: float) -> int:
        """Map spline arclength progress to a monotonic discrete bin index."""
        if self._path_arclength is None or len(self._path_arclength) == 0:
            return 0
        total_length = float(self._path_arclength[-1])
        if total_length <= 1e-6:
            return 0
        normalized = float(np.clip(progress / total_length, 0.0, 1.0))
        return int(np.floor(normalized * self.spec_cfg.path_progress_bins + 1e-6))

    def _path_lookahead_point(self, progress: float) -> NDArray[np.float32]:
        """Return a point slightly ahead on the sampled spline."""
        assert self._path_points is not None and self._path_arclength is not None
        lookahead_progress = progress + self.spec_cfg.path_lookahead
        lookahead_idx = int(np.searchsorted(self._path_arclength, lookahead_progress, side="left"))
        lookahead_idx = int(np.clip(lookahead_idx, 0, len(self._path_points) - 1))
        return self._path_points[lookahead_idx]

    def _path_direction(self, progress: float) -> NDArray[np.float32]:
        """Return the local forward tangent direction of the sampled spline."""
        assert self._path_points is not None and self._path_arclength is not None
        idx = int(np.searchsorted(self._path_arclength, progress, side="left"))
        idx = int(np.clip(idx, 0, len(self._path_points) - 2))
        direction = self._path_points[idx + 1] - self._path_points[idx]
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm > 1e-6:
            return (direction / direction_norm).astype(np.float32)
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

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

    def _true_obstacles(self) -> NDArray[np.floating]:
        """Read the exact obstacle positions for the current episode from the wrapped env."""
        env = self._base_env.unwrapped
        obstacles_pos = np.asarray(env.data.obstacles_pos, dtype=np.float32)
        if obstacles_pos.ndim == 3:
            obstacles_pos = obstacles_pos[0]
        return obstacles_pos

    @staticmethod
    def _gate_axis(gate_quat: NDArray[np.floating], fallback: NDArray[np.floating]) -> NDArray[np.float32]:
        """Gate-forward axis used to thread the spline through the square opening."""
        gate_quat = np.asarray(gate_quat, dtype=np.float32)
        if np.linalg.norm(gate_quat) > 0.5:
            axis = R.from_quat(gate_quat / np.linalg.norm(gate_quat)).apply(np.array([1.0, 0.0, 0.0], dtype=np.float32))
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

    def _hit_obstacle(self, pos: NDArray[np.floating]) -> bool:
        """Check whether the drone is effectively colliding with an obstacle rod."""
        obstacles_pos = self._true_obstacles()
        if obstacles_pos.size == 0:
            return False
        pos = np.asarray(pos, dtype=np.float32)
        dist_xy = np.linalg.norm(obstacles_pos[:, :2] - pos[None, :2], axis=1)
        return bool(np.any(dist_xy < 0.12))

    def _gates_passed(self, obs: dict[str, NDArray[np.floating]]) -> int:
        """Number of gates already passed in the current episode."""
        target_gate = self._target_gate(obs)
        n_gates = len(self.config.env.track.gates)
        if target_gate == -1:
            return n_gates
        return int(np.clip(target_gate, 0, n_gates))

    def _action_to_state_command(
        self, obs: dict[str, NDArray[np.floating]], action: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Turn a bounded PPO action into a 13-D absolute state setpoint."""
        pos = np.asarray(obs["pos"], dtype=np.float32)
        guide_progress = self._path_progress(pos, self._prev_path_progress)
        lookahead_point = self._path_lookahead_point(guide_progress)
        guide_dir = self._guide_direction(pos, guide_progress, lookahead_point)
        yaw = self._guide_yaw(obs, guide_dir) + float(action[9]) * self.spec_cfg.yaw_offset_scale

        pos_sp = lookahead_point + np.array(
            [
                action[0] * self.spec_cfg.pos_scale_xy,
                action[1] * self.spec_cfg.pos_scale_xy,
                action[2] * self.spec_cfg.pos_scale_z,
            ],
            dtype=np.float32,
        )
        pos_sp[2] = np.clip(pos_sp[2], self.spec_cfg.min_height, self.spec_cfg.max_height)

        vel_sp = self.spec_cfg.guide_speed * self._path_direction(guide_progress) + np.array(
            [
                action[3] * self.spec_cfg.vel_scale_xy,
                action[4] * self.spec_cfg.vel_scale_xy,
                action[5] * self.spec_cfg.vel_scale_z,
            ],
            dtype=np.float32,
        )
        acc_sp = np.array(
            [
                action[6] * self.spec_cfg.acc_scale_xy,
                action[7] * self.spec_cfg.acc_scale_xy,
                action[8] * self.spec_cfg.acc_scale_z,
            ],
            dtype=np.float32,
        )

        return np.concatenate(
            (
                pos_sp,
                vel_sp.astype(np.float32),
                acc_sp.astype(np.float32),
                np.array([_wrap_angle(yaw), 0.0, 0.0, 0.0], dtype=np.float32),
            )
        ).astype(np.float32)

    def _guide_direction(
        self,
        pos: NDArray[np.floating],
        guide_progress: float,
        lookahead_point: NDArray[np.floating],
    ) -> NDArray[np.float32]:
        """Guide direction from the current position toward a spline lookahead point."""
        pos = np.asarray(pos, dtype=np.float32)
        lookahead_point = np.asarray(lookahead_point, dtype=np.float32)
        direction = lookahead_point - pos
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm > 1e-6:
            return (direction / direction_norm).astype(np.float32)
        return self._path_direction(guide_progress)

    @staticmethod
    def _guide_yaw(
        obs: dict[str, NDArray[np.floating]], guide_dir: NDArray[np.float32]
    ) -> float:
        """Default yaw aligned with motion toward the target gate."""
        vxy = np.asarray(guide_dir[:2], dtype=np.float32)
        if np.linalg.norm(vxy) > 1e-6:
            return float(np.arctan2(vxy[1], vxy[0]))
        quat = np.asarray(obs["quat"], dtype=np.float32)
        return float(R.from_quat(quat).as_euler("xyz")[2])


def _make_monitored_env(config_name: str, render: bool = False):
    """Factory used by DummyVecEnv."""

    def thunk():
        return Monitor(DroneRaceSB3Env(config_name=config_name, render=render))

    return thunk


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
) -> Path:
    """Train a PPO policy and save it next to this file."""
    _require_sb3()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_LOG_DIR.mkdir(parents=True, exist_ok=True)

    train_env = DummyVecEnv([_make_monitored_env(config) for _ in range(max(1, int(n_envs)))])
    eval_env = DummyVecEnv([_make_monitored_env(config, render=False)])
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

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
        tensorboard_log=str(RUNS_DIR),
        verbose=1,
        device=device,
        seed=seed,
    )

    callbacks = [
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

    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks, tb_log_name=run_name)
        model.save(MODEL_PATH)
        return MODEL_PATH
    finally:
        train_env.close()
        eval_env.close()


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
            final_target = 0
            while not done and not truncated:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                episode_reward += float(reward)
                final_target = env._target_gate(env._obs) if env._obs is not None else 0
                if render:
                    env.render()
            rewards.append(episode_reward)
            gates_passed.append(float(env._gates_passed(env._obs)) if env._obs is not None else 0.0)
            gate_successes += int(final_target == -1)
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
    """Deployable controller that runs the saved SB3 model and emits state setpoints."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        _require_sb3()
        self._policy = PPO.load(str(_resolve_model_path()), device="cpu")
        self._env_adapter = DroneRaceSB3Env.__new__(DroneRaceSB3Env)
        self._env_adapter.config = config
        self._env_adapter.spec_cfg = EnvSpec(sensor_range=float(config.env.sensor_range))
        self._env_adapter._path_points = None
        self._env_adapter._path_arclength = None
        self._env_adapter._prev_path_progress = 0.0
        self._env_adapter._best_progress_bin = 0
        self._env_adapter._current_gates_pos = np.asarray(obs["gates_pos"], dtype=np.float32)
        self._env_adapter._current_gates_quat = np.asarray(obs["gates_quat"], dtype=np.float32)
        self._env_adapter._true_gates = lambda: (
            self._env_adapter._current_gates_pos, self._env_adapter._current_gates_quat
        )
        self._env_adapter._path_points, self._env_adapter._path_arclength = (
            self._env_adapter._build_reward_path(np.asarray(obs["pos"], dtype=np.float32))
        )
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.float32]:
        """Convert the live race observation into a deterministic PPO action."""
        obs_np = DroneRaceSB3Env._np_obs(obs)
        self._env_adapter._current_gates_pos = np.asarray(obs_np["gates_pos"], dtype=np.float32)
        self._env_adapter._current_gates_quat = np.asarray(obs_np["gates_quat"], dtype=np.float32)
        self._env_adapter._path_points, self._env_adapter._path_arclength = (
            self._env_adapter._build_reward_path(obs_np["pos"])
        )
        self._env_adapter._prev_path_progress = self._env_adapter._path_progress(
            obs_np["pos"], self._env_adapter._prev_path_progress
        )
        flat_obs = self._env_adapter._build_observation(obs_np)
        model_action, _ = self._policy.predict(flat_obs, deterministic=True)
        return self._env_adapter._action_to_state_command(obs_np, np.asarray(model_action))

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








