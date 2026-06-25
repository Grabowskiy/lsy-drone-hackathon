"""Guide-free JAX PPO training on GPU (DGX Spark port of sb3_state_race_noguide.py).

Runs thousands of drone-racing worlds in parallel on a single Blackwell GPU using
crazyflow's batched JAX sim (VecDroneRaceEnv, device="gpu"), driven by a pure
Flax/optax PPO — no SB3, no SubprocVecEnv, no per-step host sync of the *policy*.

Architecture:
  - VecDroneRaceEnv with device="gpu" and n_envs=2048 (tweak with --n_envs).
  - Rollout: Python loop stepping the batched env (the GPU sim is inside a JAX jit;
    the Python overhead is O(1) per step regardless of n_envs).
  - Policy: Flax NNX actor-critic [256, 256] on GPU.
  - Reward: faithful port of the CPU spline reward from sb3_state_race_noguide.py —
    Cubic-Hermite spline per world, reset per world on episode end.
  - Update: vectorised GAE + PPO clip loss, entirely in JAX, no sync until wandb log.

Semantics preserved from the CPU baseline (sb3_state_race_noguide.py):
  - RaceCodec.encode / decode: identical maths, vmapped in JAX.
  - EnvSpec constants: identical.
  - Reward: identical Hermite-spline + gate-bin + sparse bonuses/penalties.

Usage:
    # from repo root in the gpu pixi env:
    python lsy_drone_racing/control/train_noguide_jax.py train --config=level2.toml
    python lsy_drone_racing/control/train_noguide_jax.py evaluate --config=level2.toml
    python scripts/sim.py --config level2.toml --controller train_noguide_jax.py
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

os.environ.setdefault("SCIPY_ARRAY_API", "1")

# Python 3.13 tightened __class_getitem__ for Generic subclasses, breaking
# warp 1.6.2's `wp.array[int]` annotation syntax inside mujoco_warp @wp.struct
# decorators. We patch the mujoco mjx warp __init__.py except clause at import
# time so TypeError is treated as a graceful "warp backend unavailable" rather
# than a fatal crash. This lets the JAX backend (which we use) load normally.
def _patch_mujoco_warp_compat() -> None:
    """Broaden the mujoco.mjx.warp exception handler to swallow Python 3.13 TypeErrors."""
    import importlib.util
    import importlib.abc
    import sys

    _WARP_INIT = "mujoco.mjx.warp"

    class _MujocoWarpFinder(importlib.abc.MetaPathFinder):
        """Intercept mujoco.mjx.warp import and patch its source on the fly."""

        def find_spec(self, fullname, path, target=None):
            if fullname != _WARP_INIT:
                return None
            # Let the real finder locate the file first.
            for finder in sys.meta_path:
                if finder is self:
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None and spec.origin:
                    return _PatchedLoader.wrap(spec)
            return None

    class _PatchedLoader(importlib.abc.Loader):
        def __init__(self, original_loader, origin):
            self._loader = original_loader
            self._origin = origin

        @staticmethod
        def wrap(spec):
            patched = _PatchedLoader(spec.loader, spec.origin)
            spec.loader = patched
            return spec

        def create_module(self, spec):
            return self._loader.create_module(spec) if hasattr(self._loader, "create_module") else None

        def exec_module(self, module):
            with open(self._origin, "r") as f:
                src = f.read()
            src = src.replace(
                "except (ImportError, RuntimeError) as e:",
                "except (ImportError, RuntimeError, TypeError, AttributeError) as e:",
            )
            exec(compile(src, self._origin, "exec"), module.__dict__)

    sys.meta_path.insert(0, _MujocoWarpFinder())

_patch_mujoco_warp_compat()

import fire
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.utils import load_config

try:
    import jax
    import jax.numpy as jnp
    from flax import nnx
    import optax
except ImportError as e:
    raise ImportError(
        "JAX, Flax, and optax are required. Install with: pip install jax[cuda12] flax optax"
    ) from e

try:
    import wandb
except ImportError:
    wandb = None

if TYPE_CHECKING:
    from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
RUNS_DIR = PROJECT_ROOT / "runs" / Path(__file__).stem
CHECKPOINT_DIR = RUNS_DIR / "checkpoints"
MODEL_PATH = Path(__file__).with_suffix(".pkl")
SPEC_PATH = Path(__file__).with_suffix(".json")
OBS_CLIP = 20.0


# ---------------------------------------------------------------------------
# EnvSpec — identical to sb3_state_race_noguide.py; copy values exactly.
# ---------------------------------------------------------------------------


@dataclass
class EnvSpec:
    """Action scaling, observation layout, and reward parameters."""

    gate_count: int = 3
    obstacle_count: int = 4

    pos_scale_xy: float = 0.55
    pos_scale_z: float = 0.35
    vel_scale_xy: float = 1.2
    vel_scale_z: float = 0.8
    acc_scale_xy: float = 2.0
    acc_scale_z: float = 1.6
    yaw_scale: float = np.pi

    min_height: float = 0.15
    max_height: float = 2.00

    gate_axis_offset: float = 0.35
    path_samples: int = 600
    path_progress_window: int = 80
    path_progress_bins: int = 100
    path_bin_reward: float = 1.0
    path_reward_radius: float = 0.30

    gate_pass_bonus: float = 8.0
    finish_bonus: float = 40.0
    crash_penalty: float = 20.0
    timeout_penalty: float = 6.0

    @property
    def obs_dim(self) -> int:
        return 4 + 3 + 3 + 4 * self.obstacle_count + self.gate_count * (1 + 3 + 4)

    @property
    def action_dim(self) -> int:
        return 10


# ---------------------------------------------------------------------------
# JAX codec — faithful port of RaceCodec, vmapped over n_worlds.
# ---------------------------------------------------------------------------


def _jax_encode_single(
    pos: jnp.ndarray,       # (3,)
    quat: jnp.ndarray,      # (4,)
    vel: jnp.ndarray,       # (3,)
    ang_vel: jnp.ndarray,   # (3,)
    target_gate: jnp.ndarray,   # () int
    gates_pos: jnp.ndarray,     # (n_gates, 3)
    gates_quat: jnp.ndarray,    # (n_gates, 4)
    obstacles_pos: jnp.ndarray,     # (n_obs, 3)
    obstacles_visited: jnp.ndarray, # (n_obs,) bool
    obstacle_count: int,
    gate_count: int,
) -> jnp.ndarray:
    """Encode a single-world obs dict into a flat float32 vector."""
    n_gates = gates_pos.shape[0]
    n_obs = obstacles_pos.shape[0]

    # Nearest K obstacles sorted by distance, zero-padded.
    rel_obs = obstacles_pos - pos[None, :]                   # (n_obs, 3)
    dists = jnp.linalg.norm(rel_obs, axis=1)                # (n_obs,)
    sort_idx = jnp.argsort(dists)                            # (n_obs,)
    top_k = sort_idx[:obstacle_count]                        # (K,)
    known = obstacles_visited[top_k].astype(jnp.float32)    # (K,)
    rel_top = rel_obs[top_k]                                 # (K, 3)
    # Pad to obstacle_count if n_obs < obstacle_count (shouldn't happen in practice).
    obs_feats = jnp.concatenate([known[:, None], rel_top], axis=1).reshape(-1)  # (K*4,)

    # Next N gates (zero-padded when gate doesn't exist).
    gate_offsets = jnp.arange(gate_count, dtype=jnp.int32)  # (N,)
    gate_indices = target_gate + gate_offsets                 # (N,)
    valid = (target_gate >= 0) & (gate_indices < n_gates)    # (N,) bool
    clamped = jnp.clip(gate_indices, 0, n_gates - 1)         # (N,)
    rel_gates = gates_pos[clamped] - pos[None, :]            # (N, 3)
    gate_quats = gates_quat[clamped]                         # (N, 4)
    exists = valid.astype(jnp.float32)[:, None]              # (N, 1)
    gate_feats = jnp.concatenate(
        [exists,
         jnp.where(valid[:, None], rel_gates, jnp.zeros_like(rel_gates)),
         jnp.where(valid[:, None], gate_quats, jnp.zeros_like(gate_quats))],
        axis=1,
    ).reshape(-1)  # (N*8,)

    obs_vec = jnp.concatenate([quat, vel, ang_vel, obs_feats, gate_feats])
    return jnp.clip(obs_vec, -OBS_CLIP, OBS_CLIP).astype(jnp.float32)


def make_jax_encode(spec: EnvSpec):
    """Return a vmapped encode function for batched obs dicts (n_worlds, ...)."""
    obstacle_count = spec.obstacle_count
    gate_count = spec.gate_count

    def encode_single(pos, quat, vel, ang_vel, target_gate, gates_pos, gates_quat, obstacles_pos, obstacles_visited):
        return _jax_encode_single(
            pos, quat, vel, ang_vel, target_gate, gates_pos, gates_quat,
            obstacles_pos, obstacles_visited, obstacle_count, gate_count,
        )

    encode_batch = jax.vmap(encode_single)

    @jax.jit
    def encode(obs: dict) -> jnp.ndarray:
        return encode_batch(
            obs["pos"].astype(jnp.float32),
            obs["quat"].astype(jnp.float32),
            obs["vel"].astype(jnp.float32),
            obs["ang_vel"].astype(jnp.float32),
            obs["target_gate"].astype(jnp.int32),
            obs["gates_pos"].astype(jnp.float32),
            obs["gates_quat"].astype(jnp.float32),
            obs["obstacles_pos"].astype(jnp.float32),
            obs["obstacles_visited"].astype(jnp.float32),
        )

    return encode


def make_jax_decode(spec: EnvSpec):
    """Return a vmapped decode function: (pos_batch, action_batch) -> command_batch."""
    pos_scale = np.array([spec.pos_scale_xy, spec.pos_scale_xy, spec.pos_scale_z], dtype=np.float32)
    vel_scale = np.array([spec.vel_scale_xy, spec.vel_scale_xy, spec.vel_scale_z], dtype=np.float32)
    acc_scale = np.array([spec.acc_scale_xy, spec.acc_scale_xy, spec.acc_scale_z], dtype=np.float32)
    min_h, max_h = spec.min_height, spec.max_height
    yaw_scale = spec.yaw_scale

    def decode_single(pos: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        action = jnp.clip(action, -1.0, 1.0)
        pos_sp = pos + action[:3] * jnp.array(pos_scale)
        pos_sp = pos_sp.at[2].set(jnp.clip(pos_sp[2], min_h, max_h))
        vel_sp = action[3:6] * jnp.array(vel_scale)
        acc_sp = action[6:9] * jnp.array(acc_scale)
        raw_yaw = action[9] * yaw_scale
        yaw = jnp.arctan2(jnp.sin(raw_yaw), jnp.cos(raw_yaw))
        return jnp.concatenate([pos_sp, vel_sp, acc_sp, jnp.array([yaw, 0.0, 0.0, 0.0])]).astype(jnp.float32)

    decode_batch = jax.vmap(decode_single)

    @jax.jit
    def decode(pos_batch: jnp.ndarray, action_batch: jnp.ndarray) -> jnp.ndarray:
        return decode_batch(pos_batch.astype(jnp.float32), action_batch.astype(jnp.float32))

    return decode


# ---------------------------------------------------------------------------
# Reward tracker — faithful port of the CPU Hermite-spline reward, but
# vectorised over n_worlds in numpy (runs on host, amortised across batch).
# ---------------------------------------------------------------------------

# Numpy-only helpers (identical maths to sb3_state_race_noguide.py).

def _gate_axis_np(gate_quat: NDArray, fallback: NDArray) -> NDArray:
    gate_quat = np.asarray(gate_quat, dtype=np.float64)
    if np.linalg.norm(gate_quat) > 0.5:
        axis = R.from_quat(gate_quat / np.linalg.norm(gate_quat)).apply([1.0, 0.0, 0.0])
        fallback = np.asarray(fallback, dtype=np.float64)
        if np.dot(axis, fallback) < 0.0:
            axis = -axis
        n = float(np.linalg.norm(axis))
        if n > 1e-6:
            return (axis / n).astype(np.float32)
    fallback = np.asarray(fallback, dtype=np.float64)
    n = float(np.linalg.norm(fallback))
    if n > 1e-6:
        return (fallback / n).astype(np.float32)
    return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def _build_reward_path(
    start_pos: NDArray,
    gates_pos: NDArray,   # (n_gates, 3)
    gates_quat: NDArray,  # (n_gates, 4)
    spec: EnvSpec,
) -> tuple[NDArray, NDArray]:
    """Identical to DroneRaceSB3Env._build_reward_path; returns (path_points, path_arclength)."""
    start_pos = np.asarray(start_pos, dtype=np.float64)
    if gates_pos.size == 0:
        path = np.repeat(start_pos[None, :], 2, axis=0).astype(np.float32)
        return path, np.array([0.0, 1e-3], dtype=np.float32)

    offset = float(spec.gate_axis_offset)
    points: list[NDArray] = [start_pos]
    preferred_dirs: list[NDArray | None] = [None]
    prev_point = start_pos
    for gp, gq in zip(gates_pos, gates_quat, strict=False):
        gp = np.asarray(gp, dtype=np.float64)
        axis = _gate_axis_np(gq, gp - prev_point).astype(np.float64)
        points.extend([gp - offset * axis, gp, gp + offset * axis])
        preferred_dirs.extend([axis, axis, axis])
        prev_point = gp + offset * axis

    knot_pts = np.asarray(points, dtype=np.float64)
    tangents = np.zeros_like(knot_pts)
    for i in range(len(knot_pts)):
        if i == 0:
            t = knot_pts[1] - knot_pts[0]
        elif i == len(knot_pts) - 1:
            t = knot_pts[-1] - knot_pts[-2]
        else:
            t = 0.5 * (knot_pts[i + 1] - knot_pts[i - 1])
        t_scale = max(0.25, float(np.linalg.norm(t)))
        pd = preferred_dirs[i]
        if pd is not None:
            pd = np.asarray(pd, dtype=np.float64)
            pn = float(np.linalg.norm(pd))
            if pn > 1e-6:
                t = t_scale * pd / pn
        tangents[i] = t

    knots = np.arange(len(knot_pts), dtype=np.float64)
    spline = CubicHermiteSpline(knots, knot_pts, tangents, axis=0)
    sample_count = max(int(spec.path_samples), 50 * len(knot_pts))
    sample_t = np.linspace(knots[0], knots[-1], sample_count)
    path_pts = np.asarray(spline(sample_t), dtype=np.float32)
    seg = np.linalg.norm(np.diff(path_pts, axis=0), axis=1)
    arclength = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)
    return path_pts, arclength


class RewardTracker:
    """Manages per-world spline state and computes faithful Hermite-spline rewards.

    On each step call, it detects which worlds just reset (via the previous step's
    done flags), rebuilds their splines from the new gate poses, and computes rewards
    for all worlds in one vectorised numpy pass.
    """

    def __init__(self, n_worlds: int, spec: EnvSpec):
        self.n_worlds = n_worlds
        self.spec = spec
        # Per-world path data: list of (path_pts, arclength) tuples.
        self._paths: list[tuple[NDArray, NDArray] | None] = [None] * n_worlds
        # Per-world scalar state.
        self._prev_progress = np.zeros(n_worlds, dtype=np.float64)
        self._best_bin = np.zeros(n_worlds, dtype=np.int32)
        self._prev_target = np.zeros(n_worlds, dtype=np.int32)
        self._initialized = np.zeros(n_worlds, dtype=bool)

    def reset_all(self, start_pos: NDArray, gates_pos: NDArray, gates_quat: NDArray) -> None:
        """Build splines for every world; called once after the very first env.reset().

        Args:
            start_pos:  (n_worlds, 3)
            gates_pos:  (n_worlds, n_gates, 3)
            gates_quat: (n_worlds, n_gates, 4)
        """
        for w in range(self.n_worlds):
            self._build_world(w, start_pos[w], gates_pos[w], gates_quat[w])
        self._prev_progress[:] = 0.0
        self._best_bin[:] = 0
        self._initialized[:] = True

    def reset_worlds(
        self,
        mask: NDArray,          # (n_worlds,) bool — worlds that just reset
        start_pos: NDArray,     # (n_worlds, 3)
        gates_pos: NDArray,     # (n_worlds, n_gates, 3)
        gates_quat: NDArray,    # (n_worlds, n_gates, 4)
    ) -> None:
        """Rebuild splines for worlds indicated by mask (they just auto-reset)."""
        for w in np.where(mask)[0]:
            self._build_world(w, start_pos[w], gates_pos[w], gates_quat[w])
        self._prev_progress[mask] = 0.0
        self._best_bin[mask] = 0

    def _build_world(self, w: int, start_pos: NDArray, gates_pos: NDArray, gates_quat: NDArray):
        pts, arc = _build_reward_path(start_pos, gates_pos, gates_quat, self.spec)
        self._paths[w] = (pts, arc)

    def compute_rewards(
        self,
        prev_target: NDArray,  # (n_worlds,) int  — target gate BEFORE step
        next_target: NDArray,  # (n_worlds,) int  — target gate AFTER step
        pos: NDArray,          # (n_worlds, 3) drone position AFTER step
        terminated: NDArray,   # (n_worlds,) bool
        truncated: NDArray,    # (n_worlds,) bool
    ) -> NDArray:              # (n_worlds,) float32
        """Compute per-world rewards using the same logic as DroneRaceSB3Env._reward."""
        spec = self.spec
        rewards = np.zeros(self.n_worlds, dtype=np.float32)

        # Per-world: find nearest path point and compute progress.
        for w in range(self.n_worlds):
            if self._paths[w] is None:
                continue
            pts, arc = self._paths[w]
            # Full search at episode start (prev_progress ≈ 0); windowed otherwise.
            if self._prev_progress[w] < 1e-6:
                dists = np.linalg.norm(pts - pos[w], axis=1)
                best_i = int(np.argmin(dists))
                path_dist = float(dists[best_i])
            else:
                ref_idx = int(np.searchsorted(arc, self._prev_progress[w], side="left"))
                start_i = max(0, ref_idx - 3)
                end_i = min(len(pts), ref_idx + spec.path_progress_window)
                dists = np.linalg.norm(pts[start_i:end_i] - pos[w], axis=1)
                best_local = int(np.argmin(dists))
                best_i = start_i + best_local
                path_dist = float(dists[best_local])
            next_prog = max(float(self._prev_progress[w]), float(arc[best_i]))

            if path_dist <= spec.path_reward_radius:
                total = float(arc[-1])
                if total > 1e-6:
                    current_bin = int(np.floor(next_prog / total * spec.path_progress_bins + 1e-6))
                    newly = max(0, current_bin - self._best_bin[w])
                    rewards[w] += spec.path_bin_reward * float(newly)
                    self._best_bin[w] = max(self._best_bin[w], current_bin)

            self._prev_progress[w] = next_prog

        # Sparse event rewards (vectorised).
        gate_passed = (next_target != -1) & (prev_target != -1) & (next_target > prev_target)
        rewards += spec.gate_pass_bonus * gate_passed.astype(np.float32) * (next_target - prev_target).clip(min=0)
        finished = (prev_target >= 0) & (next_target == -1)
        rewards += spec.finish_bonus * finished.astype(np.float32)
        crashed = terminated & (next_target != -1)
        rewards -= spec.crash_penalty * crashed.astype(np.float32)
        timed_out = truncated & (next_target != -1)
        rewards -= spec.timeout_penalty * timed_out.astype(np.float32)

        self._prev_target = next_target.copy()
        return rewards


# ---------------------------------------------------------------------------
# Actor-Critic network (Flax NNX, [256, 256] MLP).
# ---------------------------------------------------------------------------


class MLP(nnx.Module):
    """Two-hidden-layer MLP used for both actor and critic."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(in_dim, hidden, rngs=rngs)
        self.l2 = nnx.Linear(hidden, hidden, rngs=rngs)
        self.out = nnx.Linear(hidden, out_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nnx.tanh(self.l1(x))
        x = nnx.tanh(self.l2(x))
        return self.out(x)


class ActorCritic(nnx.Module):
    """Separate actor (policy mean) and critic (value) heads sharing no weights."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256, rngs: nnx.Rngs = None):
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.actor = MLP(obs_dim, hidden, action_dim, rngs=rngs)
        self.critic = MLP(obs_dim, hidden, 1, rngs=rngs)
        # Log std as a learnable parameter (initialised to 0 → std = 1).
        self.log_std = nnx.Param(jnp.zeros(action_dim))

    def policy(self, obs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return (mean, std) for the Gaussian policy."""
        mean = self.actor(obs)
        std = jnp.exp(self.log_std.value)
        return mean, std

    def value(self, obs: jnp.ndarray) -> jnp.ndarray:
        return self.critic(obs).squeeze(-1)

    def act(
        self, obs: jnp.ndarray, rng: jax.random.PRNGKey
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample action, return (action, log_prob, value)."""
        mean, std = self.policy(obs)
        eps = jax.random.normal(rng, mean.shape)
        action = jnp.clip(mean + std * eps, -1.0, 1.0)
        log_prob = -0.5 * jnp.sum(((action - mean) / (std + 1e-8)) ** 2 + jnp.log(2 * jnp.pi * (std + 1e-8) ** 2), axis=-1)
        val = self.value(obs)
        return action, log_prob, val


# ---------------------------------------------------------------------------
# PPO update — JAX, on-device.
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: jnp.ndarray,    # (T, B)
    values: jnp.ndarray,     # (T, B)
    dones: jnp.ndarray,      # (T, B) — episode ends (terminated | truncated)
    last_value: jnp.ndarray, # (B,)
    gamma: float,
    gae_lambda: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns (advantages, returns), both (T, B)."""
    T = rewards.shape[0]
    advantages = jnp.zeros_like(rewards)
    gae = jnp.zeros(rewards.shape[1])

    for t in reversed(range(T)):
        next_val = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages = advantages.at[t].set(gae)

    returns = advantages + values
    return advantages, returns


def _ppo_loss_fn(
    graphdef: nnx.GraphDef,
    params: nnx.State,
    obs_b: jnp.ndarray,
    act_b: jnp.ndarray,
    adv_b: jnp.ndarray,
    ret_b: jnp.ndarray,
    logp_old_b: jnp.ndarray,
    clip_range: float,
    ent_coef: float,
    vf_coef: float,
) -> tuple[jnp.ndarray, dict]:
    """PPO loss; graphdef treated as static by the jit wrapper in ppo_update."""
    model = nnx.merge(graphdef, params)
    mean, std = model.policy(obs_b)
    log_prob = -0.5 * jnp.sum(
        ((act_b - mean) / (std + 1e-8)) ** 2 + jnp.log(2 * jnp.pi * (std + 1e-8) ** 2), axis=-1
    )
    ratio = jnp.exp(log_prob - logp_old_b)
    adv_norm = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
    pg_loss = jnp.mean(jnp.maximum(
        -adv_norm * ratio,
        -adv_norm * jnp.clip(ratio, 1 - clip_range, 1 + clip_range),
    ))

    val = model.value(obs_b)
    vf_loss = 0.5 * jnp.mean((val - ret_b) ** 2)
    entropy = jnp.mean(0.5 * jnp.sum(jnp.log(2 * jnp.pi * jnp.e * (std + 1e-8) ** 2), axis=-1))
    loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy
    return loss, {"loss/policy": pg_loss, "loss/value": vf_loss, "loss/entropy": entropy, "loss/total": loss}


def ppo_update(
    model: ActorCritic,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    obs: jnp.ndarray,            # (T, B, obs_dim)
    actions: jnp.ndarray,        # (T, B, act_dim)
    advantages: jnp.ndarray,     # (T, B)
    returns: jnp.ndarray,        # (T, B)
    log_probs_old: jnp.ndarray,  # (T, B)
    n_epochs: int,
    batch_size: int,
    clip_range: float,
    ent_coef: float,
    vf_coef: float,
    rng: jax.random.PRNGKey,
) -> tuple[ActorCritic, optax.OptState, dict, jax.random.PRNGKey]:
    T, B = obs.shape[:2]
    total = T * B
    obs_flat = obs.reshape(total, -1)
    act_flat = actions.reshape(total, -1)
    adv_flat = advantages.reshape(total)
    ret_flat = returns.reshape(total)
    logp_flat = log_probs_old.reshape(total)

    graphdef, params = nnx.split(model)
    all_metrics: list[dict] = []

    # JIT with graphdef as static (it contains no array leaves, just metadata).
    @functools.partial(jax.jit, static_argnums=(0,))
    def grad_step(graphdef, params, opt_state, obs_b, act_b, adv_b, ret_b, logp_b):
        loss_grad_fn = jax.value_and_grad(_ppo_loss_fn, argnums=1, has_aux=True)
        (_, metrics), grads = loss_grad_fn(
            graphdef, params, obs_b, act_b, adv_b, ret_b, logp_b,
            clip_range, ent_coef, vf_coef,
        )
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, metrics

    for _ in range(n_epochs):
        rng, perm_rng = jax.random.split(rng)
        perm = jax.random.permutation(perm_rng, total)
        for start in range(0, total, batch_size):
            idx = perm[start : start + batch_size]
            params, opt_state, metrics = grad_step(
                graphdef, params, opt_state,
                obs_flat[idx], act_flat[idx], adv_flat[idx], ret_flat[idx], logp_flat[idx],
            )
            all_metrics.append({k: float(v) for k, v in metrics.items()})

    model = nnx.merge(graphdef, params)
    avg_metrics: dict[str, float] = {}
    if all_metrics:
        for key in all_metrics[0]:
            avg_metrics[key] = float(np.mean([m[key] for m in all_metrics]))
    return model, opt_state, avg_metrics, rng


# ---------------------------------------------------------------------------
# Model save / load.
# ---------------------------------------------------------------------------


def save_model(model: ActorCritic, spec: EnvSpec, path: Path) -> None:
    """Save model weights (pickle) and spec (JSON) next to each other."""
    _, state = nnx.split(model)
    # Convert JAX arrays to numpy before pickling (portable across devices).
    np_state = jax.tree_util.tree_map(np.array, state)
    with open(path.with_suffix(".pkl"), "wb") as f:
        pickle.dump(np_state, f)
    spec_path = path.with_suffix(".json")
    spec_path.write_text(json.dumps(dataclasses.asdict(spec)))


def load_model(path: Path, spec: EnvSpec | None = None) -> tuple[ActorCritic, EnvSpec]:
    """Load model weights from a pickle file; path may end in .pkl or .npz."""
    pkl_path = path.with_suffix(".pkl")
    spec_path = path.with_suffix(".json")
    if spec_path.exists() and spec is None:
        spec = EnvSpec(**json.loads(spec_path.read_text()))
    if spec is None:
        spec = EnvSpec()
    # Reconstruct graph structure from a freshly-initialised model.
    model = ActorCritic(spec.obs_dim, spec.action_dim, rngs=nnx.Rngs(0))
    graphdef, _ = nnx.split(model)
    with open(pkl_path, "rb") as f:
        np_state = pickle.load(f)
    state = jax.tree_util.tree_map(jnp.array, np_state)
    model = nnx.merge(graphdef, state)
    return model, spec


# ---------------------------------------------------------------------------
# Train.
# ---------------------------------------------------------------------------


def train(
    config: str = "level2.toml",
    total_timesteps: int = 50_000_000,
    n_envs: int = 2048,
    seed: int = 7,
    learning_rate: float = 3e-4,
    n_steps: int = 64,
    batch_size: int = 8192,
    n_epochs: int = 4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    wandb_project: str = "lsy-drone-racing",
    wandb_entity: str | None = None,
    wandb_mode: str = "online",
    log_freq: int = 10,
    checkpoint_freq: int = 100,
) -> Path:
    """Train a guide-free PPO policy on thousands of parallel GPU worlds.

    Args:
        config: Config file name (relative to config/ dir).
        total_timesteps: Total env steps across all worlds.
        n_envs: Number of parallel worlds (2048 fits comfortably on a DGX Blackwell).
        seed: Random seed.
        learning_rate: Adam learning rate.
        n_steps: Steps per rollout per world (rollout buffer = n_envs * n_steps).
        batch_size: Mini-batch size for PPO update.
        n_epochs: PPO epochs per update.
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        clip_range: PPO clip range.
        ent_coef: Entropy bonus coefficient.
        vf_coef: Value function loss coefficient.
        wandb_project: W&B project name.
        wandb_entity: W&B entity (optional).
        wandb_mode: "online", "offline", or "disabled".
        log_freq: Log to W&B every this many updates.
        checkpoint_freq: Save checkpoint every this many updates.
    """
    from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

    cfg = load_config(CONFIG_DIR / config)
    cfg.sim.render = False  # No GUI during GPU training.
    spec = EnvSpec()

    # Confirm GPU.
    devices = jax.devices()
    print(f"JAX devices: {devices}", flush=True)
    gpu_devices = [d for d in devices if d.device_kind != "cpu"]
    if not gpu_devices:
        print("WARNING: No GPU found, training on CPU. Expect low throughput.", flush=True)

    # Build the vectorised GPU env.
    env = VecDroneRaceEnv(
        num_envs=n_envs,
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        track=cfg.env.track,
        sensor_range=cfg.env.sensor_range,
        control_mode="state",
        disturbances=cfg.env.get("disturbances"),
        randomizations=cfg.env.get("randomizations"),
        seed=seed,
        device="gpu",
    )

    encode = make_jax_encode(spec)
    decode = make_jax_decode(spec)

    # Initialise reward tracker.
    tracker = RewardTracker(n_envs, spec)

    # Model + optimiser.
    rng = jax.random.PRNGKey(seed)
    model = ActorCritic(spec.obs_dim, spec.action_dim, rngs=nnx.Rngs(seed))
    optimizer = optax.adam(learning_rate)
    _, state = nnx.split(model)
    opt_state = optimizer.init(state)

    use_wandb = wandb_mode != "disabled" and wandb is not None
    run = None
    if use_wandb:
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config={
                "config": config, "total_timesteps": total_timesteps, "n_envs": n_envs,
                "n_steps": n_steps, "batch_size": batch_size, "n_epochs": n_epochs,
                "gamma": gamma, "gae_lambda": gae_lambda, "clip_range": clip_range,
                "ent_coef": ent_coef, "vf_coef": vf_coef, "learning_rate": learning_rate,
            },
            mode=wandb_mode,
        )

    # First reset.
    obs_dict, _ = env.reset(seed=seed)
    obs_np = {k: np.asarray(v) for k, v in obs_dict.items()}
    # env.data.gates_pos shape: (n_worlds, n_gates, 3) — no drone dimension here.
    true_gates_pos = np.asarray(env.data.gates_pos)   # (n_worlds, n_gates, 3)
    true_gates_quat = np.asarray(env.data.gates_quat) # (n_worlds, n_gates, 4)
    tracker.reset_all(obs_np["pos"], true_gates_pos, true_gates_quat)
    prev_target = np.asarray(obs_np["target_gate"], dtype=np.int32)
    # pending_rebuild[w]=True means world w reset at end of previous step;
    # rebuild its spline at the start of the current step (env.data now has fresh poses).
    pending_rebuild = np.zeros(n_envs, dtype=bool)

    total_updates = total_timesteps // (n_envs * n_steps)
    steps_done = 0
    t_start = time.time()

    try:
        for update in range(1, total_updates + 1):
            # --- Rollout collection ---
            buf_obs   = np.zeros((n_steps, n_envs, spec.obs_dim), dtype=np.float32)
            buf_act   = np.zeros((n_steps, n_envs, spec.action_dim), dtype=np.float32)
            buf_logp  = np.zeros((n_steps, n_envs), dtype=np.float32)
            buf_val   = np.zeros((n_steps, n_envs), dtype=np.float32)
            buf_rew   = np.zeros((n_steps, n_envs), dtype=np.float32)
            buf_done  = np.zeros((n_steps, n_envs), dtype=np.float32)

            ep_gates: list[float] = []

            for t in range(n_steps):
                # Rebuild splines for worlds that reset at the end of the PREVIOUS step.
                # env.data now contains the fresh gate poses from the auto-reset.
                if pending_rebuild.any():
                    true_gp = np.asarray(env.data.gates_pos)
                    true_gq = np.asarray(env.data.gates_quat)
                    tracker.reset_worlds(pending_rebuild, obs_np["pos"], true_gp, true_gq)
                    pending_rebuild[:] = False

                obs_jax = encode(obs_np)  # (n_worlds, obs_dim) on device
                rng, step_rng = jax.random.split(rng)
                action, log_prob, value = model.act(obs_jax, step_rng)

                # Decode action to 13-D state command and step env.
                pos_jax = jnp.array(obs_np["pos"])
                cmd = decode(pos_jax, action)                # (n_worlds, 13)
                cmd_np = np.asarray(cmd)

                obs_dict, _, terminated_arr, truncated_arr, _ = env.step(cmd_np)
                obs_np = {k: np.asarray(v) for k, v in obs_dict.items()}

                terminated = np.asarray(terminated_arr, dtype=bool)
                truncated = np.asarray(truncated_arr, dtype=bool)
                done = terminated | truncated
                next_target = np.asarray(obs_np["target_gate"], dtype=np.int32)

                rewards = tracker.compute_rewards(
                    prev_target, next_target, obs_np["pos"], terminated, truncated
                )

                buf_obs[t] = np.asarray(obs_jax)
                buf_act[t] = np.asarray(action)
                buf_logp[t] = np.asarray(log_prob)
                buf_val[t] = np.asarray(value)
                buf_rew[t] = rewards
                buf_done[t] = done.astype(np.float32)

                # Collect episode stats from finished worlds.
                for w in np.where(done)[0]:
                    n_gates_total = len(cfg.env.track.gates)
                    tg = next_target[w]
                    gates_p = n_gates_total if tg == -1 else max(0, int(tg))
                    ep_gates.append(float(gates_p))

                # Schedule spline rebuild for done worlds (reset happens at start of next step).
                pending_rebuild |= done

                prev_target = next_target
                steps_done += n_envs

            # Bootstrap value for last observation.
            last_obs_jax = encode(obs_np)
            last_val = np.asarray(model.value(last_obs_jax))  # (n_worlds,)

            # --- Compute GAE on GPU ---
            advantages, returns = compute_gae(
                jnp.array(buf_rew), jnp.array(buf_val), jnp.array(buf_done),
                jnp.array(last_val), gamma, gae_lambda,
            )

            # --- PPO update ---
            model, opt_state, loss_metrics, rng = ppo_update(
                model, opt_state, optimizer,
                jnp.array(buf_obs), jnp.array(buf_act),
                advantages, returns, jnp.array(buf_logp),
                n_epochs, batch_size, clip_range, ent_coef, vf_coef, rng,
            )

            # --- Logging ---
            if update % log_freq == 0:
                elapsed = time.time() - t_start
                sps = steps_done / elapsed
                gates_mean = float(np.mean(ep_gates)) if ep_gates else 0.0
                success_rate = float(np.mean([g == len(cfg.env.track.gates) for g in ep_gates])) if ep_gates else 0.0
                print(
                    f"update={update}/{total_updates}  steps={steps_done:,}  sps={sps:,.0f}  "
                    f"gates_passed={gates_mean:.2f}  success={success_rate:.3f}  "
                    f"loss={loss_metrics.get('loss/total', 0):.4f}",
                    flush=True,
                )
                if use_wandb and run is not None:
                    wandb.log({
                        "rollout/gates_passed_mean": gates_mean,
                        "rollout/success_rate": success_rate,
                        "train/steps_per_sec": sps,
                        "train/steps": steps_done,
                        **loss_metrics,
                    }, step=steps_done)
                ep_gates.clear()

            # --- Checkpoint ---
            if update % checkpoint_freq == 0:
                ckpt_path = CHECKPOINT_DIR / f"model_{steps_done:012d}.pkl"
                save_model(model, spec, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}", flush=True)

    finally:
        env.close()
        if run is not None:
            run.finish()

    save_model(model, spec, MODEL_PATH)
    print(f"Saved final model: {MODEL_PATH}", flush=True)
    return MODEL_PATH


# ---------------------------------------------------------------------------
# Evaluate.
# ---------------------------------------------------------------------------


def evaluate(
    config: str = "level2.toml",
    model_path: str | None = None,
    n_episodes: int = 5,
) -> dict[str, float]:
    """Run deterministic evaluation episodes for a trained policy."""
    path = Path(model_path) if model_path else MODEL_PATH
    model, spec = load_model(path)

    cfg = load_config(CONFIG_DIR / config)
    base_env = gym_make_env(cfg)
    encode = make_jax_encode(spec)
    decode_single = make_jax_decode(spec)

    rewards_list: list[float] = []
    gates_list: list[float] = []

    for ep in range(n_episodes):
        obs_raw, _ = base_env.reset(seed=ep + 1)
        obs_np = {k: np.asarray(v, dtype=np.float32) for k, v in obs_raw.items()}
        # Add batch dim for encode.
        obs_batch = {k: v[None] for k, v in obs_np.items()}
        ep_reward = 0.0
        done = False

        while not done:
            obs_jax = encode(obs_batch)  # (1, obs_dim)
            mean, _ = model.policy(obs_jax)
            action = jnp.clip(mean, -1.0, 1.0)  # deterministic
            pos = jnp.array(obs_np["pos"][None])  # (1, 3)
            cmd = np.asarray(decode_single(pos, action))[0]  # (13,)

            obs_raw, r, terminated, truncated, _ = base_env.step(cmd)
            obs_np = {k: np.asarray(v, dtype=np.float32) for k, v in obs_raw.items()}
            obs_batch = {k: v[None] for k, v in obs_np.items()}
            ep_reward += float(r)
            done = terminated or truncated

        n_gates_total = len(cfg.env.track.gates)
        tg = int(obs_np["target_gate"].item())
        gates_list.append(float(n_gates_total if tg == -1 else max(0, tg)))
        rewards_list.append(ep_reward)

    base_env.close()
    summary = {
        "mean_reward": float(np.mean(rewards_list)) if rewards_list else 0.0,
        "std_reward": float(np.std(rewards_list)) if rewards_list else 0.0,
        "mean_gates_passed": float(np.mean(gates_list)) if gates_list else 0.0,
        "success_rate": float(np.mean([g == len(cfg.env.track.gates) for g in gates_list])),
    }
    print(summary)
    return summary


def gym_make_env(cfg: Any):
    """Build a single-world CPU env wrapped to NumPy (for evaluate())."""
    import gymnasium as gym
    from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
    env = gym.make(
        cfg.env.id,
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        sensor_range=cfg.env.sensor_range,
        control_mode="state",
        track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"),
        randomizations=cfg.env.get("randomizations"),
        seed=cfg.env.seed,
    )
    return JaxToNumpy(env)


# ---------------------------------------------------------------------------
# Deployable controller (scripts/sim.py compatible).
# ---------------------------------------------------------------------------


class JaxStateRaceController(Controller):
    """Deployable controller: encode obs -> Flax policy -> decode to state command.

    Loads the trained Flax actor-critic weights saved by train() and chains the
    same RaceCodec encode/decode logic (identical maths, just numpy in inference).
    """

    def __init__(self, obs: dict, info: dict, config: dict):
        super().__init__(obs, info, config)
        model_path = MODEL_PATH
        self._model, self._spec = load_model(model_path)
        self._encode = make_jax_encode(self._spec)
        self._decode = make_jax_decode(self._spec)

    def compute_control(self, obs: dict, info: dict | None = None) -> NDArray:
        """Convert live race observation into a deterministic policy command."""
        obs_np = {k: np.asarray(v, dtype=np.float32) for k, v in obs.items()}
        obs_batch = {k: v[None] for k, v in obs_np.items()}
        obs_jax = self._encode(obs_batch)  # (1, obs_dim)
        mean, _ = self._model.policy(obs_jax)
        action = jnp.clip(mean, -1.0, 1.0)
        pos = jnp.array(obs_np["pos"][None])
        cmd = np.asarray(self._decode(pos, action))[0]
        return cmd.astype(np.float32)

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        return False


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main():
    fire.Fire({"train": train, "evaluate": evaluate})


if __name__ == "__main__":
    main()
