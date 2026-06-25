from __future__ import annotations

import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation


# Observation layout (9 values total):
#   [0:3]  rel_pos  — vector from drone to target gate (x, y, z)
#   [3:6]  vel      — drone linear velocity
#   [6:9]  rpy      — drone roll, pitch, yaw (radians, derived from quaternion)
OBS_SIZE = 9


class RelativeDroneEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_SIZE,),
            dtype=np.float32,
        )

        self._prev_distance_to_gate: float | None = None
        self._prev_target_gate: int | None = None


    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)

        self._prev_target_gate = int(obs["target_gate"])
        gate_pos = self._current_gate_pos(obs)
        self._prev_distance_to_gate = float(np.linalg.norm(gate_pos - obs["pos"]))

        return self._get_obs(obs), info
    
    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        formatted_obs = self._get_obs(obs)
        reward = self._compute_reward(obs, terminated)

        self._prev_target_gate = int(obs["target_gate"])
        gate_pos = self._current_gate_pos(obs)
        self._prev_distance_to_gate = float(np.linalg.norm(gate_pos - obs["pos"]))

        return formatted_obs, reward, terminated, truncated, info


    def _get_obs(self, obs: dict) -> np.ndarray:
        gate_pos = self._current_gate_pos(obs)
        rel_pos = gate_pos - obs["pos"]
        rpy = Rotation.from_quat(obs["quat"]).as_euler("xyz").astype(np.float32)

        return np.concatenate([
            rel_pos.astype(np.float32),   # 3 values
            obs["vel"].astype(np.float32),  # 3 values
            rpy,                            # 3 values
        ])

    def _compute_reward(self, obs: dict, terminated: bool) -> float:
        """Shape the reward signal.

        Components:
          +progress * 10   — dense signal for moving toward the target gate
          +50              — bonus for each gate passage
          -100             — penalty for crashing before finishing
        """
        gate_pos = self._current_gate_pos(obs)
        current_distance = float(np.linalg.norm(gate_pos - obs["pos"]))

        # Dense progress signal: positive when closer, negative when further away
        progress = self._prev_distance_to_gate - current_distance
        reward = progress * 10.0

        # Gate passage: target_gate index advanced this step
        if self._gate_was_passed(obs):
            reward += 50.0

        # Crash penalty: terminated without having cleared all gates.
        # (target_gate == -1 means all gates were passed successfully and also
        #  triggers termination, so we must exclude that case.)
        if terminated and int(obs["target_gate"]) != -1:
            reward -= 100.0

        return float(reward)

    def _current_gate_pos(self, obs: dict) -> np.ndarray:
        """Return the (3,) position of the current target gate.

        When target_gate is -1 the drone has finished; we clamp to gate 0 so
        the observation stays well-defined until the episode ends.
        """
        target = int(obs["target_gate"])
        n_gates = obs["gates_pos"].shape[0]
        idx = max(0, target % n_gates)
        return obs["gates_pos"][idx]

    def _gate_was_passed(self, obs: dict) -> bool:
        """Return True if the drone passed through a gate this step."""
        current = int(obs["target_gate"])
        if self._prev_target_gate is None:
            return False
        # Gate index advances when passed; -1 means the final gate was cleared
        return current != self._prev_target_gate
