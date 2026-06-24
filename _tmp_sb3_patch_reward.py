from pathlib import Path
import re

path = Path(r"code/lsy-drone-hackathon/lsy_drone_racing/control/sb3_state_race.py")
text = path.read_text()
text = text.replace(
    "    progress_gain: float = 10.0\n    finish_bonus: float = 40.0\n",
    "    path_progress_bins: int = 100\n    path_bin_reward: float = 1.0\n    gate_pass_bonus: float = 8.0\n    finish_bonus: float = 40.0\n",
)
text = text.replace(
    "        self._path_points: NDArray[np.floating] | None = None\n        self._path_arclength: NDArray[np.floating] | None = None\n        self._prev_path_progress = 0.0\n",
    "        self._path_points: NDArray[np.floating] | None = None\n        self._path_arclength: NDArray[np.floating] | None = None\n        self._prev_path_progress = 0.0\n        self._best_progress_bin = 0\n",
)
text = text.replace(
    "        self._path_points, self._path_arclength = self._build_reward_path(self._obs[\"pos\"])\n        self._prev_path_progress = self._path_progress(self._obs[\"pos\"])\n        return self._build_observation(self._obs), info\n",
    "        self._path_points, self._path_arclength = self._build_reward_path(self._obs[\"pos\"])\n        self._prev_path_progress = self._path_progress(self._obs[\"pos\"])\n        self._best_progress_bin = self._progress_bin(self._prev_path_progress)\n        return self._build_observation(self._obs), info\n",
)
text = text.replace(
    "        self._obs = next_obs\n        self._prev_path_progress = next_path_progress\n        self._prev_target_gate = self._target_gate(next_obs)\n",
    "        self._obs = next_obs\n        self._prev_path_progress = next_path_progress\n        self._best_progress_bin = max(self._best_progress_bin, self._progress_bin(next_path_progress))\n        self._prev_target_gate = self._target_gate(next_obs)\n",
)
text = re.sub(
    r"    def _reward\(\n        self,\n        prev_obs: dict\[str, NDArray\[np\.floating\]\],\n        next_obs: dict\[str, NDArray\[np\.floating\]\],\n        action: NDArray\[np\.float32\],\n        next_progress: float,\n        terminated: bool,\n        truncated: bool,\n    \) -> float:\n        \"\"\".*?\n        return reward\n",
    "    def _reward(\n        self,\n        prev_obs: dict[str, NDArray[np.floating]],\n        next_obs: dict[str, NDArray[np.floating]],\n        action: NDArray[np.float32],\n        next_progress: float,\n        terminated: bool,\n        truncated: bool,\n    ) -> float:\n        \"\"\"Discrete reward for newly reached path bins plus explicit collision penalties.\"\"\"\n        del action\n        prev_target = self._target_gate(prev_obs)\n        next_target = self._target_gate(next_obs)\n        current_bin = self._progress_bin(next_progress)\n        newly_reached_bins = max(0, current_bin - self._best_progress_bin)\n        reward = self.spec_cfg.path_bin_reward * float(newly_reached_bins)\n\n        if next_target != -1 and prev_target != -1 and next_target > prev_target:\n            reward += self.spec_cfg.gate_pass_bonus * float(next_target - prev_target)\n        if prev_target >= 0 and next_target == -1:\n            reward += self.spec_cfg.finish_bonus\n        if terminated and next_target != -1:\n            reward -= self.spec_cfg.crash_penalty\n            if self._hit_obstacle(next_obs[\"pos\"]):\n                reward -= self.spec_cfg.obstacle_hit_penalty\n        if truncated and next_target != -1:\n            reward -= self.spec_cfg.timeout_penalty\n\n        return reward\n",
    text,
    count=1,
    flags=re.S,
)
path.write_text(text)
