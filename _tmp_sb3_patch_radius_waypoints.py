from pathlib import Path

path = Path(r"code/lsy-drone-hackathon/lsy_drone_racing/control/sb3_state_race.py")
text = path.read_text()

text = text.replace(
    "    path_progress_bins: int = 100\n    path_bin_reward: float = 1.0\n    gate_pass_bonus: float = 8.0\n",
    "    path_progress_bins: int = 100\n    path_bin_reward: float = 1.0\n    path_reward_radius: float = 0.30\n    gate_pass_bonus: float = 8.0\n",
)
text = text.replace(
    "    path_samples: int = 600\n    path_lookahead: float = 0.55\n    path_progress_window: int = 80\n",
    "    path_samples: int = 600\n    path_lookahead: float = 0.55\n    path_progress_window: int = 80\n    path_waypoint_count: int = 4\n    path_waypoint_spacing: float = 0.40\n",
)
text = text.replace(
    "        obs_dim = 4 + 3 + 3 + 4 * n_obstacles + self.spec_cfg.gate_count * (1 + 3 + 4)\n",
    "        obs_dim = (\n            4\n            + 3\n            + 3\n            + 4 * n_obstacles\n            + self.spec_cfg.gate_count * (1 + 3 + 4)\n            + 3 * self.spec_cfg.path_waypoint_count\n        )\n",
)
text = text.replace(
    "        self._prev_path_progress = self._path_progress(self._obs[\"pos\"])\n",
    "        self._prev_path_progress, _ = self._path_progress_and_distance(self._obs[\"pos\"])\n",
    1,
)
text = text.replace(
    "        next_path_progress = self._path_progress(next_obs[\"pos\"], self._prev_path_progress)\n        reward = self._reward(\n            self._obs, next_obs, clipped_action, next_path_progress, terminated, truncated\n        )\n",
    "        next_path_progress, path_distance = self._path_progress_and_distance(\n            next_obs[\"pos\"], self._prev_path_progress\n        )\n        reward = self._reward(\n            self._obs,\n            next_obs,\n            clipped_action,\n            next_path_progress,\n            path_distance,\n            terminated,\n            truncated,\n        )\n",
)
text = text.replace(
    "        self._best_progress_bin = max(self._best_progress_bin, self._progress_bin(next_path_progress))\n",
    "        if path_distance <= self.spec_cfg.path_reward_radius:\n            self._best_progress_bin = max(self._best_progress_bin, self._progress_bin(next_path_progress))\n",
)
text = text.replace(
    "        obs_vec = np.concatenate(pieces, dtype=np.float32)\n",
    "        path_progress, _ = self._path_progress_and_distance(pos, self._prev_path_progress)\n        for waypoint_idx in range(self.spec_cfg.path_waypoint_count):\n            waypoint_progress = path_progress + (waypoint_idx + 1) * self.spec_cfg.path_waypoint_spacing\n            waypoint = self._path_lookahead_point(waypoint_progress)\n            pieces.append((waypoint - pos).astype(np.float32))\n\n        obs_vec = np.concatenate(pieces, dtype=np.float32)\n",
)
text = text.replace(
    "        action: NDArray[np.float32],\n        next_progress: float,\n        terminated: bool,\n",
    "        action: NDArray[np.float32],\n        next_progress: float,\n        path_distance: float,\n        terminated: bool,\n",
)
text = text.replace(
    "        current_bin = self._progress_bin(next_progress)\n        newly_reached_bins = max(0, current_bin - self._best_progress_bin)\n",
    "        current_bin = self._progress_bin(next_progress)\n        reward_active = path_distance <= self.spec_cfg.path_reward_radius\n        newly_reached_bins = max(0, current_bin - self._best_progress_bin) if reward_active else 0\n",
)
old = '''    def _path_progress(\n        self, pos: NDArray[np.floating], reference_progress: float | None = None\n    ) -> float:\n        """Project the drone position onto the sampled reward spline with forward-only search."""\n        if self._path_points is None or self._path_arclength is None:\n            return 0.0\n        pos = np.asarray(pos, dtype=np.float32)\n        if reference_progress is None:\n            distances = np.linalg.norm(self._path_points - pos[None, :], axis=1)\n            return float(self._path_arclength[int(np.argmin(distances))])\n\n        ref_idx = int(np.searchsorted(self._path_arclength, reference_progress, side="left"))\n        start_idx = max(0, ref_idx - 3)\n        end_idx = min(len(self._path_points), ref_idx + self.spec_cfg.path_progress_window)\n        distances = np.linalg.norm(self._path_points[start_idx:end_idx] - pos[None, :], axis=1)\n        best_idx = start_idx + int(np.argmin(distances))\n        return max(float(reference_progress), float(self._path_arclength[best_idx]))\n'''
new = '''    def _path_progress_and_distance(\n        self, pos: NDArray[np.floating], reference_progress: float | None = None\n    ) -> tuple[float, float]:\n        """Project onto the sampled reward spline and return both progress and distance."""\n        if self._path_points is None or self._path_arclength is None:\n            return 0.0, np.inf\n        pos = np.asarray(pos, dtype=np.float32)\n        if reference_progress is None:\n            distances = np.linalg.norm(self._path_points - pos[None, :], axis=1)\n            best_idx = int(np.argmin(distances))\n            return float(self._path_arclength[best_idx]), float(distances[best_idx])\n\n        ref_idx = int(np.searchsorted(self._path_arclength, reference_progress, side="left"))\n        start_idx = max(0, ref_idx - 3)\n        end_idx = min(len(self._path_points), ref_idx + self.spec_cfg.path_progress_window)\n        distances = np.linalg.norm(self._path_points[start_idx:end_idx] - pos[None, :], axis=1)\n        best_local_idx = int(np.argmin(distances))\n        best_idx = start_idx + best_local_idx\n        return (\n            max(float(reference_progress), float(self._path_arclength[best_idx])),\n            float(distances[best_local_idx]),\n        )\n\n    def _path_progress(\n        self, pos: NDArray[np.floating], reference_progress: float | None = None\n    ) -> float:\n        """Project the drone position onto the sampled reward spline."""\n        progress, _ = self._path_progress_and_distance(pos, reference_progress)\n        return progress\n'''
if old not in text:
    raise SystemExit('expected path_progress block not found')
text = text.replace(old, new)

path.write_text(text)
