from pathlib import Path
path = Path(r"code/lsy-drone-hackathon/lsy_drone_racing/control/sb3_state_race.py")
text = path.read_text()
old = "        self._env_adapter._path_points = None\n        self._env_adapter._path_arclength = None\n        self._env_adapter._prev_path_progress = 0.0\n"
new = "        self._env_adapter._path_points = None\n        self._env_adapter._path_arclength = None\n        self._env_adapter._prev_path_progress = 0.0\n        self._env_adapter._best_progress_bin = 0\n"
if old in text:
    text = text.replace(old, new, 1)
path.write_text(text)
