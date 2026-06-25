from pathlib import Path
path = Path(r"code/lsy-drone-hackathon/lsy_drone_racing/control/sb3_state_race.py")
text = path.read_text()
text = text.replace(
    'def train(\n    config: str = "level2.toml",\n    total_timesteps: int = 400_000,\n    n_envs: int = 8,\n    seed: int = 7,\n    device: str = "auto",\n    learning_rate: float = 3e-4,\n    batch_size: int = 256,\n    n_steps: int = 512,\n    eval_freq: int = 20_000,\n    save_freq: int = 50_000,\n) -> Path:',
    'def train(\n    config: str = "level2.toml",\n    total_timesteps: int = 10_000_000,\n    n_envs: int = 8,\n    seed: int = 7,\n    device: str = "auto",\n    learning_rate: float = 3e-4,\n    batch_size: int = 256,\n    n_steps: int = 4096,\n    gamma: float = 0.99,\n    eval_freq: int = 20_000,\n    save_freq: int = 50_000,\n) -> Path:'
)
text = text.replace('        gamma=0.99,\n', '        gamma=gamma,\n')
path.write_text(text)
