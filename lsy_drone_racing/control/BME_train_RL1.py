"""PPO training script for the BME drone racing environment.

Usage:
    # Train (default: 1 000 000 steps, level1.toml)
    python lsy_drone_racing/control/BME_train_RL1.py

    # Custom config / timesteps / output path
    python lsy_drone_racing/control/BME_train_RL1.py \
        --config level2.toml \
        --timesteps 2000000 \
        --save_path my_policy.zip

    # Evaluate a saved policy (no training)
    python lsy_drone_racing/control/BME_train_RL1.py \
        --train False \
        --eval_episodes 5
"""

from __future__ import annotations

from pathlib import Path

import fire
import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from lsy_drone_racing.control.BME_env_RL1 import RelativeDroneEnv
from lsy_drone_racing.utils import load_config


class JaxToNumpy(gymnasium.Wrapper):
    """Convert JAX arrays in obs/info/reward to numpy without requiring array-api-compat."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._to_numpy(obs), self._to_numpy(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._to_numpy(obs), float(reward), bool(terminated), bool(truncated), self._to_numpy(info)

    @staticmethod
    def _to_numpy(value):
        if isinstance(value, dict):
            return {k: JaxToNumpy._to_numpy(v) for k, v in value.items()}
        try:
            return np.asarray(value)
        except Exception:
            return value

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parents[2]
_CONFIG_DIR = _ROOT / "config"
_DEFAULT_SAVE = Path(__file__).parent / "BME_ppo_policy"


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(config_name: str = "level1.toml", render: bool = False) -> RelativeDroneEnv:
    """Build the wrapped racing environment.

    Stack:
        DroneRaceEnv  (JAX-backed simulator)
        └─ JaxToNumpy (converts JAX arrays → numpy)
           └─ RelativeDroneEnv (relative obs + custom reward)
              └─ Monitor (SB3 episode stats)
    """
    config = load_config(_CONFIG_DIR / config_name)
    config.sim.render = render

    # Force attitude control: its action space [-π/2, π/2, thrust_min, thrust_max]
    # has finite bounds required by SB3. State mode uses (-inf, +inf) which SB3 rejects.
    base_env = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode="attitude",
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    env = JaxToNumpy(base_env)
    env = RelativeDroneEnv(env)
    env = Monitor(env)
    return env


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    config: str = "level1.toml",
    timesteps: int = 1_000_000,
    save_path: str = str(_DEFAULT_SAVE),
    checkpoint_freq: int = 100_000,
    seed: int = 42,
) -> PPO:
    """Train a PPO policy and save it.

    Args:
        config: Config file name inside config/ (e.g. "level1.toml").
        timesteps: Total environment steps for training.
        save_path: Where to save the final policy (no extension needed).
        checkpoint_freq: Steps between intermediate checkpoints.
        seed: Random seed.

    Returns:
        The trained PPO model.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building environment from {config} ...")
    train_env = make_env(config, render=False)
    eval_env = make_env(config, render=False)

    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(save_path.parent / "checkpoints"),
        name_prefix="bme_ppo",
        verbose=1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path.parent / "best"),
        log_path=str(save_path.parent / "logs"),
        eval_freq=checkpoint_freq,
        n_eval_episodes=3,
        deterministic=True,
        verbose=1,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        learning_rate=3e-4,
        verbose=1,
        seed=seed,
        tensorboard_log=str(save_path.parent / "tb_logs"),
    )

    print(f"Training for {timesteps:,} timesteps ...")
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    model.save(str(save_path))
    print(f"Policy saved to {save_path}.zip")

    train_env.close()
    eval_env.close()
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    config: str = "level1.toml",
    load_path: str = str(_DEFAULT_SAVE),
    eval_episodes: int = 5,
    render: bool = True,
) -> None:
    """Load a saved policy and run evaluation episodes.

    Args:
        config: Config file name inside config/.
        load_path: Path to the saved policy (.zip, extension optional).
        eval_episodes: Number of evaluation episodes to run.
        render: Enable simulation rendering.
    """
    env = make_env(config, render=render)
    model = PPO.load(load_path, env=env)

    rewards, lengths = [], []
    for ep in range(eval_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward, ep_len = 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            if render:
                env.render()
            ep_reward += reward
            ep_len += 1
            done = terminated or truncated
        rewards.append(ep_reward)
        lengths.append(ep_len)
        print(f"Episode {ep + 1:2d}: reward={ep_reward:8.2f}  steps={ep_len}")

    print(
        f"\nMean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}  "
        f"| Mean steps: {np.mean(lengths):.0f}"
    )
    env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    train_: bool = True,
    config: str = "level1.toml",
    timesteps: int = 1_000_000,
    save_path: str = str(_DEFAULT_SAVE),
    checkpoint_freq: int = 100_000,
    eval_episodes: int = 0,
    render_eval: bool = False,
    seed: int = 42,
) -> None:
    """Main entry point.

    Args:
        train_: Run training. Set to False to skip (load existing policy).
        config: Config file name inside config/.
        timesteps: Total training timesteps.
        save_path: Path to save / load the policy.
        checkpoint_freq: Steps between checkpoints.
        eval_episodes: Evaluation episodes after training (0 = skip).
        render_eval: Enable rendering during evaluation.
        seed: Random seed.
    """
    if train_:
        train(
            config=config,
            timesteps=timesteps,
            save_path=save_path,
            checkpoint_freq=checkpoint_freq,
            seed=seed,
        )

    if eval_episodes > 0:
        evaluate(
            config=config,
            load_path=save_path,
            eval_episodes=eval_episodes,
            render=render_eval,
        )


if __name__ == "__main__":
    fire.Fire(main)
