"""Parity test: JAX codec vs numpy codec from sb3_state_race_noguide.py.

Run with: python tests/test_jax_codec_parity.py
(or: pytest tests/test_jax_codec_parity.py)

Checks that jax_encode and jax_decode produce the same numbers as the
original numpy RaceCodec on identical inputs, with atol=1e-5.
"""

import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

# lsy_drone_racing.__init__ applies the Python 3.13/warp compat patch before
# any mujoco imports. Import it first to ensure the patch is in place.
import lsy_drone_racing  # noqa: F401


def _make_fake_obs(rng: np.random.Generator, n_gates: int = 4, n_obstacles: int = 4):
    """Construct a realistic fake obs dict (single-world, not batched)."""
    pos = rng.uniform(-2, 2, (3,)).astype(np.float32)
    return {
        "pos": pos,
        "quat": rng.uniform(-1, 1, (4,)).astype(np.float32),
        "vel": rng.uniform(-2, 2, (3,)).astype(np.float32),
        "ang_vel": rng.uniform(-1, 1, (3,)).astype(np.float32),
        "target_gate": np.array(rng.integers(0, n_gates), dtype=np.int32),
        "gates_pos": rng.uniform(-2, 2, (n_gates, 3)).astype(np.float32),
        "gates_quat": rng.uniform(-1, 1, (n_gates, 4)).astype(np.float32),
        "obstacles_pos": rng.uniform(-2, 2, (n_obstacles, 3)).astype(np.float32),
        "obstacles_visited": rng.integers(0, 2, (n_obstacles,)).astype(np.float32),
    }


def test_encode_parity():
    from lsy_drone_racing.control.sb3_state_race_noguide import RaceCodec, EnvSpec as CPUEnvSpec
    from lsy_drone_racing.control.train_noguide_jax import EnvSpec as JaxEnvSpec, make_jax_encode
    import jax.numpy as jnp

    rng = np.random.default_rng(42)
    cpu_spec = CPUEnvSpec()
    jax_spec = JaxEnvSpec()
    cpu_codec = RaceCodec(cpu_spec)
    jax_encode = make_jax_encode(jax_spec)

    for trial in range(20):
        obs = _make_fake_obs(rng)

        # CPU encode (single world).
        cpu_obs_vec = cpu_codec.encode(obs)

        # JAX encode (batched with batch=1).
        obs_batch = {k: v[None] for k, v in obs.items()}
        jax_obs_vec = np.asarray(jax_encode(obs_batch))[0]

        np.testing.assert_allclose(
            cpu_obs_vec, jax_obs_vec, atol=1e-5,
            err_msg=f"encode mismatch on trial {trial}",
        )

    print("✓ encode parity: all 20 random trials match (atol=1e-5)")


def test_encode_parity_target_gate_minus1():
    """When target_gate=-1 (finished), all gate features must be zero."""
    from lsy_drone_racing.control.sb3_state_race_noguide import RaceCodec, EnvSpec as CPUEnvSpec
    from lsy_drone_racing.control.train_noguide_jax import EnvSpec as JaxEnvSpec, make_jax_encode

    rng = np.random.default_rng(7)
    cpu_spec = CPUEnvSpec()
    jax_spec = JaxEnvSpec()
    cpu_codec = RaceCodec(cpu_spec)
    jax_encode = make_jax_encode(jax_spec)

    obs = _make_fake_obs(rng)
    obs["target_gate"] = np.array(-1, dtype=np.int32)

    cpu_vec = cpu_codec.encode(obs)
    jax_vec = np.asarray(jax_encode({k: v[None] for k, v in obs.items()}))[0]

    np.testing.assert_allclose(cpu_vec, jax_vec, atol=1e-5,
                               err_msg="encode mismatch when target_gate=-1")
    print("✓ encode parity: target_gate=-1 case matches")


def test_decode_parity():
    from lsy_drone_racing.control.sb3_state_race_noguide import RaceCodec, EnvSpec as CPUEnvSpec
    from lsy_drone_racing.control.train_noguide_jax import EnvSpec as JaxEnvSpec, make_jax_decode
    import jax.numpy as jnp

    rng = np.random.default_rng(99)
    cpu_spec = CPUEnvSpec()
    jax_spec = JaxEnvSpec()
    cpu_codec = RaceCodec(cpu_spec)
    jax_decode = make_jax_decode(jax_spec)

    for trial in range(20):
        obs = _make_fake_obs(rng)
        action = rng.uniform(-1, 1, (10,)).astype(np.float32)

        # CPU decode.
        cpu_cmd = cpu_codec.decode(obs, action)

        # JAX decode (batched with batch=1).
        pos_batch = jnp.array(obs["pos"][None])
        action_batch = jnp.array(action[None])
        jax_cmd = np.asarray(jax_decode(pos_batch, action_batch))[0]

        np.testing.assert_allclose(
            cpu_cmd, jax_cmd, atol=1e-5,
            err_msg=f"decode mismatch on trial {trial}",
        )

    print("✓ decode parity: all 20 random trials match (atol=1e-5)")


if __name__ == "__main__":
    test_encode_parity()
    test_encode_parity_target_gate_minus1()
    test_decode_parity()
    print("\nAll parity tests passed.")
