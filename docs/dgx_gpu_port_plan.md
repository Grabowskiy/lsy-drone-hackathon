# Plan: Port guide-free drone-racing RL training to GPU (NVIDIA DGX Spark / GB10 Blackwell)

## Context (read first — assume no prior knowledge of this repo)

This is the **`lsy-drone-racing`** hackathon repo. The simulator is **crazyflow** (JAX + MuJoCo **MJX**). Work was started on an M1 Mac.

**The file to port:** `lsy_drone_racing/control/sb3_state_race_noguide.py`. Read it fully before doing anything. It contains a working, *simple* definition of the RL problem whose semantics you must **preserve**:

- **`RaceCodec.encode`** — observation layout (all relative to the drone; absolute drone position omitted): `quat(4), vel(3), ang_vel(3)`, then a fixed zero-padded window of the **nearest-K obstacles** `[known, rel_xyz]`, then the **next-N gates** `[exists, rel_xyz, quat_xyzw]`.
- **`RaceCodec.decode`** — 10-D action in [-1,1] → 13-D state command: `pos = drone_pos + action·pos_scale`, absolute `vel`, absolute `acc`, absolute `yaw`, body-rates = 0.
- **`DroneRaceSB3Env._reward`** + `_build_reward_path` — dense reward = progress along a Cubic-Hermite spline threaded through the true gate centers (binned), plus gate-pass / finish bonuses minus crash / timeout penalties.
- **`EnvSpec`** — all the scales and reward constants. Copy these values exactly.

**Critical caveat:** the CPU training runs **never learned to pass a single gate** (`gates_passed_mean` stayed 0). So this is *not* purely a speed port — part of the goal is enough sample throughput to actually debug whether the reward/action shaping works. **Validate that the policy actually learns**, don't just benchmark fps.

## Why GPU helps here (where the bottleneck is)

- The policy is a tiny `[256,256]` MLP — gradient steps are free. **The bottleneck is the physics sim**, which on the Mac ran on **CPU** via SB3's `SubprocVecEnv` + a `JaxToNumpy` wrapper (forces a CPU↔GPU sync every step). That CPU-bound, ~8-parallel-env pipeline is the cap.
- **crazyflow's `Sim` is already built for GPU batching**: `crazyflow/sim/sim.py` `__init__` takes **`n_worlds: int`** and **`device: str = "cpu"`**, and internally does `mjx.put_model(device=...)` + `jax.vmap` over `n_worlds`. So you can simulate **thousands of worlds in parallel on the GPU** — that's the 50–100x+ lever, not faster per-env.

## Step 0 — Investigate before coding

1. Confirm JAX sees CUDA: `python -c "import jax; print(jax.devices())"` → must show `CudaDevice`. If not, `pip install -U "jax[cuda12]"` matching the box's CUDA.
2. Find how **device + n_worlds** flow from config into the sim: look at `config/level2.toml` (`[sim]` section) and how `lsy_drone_racing/envs/` builds the env from `sim_config`. Check `lsy_drone_racing/envs/race_core.py`.
3. **Decisive question:** does `lsy_drone_racing/envs` expose a **vectorized / batched gym env** (returns `(n_worlds, ...)` batched obs, driven as a single batched env), or only the single-world gym wrapper? Grep `lsy_drone_racing/envs/__init__.py` and `race_core.py` for `VectorEnv` / `n_worlds` / `autoreset`. The answer decides the path below.

## Step 1 — Choose the path

**Path A (preferred — true GPU throughput): end-to-end JAX PPO.**
- Drive crazyflow's batched sim directly (`device="gpu"`, `n_worlds≈2048–8192`) via its functional/jitted `reset`/`step` (see `crazyflow/sim/functional.py`).
- **Reimplement `RaceCodec.encode` / `decode` / the reward as pure JAX**, `vmap`-ped over the world batch — identical math to the file, just `jnp` instead of `np` and no Python loops (vectorize the nearest-K obstacle sort and the next-N gate window).
- Use a JAX-native PPO (**PureJaxRL** single-file, **`rejax`**, or flax+optax) with the **entire rollout + update jitted on-device**. No `JaxToNumpy`, no SB3, no per-step host sync.
- Log scalars to **wandb** from the host periodically (every K updates, pull metrics off-device).

**Path B (fallback — smaller change, less speedup): SB3 + GPU-batched env bridge.**
- Only if Step 0.3 finds a ready batched gym `VectorEnv`. Wrap it so SB3 PPO consumes the batched obs with PPO on `cuda`. Still incurs some host sync, so expect well under Path A's ceiling — use only as a stopgap.

## Step 2 — Build it

- New entrypoint, e.g. `lsy_drone_racing/control/train_noguide_jax.py`, with `fire`-style `train` / `evaluate` like the original.
- Keep the **same `EnvSpec` constants** and obs/action/reward semantics so results are comparable to the CPU baseline.
- Save the trained policy in a form the deployable `Controller` can load for `scripts/sim.py` (either export weights the existing `SB3StateRaceController`-style class can consume, or write a small JAX-policy controller that implements `compute_control` using the same `RaceCodec` logic).

## Step 3 — Validate (do not skip)

- **Throughput:** print env-steps/sec; should be ≫10× the CPU baseline (target >1M steps/sec with thousands of worlds).
- **Learning (the real test):** the reward curve must climb and **`gates_passed_mean` must exceed 0** — the CPU run never did. If throughput is huge but it still never passes a gate, the bug is in **reward/action shaping**, not speed: inspect the progress-bin reward (is `path_reward_radius` reachable given the action scales?), the action scales in `EnvSpec`, and whether the spline/start alignment is correct. Use the higher throughput to sweep these quickly.
- Sanity-check obs/reward parity against the CPU `RaceCodec` on a few fixed states (same inputs → same numbers) before trusting the JAX port.

## Keep / reuse

- `RaceCodec` obs/action layout + reward math (port to JAX, identical).
- `EnvSpec` constants.
- wandb logging (online).

## Deliverable

A GPU training script that trains thousands of parallel worlds on the Blackwell GPU, demonstrably learns to pass gates, and produces a policy loadable by the existing controller interface for `scripts/sim.py`.
