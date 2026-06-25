"""Figure-eight benchmark with controlled landing and sim-to-real logging.

Phases:
    0: takeoff
    1: settle
    2: figure eight
    3: finish hold
    4: landing
    5: landed hold

The controller finishes only after reaching the landing setpoint and holding it.
On the real deployment runner, env.close() then stops the motors and closes the
radio connection.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class FigureEightLandingController(Controller):
    """Fly a curvature-speed figure eight, land, and save tracking data."""

    TAKEOFF = 0
    SETTLE = 1
    FIGURE_EIGHT = 2
    FINISH_HOLD = 3
    LANDING = 4
    LANDED_HOLD = 5

    def __init__(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict,
        config: dict,
    ):
        super().__init__(obs, info, config)

        if str(config.env.control_mode) != "state":
            raise ValueError(
                "FigureEightLandingController requires env.control_mode = 'state'."
            )

        cfg = config.get("figure8", {})
        self._freq = float(config.env.freq)
        if self._freq <= 0.0:
            raise ValueError("env.freq must be positive.")
        self._dt = 1.0 / self._freq

        # Main path parameters.
        self._size_m = self._positive(cfg.get("size_m", 1.2), "size_m")
        self._straight_speed = self._positive(
            cfg.get("straight_speed_mps", 0.5),
            "straight_speed_mps",
        )
        self._curve_speed = self._positive(
            cfg.get("curve_speed_mps", 0.3),
            "curve_speed_mps",
        )
        self._altitude_m = float(cfg.get("altitude_m", 0.8))
        self._laps = max(1, int(cfg.get("laps", 1)))
        self._max_accel = self._positive(
            cfg.get("max_acceleration_mps2", 1.0),
            "max_acceleration_mps2",
        )
        self._yaw_rad = float(cfg.get("yaw_rad", 0.0))
        self._geometry_samples = max(
            1000,
            int(cfg.get("geometry_samples", 4096)),
        )

        # Experiment phases.
        self._takeoff_time = self._nonnegative(
            cfg.get("takeoff_time_s", 3.0),
            "takeoff_time_s",
        )
        self._settle_time = self._nonnegative(
            cfg.get("settle_time_s", 1.0),
            "settle_time_s",
        )
        self._finish_hold_time = self._nonnegative(
            cfg.get("finish_hold_time_s", 0.5),
            "finish_hold_time_s",
        )
        self._landing_time = self._positive(
            cfg.get("landing_time_s", 3.0),
            "landing_time_s",
        )
        self._landed_hold_time = self._nonnegative(
            cfg.get("landed_hold_time_s", 0.5),
            "landed_hold_time_s",
        )

        # A small positive state setpoint is safer than commanding z < 0.
        # The real deployment runner stops the motors after the controller finishes.
        self._landing_height_m = max(
            0.0,
            float(cfg.get("landing_height_m", 0.05)),
        )

        # Logging.
        self._output_dir = Path(
            str(cfg.get("output_dir", "logs/figure8_sim_to_real"))
        )
        self._run_label = str(cfg.get("run_label", "sim"))
        self._log_prefix = str(cfg.get("log_prefix", "figure8_landing"))

        self._start_pos = np.asarray(obs["pos"], dtype=np.float64).copy()
        self._center = self._start_pos.copy()
        self._center[2] = self._altitude_m

        self._landing_target = self._start_pos.copy()
        self._landing_target[2] = self._landing_height_m

        self._reference = self._build_reference()

        self._tick = 0
        self._finished = False
        self._saved = False
        self._last_action: NDArray[np.floating] | None = None

        self._log: list[dict[str, Any]] = []
        self._append_log(
            time_s=0.0,
            obs=obs,
            applied_action=None,
            terminated=False,
            truncated=False,
        )

    @staticmethod
    def _positive(value: float, name: str) -> float:
        value = float(value)
        if value <= 0.0:
            raise ValueError(f"figure8.{name} must be positive, got {value}.")
        return value

    @staticmethod
    def _nonnegative(value: float, name: str) -> float:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"figure8.{name} must be nonnegative, got {value}.")
        return value

    @staticmethod
    def _smoothstep(x: np.ndarray | float) -> np.ndarray | float:
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _geometry(
        self,
        theta: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return local Gerono position, derivatives, and curvature."""
        theta = np.asarray(theta, dtype=np.float64)
        half_width = 0.5 * self._size_m
        half_height = 0.25 * self._size_m

        x = half_width * np.sin(theta)
        y = half_height * np.sin(2.0 * theta)
        z = np.zeros_like(theta)

        dx = half_width * np.cos(theta)
        dy = 2.0 * half_height * np.cos(2.0 * theta)
        dz = np.zeros_like(theta)

        ddx = -half_width * np.sin(theta)
        ddy = -4.0 * half_height * np.sin(2.0 * theta)
        ddz = np.zeros_like(theta)

        pos = np.stack((x, y, z), axis=-1)
        dpos = np.stack((dx, dy, dz), axis=-1)
        ddpos = np.stack((ddx, ddy, ddz), axis=-1)

        tangent_norm = np.linalg.norm(dpos, axis=-1)
        numerator = np.abs(dx * ddy - dy * ddx)
        curvature = numerator / np.maximum(tangent_norm**3, 1e-9)
        return pos, dpos, ddpos, curvature

    def _speed_from_curvature(
        self,
        curvature: np.ndarray | float,
        curvature_reference: float,
    ) -> np.ndarray:
        normalized = np.clip(
            np.asarray(curvature) / max(curvature_reference, 1e-9),
            0.0,
            1.0,
        )
        curve_weight = self._smoothstep(normalized)
        return self._straight_speed + curve_weight * (
            self._curve_speed - self._straight_speed
        )

    def _build_path_segment(self) -> dict[str, np.ndarray]:
        """Time-parameterize the figure eight with acceleration and braking limits."""
        theta_grid = np.linspace(
            0.0,
            2.0 * np.pi,
            self._geometry_samples + 1,
        )
        _, dpos_grid, _, curvature_grid = self._geometry(theta_grid)
        ds_dtheta_grid = np.linalg.norm(dpos_grid, axis=1)

        dtheta = np.diff(theta_grid)
        ds = 0.5 * (
            ds_dtheta_grid[:-1] + ds_dtheta_grid[1:]
        ) * dtheta
        s_grid = np.concatenate(([0.0], np.cumsum(ds)))
        lap_length = float(s_grid[-1])
        total_length = self._laps * lap_length
        total_theta = self._laps * 2.0 * np.pi
        curvature_reference = float(np.percentile(curvature_grid, 95.0))

        position_values: list[np.ndarray] = []
        velocity_values: list[np.ndarray] = []
        phase_values: list[float] = []
        curvature_values: list[float] = []
        nominal_speed_values: list[float] = []

        theta_global = 0.0
        speed = 0.0
        slowest_speed = min(self._straight_speed, self._curve_speed)
        fastest_speed = max(self._straight_speed, self._curve_speed)
        max_steps = int(
            np.ceil(
                3.0
                * self._freq
                * (
                    total_length / slowest_speed
                    + 2.0 * fastest_speed / self._max_accel
                    + 2.0
                )
            )
        )

        for _ in range(max_steps):
            lap_index = min(
                int(theta_global // (2.0 * np.pi)),
                self._laps - 1,
            )
            phase = theta_global - lap_index * 2.0 * np.pi
            if theta_global >= total_theta:
                phase = 0.0

            local_pos, local_dpos, _, curvature = self._geometry(phase)
            ds_dtheta = float(np.linalg.norm(local_dpos))

            s_in_lap = float(np.interp(phase, theta_grid, s_grid))
            s_global = lap_index * lap_length + s_in_lap
            if theta_global >= total_theta:
                s_global = total_length
            remaining = max(total_length - s_global, 0.0)

            nominal_speed = float(
                self._speed_from_curvature(
                    curvature,
                    curvature_reference,
                )
            )
            braking_speed = np.sqrt(
                max(2.0 * self._max_accel * remaining, 0.0)
            )
            requested_speed = min(nominal_speed, braking_speed)
            speed = float(
                np.clip(
                    requested_speed,
                    max(0.0, speed - self._max_accel * self._dt),
                    speed + self._max_accel * self._dt,
                )
            )

            tangent = local_dpos / max(ds_dtheta, 1e-9)
            position_values.append(self._center + local_pos)
            velocity_values.append(tangent * speed)
            phase_values.append(float(phase))
            curvature_values.append(float(curvature))
            nominal_speed_values.append(nominal_speed)

            if (
                remaining <= 1e-5
                and speed <= 2.0 * self._max_accel * self._dt
            ):
                break

            theta_global = min(
                total_theta,
                theta_global
                + speed / max(ds_dtheta, 1e-9) * self._dt,
            )
        else:
            raise RuntimeError(
                "Figure-eight reference generation exceeded its safety limit."
            )

        # Exact final center point with zero velocity.
        position_values.append(self._center.copy())
        velocity_values.append(np.zeros(3, dtype=np.float64))
        phase_values.append(0.0)
        curvature_values.append(0.0)
        nominal_speed_values.append(0.0)

        position = np.asarray(position_values, dtype=np.float64)
        velocity = np.asarray(velocity_values, dtype=np.float64)
        acceleration = self._time_gradient(velocity)

        return {
            "pos": position,
            "vel": velocity,
            "acc": acceleration,
            "path_phase": np.asarray(phase_values, dtype=np.float64),
            "curvature": np.asarray(curvature_values, dtype=np.float64),
            "nominal_speed": np.asarray(
                nominal_speed_values,
                dtype=np.float64,
            ),
            "segment_id": np.full(
                len(position),
                self.FIGURE_EIGHT,
                dtype=np.int8,
            ),
        }

    def _minimum_jerk_segment(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        duration_s: float,
        segment_id: int,
    ) -> dict[str, np.ndarray]:
        """Create an exact endpoint-to-endpoint rest-to-rest segment."""
        if duration_s <= 0.0:
            return self._constant_segment(
                goal,
                0.0,
                segment_id,
            )

        n = max(2, int(np.ceil(duration_s * self._freq)) + 1)
        time = np.linspace(0.0, duration_s, n)
        u = time / duration_s

        blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        blend_dot = (
            30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
        ) / duration_s
        blend_ddot = (
            60.0 * u - 180.0 * u**2 + 120.0 * u**3
        ) / duration_s**2

        displacement = goal - start
        pos = start[None, :] + blend[:, None] * displacement[None, :]
        vel = blend_dot[:, None] * displacement[None, :]
        acc = blend_ddot[:, None] * displacement[None, :]

        return {
            "pos": pos,
            "vel": vel,
            "acc": acc,
            "path_phase": np.full(n, np.nan),
            "curvature": np.zeros(n),
            "nominal_speed": np.linalg.norm(vel, axis=1),
            "segment_id": np.full(n, segment_id, dtype=np.int8),
        }

    def _constant_segment(
        self,
        position: np.ndarray,
        duration_s: float,
        segment_id: int,
    ) -> dict[str, np.ndarray]:
        n = max(1, int(np.ceil(duration_s * self._freq)) + 1)
        return {
            "pos": np.repeat(position[None, :], n, axis=0),
            "vel": np.zeros((n, 3), dtype=np.float64),
            "acc": np.zeros((n, 3), dtype=np.float64),
            "path_phase": np.full(n, np.nan),
            "curvature": np.zeros(n),
            "nominal_speed": np.zeros(n),
            "segment_id": np.full(n, segment_id, dtype=np.int8),
        }

    def _build_reference(self) -> dict[str, np.ndarray]:
        segments = [
            self._minimum_jerk_segment(
                self._start_pos,
                self._center,
                self._takeoff_time,
                self.TAKEOFF,
            ),
            self._constant_segment(
                self._center,
                self._settle_time,
                self.SETTLE,
            ),
            self._build_path_segment(),
            self._constant_segment(
                self._center,
                self._finish_hold_time,
                self.FINISH_HOLD,
            ),
            self._minimum_jerk_segment(
                self._center,
                self._landing_target,
                self._landing_time,
                self.LANDING,
            ),
            self._constant_segment(
                self._landing_target,
                self._landed_hold_time,
                self.LANDED_HOLD,
            ),
        ]

        reference: dict[str, np.ndarray] = {}
        keys = (
            "pos",
            "vel",
            "acc",
            "path_phase",
            "curvature",
            "nominal_speed",
            "segment_id",
        )
        for key in keys:
            parts = []
            for index, segment in enumerate(segments):
                values = segment[key]
                # Skip repeated boundary samples after the first segment.
                parts.append(values if index == 0 else values[1:])
            reference[key] = np.concatenate(parts, axis=0)

        reference["time"] = (
            np.arange(len(reference["pos"]), dtype=np.float64) * self._dt
        )
        reference["yaw"] = np.full(
            len(reference["pos"]),
            self._yaw_rad,
            dtype=np.float64,
        )
        return reference

    def _time_gradient(self, values: np.ndarray) -> np.ndarray:
        if len(values) < 3:
            return np.zeros_like(values)
        return np.gradient(
            values,
            self._dt,
            axis=0,
            edge_order=2,
        )

    def _reference_index(self, tick: int) -> int:
        return min(max(tick, 0), len(self._reference["time"]) - 1)

    def compute_control(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict | None = None,
    ) -> NDArray[np.floating]:
        index = self._reference_index(self._tick)

        action = np.zeros(13, dtype=np.float32)
        action[0:3] = self._reference["pos"][index]
        action[3:6] = self._reference["vel"][index]
        action[6:9] = self._reference["acc"][index]
        action[9] = self._reference["yaw"][index]

        self._last_action = action.copy()
        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        self._tick += 1
        time_s = min(
            self._tick * self._dt,
            float(self._reference["time"][-1]),
        )

        self._append_log(
            time_s=time_s,
            obs=obs,
            applied_action=action,
            terminated=terminated,
            truncated=truncated,
        )

        self._finished = (
            self._tick >= len(self._reference["time"]) - 1
        )
        done = bool(terminated or truncated or self._finished)

        # deploy.py does not call episode_callback(), so save here on hardware.
        if done:
            self._save_logs()

        return done

    def _append_log(
        self,
        time_s: float,
        obs: dict[str, NDArray[np.floating]],
        applied_action: NDArray[np.floating] | None,
        terminated: bool,
        truncated: bool,
    ) -> None:
        tick = int(round(time_s * self._freq))
        index = self._reference_index(tick)

        command = (
            np.full(13, np.nan, dtype=np.float64)
            if applied_action is None
            else np.asarray(applied_action, dtype=np.float64).copy()
        )

        self._log.append(
            {
                "time_s": float(time_s),
                "segment_id": int(self._reference["segment_id"][index]),
                "intended_pos": self._reference["pos"][index].copy(),
                "intended_vel": self._reference["vel"][index].copy(),
                "intended_acc": self._reference["acc"][index].copy(),
                "intended_yaw": float(self._reference["yaw"][index]),
                "intended_speed": float(
                    self._reference["nominal_speed"][index]
                ),
                "path_phase": float(
                    self._reference["path_phase"][index]
                ),
                "curvature": float(
                    self._reference["curvature"][index]
                ),
                "measured_pos": np.asarray(
                    obs["pos"],
                    dtype=np.float64,
                ).copy(),
                "measured_vel": np.asarray(
                    obs["vel"],
                    dtype=np.float64,
                ).copy(),
                "measured_quat": np.asarray(
                    obs["quat"],
                    dtype=np.float64,
                ).copy(),
                "measured_ang_vel": np.asarray(
                    obs["ang_vel"],
                    dtype=np.float64,
                ).copy(),
                "applied_action": command,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

    def _save_logs(self) -> Path | None:
        if self._saved or not self._log:
            return None

        self._output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S_%fZ"
        )
        stem = (
            f"{self._log_prefix}_{self._run_label}_{stamp}"
        )
        prefix = self._output_dir / stem

        time_s = np.asarray(
            [sample["time_s"] for sample in self._log],
            dtype=np.float64,
        )
        segment_id = np.asarray(
            [sample["segment_id"] for sample in self._log],
            dtype=np.int8,
        )

        intended_pos = np.stack(
            [sample["intended_pos"] for sample in self._log]
        )
        intended_vel = np.stack(
            [sample["intended_vel"] for sample in self._log]
        )
        intended_acc = np.stack(
            [sample["intended_acc"] for sample in self._log]
        )
        intended_yaw = np.asarray(
            [sample["intended_yaw"] for sample in self._log]
        )
        intended_speed = np.asarray(
            [sample["intended_speed"] for sample in self._log]
        )
        path_phase = np.asarray(
            [sample["path_phase"] for sample in self._log]
        )
        curvature = np.asarray(
            [sample["curvature"] for sample in self._log]
        )

        measured_pos = np.stack(
            [sample["measured_pos"] for sample in self._log]
        )
        measured_vel = np.stack(
            [sample["measured_vel"] for sample in self._log]
        )
        measured_quat = np.stack(
            [sample["measured_quat"] for sample in self._log]
        )
        measured_ang_vel = np.stack(
            [sample["measured_ang_vel"] for sample in self._log]
        )
        applied_action = np.stack(
            [sample["applied_action"] for sample in self._log]
        )

        position_error = measured_pos - intended_pos
        velocity_error = measured_vel - intended_vel
        figure8_mask = segment_id == self.FIGURE_EIGHT

        if np.any(figure8_mask):
            figure8_error = position_error[figure8_mask]
            figure8_rmse_3d = float(
                np.sqrt(
                    np.mean(
                        np.sum(figure8_error**2, axis=1)
                    )
                )
            )
            figure8_max_error = float(
                np.max(
                    np.linalg.norm(
                        figure8_error,
                        axis=1,
                    )
                )
            )
        else:
            figure8_rmse_3d = np.nan
            figure8_max_error = np.nan

        metadata = {
            "run_label": self._run_label,
            "size_m": self._size_m,
            "straight_speed_mps": self._straight_speed,
            "curve_speed_mps": self._curve_speed,
            "altitude_m": self._altitude_m,
            "laps": self._laps,
            "landing_height_m": self._landing_height_m,
            "landing_time_s": self._landing_time,
            "sample_frequency_hz": self._freq,
            "figure8_segment_id": self.FIGURE_EIGHT,
            "coordinate_frame": "world",
            "measurement_note": (
                "Simulation uses simulator state. Real flight uses the "
                "state estimate exposed by the deployment environment."
            ),
        }

        npz_path = prefix.with_suffix(".npz")
        np.savez_compressed(
            npz_path,
            time_s=time_s,
            segment_id=segment_id,
            intended_pos=intended_pos,
            intended_vel=intended_vel,
            intended_acc=intended_acc,
            intended_yaw=intended_yaw,
            intended_speed=intended_speed,
            path_phase_rad=path_phase,
            path_curvature=curvature,
            measured_pos=measured_pos,
            measured_vel=measured_vel,
            measured_quat=measured_quat,
            measured_ang_vel=measured_ang_vel,
            applied_action=applied_action,
            position_error=position_error,
            position_error_norm=np.linalg.norm(
                position_error,
                axis=1,
            ),
            velocity_error=velocity_error,
            figure8_mask=figure8_mask,
            figure8_position_rmse_3d_m=np.asarray(
                figure8_rmse_3d
            ),
            figure8_position_max_error_m=np.asarray(
                figure8_max_error
            ),
            metadata_json=np.asarray(json.dumps(metadata)),
        )

        self._write_comparison_csv(
            prefix.with_name(prefix.name + "_comparison.csv"),
            time_s,
            segment_id,
            intended_pos,
            measured_pos,
            intended_vel,
            measured_vel,
        )

        self._saved = True
        print(f"Saved figure-eight log: {npz_path}")
        return npz_path

    @staticmethod
    def _write_comparison_csv(
        path: Path,
        time_s: np.ndarray,
        segment_id: np.ndarray,
        intended_pos: np.ndarray,
        measured_pos: np.ndarray,
        intended_vel: np.ndarray,
        measured_vel: np.ndarray,
    ) -> None:
        position_error = measured_pos - intended_pos
        velocity_error = measured_vel - intended_vel

        header = [
            "time_s",
            "segment_id",
            "intended_x_m",
            "intended_y_m",
            "intended_z_m",
            "measured_x_m",
            "measured_y_m",
            "measured_z_m",
            "error_x_m",
            "error_y_m",
            "error_z_m",
            "error_norm_m",
            "intended_vx_mps",
            "intended_vy_mps",
            "intended_vz_mps",
            "measured_vx_mps",
            "measured_vy_mps",
            "measured_vz_mps",
            "velocity_error_norm_mps",
        ]

        rows = np.column_stack(
            (
                time_s,
                segment_id,
                intended_pos,
                measured_pos,
                position_error,
                np.linalg.norm(position_error, axis=1),
                intended_vel,
                measured_vel,
                np.linalg.norm(velocity_error, axis=1),
            )
        )

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)

    def episode_callback(self) -> None:
        """Simulation calls this; real deployment is saved in step_callback."""
        self._save_logs()

    def render_callback(self, sim: Sim) -> None:
        """Draw only the figure-eight measurement segment and the current target."""
        mask = (
            self._reference["segment_id"]
            == self.FIGURE_EIGHT
        )
        path = self._reference["pos"][mask]

        if len(path) > 1:
            # A moderate point count keeps the rendered line smooth without
            # exhausting Crazyflow visual geometries.
            count = min(len(path), 120)
            indices = np.linspace(
                0,
                len(path) - 1,
                count,
                dtype=int,
            )
            try:
                draw_line(
                    sim,
                    path[indices],
                    rgba=(0.0, 1.0, 0.0, 1.0),
                )
            except RuntimeError:
                pass

        index = self._reference_index(self._tick)
        try:
            draw_points(
                sim,
                self._reference["pos"][index].reshape(1, 3),
                rgba=(1.0, 0.0, 0.0, 1.0),
                size=0.025,
            )
        except RuntimeError:
            pass

    def reset(self) -> None:
        self._tick = 0
        self._finished = False

    def episode_reset(self) -> None:
        self.reset()
