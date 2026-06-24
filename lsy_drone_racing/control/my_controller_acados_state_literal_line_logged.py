"""Literal straight-line acados state-setpoint controller.

Purpose
-------
This is a diagnostic controller for isolating the MPC/tracker interface.
It ignores gates and obstacles completely.  The global path is one literal
straight line:

    p(s) = p_start + s * direction,  0 <= s <= line_length

where ``p_start`` is the initial x/y position at a fixed test altitude, and
``direction`` defaults to world +x.  If this controller cannot track the line,
then the issue is in the state-setpoint / MPC tracking stack, not in gate path
planning or obstacle handling.

Output mode
-----------
It keeps the default challenge ``control_mode = "state"`` and returns the 13-D
state setpoint:

    [x, y, z, vx, vy, vz, ax, ay, az, yaw, roll_rate, pitch_rate, yaw_rate]

Debug output
------------
If ``controller.debug_plot_path`` is true, saves:

    debug_plots/literal_line_latest.png
    debug_plots/literal_line_latest.npz

If ``controller.log_tracking`` is true, saves:

    debug_logs/literal_line_tracking.csv

The CSV is intended to be post-processed with analyze_mpc_tracking.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import csv
import math
import platform
import shutil
from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import (
    AcadosModel,
    AcadosOcp,
    AcadosOcpSolver,
    ocp_get_default_cmake_builder,
)

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


SOLVER_VERSION = "literal_line_state_logged_v1"


def make_windows_cmake_builder():
    cmake_builder = ocp_get_default_cmake_builder()
    cmake_builder.generator = "Visual Studio 17 2022"
    cmake_builder.host = "x64"
    return cmake_builder


def _cfg_get(config: object, dotted_name: str, default):
    cur = config
    for name in dotted_name.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            if name not in cur:
                return default
            cur = cur[name]
        else:
            if not hasattr(cur, name):
                return default
            cur = getattr(cur, name)
    return default if cur is None else cur


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_vec3(value, default: tuple[float, float, float]) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
            return arr[:3].astype(float)
    except Exception:
        pass
    return np.asarray(default, dtype=float)


def _unit(vec: np.ndarray, fallback: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> np.ndarray:
    vec = np.asarray(vec, dtype=float).reshape(3)
    n = float(np.linalg.norm(vec))
    if n > 1e-8:
        return vec / n
    fb = np.asarray(fallback, dtype=float).reshape(3)
    nf = float(np.linalg.norm(fb))
    if nf > 1e-8:
        return fb / nf
    return np.array([1.0, 0.0, 0.0])


def create_point_mass_model() -> AcadosModel:
    """6-state double-integrator model: x=[p,v], u=[a]."""
    x = ca.SX.sym("x", 6)
    u = ca.SX.sym("u", 3)
    xdot = ca.SX.sym("xdot", 6)

    f_expl = ca.vertcat(x[3], x[4], x[5], u[0], u[1], u[2])

    model = AcadosModel()
    model.name = "lsy_" + SOLVER_VERSION
    model.x = x
    model.u = u
    model.xdot = xdot
    model.f_expl_expr = f_expl
    model.f_impl_expr = xdot - f_expl
    return model


def create_state_ocp_solver(
    Tf: float,
    N: int,
    a_xy_max: float,
    a_z_max: float,
    force_rebuild: bool,
    verbose: bool = False,
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    ocp = AcadosOcp()
    ocp.model = create_point_mass_model()

    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = nx + nu

    ocp.solver_options.N_horizon = N

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Strong tracking, modest acceleration penalty.  This should make failures
    # easy to diagnose: if the reference is a line and it does not follow it,
    # the issue is probably outside the global path builder.
    Q = np.diag([
        180.0, 180.0, 260.0,  # position x/y/z
        20.0, 20.0, 24.0,     # velocity x/y/z
    ])
    R_acc = np.diag([
        0.8, 0.8, 1.4,        # acceleration command smoothness
    ])
    ocp.cost.W = np.zeros((ny, ny))
    ocp.cost.W[:nx, :nx] = Q
    ocp.cost.W[nx:nx + nu, nx:nx + nu] = R_acc

    ocp.cost.W_e = np.diag([
        220.0, 220.0, 320.0,
        26.0, 26.0, 30.0,
    ])

    Vx = np.zeros((ny, nx))
    Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx:nx + nu, :] = np.eye(nu)
    ocp.cost.Vu = Vu

    Vx_e = np.eye(nx)
    ocp.cost.Vx_e = Vx_e

    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(nx)

    ocp.constraints.x0 = np.zeros(nx)

    # Keep only input bounds.  No state bounds here: hard state bounds can make
    # the very first stage infeasible when the measured drone is on/near ground.
    ocp.constraints.idxbu = np.array([0, 1, 2], dtype=np.int64)
    ocp.constraints.lbu = np.array([-a_xy_max, -a_xy_max, -a_z_max])
    ocp.constraints.ubu = np.array([+a_xy_max, +a_xy_max, +a_z_max])

    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = 4
    ocp.solver_options.qp_solver_iter_max = 50
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.tol = 2e-4
    ocp.solver_options.tf = Tf

    export_dir = Path("c_generated_code") / ("lsy_" + SOLVER_VERSION)
    json_file = export_dir / ("lsy_" + SOLVER_VERSION + ".json")
    if force_rebuild and export_dir.exists():
        shutil.rmtree(export_dir, ignore_errors=True)
    ocp.code_export_directory = str(export_dir)

    cmake_builder = None
    if platform.system() == "Windows":
        ocp.code_gen_opts.acados_include_path = r"C:/tools/acados/include"
        ocp.code_gen_opts.acados_lib_path = r"C:/tools/acados/lib"
        cmake_builder = make_windows_cmake_builder()

    if verbose or force_rebuild:
        print(f"[MyController] generating/building acados solver in {export_dir}")

    solver = AcadosOcpSolver(
        ocp,
        json_file=str(json_file),
        verbose=verbose,
        build=True,
        generate=True,
        cmake_builder=cmake_builder,
    )
    return solver, ocp


class MyController(Controller):
    """Literal straight-line path + simple acados state-setpoint MPC."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        self.freq = float(_cfg_get(config, "env.freq", 50.0))
        self.dt = 1.0 / self.freq

        self.N = int(_cfg_get(config, "controller.nmpc_horizon", 18))
        self.Tf = self.N * self.dt

        self.a_xy_max = float(_cfg_get(config, "controller.a_xy_max", 2.2))
        self.a_z_max = float(_cfg_get(config, "controller.a_z_max", 2.6))
        self.v_xy_max = float(_cfg_get(config, "controller.v_xy_max", 0.65))
        self.v_z_max = float(_cfg_get(config, "controller.v_z_max", 0.55))
        self.v_ref = min(float(_cfg_get(config, "controller.literal_line_speed", 0.45)), self.v_xy_max)
        self.lookahead_time = float(_cfg_get(config, "controller.lookahead_time", 0.10))

        self.z_min_cmd = float(_cfg_get(config, "controller.z_min_cmd", 0.15))
        self.z_max_cmd = float(_cfg_get(config, "controller.z_max_cmd", 1.80))
        self.line_z = float(_cfg_get(config, "controller.literal_line_z", 1.00))
        self.line_length = float(_cfg_get(config, "controller.literal_line_length", 2.00))
        self.line_direction = _unit(
            _as_vec3(_cfg_get(config, "controller.literal_line_direction", [1.0, 0.0, 0.0]), (1.0, 0.0, 0.0)),
            fallback=(1.0, 0.0, 0.0),
        )

        # A purely literal line.  It does not depend on gates, obstacles, or time.
        start = np.asarray(obs["pos"], dtype=float).reshape(3).copy()
        start[2] = np.clip(self.line_z, self.z_min_cmd, self.z_max_cmd)
        self.line_start = start
        self.line_end = self.line_start + self.line_length * self.line_direction
        self.line_end[2] = self.line_start[2]
        # If the user sets a direction with z component, preserve the line but
        # still respect command z bounds.
        self.line_end[2] = np.clip(self.line_end[2], self.z_min_cmd, self.z_max_cmd)
        self.line_vec = self.line_end - self.line_start
        self.line_length_actual = max(1e-6, float(np.linalg.norm(self.line_vec)))
        self.line_tangent = self.line_vec / self.line_length_actual
        self.progress_s = 0.0

        self.force_rebuild_solver = _as_bool(_cfg_get(config, "controller.force_rebuild_solver", True))
        self.verbose_acados = _as_bool(_cfg_get(config, "controller.verbose_acados", False))
        self.debug_plot_path = _as_bool(_cfg_get(config, "controller.debug_plot_path", True))
        self.debug_plot_dir = Path(str(_cfg_get(config, "controller.debug_plot_dir", "debug_plots")))

        self.log_tracking = _as_bool(_cfg_get(config, "controller.log_tracking", True))
        self.log_tracking_dir = Path(str(_cfg_get(config, "controller.log_tracking_dir", "debug_logs")))
        self.log_tracking_file = str(_cfg_get(config, "controller.log_tracking_file", "literal_line_tracking.csv"))
        self._tracking_log_path = self.log_tracking_dir / self.log_tracking_file
        self._log_initialized = False

        self._tick = 0
        self._finished = False
        self._last_good_action: NDArray[np.float32] | None = None

        self._solver_available = True
        try:
            self._acados_ocp_solver, self._ocp = create_state_ocp_solver(
                Tf=self.Tf,
                N=self.N,
                a_xy_max=self.a_xy_max,
                a_z_max=self.a_z_max,
                force_rebuild=self.force_rebuild_solver,
                verbose=self.verbose_acados,
            )
            self._nx = self._ocp.model.x.rows()
            self._nu = self._ocp.model.u.rows()
            self._ny = self._nx + self._nu
        except Exception as exc:  # pragma: no cover
            print(f"[MyController] acados solver creation failed, using fallback: {exc}")
            self._solver_available = False
            self._acados_ocp_solver = None
            self._ocp = None
            self._nx = 6
            self._nu = 3
            self._ny = 9

        if self.debug_plot_path:
            self._save_debug_plot(obs)
        if self.log_tracking:
            self._init_tracking_log()

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.float32]:
        pos = np.asarray(obs["pos"], dtype=float).reshape(3)
        vel = np.asarray(obs["vel"], dtype=float).reshape(3)
        x0 = np.concatenate([pos, vel])

        ref_pos, ref_vel = self._build_literal_line_reference(pos)

        mode = "fallback"
        action: NDArray[np.float32] | None = None
        if self._solver_available and self._acados_ocp_solver is not None:
            action = self._try_solve_acados(x0, ref_pos, ref_vel)
            if action is not None:
                mode = "acados"

        if action is None:
            action = self._fallback_action(pos, vel, ref_pos, ref_vel)

        self._last_good_action = action.copy()
        self._log_tracking_row(obs, ref_pos, ref_vel, action, mode)
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
        return self._finished

    def episode_callback(self):
        self._tick = 0
        self._finished = False
        self.progress_s = 0.0
        self._last_good_action = None
        if self.log_tracking:
            self._init_tracking_log()

    def _init_tracking_log(self) -> None:
        """Create/overwrite the CSV tracking log and write metadata/header."""
        try:
            self.log_tracking_dir.mkdir(parents=True, exist_ok=True)
            header = [
                "tick", "t", "mode",
                "pos_x", "pos_y", "pos_z",
                "vel_x", "vel_y", "vel_z",
                "cmd_x", "cmd_y", "cmd_z",
                "cmd_vx", "cmd_vy", "cmd_vz",
                "cmd_ax", "cmd_ay", "cmd_az",
                "ref0_x", "ref0_y", "ref0_z",
                "ref0_vx", "ref0_vy", "ref0_vz",
                "refN_x", "refN_y", "refN_z",
                "line_start_x", "line_start_y", "line_start_z",
                "line_tangent_x", "line_tangent_y", "line_tangent_z",
                "s_actual", "s_cmd", "s_ref0",
                "err_cmd_3d", "err_cmd_xy", "err_cmd_z",
                "err_ref0_3d", "err_ref0_xy", "err_ref0_z",
                "cross_track_3d", "cross_track_xy", "along_err_cmd",
                "solver_available",
            ]
            with self._tracking_log_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["# literal_line_logged_controller"])
                writer.writerow(["# line_start", *[f"{v:.9g}" for v in self.line_start]])
                writer.writerow(["# line_end", *[f"{v:.9g}" for v in self.line_end]])
                writer.writerow(["# line_tangent", *[f"{v:.9g}" for v in self.line_tangent]])
                writer.writerow(["# line_length", f"{self.line_length_actual:.9g}"])
                writer.writerow(["# env_freq", f"{self.freq:.9g}"])
                writer.writerow(header)
            self._log_initialized = True
            print(f"[MyController] tracking log: {self._tracking_log_path}")
        except Exception as exc:  # pragma: no cover
            print(f"[MyController] tracking log disabled; failed to create CSV: {exc}")
            self.log_tracking = False
            self._log_initialized = False

    def _log_tracking_row(
        self,
        obs: dict[str, NDArray[np.floating]],
        ref_pos: NDArray[np.floating],
        ref_vel: NDArray[np.floating],
        action: NDArray[np.floating],
        mode: str,
    ) -> None:
        if not self.log_tracking:
            return
        if not self._log_initialized:
            self._init_tracking_log()
            if not self._log_initialized:
                return

        try:
            pos = np.asarray(obs["pos"], dtype=float).reshape(3)
            vel = np.asarray(obs["vel"], dtype=float).reshape(3)
            act = np.asarray(action, dtype=float).reshape(-1)
            cmd_p = act[0:3]
            cmd_v = act[3:6]
            cmd_a = act[6:9]
            r0p = np.asarray(ref_pos[0], dtype=float).reshape(3)
            r0v = np.asarray(ref_vel[0], dtype=float).reshape(3)
            rNp = np.asarray(ref_pos[-1], dtype=float).reshape(3)

            def progress(p: np.ndarray) -> float:
                return float(np.dot(np.asarray(p, dtype=float).reshape(3) - self.line_start, self.line_tangent))

            s_actual = progress(pos)
            s_cmd = progress(cmd_p)
            s_ref0 = progress(r0p)
            closest = self.line_start + np.clip(s_actual, 0.0, self.line_length_actual) * self.line_tangent
            cross_vec = pos - closest
            err_cmd = pos - cmd_p
            err_ref0 = pos - r0p

            row = [
                int(self._tick), self._tick * self.dt, mode,
                *pos.tolist(),
                *vel.tolist(),
                *cmd_p.tolist(),
                *cmd_v.tolist(),
                *cmd_a.tolist(),
                *r0p.tolist(),
                *r0v.tolist(),
                *rNp.tolist(),
                *self.line_start.tolist(),
                *self.line_tangent.tolist(),
                s_actual, s_cmd, s_ref0,
                float(np.linalg.norm(err_cmd)),
                float(np.linalg.norm(err_cmd[:2])),
                float(err_cmd[2]),
                float(np.linalg.norm(err_ref0)),
                float(np.linalg.norm(err_ref0[:2])),
                float(err_ref0[2]),
                float(np.linalg.norm(cross_vec)),
                float(np.linalg.norm(cross_vec[:2])),
                float(s_actual - s_cmd),
                bool(self._solver_available),
            ]
            with self._tracking_log_path.open("a", newline="") as f:
                csv.writer(f).writerow(row)
        except Exception as exc:  # pragma: no cover
            if self._tick % int(max(1.0, self.freq)) == 0:
                print(f"[MyController] tracking log row failed: {exc}")

    def _build_literal_line_reference(self, pos: NDArray[np.floating]) -> tuple[np.ndarray, np.ndarray]:
        # Project current position onto the line, but only allow monotonic
        # progress.  This avoids the reference jumping backwards.
        s_closest = float(np.dot(np.asarray(pos, dtype=float).reshape(3) - self.line_start, self.line_tangent))
        s_closest = float(np.clip(s_closest, 0.0, self.line_length_actual))
        self.progress_s = max(self.progress_s, s_closest)

        ref_pos = np.zeros((self.N + 1, 3), dtype=float)
        ref_vel = np.zeros((self.N + 1, 3), dtype=float)

        for j in range(self.N + 1):
            s = self.progress_s + self.v_ref * (j * self.dt + self.lookahead_time)
            s = float(np.clip(s, 0.0, self.line_length_actual))
            remaining = self.line_length_actual - s

            # Reference brakes near the end instead of asking the drone to fly
            # through the endpoint indefinitely.
            vmag = min(self.v_ref, math.sqrt(max(0.0, 2.0 * self.a_xy_max * remaining)))
            if remaining < 0.04:
                vmag = 0.0
                self._finished = True

            ref_pos[j] = self.line_start + s * self.line_tangent
            ref_pos[j, 2] = np.clip(ref_pos[j, 2], self.z_min_cmd, self.z_max_cmd)
            ref_vel[j] = self._clip_velocity(vmag * self.line_tangent)

        return ref_pos, ref_vel

    def _try_solve_acados(
        self,
        x0: NDArray[np.floating],
        ref_pos: NDArray[np.floating],
        ref_vel: NDArray[np.floating],
    ) -> NDArray[np.float32] | None:
        solver = self._acados_ocp_solver
        assert solver is not None

        try:
            solver.set(0, "lbx", x0)
            solver.set(0, "ubx", x0)

            x_guess = np.asarray(x0, dtype=float).copy()
            for j in range(self.N):
                u_guess = np.clip(
                    2.4 * (ref_pos[j] - x_guess[:3]) + 1.6 * (ref_vel[j] - x_guess[3:6]),
                    [-self.a_xy_max, -self.a_xy_max, -self.a_z_max],
                    [+self.a_xy_max, +self.a_xy_max, +self.a_z_max],
                )
                solver.set(j, "x", x_guess)
                solver.set(j, "u", u_guess)
                x_guess = self._integrate_point_mass(x_guess, u_guess, self.dt)
                x_ref = np.concatenate([ref_pos[j], ref_vel[j]])
                x_guess = 0.65 * x_guess + 0.35 * x_ref
            solver.set(self.N, "x", x_guess)

            for j in range(self.N):
                yref = np.zeros(self._ny)
                yref[0:3] = ref_pos[j]
                yref[3:6] = ref_vel[j]
                # yref[6:9] = acceleration reference = 0
                solver.set(j, "yref", yref)

            yref_e = np.zeros(self._nx)
            yref_e[0:3] = ref_pos[self.N]
            yref_e[3:6] = ref_vel[self.N]
            solver.set(self.N, "yref", yref_e)

            status = solver.solve()
            if status != 0:
                if self._tick % int(max(1.0, self.freq)) == 0:
                    print(f"[MyController] acados returned status {status}; using fallback")
                return None

            x1 = np.asarray(solver.get(1, "x"), dtype=float).reshape(6)
            u0 = np.asarray(solver.get(0, "u"), dtype=float).reshape(3)

            p_sp = x1[0:3].copy()
            v_sp = self._clip_velocity(x1[3:6])
            a_sp = self._clip_accel(u0)
            p_sp[2] = np.clip(p_sp[2], self.z_min_cmd, self.z_max_cmd)
            yaw = self._desired_yaw(v_sp)

            return np.array([
                p_sp[0], p_sp[1], p_sp[2],
                v_sp[0], v_sp[1], v_sp[2],
                a_sp[0], a_sp[1], a_sp[2],
                yaw, 0.0, 0.0, 0.0,
            ], dtype=np.float32)
        except Exception as exc:  # pragma: no cover
            if self._tick % int(max(1.0, self.freq)) == 0:
                print(f"[MyController] acados exception; using fallback: {exc}")
            return None

    def _fallback_action(
        self,
        pos: NDArray[np.floating],
        vel: NDArray[np.floating],
        ref_pos: NDArray[np.floating],
        ref_vel: NDArray[np.floating],
    ) -> NDArray[np.float32]:
        p_sp = ref_pos[min(1, len(ref_pos) - 1)].copy()
        v_sp = self._clip_velocity(ref_vel[min(1, len(ref_vel) - 1)].copy())

        kp = np.array([2.0, 2.0, 3.0])
        kd = np.array([1.3, 1.3, 1.6])
        a_sp = self._clip_accel(kp * (p_sp - pos) + kd * (v_sp - vel))
        yaw = self._desired_yaw(v_sp)

        return np.array([
            p_sp[0], p_sp[1], p_sp[2],
            v_sp[0], v_sp[1], v_sp[2],
            a_sp[0], a_sp[1], a_sp[2],
            yaw, 0.0, 0.0, 0.0,
        ], dtype=np.float32)

    def _save_debug_plot(self, obs: dict[str, NDArray[np.floating]]) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            print(f"[MyController] debug path plot skipped; matplotlib unavailable: {exc}")
            return

        try:
            self.debug_plot_dir.mkdir(parents=True, exist_ok=True)
            drone_pos = np.asarray(obs.get("pos", self.line_start), dtype=float).reshape(3)
            path = np.vstack([self.line_start, self.line_end])

            fig = plt.figure(figsize=(12, 5))
            ax3 = fig.add_subplot(1, 2, 1, projection="3d")
            ax2 = fig.add_subplot(1, 2, 2)

            ax3.plot(path[:, 0], path[:, 1], path[:, 2], marker="o", label="literal straight line")
            ax3.scatter([drone_pos[0]], [drone_pos[1]], [drone_pos[2]], marker="x", s=60, label="initial drone")
            ax3.set_title("Literal straight-line test path")
            ax3.set_xlabel("x")
            ax3.set_ylabel("y")
            ax3.set_zlabel("z")
            ax3.legend(loc="best")

            ax2.plot(path[:, 0], path[:, 1], marker="o", label="literal straight line")
            ax2.scatter([drone_pos[0]], [drone_pos[1]], marker="x", s=60, label="initial drone")
            ax2.axis("equal")
            ax2.grid(True)
            ax2.set_title("Top-down XY")
            ax2.set_xlabel("x")
            ax2.set_ylabel("y")
            ax2.legend(loc="best")

            fig.tight_layout()
            fig.savefig(self.debug_plot_dir / "literal_line_latest.png", dpi=160)
            plt.close(fig)

            np.savez(
                self.debug_plot_dir / "literal_line_latest.npz",
                line_start=self.line_start,
                line_end=self.line_end,
                line_tangent=self.line_tangent,
                line_length=np.asarray([self.line_length_actual], dtype=float),
            )
        except Exception as exc:  # pragma: no cover
            print(f"[MyController] debug path plot failed: {exc}")

    @staticmethod
    def _integrate_point_mass(x: NDArray[np.floating], u: NDArray[np.floating], dt: float) -> np.ndarray:
        x = np.asarray(x, dtype=float).copy()
        u = np.asarray(u, dtype=float).reshape(3)
        x[:3] = x[:3] + dt * x[3:6] + 0.5 * dt * dt * u
        x[3:6] = x[3:6] + dt * u
        return x

    def _clip_velocity(self, v: NDArray[np.floating]) -> np.ndarray:
        v = np.asarray(v, dtype=float).reshape(3).copy()
        speed_xy = float(np.linalg.norm(v[:2]))
        if speed_xy > self.v_xy_max:
            v[:2] *= self.v_xy_max / max(speed_xy, 1e-9)
        v[2] = np.clip(v[2], -self.v_z_max, self.v_z_max)
        return v

    def _clip_accel(self, a: NDArray[np.floating]) -> np.ndarray:
        a = np.asarray(a, dtype=float).reshape(3).copy()
        a_xy = float(np.linalg.norm(a[:2]))
        if a_xy > self.a_xy_max:
            a[:2] *= self.a_xy_max / max(a_xy, 1e-9)
        a[2] = np.clip(a[2], -self.a_z_max, self.a_z_max)
        return a

    def _desired_yaw(self, vel_ref: NDArray[np.floating]) -> float:
        vxy = np.asarray(vel_ref[:2], dtype=float)
        if np.linalg.norm(vxy) > 0.03:
            return float(math.atan2(vxy[1], vxy[0]))
        return float(math.atan2(self.line_tangent[1], self.line_tangent[0]))
