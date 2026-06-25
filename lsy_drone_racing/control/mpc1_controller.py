"""MPC controller that dynamically generates a trajectory through the gates.

This controller implements a Model Predictive Control (MPC) strategy to fly through a series of
gates. It adapts the example from `attitude_mpc.py` to solve the core challenge requirements:

1.  **Generalization**: It reads gate positions from the observation `obs` at runtime and generates
    a smooth `CubicSpline` trajectory through them. It does not use hardcoded waypoints.
2.  **Sim-to-Real**: It uses `acados` for fast, real-time optimization, making it suitable for
    both simulation and real-world deployment.

NOTE: This controller requires `control_mode = "attitude"` to be set in the .toml config file.
"""

from __future__ import annotations

import platform
from typing import TYPE_CHECKING

import numpy as np
import scipy
from acados_template import (
    AcadosModel,
    AcadosOcp,
    AcadosOcpSolver,
    ocp_get_default_cmake_builder,
)
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from crazyflow import Sim

def make_windows_cmake_builder():
    """Configure CMake for Windows if necessary."""
    cmake_builder = ocp_get_default_cmake_builder()
    cmake_builder.generator = "Visual Studio 17 2022"
    cmake_builder.host = "x64"
    return cmake_builder


def create_acados_model(parameters: dict) -> AcadosModel:
    """Creates an acados model from a symbolic drone_model."""
    X_dot, X, U, _ = symbolic_dynamics_euler(
        mass=parameters["mass"],
        gravity_vec=parameters["gravity_vec"],
        J=parameters["J"],
        J_inv=parameters["J_inv"],
        acc_coef=parameters["acc_coef"],
        cmd_f_coef=parameters["cmd_f_coef"],
        rpy_coef=parameters["rpy_coef"],
        rpy_rates_coef=parameters["rpy_rates_coef"],
        cmd_rpy_coef=parameters["cmd_rpy_coef"],
    )
    model = AcadosModel()
    model.name = "mpc1_controller_model"
    model.f_expl_expr = X_dot
    model.x = X
    model.u = U
    return model


def create_ocp_solver(Tf: float, N: int, parameters: dict) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()
    ocp.model = create_acados_model(parameters)
    nx, nu = ocp.model.x.rows(), ocp.model.u.rows()
    ny, ny_e = nx + nu, nx
    ocp.solver_options.N_horizon = N

    # Cost function
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # --- TUNING PARAMETERS ---
    # State weights (Q matrix)
    w_pos = np.array([250., 250., 400.])  # x, y, z: Higher values for tighter path following.
    w_rpy = np.array([5., 5., 50.])      # roll, pitch, yaw: Higher yaw weight for better heading control.
    w_vel = np.array([10., 10., 10.])    # vx, vy, vz
    w_drpy = np.array([5., 5., 5.])      # roll_rate, pitch_rate, yaw_rate
    Q = np.diag(np.concatenate([w_pos, w_rpy, w_vel, w_drpy]))

    # Input weights (R matrix)
    w_rpy_cmd = np.array([0.5, 0.5, 0.5]) # Penalizes aggressive roll/pitch/yaw commands.
    w_thrust_cmd = np.array([15.])        # Penalizes aggressive thrust commands. Lower for better climb response.
    R_mat = np.diag(np.concatenate([w_rpy_cmd, w_thrust_cmd]))
    # --- END TUNING PARAMETERS ---

    ocp.cost.W = scipy.linalg.block_diag(Q, R_mat)
    ocp.cost.W_e = Q

    ocp.cost.Vx = np.zeros((ny, nx)); ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vu = np.zeros((ny, nu)); ocp.cost.Vu[nx:, :] = np.eye(nu)
    ocp.cost.Vx_e = np.eye(nx)

    ocp.cost.yref, ocp.cost.yref_e = np.zeros((ny,)), np.zeros((ny_e,))

    # Constraints
    ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5])
    ocp.constraints.ubx = np.array([0.5, 0.5, 0.5])
    ocp.constraints.idxbx = np.array([3, 4, 5])
    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, parameters["thrust_min"] * 4])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, parameters["thrust_max"] * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    ocp.constraints.x0 = np.zeros((nx))

    # Solver options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"      # Explicit Runge-Kutta integrator
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # Use Real-Time Iteration for fast, consistent solve times
    ocp.solver_options.tf = Tf

    # Windows specific build options
    cmake_builder = make_windows_cmake_builder() if platform.system() == "Windows" else None

    acados_ocp_solver = AcadosOcpSolver(
        ocp, json_file="mpc1_controller.json", cmake_builder=cmake_builder
    )
    return acados_ocp_solver, ocp


class MPCController(Controller):
    """MPC controller that generates a dynamic trajectory through the gates."""

    @staticmethod
    def _nudge(point: np.ndarray, all_poles_pos: np.ndarray, margin: float) -> np.ndarray:
        """Push a 3-D point away from vertical poles (obstacles and gate frames)."""
        pt = point.copy()
        for pole_pos in all_poles_pos:
            # We only consider the horizontal (xy) plane for repulsion
            diff = pt[:2] - pole_pos[:2]
            dist = float(np.linalg.norm(diff))

            if dist < margin:
                # If the point is exactly on a pole, push it in an arbitrary direction (e.g., x-axis)
                if dist < 1e-6:
                    push_vec = np.array([margin, 0.0])
                else:
                    # Push point radially away from the pole's center
                    push_vec = (margin - dist) / dist * diff
                pt[:2] += push_vec
        return pt

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the MPC controller, trajectory, and acados solver."""
        super().__init__(obs, info, config)

        # MPC parameters
        self._N = 35  # Prediction horizon. Longer horizon for smoother cornering.
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt

        # --- 1. Generate a dynamic trajectory from the gate positions ---
        start_pos = np.asarray(obs["pos"], dtype=float)

        takeoff_pos = start_pos + np.array([0.0, 0.0, 0.5]) # Add a takeoff point
        gate_poses = np.asarray(obs["gates_pos"], dtype=float)
        gate_quats = np.asarray(obs["gates_quat"], dtype=float)
        obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=float)
        
        # --- 1a. Create a unified list of all vertical poles to avoid ---
        # This includes both obstacles and the four posts of each gate frame.
        gate_frame_half_width = 0.36  # Outer dimension of gate is 0.72m
        all_poles_to_avoid = list(obstacles_pos)
        # Local positions of the 4 posts of a gate frame
        local_post_positions = np.array([
            [0,  gate_frame_half_width,  gate_frame_half_width],
            [0,  gate_frame_half_width, -gate_frame_half_width],
            [0, -gate_frame_half_width,  gate_frame_half_width],
            [0, -gate_frame_half_width, -gate_frame_half_width],
        ])
        for i in range(len(gate_poses)):
            gate_pos = gate_poses[i]
            gate_rot = R.from_quat(gate_quats[i])
            # Transform local post positions to world frame and add to the list
            world_post_positions = gate_rot.apply(local_post_positions) + gate_pos
            all_poles_to_avoid.extend(world_post_positions)
        all_poles_to_avoid = np.array(all_poles_to_avoid)

        # --- 1b. Generate coarse waypoints (start, takeoff, gate approach/exit) ---
        coarse_waypoints = [start_pos, takeoff_pos]
        approach_dist = 0.4 # Distance to approach/exit gates

        for i in range(len(gate_poses)):
            center = gate_poses[i]
            # Get gate normal vector from its quaternion
            normal = R.from_quat(gate_quats[i]).apply([1.0, 0.0, 0.0])

            # Ensure the normal points from the previous waypoint towards the gate
            if np.dot(center - coarse_waypoints[-1], normal) < 0:
                normal = -normal

            coarse_waypoints.append(center - approach_dist * normal)
            coarse_waypoints.append(center + approach_dist * normal)

        coarse_waypoints = np.array(coarse_waypoints)

        # --- 2. Subdivide path and nudge points away from all poles to create curves ---
        OBSTACLE_MARGIN = 0.3  # Safety margin around all poles.
        finer_waypoints = []
        # For each segment between coarse waypoints, create finer points
        for i in range(len(coarse_waypoints) - 1):
            p1, p2 = coarse_waypoints[i], coarse_waypoints[i+1]
            dist = np.linalg.norm(p2 - p1)
            # Create a finer point roughly every 20cm
            num_subdivisions = int(np.ceil(dist / 0.2))
            if num_subdivisions < 2: num_subdivisions = 2

            # Nudge each finer point on the segment
            for point in np.linspace(p1, p2, num_subdivisions, endpoint=False):
                finer_waypoints.append(self._nudge(point, all_poles_to_avoid, OBSTACLE_MARGIN))

        # Add the final waypoint, also nudged for consistency
        finer_waypoints.append(self._nudge(coarse_waypoints[-1], all_poles_to_avoid, OBSTACLE_MARGIN))
        waypoints = np.array(finer_waypoints)

        # Filter out consecutive duplicate waypoints which cause CubicSpline to fail.
        unique_waypoints = []
        if len(waypoints) > 0:
            unique_waypoints.append(waypoints[0])
            for i in range(1, len(waypoints)):
                if not np.allclose(unique_waypoints[-1], waypoints[i], atol=1e-5):
                    unique_waypoints.append(waypoints[i])
        waypoints = np.array(unique_waypoints)

        # --- 3. Create time parameterization based on a constant average speed ---
        avg_speed = 1.0  # m/s
        distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        time_allocations = distances / np.maximum(avg_speed, 1e-6) # Avoid division by zero
        self._t_waypoints = np.concatenate(([0], np.cumsum(time_allocations)))
        self._t_total = self._t_waypoints[-1]

        # Create the cubic spline for position and its derivative for velocity
        self._des_pos_spline = CubicSpline(self._t_waypoints, waypoints)
        self._des_vel_spline = self._des_pos_spline.derivative()

        # Setup the acados MPC solver
        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        # Controller state
        self._tick = 0
        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next attitude command using MPC."""
        current_time = self._tick * self._dt

        # If the planned trajectory is finished, just hover at the last known position.
        if current_time > self._t_total:
            self._finished = True
            # Create a hover command
            hover_att = np.array([0.0, 0.0, 0.0]) # Zero roll, pitch, yaw
            hover_thrust = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]
            return np.concatenate([hover_att, [hover_thrust]])

        # Step 3: Set the initial state for the MPC problem
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        # Step 4: Set the reference trajectory for the prediction horizon
        t_horizon = np.linspace(current_time, current_time + self._T_HORIZON, self._N + 1)
        ref_pos = self._des_pos_spline(t_horizon)
        ref_vel = self._des_vel_spline(t_horizon)

        last_yaw = obs["rpy"][2] # Start with current yaw

        for j in range(self._N):
            yref = np.zeros((self._ny,))
            yref[0:3] = ref_pos[j]  # Desired position

            # Calculate desired yaw to face the direction of travel
            ref_vel_xy = ref_vel[j, :2]
            if np.linalg.norm(ref_vel_xy) > 0.1:
                last_yaw = np.arctan2(ref_vel_xy[1], ref_vel_xy[0])
            yref[5] = last_yaw # Set desired yaw

            yref[6:9] = ref_vel[j]  # Desired velocity

            # Reference for inputs (hover thrust)
            yref[15] = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]
            self._acados_ocp_solver.set(j, "yref", yref)

        # Set terminal reference
        yref_e = np.zeros((self._ny_e,))
        yref_e[0:3] = ref_pos[self._N]
        yref_e[5] = last_yaw
        yref_e[6:9] = ref_vel[self._N]
        self._acados_ocp_solver.set(self._N, "yref", yref_e)

        # Step 5: Solve the optimization problem
        self._acados_ocp_solver.solve()

        # Step 6: Get the first optimal control input
        u0 = self._acados_ocp_solver.get(0, "u")

        return u0

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the time step counter."""
        self._tick += 1
        if not self._finished and self._tick >= int(self._t_total * (1/self._dt)):
            self._finished = True
        return self._finished

    def episode_callback(self):
        """Reset the controller's internal state for the next episode."""
        self._tick = 0
        self._finished = False


    def render_callback(self, sim: Sim):
        """Visualize the planned trajectory."""
        # Draw the full planned trajectory in green
        trajectory_points = self._des_pos_spline(np.linspace(0, self._t_total, 200))
        draw_line(sim, trajectory_points, rgba=(0.0, 1.0, 0.0, 0.8))

        # Draw the upcoming reference points for the MPC horizon in red
        current_time = self._tick * self._dt
        if current_time <= self._t_total:
            t_horizon = np.linspace(current_time, current_time + self._T_HORIZON, self._N + 1)
            ref_pos = self._des_pos_spline(t_horizon)
            draw_points(sim, ref_pos, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
