"""Your controller — START HERE.

This is the only file you need to write for the challenge. Implement a subclass of ``Controller``
that, given the current observation, returns the next command for the drone. The same controller
runs in simulation and on the real drone.

This template just takes off and hovers (so it runs out of the box). Replace ``compute_control``
with your racing logic. Good starting points to copy from:
    - ``state_controllxer.py``  : trajectory tracking with state setpoints (easiest)
    - ``attitude_mpc.py``      : model predictive control with attitude/thrust commands
    - ``attitude_rl.py`` + ``train_rl.py`` : a trained RL policy

Key rule: the demo track is HELD OUT. Read the gate poses from ``obs`` at runtime
(``gates_pos``, ``gates_quat``, ``target_gate``) — do NOT hardcode gate coordinates, or you will
fail on the unseen track. See CHALLENGE.md.
"""
from __future__ import annotations

from this import d
from turtle import shape
from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control import Controller

from crazyflow.sim.visualize import draw_line, draw_points

from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from numpy.typing import NDArray

MODE = "sim"
SHOW_OBSTACLES = True


class MyController(Controller):
    """A minimal example controller (takes off and hovers). Replace with your racing logic."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the controller.

        Args:
            obs: Initial observation. Keys include ``pos``, ``quat`` (xyzw), ``vel``, ``ang_vel``,
                ``target_gate``, ``gates_pos``, ``gates_quat``, ``gates_visited``,
                ``obstacles_pos``, ``obstacles_visited``. Gate/obstacle poses are exact only within
                the sensor range, otherwise their nominal (config) pose is reported.
            info: Reset info.
            config: Race configuration (``config.env.freq`` is the control frequency, etc.).
        """
        super().__init__(obs, info, config)
        self.freq = config.env.freq
        # Hover target: 1 m above the start position. TODO: replace with a plan through the gates.
        self._hover = np.asarray(obs["pos"], dtype=np.float32).copy()
        self._hover[2] = 1.0

        # Parameters
        self.entrance_clearance = 0.4
        self.safe_height = 1.85
        self.safe_horizontal_waypoint_dist = 0.3
        self.safe_up_waypoint_dist = 0.2
        self.safe_down_waypoint_dist = 0.3
        self.safe_stop_dist = 0.1
        self.safe_stop_vel = 0.05

        # Mode
        self.mode = ["rising", "lowering", "crossing_gate", "navigating_to_gate"][0]

        # Initial observation
        self.initial_obs = obs
        self.target_gate = self.initial_obs["target_gate"]

        # General
        self.last_waypoint = obs["pos"]
        self.last_goal = obs["pos"]
        self.rising_hover_saved = np.asarray(obs["pos"], dtype=np.float32).copy()
        self.rising_hover_saved[2]=self.safe_height


    def next_target_gate(self, obs):
        """Gets the position and normal vector of the next target gate."""
        self.target_gate:int = obs["target_gate"]
        target_gate_pos = obs["gates_pos"][self.target_gate]
        gate_quat = obs["gates_quat"][self.target_gate]
        target_gate_normal = Rotation.from_quat(gate_quat).as_matrix() @ np.array([1, 0, 0])
        return target_gate_pos, target_gate_normal

    def next_gate_entrance(self, obs) -> NDArray[np.floating]:
        """Gets the position of the next gates entrance."""
        target_gate_pos, target_gate_normal = self.next_target_gate(obs)
        return target_gate_pos - target_gate_normal*self.entrance_clearance

    def next_gate_exit(self, obs) -> NDArray[np.floating]:
        """Gets the position of the next gates exit."""
        target_gate_pos, target_gate_normal = self.next_target_gate(obs)
        return target_gate_pos + target_gate_normal*self.entrance_clearance

    def get_waypoint(self, obs, goal):
        """Returns an achievable waypoint between the current position and the goal."""
        vec = goal - obs["pos"] # vector from the current position to the goal
        vec[0] = np.clip(vec[0], -self.safe_horizontal_waypoint_dist, self.safe_horizontal_waypoint_dist)
        vec[1] = np.clip(vec[1], -self.safe_horizontal_waypoint_dist, self.safe_horizontal_waypoint_dist)
        vec[2] = np.clip(vec[2], -self.safe_up_waypoint_dist, self.safe_down_waypoint_dist)
        self.last_waypoint = obs["pos"] + vec
        self.last_goal = goal
        if np.linalg.norm(obs["vel"])<self.safe_stop_vel:
            return obs["pos"] + vec*3
        return obs["pos"] + vec

    def waypoint_achieved(self, obs, goal) -> bool:
        """Returns whether the waypoint has been achieved."""
        if np.linalg.norm(obs["pos"]-goal)<self.safe_stop_dist and np.linalg.norm(obs["vel"])<self.safe_stop_vel:
            return True
        return False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Return the next command.

        With ``control_mode = "state"`` (default) return a 13-D state setpoint
        ``[x, y, z, vx, vy, vz, ax, ay, az, yaw, roll_rate, pitch_rate, yaw_rate]``.
        With ``control_mode = "attitude"`` return ``[collective_thrust, roll, pitch, yaw]``.

        Args:
            obs: Current observation (see ``__init__`` for the keys).
            info: Optional additional info.

        Returns:
            The command as a numpy array.
        """
        # TODO: your racing logic here. Use obs["target_gate"] and obs["gates_pos"]/["gates_quat"]
        #       to fly through the gates. For now we just hover at the start position.

        if self.mode == "rising":
            if self.waypoint_achieved(obs, self.rising_hover_saved):
                self.mode = "navigating_to_gate"
            return np.concatenate((self.get_waypoint(obs, self.rising_hover_saved), np.zeros(10)), dtype=np.float32)
        elif self.mode == "lowering":
            if self.waypoint_achieved(obs, self.next_gate_entrance(obs)):
                self.mode = "crossing_gate"
                self.next_gate_exit_saved = self.next_gate_exit(obs)
            return np.concatenate((self.get_waypoint(obs, self.next_gate_entrance(obs)), np.zeros(10)), dtype=np.float32)
        elif self.mode == "crossing_gate":
            if self.waypoint_achieved(obs, self.next_gate_exit_saved):
                self.mode = "rising"
                self.rising_hover_saved = np.asarray(obs["pos"], dtype=np.float32).copy()
                self.rising_hover_saved[2]=self.safe_height
            return np.concatenate((self.get_waypoint(obs, self.next_gate_exit_saved), np.zeros(10)), dtype=np.float32)
        elif self.mode == "navigating_to_gate":
            hover = np.asarray(self.next_gate_entrance(obs), dtype=np.float32).copy()
            hover[2]=self.safe_height
            if self.waypoint_achieved(obs, hover):
                self.mode = "lowering"

            print(f"height: {obs['pos'][2]}, hover height: {hover[2]}, get_waypoint_height: {self.get_waypoint(obs, hover)[2]}")
            return np.concatenate((self.get_waypoint(obs, hover), np.zeros(10)), dtype=np.float32)

        return np.concatenate((self._hover, np.zeros(10)), dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Called once after each step. Return True to signal the controller is finished."""
        return False

    def render_callback(self, sim: Sim):
        """Special rendering for the pathfinding algorithm.

        Currently renders the edges of the obstacles.
        """
        assert MODE == "sim"
        assert self.initial_obs is not None


        if SHOW_OBSTACLES:
            # Draw world edges

            draw_points(sim, \
                np.array([[[[x, y, z] for z in [0, 2.0]] for y in [-1.5, 1.5]] for x in [-2.5, 2.5]]).reshape(-1, 3), \
                rgba=(0.0, 1.0, 0.0, 1.0), size = 0.02)

            # Draw gate spheres
            #draw_points(sim, np.array(self.initial_obs["gates_pos"]), rgba=(1.0, 0.0, 0.0, 1.0), size=0.1)

            # Draw axes
            draw_line(sim, np.array([[0, 0, 0], [0.5, 0, 0], [1, 0, 0]]), rgba=(1.0, 0.0, 0.0, 1.0), start_size=5, end_size=5)
            draw_line(sim, np.array([[0, 0, 0], [0, 0.5, 0], [0, 1, 0]]), rgba=(0.0, 1.0, 0.0, 1.0), start_size=5, end_size=5)
            draw_line(sim, np.array([[0, 0, 0], [0, 0, 0.5], [0, 0, 1]]), rgba=(0.0, 0.0, 1.0, 1.0), start_size=5, end_size=5)

            # Draw normal vectors
            for i in range(len(self.initial_obs["gates_pos"])):
                gate_pos = self.initial_obs["gates_pos"][i]
                gate_quat = self.initial_obs["gates_quat"][i]
                gate_normal = Rotation.from_quat(gate_quat).as_matrix() @ np.array([1, 0, 0])
                draw_line(sim, np.array([gate_pos, gate_pos + gate_normal/4]), rgba=(0.0, 1.0, 0.0, 1.0), start_size=5, end_size=5)


            # Draw gate corners
            for i in range(len(self.initial_obs["gates_pos"])):
                gate_pos = self.initial_obs["gates_pos"][i]
                gate_quat = self.initial_obs["gates_quat"][i]
                gate_rot = Rotation.from_quat(gate_quat).as_matrix()
                gate_outer_corners = np.array([
                    gate_rot @ np.array([0, 1, 1]),
                    gate_rot @ np.array([0, -1, 1]),
                    gate_rot @ np.array([0, -1, -1]),
                    gate_rot @ np.array([0, 1, -1]),
                ]) * 0.4 / 2
                gate_inner_corners = gate_outer_corners / 0.4 * 0.72
                draw_points(sim, gate_outer_corners + gate_pos, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
                draw_points(sim, gate_inner_corners + gate_pos, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)

            # Draw the obstacles
            for i in range(len(self.initial_obs["obstacles_pos"])):
                obstacle_pos = self.initial_obs["obstacles_pos"][i]
                draw_line(sim, np.array([obstacle_pos + [0, 0, 0.2], [obstacle_pos[0], obstacle_pos[1], 0]]), rgba=(1.0, 0.0, 0.0, 1.0), start_size=100, end_size=100)

            # Draw the next gate entrance
            next_gate_entrance = self.next_gate_entrance(self.initial_obs)
            draw_points(sim, np.array([next_gate_entrance]), rgba=(0.0, 1.0, 0.0, 1.0), size=0.1)

            # Draw the next waypoint
            draw_points(sim, np.array([self.last_waypoint]), rgba=(0.0, 0.0, 1.0, 1.0), size=0.01)
            draw_points(sim, np.array([self.last_goal]), rgba=(1.0, 0.0, 0.0, 1.0), size=0.01)
            print(f"mode: {self.mode}")
