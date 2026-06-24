from __future__ import annotations

from typing import Any

import numpy as np

from lsy_drone_racing.control import Controller


class TubeVectorFieldController(Controller):
    """
    Tube-based vector-field controller for the drone racing task.

    Main idea:
        1. Build a smooth centerline tube through the active gate and a small
           lookahead of future gates.
        2. If the drone is outside the tube, command the fastest safe motion back
           into the tube.
        3. If the drone is inside the tube, flow forward along the tube while
           adding obstacle avoidance that is mostly lateral to the tube direction.

    This avoids hard waypoint hits, global path snapping, and gate-exit pullback.
    The controller uses state mode and returns the standard 13-D command:
        [x,y,z, vx,vy,vz, ax,ay,az, yaw, roll_rate,pitch_rate,yaw_rate]

    Config requirement:
        [env]
        control_mode = "state"

    Run example:
        python scripts/sim.py --config level2.toml --controller tube_vector_field_controller.py --render=True
    """

    def __init__(self, obs: dict[str, Any], info: dict | None, config: Any):
        super().__init__(obs, info, config)

        self.freq = int(getattr(config.env, "freq", 50))
        self.dt = 1.0 / max(float(self.freq), 1.0)
        self.tick = 0

        # ------------------------------------------------------------------
        # Tube geometry
        # ------------------------------------------------------------------
        self.min_z = 0.28
        self.takeoff_z = 0.48
        self.approach_dist = 0.62
        self.exit_dist = 0.72
        self.future_gate_lookahead = 1  # include active gate plus this many future gates

        # Tube radius. Around the gate opening it tightens; between gates it is
        # wider so the drone does not over-correct to an infinitesimal line.
        self.travel_tube_radius = 0.44
        self.gate_tube_radius = 0.18
        self.gate_tube_axial_radius = 0.72
        self.z_error_weight = 0.75

        # Field behavior outside/inside the tube.
        self.outside_pull_gain = 2.25
        self.outside_forward_gain = 0.22
        self.inside_centering_gain = 0.38
        self.inside_forward_gain = 1.15

        # Progress is monotonic-ish along the tube. This prevents a spatially
        # crossed tube from jumping to a later branch.
        self.progress_s = 0.0
        self.last_target_gate = int(np.asarray(obs.get("target_gate", 0)).reshape(()))
        self.progress_back_allowance = 0.10
        self.max_progress_advance_inside = 0.10
        self.max_progress_advance_outside = 0.045
        self.max_progress_advance_after_gate_change = 0.35
        self.just_changed_gate = True

        # ------------------------------------------------------------------
        # Obstacle behavior
        # ------------------------------------------------------------------
        self.use_obstacles = True
        self.use_nominal_obstacles = True
        self.obstacle_influence_seen = 0.70
        self.obstacle_influence_nominal = 0.58
        self.obstacle_core_radius = 0.22
        self.obstacle_lateral_gain = 1.45
        self.obstacle_swirl_gain = 0.58
        self.obstacle_vertical_gain = 0.10
        self.max_obstacle_field = 1.35
        self.obstacle_side_memory: dict[int, float] = {}

        # ------------------------------------------------------------------
        # State-command shaping
        # ------------------------------------------------------------------
        self.cruise_speed = 0.68
        self.outside_speed = 0.50
        self.gate_speed = 0.40
        self.obstacle_speed = 0.38
        self.max_speed = 0.82
        self.min_speed = 0.20

        self.command_horizon = 0.42
        self.inside_min_target_lead = 0.18
        self.max_target_step = 0.060
        self.max_velocity_step = 0.050
        self.max_accel_ff = 1.35
        self.yaw_step_limit = 0.10

        # Filtering. Sensor updates should bend the tube/field, not teleport it.
        self.gate_filter_alpha_nominal = 0.06
        self.gate_filter_alpha_sensor = 0.20
        self.obstacle_filter_alpha_nominal = 0.08
        self.obstacle_filter_alpha_sensor = 0.24
        self.field_alpha_inside = 0.32
        self.field_alpha_outside = 0.48

        # Runtime filtered world model.
        self.filtered_gate_pos: dict[int, np.ndarray] = {}
        self.filtered_gate_x: dict[int, np.ndarray] = {}
        self.filtered_gate_y: dict[int, np.ndarray] = {}
        self.filtered_gate_z: dict[int, np.ndarray] = {}
        self.filtered_obstacles: dict[int, np.ndarray] = {}

        pos0 = self._arr(obs["pos"], (3,))
        self.filtered_field = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.filtered_vel = np.zeros(3, dtype=np.float32)
        self.filtered_target = pos0.copy()
        self.last_yaw = self._yaw_from_quat(self._arr(obs.get("quat", [0, 0, 0, 1]), (4,)))

        # Debug rendering buffers.
        self.debug_centerline = np.empty((0, 3), dtype=np.float32)
        self.debug_projection = pos0.copy()
        self.debug_target = pos0.copy()
        self.debug_field_line = np.empty((0, 3), dtype=np.float32)
        self.debug_tube_status = np.empty((0, 3), dtype=np.float32)
        self.debug_obstacle_lines: list[np.ndarray] = []

    # ----------------------------------------------------------------------
    # Small math helpers
    # ----------------------------------------------------------------------
    @staticmethod
    def _arr(x: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if shape is not None:
            arr = arr.reshape(shape)
        return arr

    @staticmethod
    def _normalize(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n > 1e-8:
            return (v / n).astype(np.float32)
        if fallback is None:
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return np.asarray(fallback, dtype=np.float32)

    @staticmethod
    def _clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n > max_norm > 0.0:
            return (v * (max_norm / n)).astype(np.float32)
        return v.astype(np.float32)

    @staticmethod
    def _quat_rotate_xyzw(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
        q = np.asarray(q_xyzw, dtype=np.float32).copy()
        qn = float(np.linalg.norm(q))
        if qn < 1e-8:
            return np.asarray(v, dtype=np.float32).copy()
        q /= qn
        q_vec = q[:3]
        q_w = float(q[3])
        v = np.asarray(v, dtype=np.float32)
        t = 2.0 * np.cross(q_vec, v)
        return (v + q_w * t + np.cross(q_vec, t)).astype(np.float32)

    @staticmethod
    def _yaw_from_quat(q_xyzw: np.ndarray) -> float:
        x, y, z, w = [float(a) for a in q_xyzw]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    @staticmethod
    def _yaw_from_vector(v: np.ndarray, fallback: float = 0.0) -> float:
        vx, vy = float(v[0]), float(v[1])
        if vx * vx + vy * vy < 1e-8:
            return float(fallback)
        return float(np.arctan2(vy, vx))

    @staticmethod
    def _angle_wrap(a: float) -> float:
        return float((a + np.pi) % (2.0 * np.pi) - np.pi)

    @staticmethod
    def _smoothstep(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    @staticmethod
    def _remove_near_duplicates(points: np.ndarray, min_dist: float = 0.04) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32).reshape((-1, 3))
        if len(points) <= 1:
            return points.astype(np.float32)
        kept = [points[0]]
        for p in points[1:]:
            if float(np.linalg.norm(p - kept[-1])) >= min_dist:
                kept.append(p.astype(np.float32))
        return np.vstack(kept).astype(np.float32)

    # ----------------------------------------------------------------------
    # Filtered gate and obstacle model
    # ----------------------------------------------------------------------
    def _filtered_gate_frame(self, obs: dict[str, Any], gate_i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        gates_pos = self._arr(obs.get("gates_pos", []))
        gates_quat = self._arr(obs.get("gates_quat", []))
        gates_visited = np.asarray(obs.get("gates_visited", np.zeros((len(gates_pos),), dtype=bool))).astype(bool).reshape(-1)

        pos = gates_pos[gate_i].astype(np.float32).copy()
        q = gates_quat[gate_i].astype(np.float32).copy()
        pos[2] = max(float(pos[2]), self.min_z)

        x_axis = self._normalize(self._quat_rotate_xyzw(q, np.array([1.0, 0.0, 0.0], dtype=np.float32)))
        y_axis = self._normalize(
            self._quat_rotate_xyzw(q, np.array([0.0, 1.0, 0.0], dtype=np.float32)),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        z_axis = self._normalize(
            self._quat_rotate_xyzw(q, np.array([0.0, 0.0, 1.0], dtype=np.float32)),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )

        visited = bool(gate_i < len(gates_visited) and gates_visited[gate_i])
        alpha = self.gate_filter_alpha_sensor if visited else self.gate_filter_alpha_nominal

        if gate_i not in self.filtered_gate_pos:
            self.filtered_gate_pos[gate_i] = pos
            self.filtered_gate_x[gate_i] = x_axis
            self.filtered_gate_y[gate_i] = y_axis
            self.filtered_gate_z[gate_i] = z_axis
        else:
            self.filtered_gate_pos[gate_i] = ((1.0 - alpha) * self.filtered_gate_pos[gate_i] + alpha * pos).astype(np.float32)
            self.filtered_gate_x[gate_i] = self._normalize((1.0 - alpha) * self.filtered_gate_x[gate_i] + alpha * x_axis, x_axis)
            self.filtered_gate_y[gate_i] = self._normalize((1.0 - alpha) * self.filtered_gate_y[gate_i] + alpha * y_axis, y_axis)
            self.filtered_gate_z[gate_i] = self._normalize((1.0 - alpha) * self.filtered_gate_z[gate_i] + alpha * z_axis, z_axis)

        return (
            self.filtered_gate_pos[gate_i].copy(),
            self.filtered_gate_x[gate_i].copy(),
            self.filtered_gate_y[gate_i].copy(),
            self.filtered_gate_z[gate_i].copy(),
        )

    def _filtered_obstacle_positions(self, obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        obstacles = self._arr(obs.get("obstacles_pos", []))
        if obstacles.ndim != 2 or obstacles.shape[1] != 3 or len(obstacles) == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=bool)

        visited_raw = np.asarray(obs.get("obstacles_visited", np.ones((len(obstacles),), dtype=bool)))
        visited = visited_raw.astype(bool).reshape(-1)
        if len(visited) < len(obstacles):
            padded = np.zeros((len(obstacles),), dtype=bool)
            padded[: len(visited)] = visited
            visited = padded

        out = []
        for i, p in enumerate(obstacles):
            p = p.astype(np.float32).copy()
            p[2] = max(float(p[2]), self.min_z)
            alpha = self.obstacle_filter_alpha_sensor if bool(visited[i]) else self.obstacle_filter_alpha_nominal
            if i not in self.filtered_obstacles:
                self.filtered_obstacles[i] = p
            else:
                self.filtered_obstacles[i] = ((1.0 - alpha) * self.filtered_obstacles[i] + alpha * p).astype(np.float32)
            out.append(self.filtered_obstacles[i].copy())

        return np.vstack(out).astype(np.float32), visited[: len(obstacles)]

    # ----------------------------------------------------------------------
    # Tube geometry and projection
    # ----------------------------------------------------------------------
    def _build_tube_centerline(self, obs: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
        target_gate = int(np.asarray(obs.get("target_gate", 0)).reshape(()))
        gates_pos = self._arr(obs.get("gates_pos", []))
        gates_quat = self._arr(obs.get("gates_quat", []))

        if target_gate < 0 or gates_pos.ndim != 2 or gates_quat.ndim != 2:
            return np.empty((0, 3), dtype=np.float32), []

        n_gates = min(len(gates_pos), len(gates_quat))
        if target_gate >= n_gates:
            return np.empty((0, 3), dtype=np.float32), []

        points: list[np.ndarray] = []
        gate_infos: list[dict[str, Any]] = []
        last_exit: np.ndarray | None = None
        end_gate = min(n_gates, target_gate + 1 + self.future_gate_lookahead)

        for gate_i in range(target_gate, end_gate):
            center, gate_x, gate_y, gate_z = self._filtered_gate_frame(obs, gate_i)
            approach = center - self.approach_dist * gate_x
            exit_pt = center + self.exit_dist * gate_x
            approach[2] = max(float(approach[2]), self.min_z)
            exit_pt[2] = max(float(exit_pt[2]), self.min_z)

            # Connector: previous gate exit to next gate approach. This is still a
            # tube segment, not a waypoint to hit exactly.
            if last_exit is not None:
                points.append(approach.astype(np.float32))
            else:
                points.extend([approach.astype(np.float32), center.astype(np.float32), exit_pt.astype(np.float32)])

            if last_exit is not None:
                # After the connector point is added, append the actual gate tube.
                points.extend([center.astype(np.float32), exit_pt.astype(np.float32)])

            gate_infos.append(
                {
                    "i": int(gate_i),
                    "center": center,
                    "gate_x": gate_x,
                    "gate_y": gate_y,
                    "gate_z": gate_z,
                    "approach": approach,
                    "exit": exit_pt,
                }
            )
            last_exit = exit_pt

        return self._remove_near_duplicates(np.vstack(points).astype(np.float32), 0.04), gate_infos

    @staticmethod
    def _arc_lengths(path: np.ndarray) -> np.ndarray:
        if len(path) <= 1:
            return np.zeros((len(path),), dtype=np.float32)
        ds = np.linalg.norm(np.diff(path, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(ds)]).astype(np.float32)

    def _tube_metric_dist(self, pos: np.ndarray, point: np.ndarray) -> float:
        diff = (pos - point).astype(np.float32).copy()
        diff[2] *= float(self.z_error_weight)
        return float(np.linalg.norm(diff))

    def _nearest_gate_radius(self, pos: np.ndarray, gate_infos: list[dict[str, Any]]) -> float:
        if not gate_infos:
            return self.travel_tube_radius

        gate_weight = 0.0
        for info in gate_infos:
            center = info["center"]
            gate_x = info["gate_x"]
            rel = pos - center
            local_x = abs(float(np.dot(rel, gate_x)))
            w = 1.0 - self._smoothstep(local_x / max(self.gate_tube_axial_radius, 1e-6))
            gate_weight = max(gate_weight, w)
        radius = (1.0 - gate_weight) * self.travel_tube_radius + gate_weight * self.gate_tube_radius
        return float(radius)

    def _project_to_centerline(
        self,
        pos: np.ndarray,
        path: np.ndarray,
        path_s: np.ndarray,
        max_advance: float,
        allow_large_jump: bool = False,
    ) -> tuple[float, np.ndarray, np.ndarray, float]:
        if len(path) == 0:
            return 0.0, pos.copy(), np.array([1.0, 0.0, 0.0], dtype=np.float32), 0.0
        if len(path) == 1 or len(path_s) <= 1:
            d = self._tube_metric_dist(pos, path[0])
            return 0.0, path[0].copy(), np.array([1.0, 0.0, 0.0], dtype=np.float32), d

        min_s = max(0.0, float(self.progress_s) - self.progress_back_allowance)
        max_s = float(path_s[-1]) if allow_large_jump else min(float(path_s[-1]), float(self.progress_s) + max_advance)

        best_any: tuple[float, np.ndarray, np.ndarray, float] | None = None
        best_window: tuple[float, np.ndarray, np.ndarray, float] | None = None

        for k in range(len(path) - 1):
            a = path[k]
            b = path[k + 1]
            ab = b - a
            ab2 = float(np.dot(ab, ab))
            if ab2 < 1e-10:
                continue
            t = float(np.clip(np.dot(pos - a, ab) / ab2, 0.0, 1.0))
            p = (a + t * ab).astype(np.float32)
            seg_len = float(np.sqrt(ab2))
            s_here = float(path_s[k] + t * seg_len)
            tangent = self._normalize(ab)
            dist = self._tube_metric_dist(pos, p)
            item = (s_here, p, tangent, dist)

            if best_any is None or dist < best_any[3]:
                best_any = item
            if min_s <= s_here <= max_s:
                if best_window is None or dist < best_window[3]:
                    best_window = item

        chosen = best_window if best_window is not None else best_any
        if chosen is None:
            return 0.0, pos.copy(), np.array([1.0, 0.0, 0.0], dtype=np.float32), 0.0

        s_here, p, tangent, dist = chosen
        if allow_large_jump:
            self.progress_s = float(np.clip(s_here, 0.0, float(path_s[-1])))
        else:
            # Inside the tube, allow normal forward progress. Outside the tube,
            # the caller supplies a smaller max_advance so crossings cannot jump.
            self.progress_s = float(np.clip(max(self.progress_s - self.progress_back_allowance, s_here), 0.0, max_s))
            p, tangent = self._interpolate_path(path, path_s, self.progress_s)
            dist = self._tube_metric_dist(pos, p)
        return float(self.progress_s), p.astype(np.float32), tangent.astype(np.float32), float(dist)

    def _interpolate_path(self, path: np.ndarray, path_s: np.ndarray, s_query: float) -> tuple[np.ndarray, np.ndarray]:
        if len(path) == 0:
            return np.zeros(3, dtype=np.float32), np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if len(path) == 1 or len(path_s) <= 1:
            return path[0].copy(), np.array([1.0, 0.0, 0.0], dtype=np.float32)
        s_query = float(np.clip(s_query, 0.0, float(path_s[-1])))
        idx = int(np.searchsorted(path_s, s_query, side="right") - 1)
        idx = max(0, min(idx, len(path) - 2))
        s0 = float(path_s[idx])
        s1 = float(path_s[idx + 1])
        alpha = float(np.clip((s_query - s0) / max(s1 - s0, 1e-8), 0.0, 1.0))
        p = (1.0 - alpha) * path[idx] + alpha * path[idx + 1]
        tangent = self._normalize(path[idx + 1] - path[idx])
        return p.astype(np.float32), tangent.astype(np.float32)

    # ----------------------------------------------------------------------
    # Obstacle avoidance inside the tube
    # ----------------------------------------------------------------------
    def _obstacle_field(
        self,
        obs: dict[str, Any],
        pos: np.ndarray,
        tangent: np.ndarray,
        inside_tube: bool,
    ) -> tuple[np.ndarray, float, list[np.ndarray]]:
        if not self.use_obstacles:
            return np.zeros(3, dtype=np.float32), float("inf"), []

        obstacles, visited = self._filtered_obstacle_positions(obs)
        if len(obstacles) == 0:
            return np.zeros(3, dtype=np.float32), float("inf"), []

        tangent = self._normalize(tangent)
        field = np.zeros(3, dtype=np.float32)
        closest_dist = float("inf")
        debug: list[np.ndarray] = []
        world_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        for i, obs_pos in enumerate(obstacles):
            if not self.use_nominal_obstacles and not bool(visited[i]):
                continue

            rel_to_obs = pos - obs_pos
            ahead = float(np.dot(obs_pos - pos, tangent))
            # Ignore mostly-behind obstacles, especially once inside the tube.
            if inside_tube and ahead < -0.12:
                continue

            lateral = rel_to_obs - float(np.dot(rel_to_obs, tangent)) * tangent
            lateral[2] *= 0.70
            lateral_dist = float(np.linalg.norm(lateral))
            closest_dist = min(closest_dist, lateral_dist)

            influence = self.obstacle_influence_seen if bool(visited[i]) else self.obstacle_influence_nominal
            ahead_window = influence * 1.10
            if lateral_dist >= influence or ahead > ahead_window:
                continue

            if lateral_dist < 1e-5:
                side_vec = np.cross(world_z, tangent)
                if float(np.linalg.norm(side_vec)) < 1e-5:
                    side_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                side_vec = self._normalize(side_vec)
            else:
                side_vec = self._normalize(lateral)

            # Stable side choice. Do not let the avoidance side flip every step.
            mem = self.obstacle_side_memory.get(i)
            ref_side = np.cross(world_z, tangent)
            if float(np.linalg.norm(ref_side)) < 1e-5:
                ref_side = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            ref_side = self._normalize(ref_side)
            if mem is None:
                mem = 1.0 if float(np.dot(side_vec, ref_side)) >= 0.0 else -1.0
                self.obstacle_side_memory[i] = mem
            bypass = mem * ref_side

            closeness = float(np.clip((influence - lateral_dist) / max(influence - self.obstacle_core_radius, 1e-6), 0.0, 1.0))
            ahead_weight = 1.0 - self._smoothstep(max(0.0, ahead) / max(ahead_window, 1e-6))
            if ahead < 0.0:
                ahead_weight *= 0.35

            # Inside the tube, avoid laterally more than backwards. Outside the
            # tube, keep avoidance weaker so the controller first re-enters tube.
            tube_weight = 1.0 if inside_tube else 0.55
            repulse = side_vec * (self.obstacle_lateral_gain * closeness * closeness)
            swirl = bypass * (self.obstacle_swirl_gain * closeness * (1.0 - 0.3 * closeness))
            vertical = np.array([0.0, 0.0, self.obstacle_vertical_gain * closeness], dtype=np.float32)
            contrib = tube_weight * ahead_weight * (repulse + swirl + vertical)
            field += contrib.astype(np.float32)

            if float(np.linalg.norm(contrib)) > 1e-4:
                debug.append(np.vstack([obs_pos, obs_pos + 0.32 * self._normalize(contrib)]).astype(np.float32))

        return self._clip_norm(field, self.max_obstacle_field), closest_dist, debug

    def _speed(self, inside_tube: bool, tube_dist: float, tube_radius: float, dist_obstacle: float) -> float:
        speed = self.cruise_speed if inside_tube else self.outside_speed

        # Tighten speed near the gate opening. The radius shrinks there, so this
        # is a useful proxy for being in the gate-critical region.
        if tube_radius <= self.gate_tube_radius + 0.04:
            speed = min(speed, self.gate_speed)

        if dist_obstacle < self.obstacle_influence_seen:
            f = self._smoothstep(dist_obstacle / max(self.obstacle_influence_seen, 1e-6))
            speed = min(speed, self.obstacle_speed + (self.cruise_speed - self.obstacle_speed) * f)

        # If far outside, do not sprint diagonally through the course.
        if not inside_tube:
            excess = max(0.0, tube_dist - tube_radius)
            if excess > 0.25:
                speed = min(speed, 0.42)

        return float(np.clip(speed, self.min_speed, self.max_speed))

    # ----------------------------------------------------------------------
    # Main controller
    # ----------------------------------------------------------------------
    def compute_control(self, obs: dict[str, Any], info: dict | None = None) -> np.ndarray:
        pos = self._arr(obs["pos"], (3,))
        target_gate = int(np.asarray(obs.get("target_gate", 0)).reshape(()))

        cmd = np.zeros(13, dtype=np.float32)
        if target_gate < 0:
            cmd[0:3] = pos
            cmd[3:6] = 0.0
            cmd[6:9] = 0.0
            cmd[9] = self.last_yaw
            self.filtered_target = pos.copy()
            self.filtered_vel[:] = 0.0
            return cmd

        if target_gate != self.last_target_gate:
            self.last_target_gate = target_gate
            self.progress_s = 0.0
            self.just_changed_gate = True
            self.obstacle_side_memory.clear()

        if float(pos[2]) < self.takeoff_z - 0.08:
            # Simple vertical takeoff before entering the tube field.
            target = pos.copy()
            target[2] = self.takeoff_z
            direction = self._normalize(target - pos, np.array([0.0, 0.0, 1.0], dtype=np.float32))
            speed = 0.34
            inside_tube = False
            tube_radius = self.travel_tube_radius
            tube_dist = 0.0
            projection = pos.copy()
            tangent = direction.copy()
            centerline = np.vstack([pos, target]).astype(np.float32)
            obs_field = np.zeros(3, dtype=np.float32)
            dist_obstacle = float("inf")
            obs_debug = []
        else:
            centerline, gate_infos = self._build_tube_centerline(obs)
            path_s = self._arc_lengths(centerline)
            if len(centerline) < 2:
                cmd[0:3] = pos
                cmd[9] = self.last_yaw
                return cmd

            # First get a coarse projection without advancing too much. Determine
            # inside/outside, then use the appropriate cap.
            if self.just_changed_gate:
                max_adv = self.max_progress_advance_after_gate_change
                allow_large = True
                self.just_changed_gate = False
            else:
                max_adv = self.max_progress_advance_inside
                allow_large = False
            _s0, projection0, tangent0, tube_dist0 = self._project_to_centerline(pos, centerline, path_s, max_adv, allow_large)
            tube_radius = self._nearest_gate_radius(projection0, gate_infos)
            inside_tube0 = tube_dist0 <= tube_radius

            max_adv = self.max_progress_advance_inside if inside_tube0 else self.max_progress_advance_outside
            _s, projection, tangent, tube_dist = self._project_to_centerline(pos, centerline, path_s, max_adv, False)
            tube_radius = self._nearest_gate_radius(projection, gate_infos)
            inside_tube = tube_dist <= tube_radius

            to_tube = projection - pos
            to_tube_weighted = to_tube.copy()
            to_tube_weighted[2] *= self.z_error_weight
            to_tube_dir = self._normalize(to_tube_weighted, np.zeros(3, dtype=np.float32))

            obs_field, dist_obstacle, obs_debug = self._obstacle_field(obs, pos, tangent, inside_tube)

            if inside_tube:
                # Inside: mostly forward flow, mild centering, plus obstacle bypass.
                centering_mag = float(np.clip(tube_dist / max(tube_radius, 1e-6), 0.0, 1.0))
                raw_field = (
                    self.inside_forward_gain * tangent
                    + self.inside_centering_gain * centering_mag * to_tube_dir
                    + obs_field
                )
            else:
                # Outside: re-enter the tube ASAP. Obstacle avoidance is included
                # but secondary to the normal correction back to the tube.
                raw_field = (
                    self.outside_pull_gain * to_tube_dir
                    + self.outside_forward_gain * tangent
                    + 0.45 * obs_field
                )

            direction = self._normalize(raw_field, tangent)
            field_alpha = self.field_alpha_inside if inside_tube else self.field_alpha_outside
            self.filtered_field = self._normalize((1.0 - field_alpha) * self.filtered_field + field_alpha * direction, direction)
            speed = self._speed(inside_tube, tube_dist, tube_radius, dist_obstacle)

        # Desired velocity with rate limit.
        desired_vel_raw = speed * self.filtered_field
        prev_vel = self.filtered_vel.copy()
        self.filtered_vel = (self.filtered_vel + self._clip_norm(desired_vel_raw - self.filtered_vel, self.max_velocity_step)).astype(np.float32)
        desired_acc = self._clip_norm((self.filtered_vel - prev_vel) / max(self.dt, 1e-6), self.max_accel_ff)

        raw_target = pos + self.command_horizon * self.filtered_vel
        raw_target[2] = max(float(raw_target[2]), self.min_z)
        delta_target = raw_target - self.filtered_target
        self.filtered_target = (self.filtered_target + self._clip_norm(delta_target, self.max_target_step)).astype(np.float32)
        self.filtered_target[2] = max(float(self.filtered_target[2]), self.min_z)

        # If inside the tube, never let the smoothed position setpoint trail behind
        # the flow. Otherwise the inner state controller can pull backward even
        # though the velocity vector points forward.
        if inside_tube:
            lead = float(np.dot(self.filtered_target - pos, tangent))
            if lead < self.inside_min_target_lead:
                self.filtered_target = (self.filtered_target + (self.inside_min_target_lead - lead) * tangent).astype(np.float32)
                self.filtered_target[2] = max(float(self.filtered_target[2]), self.min_z)

        yaw_raw = self._yaw_from_vector(self.filtered_vel, self.last_yaw)
        yaw_err = self._angle_wrap(yaw_raw - self.last_yaw)
        yaw = self._angle_wrap(self.last_yaw + float(np.clip(yaw_err, -self.yaw_step_limit, self.yaw_step_limit)))
        self.last_yaw = yaw

        cmd[0:3] = self.filtered_target.astype(np.float32)
        cmd[3:6] = self.filtered_vel.astype(np.float32)
        cmd[6:9] = desired_acc.astype(np.float32)
        cmd[9] = np.float32(yaw)
        cmd[10:13] = 0.0

        # Debug geometry.
        self.debug_centerline = centerline.astype(np.float32)
        self.debug_projection = projection.astype(np.float32)
        self.debug_target = self.filtered_target.copy()
        self.debug_field_line = np.vstack([pos, pos + 0.55 * self.filtered_field]).astype(np.float32)
        # Two points encode tube status/radius: projection and one radius marker.
        marker = projection + tube_radius * self._normalize(np.cross(tangent, np.array([0.0, 0.0, 1.0], dtype=np.float32)), np.array([0.0, 1.0, 0.0], dtype=np.float32))
        self.debug_tube_status = np.vstack([projection, marker]).astype(np.float32)
        self.debug_obstacle_lines = obs_debug

        return cmd.astype(np.float32)

    def step_callback(
        self,
        action: np.ndarray,
        obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict | None,
    ) -> bool:
        self.tick += 1
        return False

    def episode_reset(self) -> None:
        self.tick = 0
        self.progress_s = 0.0
        self.filtered_gate_pos.clear()
        self.filtered_gate_x.clear()
        self.filtered_gate_y.clear()
        self.filtered_gate_z.clear()
        self.filtered_obstacles.clear()
        self.obstacle_side_memory.clear()
        self.filtered_field = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.filtered_vel[:] = 0.0

    # ----------------------------------------------------------------------
    # Debug rendering
    # ----------------------------------------------------------------------
    def render_callback(self, sim: Any) -> None:
        try:
            from crazyflow.sim.visualize import draw_line, draw_points
        except Exception:
            return

        try:
            # Tube centerline.
            if len(self.debug_centerline) >= 2:
                draw_line(sim, np.asarray(self.debug_centerline, dtype=np.float32), rgba=(0.0, 0.85, 1.0, 1.0))

            # Projection onto tube and approximate radius marker.
            draw_points(sim, np.asarray(self.debug_projection, dtype=np.float32).reshape(1, 3), rgba=(1.0, 1.0, 1.0, 1.0), size=0.035)
            if len(self.debug_tube_status) >= 2:
                draw_line(sim, np.asarray(self.debug_tube_status, dtype=np.float32), rgba=(1.0, 1.0, 1.0, 1.0))

            # Commanded state target.
            draw_points(sim, np.asarray(self.debug_target, dtype=np.float32).reshape(1, 3), rgba=(1.0, 0.0, 0.0, 1.0), size=0.045)

            # Current field direction.
            if len(self.debug_field_line) >= 2:
                draw_line(sim, np.asarray(self.debug_field_line, dtype=np.float32), rgba=(0.0, 1.0, 0.0, 1.0))

            # Obstacle push vectors.
            for line in self.debug_obstacle_lines:
                if len(line) >= 2:
                    draw_line(sim, np.asarray(line, dtype=np.float32), rgba=(1.0, 0.0, 1.0, 1.0))
        except Exception:
            return
