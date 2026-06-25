from __future__ import annotations

from typing import Any

import numpy as np

from lsy_drone_racing.control import Controller


class VectorFieldGateExitLockController(Controller):
    """
    Smooth vector-field controller for the drone racing task.

    This deliberately avoids explicit path replanning. Every control step builds a
    desired velocity direction from continuous vector-field terms:

        gate funnel field     -> pass current gate from -local x to +local x
        obstacle field        -> repulsion + tangential swirl around obstacles
        next-gate preview     -> weak bias after the current gate
        smoothing/rate limits -> no target teleporting when sensor data updates

    The controller still uses state mode and returns the standard 13-D command:
        [x,y,z, vx,vy,vz, ax,ay,az, yaw, roll_rate,pitch_rate,yaw_rate]

    Config requirement:
        [env]
        control_mode = "state"

    Run example:
        python scripts/sim.py --config level2.toml --controller vector_field_gate_exit_lock_controller.py --render=True
    """

    def __init__(self, obs: dict[str, Any], info: dict | None, config: Any):
        super().__init__(obs, info, config)

        self.freq = int(getattr(config.env, "freq", 50))
        self.dt = 1.0 / max(float(self.freq), 1.0)
        self.tick = 0

        # ------------------------------------------------------------------
        # Gate field tuning
        # ------------------------------------------------------------------
        self.min_z = 0.28
        self.takeoff_z = 0.48
        self.approach_dist = 0.58
        self.exit_dist = 1.62

        # Gate funnel: near the gate, follow gate_x while correcting lateral
        # error to the gate centerline. Larger values make it line up harder.
        self.funnel_lateral_gain_far = 1.15
        self.funnel_lateral_gain_near = 2.35
        self.funnel_forward_gain = 1.00
        self.approach_forward_bias = 0.25
        self.next_gate_preview_gain = 0.20

        # The gate opening is small, so slow down and tighten the funnel close to
        # the gate plane.
        self.gate_slow_radius = 0.85
        self.gate_tight_radius = 0.48

        # Exit lock: after crossing the gate plane, commit to the old gate's
        # +local-x direction until the drone is safely clear. This prevents the
        # vector field or a lagging state target from pulling the drone back
        # through the gate immediately after a valid-looking pass.
        self.exit_lock_enabled = True
        self.exit_lock_enter_x = 0.05
        self.exit_lock_release_x = self.exit_dist + 0.24
        self.exit_lock_max_x = self.exit_dist + 0.72
        self.exit_lock_lateral_radius = 0.34
        self.exit_lock_lateral_gain = 0.95
        self.exit_lock_min_steps = max(4, int(0.20 * self.freq))
        self.exit_lock_max_steps = max(20, int(1.10 * self.freq))
        self.exit_lock_speed = 0.48
        self.exit_lock_min_forward_speed = 0.30
        self.exit_lock_target_lead = 0.24
        self.exit_lock_field_alpha = 0.68

        # ------------------------------------------------------------------
        # Obstacle vector-field tuning
        # ------------------------------------------------------------------
        self.use_obstacles = True
        self.use_nominal_obstacles = True
        self.obstacle_influence_seen = 0.78
        self.obstacle_influence_nominal = 0.62
        self.obstacle_core_radius = 0.23
        self.obstacle_repulse_gain = 1.15
        self.obstacle_swirl_gain = 0.72
        self.obstacle_vertical_gain = 0.18
        self.max_obstacle_field = 1.45

        # ------------------------------------------------------------------
        # State-command shaping
        # ------------------------------------------------------------------
        self.cruise_speed = 0.70
        self.gate_speed = 0.42
        self.tight_gate_speed = 0.34
        self.obstacle_speed = 0.40
        self.max_speed = 0.85
        self.min_speed = 0.20

        self.command_horizon = 0.46       # desired position = pos + horizon * v_des
        self.max_target_step = 0.055      # target position slew per control step
        self.max_velocity_step = 0.045    # velocity setpoint slew per control step
        self.max_accel_ff = 1.40
        self.yaw_step_limit = 0.10

        # Low-pass for field/data updates. These are intentionally not too small:
        # the field should react, but it should not snap.
        self.field_alpha = 0.28
        self.gate_filter_alpha_nominal = 0.06
        self.gate_filter_alpha_sensor = 0.18
        self.obstacle_filter_alpha_nominal = 0.08
        self.obstacle_filter_alpha_sensor = 0.22

        # Runtime memory.
        self.last_target_gate = int(np.asarray(obs.get("target_gate", 0)).reshape(()))
        self.prev_gate_local_x: dict[int, float] = {}
        self.exit_lock_active = False
        self.exit_lock_gate: int | None = None
        self.exit_lock_start_tick = 0
        self.exit_lock_center: np.ndarray | None = None
        self.exit_lock_x: np.ndarray | None = None
        self.exit_lock_y: np.ndarray | None = None
        self.exit_lock_z: np.ndarray | None = None
        self.filtered_gate_pos: dict[int, np.ndarray] = {}
        self.filtered_gate_x: dict[int, np.ndarray] = {}
        self.filtered_gate_y: dict[int, np.ndarray] = {}
        self.filtered_gate_z: dict[int, np.ndarray] = {}
        self.filtered_obstacles: dict[int, np.ndarray] = {}
        self.obstacle_side_memory: dict[int, float] = {}

        pos0 = self._arr(obs["pos"], (3,))
        self.filtered_field = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.filtered_target = pos0.copy()
        self.filtered_vel = np.zeros(3, dtype=np.float32)
        self.filtered_acc = np.zeros(3, dtype=np.float32)
        self.last_yaw = self._yaw_from_quat(self._arr(obs.get("quat", [0, 0, 0, 1]), (4,)))

        # Debug rendering buffers.
        self.debug_field_line = np.empty((0, 3), dtype=np.float32)
        self.debug_gate_line = np.empty((0, 3), dtype=np.float32)
        self.debug_gate_points = np.empty((0, 3), dtype=np.float32)
        self.debug_obstacle_lines: list[np.ndarray] = []
        self.debug_target = pos0.copy()
        self.debug_raw_field = self.filtered_field.copy()

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

    # ----------------------------------------------------------------------
    # Exit-lock helpers
    # ----------------------------------------------------------------------
    def _start_exit_lock(
        self,
        gate_i: int,
        center: np.ndarray,
        gate_x: np.ndarray,
        gate_y: np.ndarray,
        gate_z: np.ndarray,
    ) -> None:
        if not self.exit_lock_enabled:
            return
        self.exit_lock_active = True
        self.exit_lock_gate = int(gate_i)
        self.exit_lock_start_tick = int(self.tick)
        self.exit_lock_center = center.astype(np.float32).copy()
        self.exit_lock_x = self._normalize(gate_x).astype(np.float32)
        self.exit_lock_y = self._normalize(gate_y, np.array([0.0, 1.0, 0.0], dtype=np.float32)).astype(np.float32)
        self.exit_lock_z = self._normalize(gate_z, np.array([0.0, 0.0, 1.0], dtype=np.float32)).astype(np.float32)

        # Remove any backward component immediately. The later target guard makes
        # sure the commanded state setpoint is also in front of the drone.
        self.filtered_field = self.exit_lock_x.copy()
        v_forward = float(np.dot(self.filtered_vel, self.exit_lock_x))
        if v_forward < self.exit_lock_min_forward_speed:
            lateral_v = self.filtered_vel - v_forward * self.exit_lock_x
            self.filtered_vel = (
                self.exit_lock_min_forward_speed * self.exit_lock_x + 0.25 * lateral_v
            ).astype(np.float32)

    def _stop_exit_lock(self) -> None:
        self.exit_lock_active = False
        self.exit_lock_gate = None
        self.exit_lock_center = None
        self.exit_lock_x = None
        self.exit_lock_y = None
        self.exit_lock_z = None

    def _exit_lock_field(
        self,
        obs: dict[str, Any],
        pos: np.ndarray,
    ) -> tuple[np.ndarray, float, dict[str, Any]] | None:
        if not self.exit_lock_active:
            return None
        if self.exit_lock_center is None or self.exit_lock_x is None or self.exit_lock_y is None or self.exit_lock_z is None:
            self._stop_exit_lock()
            return None

        target_gate = int(np.asarray(obs.get("target_gate", 0)).reshape(()))
        center = self.exit_lock_center.copy()
        gate_x = self.exit_lock_x.copy()
        gate_y = self.exit_lock_y.copy()
        gate_z = self.exit_lock_z.copy()
        rel = pos - center
        local_x = float(np.dot(rel, gate_x))
        local_y = float(np.dot(rel, gate_y))
        local_z = float(np.dot(rel, gate_z))
        lateral = local_y * gate_y + local_z * gate_z
        age = int(self.tick - self.exit_lock_start_tick)

        # Release once the simulator has advanced to the next gate AND we have
        # moved a little beyond the old gate. If target_gate has not advanced yet,
        # keep coasting forward instead of pulling back to the old center/exit.
        advanced_to_next_gate = self.exit_lock_gate is not None and target_gate != int(self.exit_lock_gate)
        safely_clear = local_x > self.exit_lock_release_x and age >= self.exit_lock_min_steps
        timed_out_far = age >= self.exit_lock_max_steps or local_x > self.exit_lock_max_x
        if advanced_to_next_gate and safely_clear:
            self._stop_exit_lock()
            return None
        if timed_out_far:
            # If the gate was not counted, do not lock forever. Releasing here lets
            # the normal field recover/retry rather than flying away indefinitely.
            self._stop_exit_lock()
            return None

        correction = -self.exit_lock_lateral_gain * lateral
        field = self._normalize(gate_x + correction, gate_x)
        exit_pt = center + self.exit_dist * gate_x
        exit_pt[2] = max(float(exit_pt[2]), self.min_z)
        approach = center - self.approach_dist * gate_x
        approach[2] = max(float(approach[2]), self.min_z)
        return field.astype(np.float32), float(np.linalg.norm(pos - center)), {
            "center": center,
            "gate_x": gate_x,
            "gate_y": gate_y,
            "gate_z": gate_z,
            "approach": approach,
            "exit": exit_pt,
            "local_x": local_x,
            "local_y": local_y,
            "local_z": local_z,
            "mode": "exit_lock",
        }

    def _guard_exit_lock_target(self, pos: np.ndarray) -> None:
        if not self.exit_lock_active or self.exit_lock_x is None:
            return
        gate_x = self.exit_lock_x
        lead = float(np.dot(self.filtered_target - pos, gate_x))
        if lead < self.exit_lock_target_lead:
            self.filtered_target = (
                self.filtered_target + (self.exit_lock_target_lead - lead) * gate_x
            ).astype(np.float32)
            self.filtered_target[2] = max(float(self.filtered_target[2]), self.min_z)


    # ----------------------------------------------------------------------
    # Filtered world model: new exact data bends the field instead of snapping
    # ----------------------------------------------------------------------
    def _filtered_gate_frame(self, obs: dict[str, Any], gate_i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        gates_pos = self._arr(obs.get("gates_pos", []))
        gates_quat = self._arr(obs.get("gates_quat", []))
        gates_visited = np.asarray(obs.get("gates_visited", np.zeros((len(gates_pos),), dtype=bool))).astype(bool).reshape(-1)

        pos = gates_pos[gate_i].astype(np.float32).copy()
        q = gates_quat[gate_i].astype(np.float32).copy()
        pos[2] = max(float(pos[2]), self.min_z)

        x_axis = self._normalize(self._quat_rotate_xyzw(q, np.array([1.0, 0.0, 0.0], dtype=np.float32)))
        y_axis = self._normalize(self._quat_rotate_xyzw(q, np.array([0.0, 1.0, 0.0], dtype=np.float32)), np.array([0, 1, 0], dtype=np.float32))
        z_axis = self._normalize(self._quat_rotate_xyzw(q, np.array([0.0, 0.0, 1.0], dtype=np.float32)), np.array([0, 0, 1], dtype=np.float32))

        visited = bool(gate_i < len(gates_visited) and gates_visited[gate_i])
        alpha = self.gate_filter_alpha_sensor if visited else self.gate_filter_alpha_nominal

        if gate_i not in self.filtered_gate_pos:
            self.filtered_gate_pos[gate_i] = pos
            self.filtered_gate_x[gate_i] = x_axis
            self.filtered_gate_y[gate_i] = y_axis
            self.filtered_gate_z[gate_i] = z_axis
        else:
            self.filtered_gate_pos[gate_i] = ((1.0 - alpha) * self.filtered_gate_pos[gate_i] + alpha * pos).astype(np.float32)
            self.filtered_gate_x[gate_i] = self._normalize((1.0 - alpha) * self.filtered_gate_x[gate_i] + alpha * x_axis)
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
            visited_padded = np.zeros((len(obstacles),), dtype=bool)
            visited_padded[: len(visited)] = visited
            visited = visited_padded

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
    # Vector-field terms
    # ----------------------------------------------------------------------
    def _gate_field(self, obs: dict[str, Any], pos: np.ndarray) -> tuple[np.ndarray, float, dict[str, Any]]:
        target_gate = int(np.asarray(obs.get("target_gate", 0)).reshape(()))
        gates_pos = self._arr(obs.get("gates_pos", []))
        gates_quat = self._arr(obs.get("gates_quat", []))

        if target_gate < 0 or gates_pos.ndim != 2 or target_gate >= len(gates_pos):
            return np.zeros(3, dtype=np.float32), 0.0, {"finished": True}
        if gates_quat.ndim != 2 or target_gate >= len(gates_quat):
            return np.zeros(3, dtype=np.float32), 0.0, {"finished": True}

        # If we recently crossed a gate, keep using the old gate frame until we
        # are safely clear. This is intentionally checked before reading the
        # current target gate frame, because target_gate may already have advanced.
        locked = self._exit_lock_field(obs, pos)
        if locked is not None:
            return locked

        center, gate_x, gate_y, gate_z = self._filtered_gate_frame(obs, target_gate)
        rel = pos - center
        local_x = float(np.dot(rel, gate_x))
        local_y = float(np.dot(rel, gate_y))
        local_z = float(np.dot(rel, gate_z))
        lateral = local_y * gate_y + local_z * gate_z

        approach = center - self.approach_dist * gate_x
        exit_pt = center + self.exit_dist * gate_x
        approach[2] = max(float(approach[2]), self.min_z)
        exit_pt[2] = max(float(exit_pt[2]), self.min_z)

        # Start exit lock when the drone crosses the gate plane from the entry
        # side to the exit side while reasonably close to the opening centerline.
        prev_local_x = self.prev_gate_local_x.get(target_gate, local_x)
        lateral_norm = float(np.sqrt(local_y * local_y + local_z * local_z))
        crossed_forward = prev_local_x <= self.exit_lock_enter_x < local_x
        self.prev_gate_local_x[target_gate] = local_x
        if (
            self.exit_lock_enabled
            and not self.exit_lock_active
            and crossed_forward
            and lateral_norm <= self.exit_lock_lateral_radius
        ):
            self._start_exit_lock(target_gate, center, gate_x, gate_y, gate_z)
            locked = self._exit_lock_field(obs, pos)
            if locked is not None:
                return locked

        # Initial takeoff helper: do not sprint horizontally when still low.
        if float(pos[2]) < self.takeoff_z - 0.08:
            takeoff = pos.copy()
            takeoff[2] = self.takeoff_z
            field = self._normalize(takeoff - pos, np.array([0.0, 0.0, 1.0], dtype=np.float32))
            return field, float(np.linalg.norm(pos - center)), {
                "center": center,
                "gate_x": gate_x,
                "approach": approach,
                "exit": exit_pt,
                "local_x": local_x,
                "mode": "takeoff",
            }

        # Phase 1: outside the gate funnel. Move toward the approach point, with
        # a weak forward bias so the field is already aligned with the crossing.
        if local_x < -self.approach_dist:
            to_approach = approach - pos
            field = self._normalize(to_approach) + self.approach_forward_bias * gate_x
            field = self._normalize(field)
            mode = "approach"
        else:
            # Phase 2: gate funnel. Move along +gate_x while correcting lateral
            # error to the gate centerline. This is the vector-field replacement
            # for approach-center-exit waypoints.
            gate_plane_weight = 1.0 - self._smoothstep(abs(local_x) / max(self.approach_dist, 1e-6))
            lateral_gain = (1.0 - gate_plane_weight) * self.funnel_lateral_gain_far + gate_plane_weight * self.funnel_lateral_gain_near
            correction = -lateral_gain * lateral

            # If still before the plane, also gently pull toward the center. If
            # just after the plane, pull toward the exit. Once past the exit point,
            # never pull backwards to it; keep coasting +gate_x until target_gate
            # advances or the exit lock releases.
            if local_x < 0.0:
                axial_bias = 0.25 * self._normalize(center - pos, gate_x)
            elif local_x < self.exit_dist:
                axial_bias = 0.35 * self._normalize(exit_pt - pos, gate_x)
            else:
                axial_bias = np.zeros(3, dtype=np.float32)

            field = self.funnel_forward_gain * gate_x + correction + axial_bias
            field = self._normalize(field, gate_x)
            mode = "funnel"

        # Weak preview of the next gate after we are almost through the current one.
        next_i = target_gate + 1
        if next_i < len(gates_pos) and local_x > 0.10:
            next_center, _nx, _ny, _nz = self._filtered_gate_frame(obs, next_i)
            preview = self._normalize(next_center - pos, gate_x)
            field = self._normalize(field + self.next_gate_preview_gain * preview, gate_x)

        dist_gate = float(np.linalg.norm(pos - center))
        return field.astype(np.float32), dist_gate, {
            "center": center,
            "gate_x": gate_x,
            "gate_y": gate_y,
            "gate_z": gate_z,
            "approach": approach,
            "exit": exit_pt,
            "local_x": local_x,
            "local_y": local_y,
            "local_z": local_z,
            "mode": mode,
        }

    def _obstacle_field(self, obs: dict[str, Any], pos: np.ndarray, nominal_direction: np.ndarray) -> tuple[np.ndarray, float, list[np.ndarray]]:
        if not self.use_obstacles:
            return np.zeros(3, dtype=np.float32), float("inf"), []

        obstacles, visited = self._filtered_obstacle_positions(obs)
        if len(obstacles) == 0:
            return np.zeros(3, dtype=np.float32), float("inf"), []

        field = np.zeros(3, dtype=np.float32)
        closest_dist = float("inf")
        debug_lines: list[np.ndarray] = []
        nom_dir = self._normalize(nominal_direction)

        for i, obs_pos in enumerate(obstacles):
            if not self.use_nominal_obstacles and not bool(visited[i]):
                continue

            rel = pos - obs_pos
            rel_xy = rel[:2]
            d_xy = float(np.linalg.norm(rel_xy))
            closest_dist = min(closest_dist, d_xy)
            influence = self.obstacle_influence_seen if bool(visited[i]) else self.obstacle_influence_nominal
            if d_xy >= influence:
                continue

            # Do not overreact to obstacles behind the drone relative to desired motion.
            obs_ahead = float(np.dot(obs_pos - pos, nom_dir))
            ahead_weight = 0.35 + 0.65 * self._smoothstep((obs_ahead + 0.15) / influence)

            if d_xy < 1e-5:
                away_xy = np.array([-nom_dir[1], nom_dir[0]], dtype=np.float32)
                d_xy = 1e-5
            else:
                away_xy = rel_xy / d_xy

            # Smooth repulsion; very strong close to the core radius.
            closeness = float(np.clip((influence - d_xy) / max(influence - self.obstacle_core_radius, 1e-6), 0.0, 1.0))
            repulse_xy = away_xy * (self.obstacle_repulse_gain * closeness * closeness)

            # Tangential swirl prevents local minima directly in front of obstacles.
            side = self.obstacle_side_memory.get(i)
            if side is None:
                # Pick the side whose tangent has positive projection on the gate field.
                tangent_plus = np.array([-away_xy[1], away_xy[0]], dtype=np.float32)
                side = 1.0 if float(np.dot(tangent_plus, nom_dir[:2])) >= 0.0 else -1.0
                self.obstacle_side_memory[i] = side
            tangent_xy = side * np.array([-away_xy[1], away_xy[0]], dtype=np.float32)
            swirl_xy = tangent_xy * (self.obstacle_swirl_gain * closeness * (1.0 - 0.35 * closeness))

            z_term = 0.0
            if d_xy < self.obstacle_core_radius + 0.08:
                z_term = self.obstacle_vertical_gain * (1.0 - d_xy / max(self.obstacle_core_radius + 0.08, 1e-6))

            contrib = np.array([repulse_xy[0] + swirl_xy[0], repulse_xy[1] + swirl_xy[1], z_term], dtype=np.float32)
            contrib *= ahead_weight
            field += contrib

            # Debug: line from obstacle to direction of its local push.
            if float(np.linalg.norm(contrib)) > 1e-4:
                debug_lines.append(np.vstack([obs_pos, obs_pos + 0.35 * self._normalize(contrib)]).astype(np.float32))

        field = self._clip_norm(field, self.max_obstacle_field)
        return field.astype(np.float32), closest_dist, debug_lines

    def _speed(self, dist_gate: float, dist_obstacle: float) -> float:
        speed = self.cruise_speed
        if dist_gate < self.gate_slow_radius:
            speed = min(speed, self.gate_speed)
        if dist_gate < self.gate_tight_radius:
            speed = min(speed, self.tight_gate_speed)
        if dist_obstacle < self.obstacle_influence_seen:
            # Slow down smoothly near obstacles.
            obstacle_factor = self._smoothstep(dist_obstacle / max(self.obstacle_influence_seen, 1e-6))
            speed = min(speed, self.obstacle_speed + (self.cruise_speed - self.obstacle_speed) * obstacle_factor)
        return float(np.clip(speed, self.min_speed, self.max_speed))

    # ----------------------------------------------------------------------
    # Main controller
    # ----------------------------------------------------------------------
    def compute_control(self, obs: dict[str, Any], info: dict | None = None) -> np.ndarray:
        pos = self._arr(obs["pos"], (3,))
        target_gate = int(np.asarray(obs.get("target_gate", 0)).reshape(()))

        cmd = np.zeros(13, dtype=np.float32)
        if target_gate < 0:
            # Finished: hold current position.
            cmd[0:3] = pos
            cmd[3:6] = 0.0
            cmd[6:9] = 0.0
            cmd[9] = self.last_yaw
            self.filtered_target = pos.copy()
            self.filtered_vel[:] = 0.0
            return cmd

        if target_gate != self.last_target_gate:
            # Allow gate transitions to reset obstacle side choices, but keep the
            # filtered field/target so the command is still continuous.
            self.obstacle_side_memory.clear()
            self.last_target_gate = target_gate

        gate_field, dist_gate, gate_debug = self._gate_field(obs, pos)
        obs_field, dist_obstacle, obs_debug_lines = self._obstacle_field(obs, pos, gate_field)

        raw_field = self._normalize(gate_field + obs_field, gate_field)
        self.debug_raw_field = raw_field.copy()

        # Low-pass the direction. This is the main "new data bends the vector
        # field" behavior: exact sensor updates change the field, but the command
        # direction does not flip instantly.
        field_alpha = self.exit_lock_field_alpha if self.exit_lock_active else self.field_alpha
        self.filtered_field = self._normalize((1.0 - field_alpha) * self.filtered_field + field_alpha * raw_field, raw_field)

        speed = self._speed(dist_gate, dist_obstacle)
        if self.exit_lock_active:
            speed = max(speed, self.exit_lock_speed)
        desired_vel_raw = speed * self.filtered_field

        prev_vel = self.filtered_vel.copy()
        dv = desired_vel_raw - self.filtered_vel
        self.filtered_vel = (self.filtered_vel + self._clip_norm(dv, self.max_velocity_step)).astype(np.float32)
        if self.exit_lock_active and self.exit_lock_x is not None:
            forward_speed = float(np.dot(self.filtered_vel, self.exit_lock_x))
            if forward_speed < self.exit_lock_min_forward_speed:
                lateral_v = self.filtered_vel - forward_speed * self.exit_lock_x
                self.filtered_vel = (
                    self.exit_lock_min_forward_speed * self.exit_lock_x + 0.20 * lateral_v
                ).astype(np.float32)
        desired_acc = self._clip_norm((self.filtered_vel - prev_vel) / max(self.dt, 1e-6), self.max_accel_ff)

        raw_target = pos + self.command_horizon * self.filtered_vel
        raw_target[2] = max(float(raw_target[2]), self.min_z)

        # Slew-limit position setpoint too. The state controller follows position,
        # so this is important even when velocity is smooth.
        delta_target = raw_target - self.filtered_target
        self.filtered_target = (self.filtered_target + self._clip_norm(delta_target, self.max_target_step)).astype(np.float32)
        self.filtered_target[2] = max(float(self.filtered_target[2]), self.min_z)
        self._guard_exit_lock_target(pos)

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
        center = gate_debug.get("center", pos)
        gate_x = gate_debug.get("gate_x", np.array([1, 0, 0], dtype=np.float32))
        approach = gate_debug.get("approach", center)
        exit_pt = gate_debug.get("exit", center)
        self.debug_target = self.filtered_target.copy()
        self.debug_field_line = np.vstack([pos, pos + 0.55 * self.filtered_field]).astype(np.float32)
        self.debug_gate_line = np.vstack([center - 0.32 * gate_x, center + 0.32 * gate_x]).astype(np.float32)
        self.debug_gate_points = np.vstack([approach, center, exit_pt]).astype(np.float32)
        self.debug_obstacle_lines = obs_debug_lines

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
        self.filtered_gate_pos.clear()
        self.filtered_gate_x.clear()
        self.filtered_gate_y.clear()
        self.filtered_gate_z.clear()
        self.filtered_obstacles.clear()
        self.obstacle_side_memory.clear()
        self.prev_gate_local_x.clear()
        self._stop_exit_lock()
        self.filtered_field = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.filtered_vel[:] = 0.0
        self.filtered_acc[:] = 0.0

    # ----------------------------------------------------------------------
    # Debug rendering
    # ----------------------------------------------------------------------
    def render_callback(self, sim: Any) -> None:
        try:
            from crazyflow.sim.visualize import draw_line, draw_points
        except Exception:
            return

        try:
            # Current smoothed field direction from the drone.
            if len(self.debug_field_line) >= 2:
                draw_line(sim, np.asarray(self.debug_field_line, dtype=np.float32), rgba=(0.0, 1.0, 1.0, 1.0))

            # Active gate approach-center-exit markers.
            if len(self.debug_gate_points) >= 1:
                draw_points(sim, np.asarray(self.debug_gate_points, dtype=np.float32), rgba=(1.0, 0.75, 0.0, 1.0), size=0.035)
            if len(self.debug_gate_points) >= 2:
                draw_line(sim, np.asarray(self.debug_gate_points, dtype=np.float32), rgba=(1.0, 0.75, 0.0, 1.0))

            # Gate crossing direction.
            if len(self.debug_gate_line) >= 2:
                draw_line(sim, np.asarray(self.debug_gate_line, dtype=np.float32), rgba=(0.0, 1.0, 0.0, 1.0))

            # Commanded state target.
            draw_points(sim, np.asarray(self.debug_target, dtype=np.float32).reshape(1, 3), rgba=(1.0, 0.0, 0.0, 1.0), size=0.045)

            # Obstacle push vectors.
            for line in self.debug_obstacle_lines:
                if len(line) >= 2:
                    draw_line(sim, np.asarray(line, dtype=np.float32), rgba=(1.0, 0.0, 1.0, 1.0))
        except Exception:
            return
