from __future__ import annotations

from typing import Any

import numpy as np

from lsy_drone_racing.control import Controller


class SharedVectorFieldController(Controller):
    """
    Shared vector-field controller for the drone racing task.

    Design idea:
        Every gate and obstacle contributes to one common desired-direction field.
        The drone follows the sum of those contributions; no global spline/path
        index is used, so there is nothing that can snap to a crossed path branch.

    Gate contributions:
        - past gates: normally zero
        - just-passed gate: short downstream exit-tail field
        - current gate: strong elongated one-way tube, local -x -> +x
        - next gate: weak preview pull to its entry mouth
        - later gates: zero by default

    Obstacle contributions:
        - all obstacles contribute
        - exact/visited obstacles are sharper/stronger
        - nominal/unvisited obstacles are wider/softer
        - each obstacle adds repulsion plus a tangential swirl term

    Config requirement:
        [env]
        control_mode = "state"

    Run example:
        python scripts/sim.py --config level2.toml --controller shared_vector_field_controller.py --render=True
    """

    def __init__(self, obs: dict[str, Any], info: dict | None, config: Any):
        super().__init__(obs, info, config)

        self.freq = int(getattr(config.env, "freq", 50))
        self.dt = 1.0 / max(float(self.freq), 1.0)
        self.tick = 0

        # ------------------------------------------------------------------
        # Gate tube field parameters
        # ------------------------------------------------------------------
        self.min_z = 0.28
        self.takeoff_z = 0.50

        self.approach_dist = 0.68        # entry mouth distance before gate
        self.exit_dist = 0.70            # strong tube distance after gate
        self.exit_fade_dist = 1.18       # downstream fade-out distance
        self.gate_tube_radius = 0.42     # radius where lateral correction is strong
        self.gate_far_radius = 1.15      # broad influence around current gate

        self.current_gate_gain_seen = 1.20
        self.current_gate_gain_nominal = 0.95
        self.future_gate_gain_seen = 0.22
        self.future_gate_gain_nominal = 0.14
        self.later_gate_gain = 0.00

        self.funnel_lateral_gain_far = 0.85
        self.funnel_lateral_gain_near = 2.55
        self.funnel_lateral_gain_exit = 0.85
        self.entry_forward_bias = 0.24
        self.future_preview_start_x = -0.10
        self.future_preview_full_x = self.exit_dist + 0.35

        # If target_gate advances before the drone has fully cleared the old gate,
        # keep a short one-way exit tail from the old gate.
        self.exit_tail_enabled = True
        self.exit_tail_gain = 0.95
        self.exit_tail_lateral_gain = 0.70
        self.exit_tail_release_x = self.exit_dist + 0.32
        self.exit_tail_max_age_steps = max(10, int(0.90 * self.freq))
        self.exit_tail_min_forward_lead = 0.22
        self.exit_tail_min_forward_speed = 0.26

        # ------------------------------------------------------------------
        # Obstacle field parameters
        # ------------------------------------------------------------------
        self.use_obstacles = True
        self.use_nominal_obstacles = True
        self.obstacle_influence_seen = 0.72
        self.obstacle_influence_nominal = 0.92
        self.obstacle_core_radius_seen = 0.25
        self.obstacle_core_radius_nominal = 0.32
        self.obstacle_repulse_gain_seen = 1.35
        self.obstacle_repulse_gain_nominal = 0.82
        self.obstacle_swirl_gain_seen = 0.62
        self.obstacle_swirl_gain_nominal = 0.36
        self.max_obstacle_field = 1.65
        self.obstacle_vertical_gain = 0.12
        self.obstacle_behind_weight = 0.20

        # ------------------------------------------------------------------
        # Filtering and command shaping
        # ------------------------------------------------------------------
        self.gate_filter_alpha_seen = 0.22
        self.gate_filter_alpha_nominal = 0.07
        self.obstacle_filter_alpha_seen = 0.26
        self.obstacle_filter_alpha_nominal = 0.08

        self.field_alpha = 0.30
        self.field_alpha_sensor_event = 0.45
        self.field_alpha_exit_tail = 0.55

        self.cruise_speed = 0.72
        self.gate_speed = 0.43
        self.tight_gate_speed = 0.34
        self.obstacle_speed = 0.42
        self.exit_tail_speed = 0.48
        self.max_speed = 0.88
        self.min_speed = 0.20

        self.command_horizon = 0.46
        self.max_target_step = 0.060
        self.max_velocity_step = 0.050
        self.max_accel_ff = 1.45
        self.yaw_step_limit = 0.10

        self.velocity_damping_gain = 0.10
        self.target_lead_min_current_gate = 0.16

        # ------------------------------------------------------------------
        # Runtime memory
        # ------------------------------------------------------------------
        self.last_target_gate = int(np.asarray(obs.get("target_gate", 0)).reshape(()))
        self.prev_gates_visited = self._bool_arr(obs.get("gates_visited", []))
        self.prev_obstacles_visited = self._bool_arr(obs.get("obstacles_visited", []))

        self.filtered_gate_pos: dict[int, np.ndarray] = {}
        self.filtered_gate_x: dict[int, np.ndarray] = {}
        self.filtered_gate_y: dict[int, np.ndarray] = {}
        self.filtered_gate_z: dict[int, np.ndarray] = {}
        self.filtered_obstacles: dict[int, np.ndarray] = {}
        self.obstacle_side_memory: dict[int, float] = {}

        self.exit_tail: dict[str, Any] | None = None

        pos0 = self._arr(obs["pos"], (3,))
        self.filtered_field = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.filtered_vel = np.zeros(3, dtype=np.float32)
        self.filtered_target = pos0.copy()
        self.last_yaw = self._yaw_from_quat(self._arr(obs.get("quat", [0, 0, 0, 1]), (4,)))

        # Debug draw buffers.
        self.debug_total_field = np.empty((0, 3), dtype=np.float32)
        self.debug_gate_lines: list[np.ndarray] = []
        self.debug_gate_points: list[np.ndarray] = []
        self.debug_obstacle_lines: list[np.ndarray] = []
        self.debug_target = pos0.copy()
        self.debug_mode = "init"

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
    def _bool_arr(x: Any) -> np.ndarray:
        return np.asarray(x, dtype=bool).reshape(-1)

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
    def _smoothstep(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    @staticmethod
    def _gaussian_weight(x: float) -> float:
        return float(np.exp(-0.5 * x * x))

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
    def _yaw_from_vector(v: np.ndarray, fallback: float) -> float:
        vx, vy = float(v[0]), float(v[1])
        if vx * vx + vy * vy < 1e-8:
            return float(fallback)
        return float(np.arctan2(vy, vx))

    @staticmethod
    def _angle_wrap(a: float) -> float:
        return float((a + np.pi) % (2.0 * np.pi) - np.pi)

    # ----------------------------------------------------------------------
    # Filtered object estimates
    # ----------------------------------------------------------------------
    def _visited_flag(self, visited: np.ndarray, i: int) -> bool:
        return bool(i < len(visited) and visited[i])

    def _gate_frame_raw(self, obs: dict[str, Any], i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        gates_pos = self._arr(obs.get("gates_pos", []))
        gates_quat = self._arr(obs.get("gates_quat", []))
        pos = gates_pos[i].astype(np.float32).copy()
        quat = gates_quat[i].astype(np.float32).copy()
        pos[2] = max(float(pos[2]), self.min_z)

        gate_x = self._normalize(self._quat_rotate_xyzw(quat, np.array([1.0, 0.0, 0.0], dtype=np.float32)))
        gate_y = self._normalize(
            self._quat_rotate_xyzw(quat, np.array([0.0, 1.0, 0.0], dtype=np.float32)),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        gate_z = self._normalize(
            self._quat_rotate_xyzw(quat, np.array([0.0, 0.0, 1.0], dtype=np.float32)),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )
        return pos, gate_x, gate_y, gate_z

    def _filtered_gate_frame(
        self, obs: dict[str, Any], i: int, gates_visited: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pos, gate_x, gate_y, gate_z = self._gate_frame_raw(obs, i)
        seen = self._visited_flag(gates_visited, i)
        alpha = self.gate_filter_alpha_seen if seen else self.gate_filter_alpha_nominal

        if i not in self.filtered_gate_pos:
            self.filtered_gate_pos[i] = pos
            self.filtered_gate_x[i] = gate_x
            self.filtered_gate_y[i] = gate_y
            self.filtered_gate_z[i] = gate_z
        else:
            self.filtered_gate_pos[i] = ((1.0 - alpha) * self.filtered_gate_pos[i] + alpha * pos).astype(np.float32)
            self.filtered_gate_x[i] = self._normalize((1.0 - alpha) * self.filtered_gate_x[i] + alpha * gate_x, gate_x)
            self.filtered_gate_y[i] = self._normalize((1.0 - alpha) * self.filtered_gate_y[i] + alpha * gate_y, gate_y)
            self.filtered_gate_z[i] = self._normalize((1.0 - alpha) * self.filtered_gate_z[i] + alpha * gate_z, gate_z)

        return (
            self.filtered_gate_pos[i].copy(),
            self.filtered_gate_x[i].copy(),
            self.filtered_gate_y[i].copy(),
            self.filtered_gate_z[i].copy(),
        )

    def _filtered_obstacle_positions(self, obs: dict[str, Any], visited: np.ndarray) -> np.ndarray:
        obstacles = self._arr(obs.get("obstacles_pos", []))
        if obstacles.ndim != 2 or obstacles.shape[1] != 3 or len(obstacles) == 0:
            return np.empty((0, 3), dtype=np.float32)

        out = []
        for i, p in enumerate(obstacles):
            p = p.astype(np.float32).copy()
            p[2] = max(float(p[2]), self.min_z)
            seen = self._visited_flag(visited, i)
            alpha = self.obstacle_filter_alpha_seen if seen else self.obstacle_filter_alpha_nominal
            if i not in self.filtered_obstacles:
                self.filtered_obstacles[i] = p
            else:
                self.filtered_obstacles[i] = ((1.0 - alpha) * self.filtered_obstacles[i] + alpha * p).astype(np.float32)
            out.append(self.filtered_obstacles[i].copy())
        return np.vstack(out).astype(np.float32)

    # ----------------------------------------------------------------------
    # Gate vector-field terms
    # ----------------------------------------------------------------------
    def _gate_local(
        self, pos: np.ndarray, center: np.ndarray, gate_x: np.ndarray, gate_y: np.ndarray, gate_z: np.ndarray
    ) -> tuple[float, float, float, np.ndarray]:
        rel = pos - center
        lx = float(np.dot(rel, gate_x))
        ly = float(np.dot(rel, gate_y))
        lz = float(np.dot(rel, gate_z))
        lateral = ly * gate_y + lz * gate_z
        return lx, ly, lz, lateral.astype(np.float32)

    def _elongated_current_gate_field(
        self,
        pos: np.ndarray,
        center: np.ndarray,
        gate_x: np.ndarray,
        gate_y: np.ndarray,
        gate_z: np.ndarray,
        seen: bool,
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
        lx, ly, lz, lateral = self._gate_local(pos, center, gate_x, gate_y, gate_z)
        lateral_norm = float(np.linalg.norm(lateral))
        gain = self.current_gate_gain_seen if seen else self.current_gate_gain_nominal

        entry = center - self.approach_dist * gate_x
        exit_pt = center + self.exit_dist * gate_x
        entry[2] = max(float(entry[2]), self.min_z)
        exit_pt[2] = max(float(exit_pt[2]), self.min_z)

        # Before the elongated tube: pull toward the entry mouth, not the center.
        if lx < -self.approach_dist:
            to_entry = self._normalize(entry - pos, gate_x)
            # Light lateral correction already starts aligning with the gate tube.
            field = to_entry + self.entry_forward_bias * gate_x - 0.20 * lateral
            field = self._normalize(field, gate_x)
            distance_scale = self._gaussian_weight(max(0.0, lateral_norm - self.gate_tube_radius) / self.gate_far_radius)
            weight = gain * (0.65 + 0.35 * distance_scale)
            mode = "entry"
        else:
            # Inside and after the gate: one-way flow. The forward part fades only
            # after exit_dist, so the gate keeps pushing downstream while clearing.
            if lx <= self.exit_dist:
                axial_weight = 1.0
            elif lx <= self.exit_fade_dist:
                axial_weight = 1.0 - self._smoothstep((lx - self.exit_dist) / max(self.exit_fade_dist - self.exit_dist, 1e-6))
            else:
                axial_weight = 0.0

            # Lateral correction is strongest near the gate plane, weaker in the
            # downstream exit lobe.
            near_plane = 1.0 - self._smoothstep(abs(lx) / max(self.approach_dist, 1e-6))
            exit_part = self._smoothstep(max(lx, 0.0) / max(self.exit_dist, 1e-6))
            lateral_gain = (
                (1.0 - near_plane) * self.funnel_lateral_gain_far
                + near_plane * self.funnel_lateral_gain_near
            )
            lateral_gain = (1.0 - exit_part) * lateral_gain + exit_part * self.funnel_lateral_gain_exit

            field = axial_weight * gate_x - lateral_gain * lateral
            if axial_weight <= 1e-4 and lateral_norm > 0.05:
                # If somehow still not counted after the fade region, do not pull
                # backwards; only weakly recover toward the tube side.
                field = -0.25 * lateral
            field = self._normalize(field, gate_x)

            tube_weight = self._gaussian_weight(lateral_norm / max(self.gate_tube_radius, 1e-6))
            broad_weight = self._gaussian_weight(lateral_norm / max(self.gate_far_radius, 1e-6))
            weight = gain * max(0.25 * broad_weight, tube_weight) * max(0.25, axial_weight)
            mode = "tube" if lx <= self.exit_dist else "exit_fade"

        return (weight * field).astype(np.float32), lateral_norm, {
            "mode": mode,
            "local_x": lx,
            "local_y": ly,
            "local_z": lz,
            "lateral_norm": lateral_norm,
            "entry": entry,
            "center": center,
            "exit": exit_pt,
            "gate_x": gate_x,
        }

    def _future_gate_field(
        self,
        pos: np.ndarray,
        center: np.ndarray,
        gate_x: np.ndarray,
        seen: bool,
        preview_scale: float,
    ) -> np.ndarray:
        if preview_scale <= 1e-5:
            return np.zeros(3, dtype=np.float32)
        entry = center - self.approach_dist * gate_x
        entry[2] = max(float(entry[2]), self.min_z)
        gain = self.future_gate_gain_seen if seen else self.future_gate_gain_nominal
        return (preview_scale * gain * self._normalize(entry - pos, gate_x)).astype(np.float32)

    def _start_exit_tail_from_gate(
        self,
        gate_i: int,
        center: np.ndarray,
        gate_x: np.ndarray,
        gate_y: np.ndarray,
        gate_z: np.ndarray,
    ) -> None:
        if not self.exit_tail_enabled:
            return
        self.exit_tail = {
            "gate_i": int(gate_i),
            "center": center.astype(np.float32).copy(),
            "x": self._normalize(gate_x).astype(np.float32),
            "y": self._normalize(gate_y, np.array([0.0, 1.0, 0.0], dtype=np.float32)).astype(np.float32),
            "z": self._normalize(gate_z, np.array([0.0, 0.0, 1.0], dtype=np.float32)).astype(np.float32),
            "start_tick": int(self.tick),
        }

    def _exit_tail_field(self, pos: np.ndarray) -> tuple[np.ndarray, bool, dict[str, Any]]:
        if self.exit_tail is None:
            return np.zeros(3, dtype=np.float32), False, {}
        center = self.exit_tail["center"]
        gate_x = self.exit_tail["x"]
        gate_y = self.exit_tail["y"]
        gate_z = self.exit_tail["z"]
        age = int(self.tick - int(self.exit_tail["start_tick"]))
        lx, ly, lz, lateral = self._gate_local(pos, center, gate_x, gate_y, gate_z)

        if lx > self.exit_tail_release_x or age > self.exit_tail_max_age_steps:
            self.exit_tail = None
            return np.zeros(3, dtype=np.float32), False, {}

        tail = gate_x - self.exit_tail_lateral_gain * lateral
        # Fade as it approaches release, but do not go to zero too soon.
        fade = 1.0 - 0.55 * self._smoothstep(max(lx, 0.0) / max(self.exit_tail_release_x, 1e-6))
        field = self.exit_tail_gain * fade * self._normalize(tail, gate_x)
        return field.astype(np.float32), True, {
            "local_x": lx,
            "center": center,
            "gate_x": gate_x,
            "mode": "exit_tail",
        }

    def _gate_field_sum(
        self, obs: dict[str, Any], pos: np.ndarray, target_gate: int, gates_visited: np.ndarray
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
        gates_pos = self._arr(obs.get("gates_pos", []))
        gates_quat = self._arr(obs.get("gates_quat", []))
        if gates_pos.ndim != 2 or gates_pos.shape[1] != 3 or gates_quat.ndim != 2 or gates_quat.shape[1] != 4:
            return np.zeros(3, dtype=np.float32), float("inf"), {"mode": "no_gates"}

        n_gates = min(len(gates_pos), len(gates_quat))
        total = np.zeros(3, dtype=np.float32)
        closest_current = float("inf")
        current_debug: dict[str, Any] = {"mode": "none"}

        # Active exit tail from the most recently passed gate.
        tail_field, tail_active, tail_debug = self._exit_tail_field(pos)
        if tail_active:
            total += tail_field
            current_debug = tail_debug

        # Need current gate x to decide when preview of next gate should fade in.
        current_lx = -float("inf")
        current_seen = False
        if 0 <= target_gate < n_gates:
            c, gx, gy, gz = self._filtered_gate_frame(obs, target_gate, gates_visited)
            current_seen = self._visited_flag(gates_visited, target_gate)
            current_lx, _ly, _lz, _lat = self._gate_local(pos, c, gx, gy, gz)

        if target_gate < 0:
            self.debug_gate_lines = []
            self.debug_gate_points = []
            return total, closest_current, current_debug

        preview_scale = self._smoothstep(
            (current_lx - self.future_preview_start_x)
            / max(self.future_preview_full_x - self.future_preview_start_x, 1e-6)
        )
        if tail_active:
            # While an old gate's exit tail is active, suppress next-gate pull a bit.
            preview_scale *= 0.35

        self.debug_gate_lines = []
        self.debug_gate_points = []

        for i in range(n_gates):
            center, gate_x, gate_y, gate_z = self._filtered_gate_frame(obs, i, gates_visited)
            seen = self._visited_flag(gates_visited, i)
            entry = center - self.approach_dist * gate_x
            exit_pt = center + self.exit_dist * gate_x
            entry[2] = max(float(entry[2]), self.min_z)
            exit_pt[2] = max(float(exit_pt[2]), self.min_z)

            # Debug: draw elongated gate axis and entry-center-exit points.
            self.debug_gate_lines.append(np.vstack([center - self.approach_dist * gate_x, center + self.exit_fade_dist * gate_x]).astype(np.float32))
            if i == target_gate or i == target_gate + 1:
                self.debug_gate_points.append(np.vstack([entry, center, exit_pt]).astype(np.float32))

            if i < target_gate:
                continue
            if i == target_gate:
                f, dist_lat, dbg = self._elongated_current_gate_field(pos, center, gate_x, gate_y, gate_z, seen)
                total += f
                closest_current = float(np.linalg.norm(pos - center))
                current_debug = dbg
                current_debug["seen"] = seen
            elif i == target_gate + 1:
                total += self._future_gate_field(pos, center, gate_x, seen, preview_scale)
            else:
                if self.later_gate_gain > 1e-6:
                    total += self.later_gate_gain * self._future_gate_field(pos, center, gate_x, seen, 1.0)

        current_debug["current_lx"] = current_lx
        current_debug["current_seen"] = current_seen
        current_debug["preview_scale"] = preview_scale
        return total.astype(np.float32), closest_current, current_debug

    # ----------------------------------------------------------------------
    # Obstacle vector-field terms
    # ----------------------------------------------------------------------
    def _obstacle_field_sum(
        self, obs: dict[str, Any], pos: np.ndarray, obstacles_visited: np.ndarray, nominal_dir: np.ndarray
    ) -> tuple[np.ndarray, float]:
        if not self.use_obstacles:
            return np.zeros(3, dtype=np.float32), float("inf")
        obstacles = self._filtered_obstacle_positions(obs, obstacles_visited)
        if len(obstacles) == 0:
            return np.zeros(3, dtype=np.float32), float("inf")

        total = np.zeros(3, dtype=np.float32)
        closest = float("inf")
        direction = self._normalize(nominal_dir, self.filtered_field)
        self.debug_obstacle_lines = []

        for i, o in enumerate(obstacles):
            seen = self._visited_flag(obstacles_visited, i)
            if not self.use_nominal_obstacles and not seen:
                continue

            influence = self.obstacle_influence_seen if seen else self.obstacle_influence_nominal
            core = self.obstacle_core_radius_seen if seen else self.obstacle_core_radius_nominal
            repulse_gain = self.obstacle_repulse_gain_seen if seen else self.obstacle_repulse_gain_nominal
            swirl_gain = self.obstacle_swirl_gain_seen if seen else self.obstacle_swirl_gain_nominal

            rel = pos - o
            rel_xy = rel[:2]
            d_xy = float(np.linalg.norm(rel_xy))
            closest = min(closest, d_xy)
            if d_xy >= influence:
                continue

            if d_xy < 1e-5:
                away_xy = self._normalize(np.array([-direction[1], direction[0]], dtype=np.float32))
                d_xy = 1e-5
            else:
                away_xy = (rel_xy / d_xy).astype(np.float32)

            closeness = float(np.clip((influence - d_xy) / max(influence - core, 1e-6), 0.0, 1.0))
            repulse_xy = away_xy * (repulse_gain * closeness * closeness)

            # Tangential swirl: choose a stable side that tends to agree with the
            # current gate flow, so the obstacle does not alternate left/right.
            side = self.obstacle_side_memory.get(i)
            tangent_plus = np.array([-away_xy[1], away_xy[0]], dtype=np.float32)
            if side is None:
                side = 1.0 if float(np.dot(tangent_plus, direction[:2])) >= 0.0 else -1.0
                self.obstacle_side_memory[i] = side
            tangent_xy = side * tangent_plus
            swirl_xy = tangent_xy * (swirl_gain * closeness * (1.0 - 0.25 * closeness))

            ahead = float(np.dot(o - pos, direction))
            ahead_weight = self.obstacle_behind_weight + (1.0 - self.obstacle_behind_weight) * self._smoothstep(
                (ahead + 0.10) / max(influence, 1e-6)
            )

            z_push = 0.0
            if d_xy < core + 0.10:
                z_push = self.obstacle_vertical_gain * (1.0 - d_xy / max(core + 0.10, 1e-6))

            contrib = np.array([repulse_xy[0] + swirl_xy[0], repulse_xy[1] + swirl_xy[1], z_push], dtype=np.float32)
            contrib *= ahead_weight
            total += contrib

            if float(np.linalg.norm(contrib)) > 1e-4:
                self.debug_obstacle_lines.append(np.vstack([o, o + 0.32 * self._normalize(contrib)]).astype(np.float32))

        return self._clip_norm(total, self.max_obstacle_field), closest

    # ----------------------------------------------------------------------
    # State machine and command shaping
    # ----------------------------------------------------------------------
    def _handle_gate_index_change(self, obs: dict[str, Any], target_gate: int, gates_visited: np.ndarray) -> None:
        old = int(self.last_target_gate)
        if target_gate == old:
            return

        # When moving from gate k to k+1, keep an exit tail from k. We use the
        # filtered frame if available; otherwise compute from current obs.
        if old >= 0 and self.exit_tail_enabled:
            gates_pos = self._arr(obs.get("gates_pos", []))
            gates_quat = self._arr(obs.get("gates_quat", []))
            if gates_pos.ndim == 2 and gates_quat.ndim == 2 and old < len(gates_pos) and old < len(gates_quat):
                center, gx, gy, gz = self._filtered_gate_frame(obs, old, gates_visited)
                self._start_exit_tail_from_gate(old, center, gx, gy, gz)

        self.obstacle_side_memory.clear()
        self.last_target_gate = target_gate

    def _sensor_event(self, gates_visited: np.ndarray, obstacles_visited: np.ndarray) -> bool:
        changed = False
        if len(gates_visited) != len(self.prev_gates_visited) or np.any(gates_visited != self.prev_gates_visited):
            changed = True
        if len(obstacles_visited) != len(self.prev_obstacles_visited) or np.any(obstacles_visited != self.prev_obstacles_visited):
            changed = True
        self.prev_gates_visited = gates_visited.copy()
        self.prev_obstacles_visited = obstacles_visited.copy()
        return changed

    def _speed(self, dist_gate: float, dist_obstacle: float, current_debug: dict[str, Any], exit_tail_active: bool) -> float:
        if exit_tail_active:
            return float(np.clip(self.exit_tail_speed, self.min_speed, self.max_speed))

        speed = self.cruise_speed
        lx = float(current_debug.get("current_lx", current_debug.get("local_x", -10.0)))
        lateral = float(current_debug.get("lateral_norm", 0.0))

        if dist_gate < 0.95 or (-self.approach_dist <= lx <= self.exit_dist):
            speed = min(speed, self.gate_speed)
        if abs(lx) < 0.35 and lateral < 0.35:
            speed = min(speed, self.tight_gate_speed)
        if dist_obstacle < self.obstacle_influence_seen:
            factor = self._smoothstep(dist_obstacle / max(self.obstacle_influence_seen, 1e-6))
            speed = min(speed, self.obstacle_speed + (self.cruise_speed - self.obstacle_speed) * factor)
        return float(np.clip(speed, self.min_speed, self.max_speed))

    def _guard_forward_target(self, pos: np.ndarray, target_gate: int, gates_visited: np.ndarray) -> None:
        # If an exit tail exists, the target may never lag behind the tail's gate_x.
        if self.exit_tail is not None:
            gx = self.exit_tail["x"]
            lead = float(np.dot(self.filtered_target - pos, gx))
            if lead < self.exit_tail_min_forward_lead:
                self.filtered_target = (self.filtered_target + (self.exit_tail_min_forward_lead - lead) * gx).astype(np.float32)
            v_forward = float(np.dot(self.filtered_vel, gx))
            if v_forward < self.exit_tail_min_forward_speed:
                self.filtered_vel = (self.filtered_vel + (self.exit_tail_min_forward_speed - v_forward) * gx).astype(np.float32)
            return

        # Soft guard only; failure just means no guard this step.
        if target_gate in self.filtered_gate_x:
            gx = self.filtered_gate_x[target_gate]
            lead = float(np.dot(self.filtered_target - pos, gx))
            if lead < self.target_lead_min_current_gate:
                self.filtered_target = (self.filtered_target + (self.target_lead_min_current_gate - lead) * gx).astype(np.float32)

    def compute_control(self, obs: dict[str, Any], info: dict | None = None) -> np.ndarray:
        pos = self._arr(obs["pos"], (3,))
        vel = self._arr(obs.get("vel", [0.0, 0.0, 0.0]), (3,))
        target_gate = int(np.asarray(obs.get("target_gate", 0)).reshape(()))
        gates_visited = self._bool_arr(obs.get("gates_visited", []))
        obstacles_visited = self._bool_arr(obs.get("obstacles_visited", []))

        sensor_event = self._sensor_event(gates_visited, obstacles_visited)
        self._handle_gate_index_change(obs, target_gate, gates_visited)

        cmd = np.zeros(13, dtype=np.float32)
        if target_gate < 0 and self.exit_tail is None:
            # Finished and no exit tail left: hold current position.
            cmd[0:3] = pos
            cmd[9] = self.last_yaw
            self.filtered_target = pos.copy()
            self.filtered_vel[:] = 0.0
            return cmd

        # Takeoff helper. It prevents the sum field from commanding a low diagonal
        # before the drone is safely above the ground.
        if float(pos[2]) < self.takeoff_z - 0.08:
            gate_field = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            dist_gate = float("inf")
            current_debug = {"mode": "takeoff"}
        else:
            gate_field, dist_gate, current_debug = self._gate_field_sum(obs, pos, target_gate, gates_visited)

        obstacle_field, dist_obstacle = self._obstacle_field_sum(obs, pos, obstacles_visited, gate_field)
        damping_field = -self.velocity_damping_gain * vel

        raw_sum = gate_field + obstacle_field + damping_field
        if float(np.linalg.norm(raw_sum)) < 1e-5:
            raw_field = self.filtered_field.copy()
        else:
            raw_field = self._normalize(raw_sum, self.filtered_field)

        exit_tail_active = self.exit_tail is not None
        if exit_tail_active:
            alpha = self.field_alpha_exit_tail
        elif sensor_event:
            alpha = self.field_alpha_sensor_event
        else:
            alpha = self.field_alpha
        self.filtered_field = self._normalize((1.0 - alpha) * self.filtered_field + alpha * raw_field, raw_field)

        speed = self._speed(dist_gate, dist_obstacle, current_debug, exit_tail_active)
        desired_vel_raw = speed * self.filtered_field

        prev_vel = self.filtered_vel.copy()
        dv = desired_vel_raw - self.filtered_vel
        self.filtered_vel = (self.filtered_vel + self._clip_norm(dv, self.max_velocity_step)).astype(np.float32)

        desired_acc = self._clip_norm((self.filtered_vel - prev_vel) / max(self.dt, 1e-6), self.max_accel_ff)

        raw_target = pos + self.command_horizon * self.filtered_vel
        raw_target[2] = max(float(raw_target[2]), self.min_z)
        target_delta = raw_target - self.filtered_target
        self.filtered_target = (self.filtered_target + self._clip_norm(target_delta, self.max_target_step)).astype(np.float32)
        self.filtered_target[2] = max(float(self.filtered_target[2]), self.min_z)
        self._guard_forward_target(pos, target_gate, gates_visited)
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

        # Debug draw data.
        self.debug_total_field = np.vstack([pos, pos + 0.55 * self.filtered_field]).astype(np.float32)
        self.debug_target = self.filtered_target.copy()
        self.debug_mode = str(current_debug.get("mode", "field"))

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
        self.exit_tail = None
        self.filtered_field = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.filtered_vel[:] = 0.0

    # ----------------------------------------------------------------------
    # Debug visualization
    # ----------------------------------------------------------------------
    def render_callback(self, sim: Any) -> None:
        try:
            from crazyflow.sim.visualize import draw_line, draw_points
        except Exception:
            return

        try:
            # Cyan: actual summed/smoothed field direction from drone.
            if len(self.debug_total_field) >= 2:
                draw_line(sim, np.asarray(self.debug_total_field, dtype=np.float32), rgba=(0.0, 1.0, 1.0, 1.0))

            # Yellow: elongated gate axes and entry-center-exit markers for current/next.
            for line in self.debug_gate_lines:
                if len(line) >= 2:
                    draw_line(sim, np.asarray(line, dtype=np.float32), rgba=(1.0, 0.75, 0.0, 1.0))
            for pts in self.debug_gate_points:
                if len(pts) >= 1:
                    draw_points(sim, np.asarray(pts, dtype=np.float32), rgba=(1.0, 0.75, 0.0, 1.0), size=0.032)

            # Magenta: obstacle contribution directions.
            for line in self.debug_obstacle_lines:
                if len(line) >= 2:
                    draw_line(sim, np.asarray(line, dtype=np.float32), rgba=(1.0, 0.0, 1.0, 1.0))

            # Red: commanded state position.
            draw_points(sim, np.asarray(self.debug_target, dtype=np.float32).reshape(1, 3), rgba=(1.0, 0.0, 0.0, 1.0), size=0.045)
        except Exception:
            return
