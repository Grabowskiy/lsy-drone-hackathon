from __future__ import annotations

from typing import Any

import numpy as np

from lsy_drone_racing.control import Controller


class HermiteReplanMPCCStableController(Controller):
    """
    Continuously replanning Hermite + simple MPCC-style state controller.

    What it does:
        - Rebuilds a gate-frame-aware Hermite path during flight.
        - Uses obs["gates_pos"] / obs["gates_quat"] at runtime, so when a gate
          becomes exact inside sensor range, the path is rebuilt from the current
          drone pose.
        - Adds simple lateral detour points around obstacles before sampling the
          Hermite path.
        - Tracks the path with a lightweight sampled MPCC-style law:
              1. project current position to path progress s,
              2. choose a forward progress target s + lookahead,
              3. command that path point plus velocity feed-forward along tangent.
        - Draws the current planned path, target point, projection point, waypoints,
          and gate crossing directions in the MuJoCo/Crazyflow viewer.

    This is NOT a full nonlinear/acados MPCC. It is deliberately simple and fast
    enough for a 50 Hz hackathon controller.

    Config requirement:
        [env]
        control_mode = "state"

    Run example:
        python scripts/sim.py --config level2.toml --controller hermite_replan_mpcc_stable_controller.py --render=True
    """

    def __init__(self, obs: dict[str, Any], info: dict | None, config: Any):
        super().__init__(obs, info, config)

        self.freq = int(getattr(config.env, "freq", 50))
        self.dt = 1.0 / max(float(self.freq), 1.0)
        self.tick = 0

        # ---------------------------
        # Planner parameters
        # ---------------------------
        self.replan_every_steps = 10      # max 5 Hz at env.freq=50; event replans still happen immediately
        self.samples_per_segment = 18
        self.tangent_scale = 0.42         # lower = less spline overshoot
        self.max_tangent_fraction = 0.75  # cap tangents by local segment lengths

        self.approach_dist = 0.60         # before gate along -local x
        self.exit_dist = 0.65             # after gate along +local x
        self.min_z = 0.28
        self.takeoff_z = 0.48             # inserted as vertical first waypoint if needed

        # Obstacle detour parameters. These are intentionally conservative but simple.
        self.use_obstacle_detours = True
        self.avoid_nominal_obstacles = True     # use nominal obstacle poses before exact sensing too
        self.obstacle_clearance_seen = 0.38
        self.obstacle_clearance_nominal = 0.46  # bigger because nominal pose may still shift
        self.max_detours_per_segment = 2
        self.detour_min_segment_len = 0.20

        # Detour hysteresis: once the planner chooses to pass an obstacle on one
        # side, keep that side. Otherwise tiny changes in the current start pose
        # can flip the detour from left to right and make the state target snap.
        self.detour_hysteresis_margin = 0.13
        self.detour_side_memory: dict[int, float] = {}
        self.detour_active_memory: dict[int, bool] = {}

        # ---------------------------
        # MPCC-style tracking parameters
        # ---------------------------
        self.cruise_speed = 0.72
        self.gate_speed = 0.46
        self.finish_speed = 0.35
        self.max_speed = 0.90
        self.min_speed = 0.25

        self.lookahead_min = 0.24
        self.lookahead_nominal = 0.42
        self.lookahead_max = 0.62

        # Progress is not advanced by hitting discrete waypoints. Instead, the
        # drone is considered to have made progress when it lies inside a tube
        # around any later portion of the path. This prevents the controller from
        # getting stuck trying to exactly hit a detour/control point.
        self.path_tube_radius = 0.24
        self.path_tube_z_weight = 0.65       # altitude lag should not block path progress too much
        self.progress_back_allowance = 0.08  # allow tiny numerical backward motion only
        self.progress_s = 0.0
        self.last_target_s = 0.0

        # Replanning stability. We still rebuild online, but not merely because
        # the current position changed by a few millimeters. New sensor data and
        # target_gate changes trigger immediate replans. Periodic replans require
        # meaningful motion since the last rebuild.
        self.replan_motion_trigger = 0.22
        self.path_blend_distance_periodic = 0.85
        self.path_blend_distance_sensor_event = 0.38

        # Command smoothing. This is separate from path smoothing and prevents
        # the state-controller setpoint from teleporting after a replan.
        self.target_slew_rate = 1.65       # m/s virtual target slew limit
        self.target_filter_max_lag = 0.34  # do not let the smoothed target fall too far behind
        self.velocity_slew_rate = 2.20     # m/s^2 desired-velocity slew limit
        self.filtered_target_pos: np.ndarray | None = None

        self.gate_slow_radius = 0.80
        self.gate_tight_radius = 0.45
        self.end_slow_distance = 0.60

        # If feed-forward is too spicy in your sim, set this to False.
        self.use_velocity_feedforward = True
        self.velocity_ff_gain = 0.85
        self.use_accel_feedforward = False
        self.max_accel_ff = 1.2

        # ---------------------------
        # Runtime path/debug state
        # ---------------------------
        self.path = np.empty((0, 3), dtype=np.float32)
        self.path_s = np.empty((0,), dtype=np.float32)
        self.waypoints = np.empty((0, 3), dtype=np.float32)
        self.gate_centers = np.empty((0, 3), dtype=np.float32)
        self.gate_normal_lines: list[np.ndarray] = []
        self.detour_points = np.empty((0, 3), dtype=np.float32)

        self.debug_target = self._arr(obs["pos"], (3,)).copy()
        self.debug_projection = self.debug_target.copy()
        self.debug_tangent_line = np.empty((0, 3), dtype=np.float32)
        self.debug_tube_projection = self.debug_projection.copy()
        self.debug_active_gate_center: np.ndarray | None = None

        self.last_obs_signature: tuple[Any, ...] | None = None
        self.last_target_gate = int(np.asarray(obs.get("target_gate", 0)).reshape(()))
        self.last_replan_pos = self._arr(obs["pos"], (3,)).copy()
        self.last_des_vel = np.zeros(3, dtype=np.float32)
        self.last_yaw = self._yaw_from_quat(self._arr(obs.get("quat", [0.0, 0.0, 0.0, 1.0]), (4,)))

        self._rebuild_plan(obs, force=True)

    # -------------------------------------------------------------------------
    # Small math helpers
    # -------------------------------------------------------------------------
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
        """Rotate vector v by quaternion q in xyzw convention, without scipy."""
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
    def _remove_near_duplicate_points(points: np.ndarray, min_dist: float) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        if len(points) <= 1:
            return points.reshape((-1, 3)).astype(np.float32)
        kept = [points[0].astype(np.float32)]
        for p in points[1:]:
            if float(np.linalg.norm(p - kept[-1])) >= min_dist:
                kept.append(p.astype(np.float32))
        return np.vstack(kept).astype(np.float32)

    # -------------------------------------------------------------------------
    # Gate-aware path construction
    # -------------------------------------------------------------------------
    def _gate_frame(self, gate_pos: np.ndarray, gate_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        center = np.asarray(gate_pos, dtype=np.float32).copy()
        center[2] = max(float(center[2]), self.min_z)

        gate_x = self._normalize(
            self._quat_rotate_xyzw(gate_quat, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        )
        gate_y = self._normalize(
            self._quat_rotate_xyzw(gate_quat, np.array([0.0, 1.0, 0.0], dtype=np.float32)),
            fallback=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        gate_z = self._normalize(
            self._quat_rotate_xyzw(gate_quat, np.array([0.0, 0.0, 1.0], dtype=np.float32)),
            fallback=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )
        return center.astype(np.float32), gate_x.astype(np.float32), gate_y.astype(np.float32), gate_z.astype(np.float32)

    def _gate_triplet(self, gate_pos: np.ndarray, gate_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        center, gate_x, _gate_y, _gate_z = self._gate_frame(gate_pos, gate_quat)
        approach = center - self.approach_dist * gate_x
        exit_pt = center + self.exit_dist * gate_x
        approach[2] = max(float(approach[2]), self.min_z)
        exit_pt[2] = max(float(exit_pt[2]), self.min_z)
        return approach.astype(np.float32), center.astype(np.float32), exit_pt.astype(np.float32), gate_x.astype(np.float32)

    def _build_gate_waypoints(self, obs: dict[str, Any]) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        pos = self._arr(obs["pos"], (3,))
        gates_pos = self._arr(obs.get("gates_pos", []))
        gates_quat = self._arr(obs.get("gates_quat", []))
        target_gate = int(np.asarray(obs.get("target_gate", 0)).reshape(()))

        if gates_pos.ndim != 2 or gates_pos.shape[1] != 3:
            return pos.reshape(1, 3).astype(np.float32), [], np.empty((0, 3), dtype=np.float32)
        if gates_quat.ndim != 2 or gates_quat.shape[1] != 4:
            return pos.reshape(1, 3).astype(np.float32), [], np.empty((0, 3), dtype=np.float32)

        n_gates = min(gates_pos.shape[0], gates_quat.shape[0])
        if target_gate < 0:
            return pos.reshape(1, 3).astype(np.float32), [], np.empty((0, 3), dtype=np.float32)
        start_idx = max(0, min(target_gate, n_gates))

        waypoints: list[np.ndarray] = [pos.astype(np.float32)]

        # First go up before moving laterally. This prevents a low diagonal sprint
        # from the start position toward the first gate.
        if float(pos[2]) < self.takeoff_z - 0.08:
            takeoff = pos.copy()
            takeoff[2] = self.takeoff_z
            waypoints.append(takeoff.astype(np.float32))

        gate_normals: list[np.ndarray] = []
        gate_centers: list[np.ndarray] = []

        for i in range(start_idx, n_gates):
            approach, center, exit_pt, gate_x = self._gate_triplet(gates_pos[i], gates_quat[i])
            gate_centers.append(center)
            gate_normals.append(np.vstack([center - 0.28 * gate_x, center + 0.28 * gate_x]).astype(np.float32))

            if i == target_gate:
                # Current gate special case:
                # If we are already in the gate plane region, don't re-command a
                # point behind us. If we are clearly on the wrong/far side while
                # target_gate did not advance, include approach again to retry.
                local_x = float(np.dot(pos - center, gate_x))
                if -0.18 <= local_x <= 0.16:
                    waypoints.extend([center, exit_pt])
                else:
                    waypoints.extend([approach, center, exit_pt])
            else:
                waypoints.extend([approach, center, exit_pt])

        wp = np.vstack(waypoints).astype(np.float32)
        wp = self._remove_near_duplicate_points(wp, min_dist=0.035)
        centers = np.vstack(gate_centers).astype(np.float32) if gate_centers else np.empty((0, 3), dtype=np.float32)
        return wp, gate_normals, centers

    # -------------------------------------------------------------------------
    # Obstacle detours
    # -------------------------------------------------------------------------
    def _obstacle_arrays(self, obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        obstacles_pos = self._arr(obs.get("obstacles_pos", []))
        if obstacles_pos.ndim != 2 or obstacles_pos.shape[1] != 3 or len(obstacles_pos) == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=bool)

        visited_raw = np.asarray(obs.get("obstacles_visited", np.ones((len(obstacles_pos),), dtype=bool)))
        visited = visited_raw.astype(bool).reshape(-1)
        if len(visited) < len(obstacles_pos):
            padded = np.zeros((len(obstacles_pos),), dtype=bool)
            padded[: len(visited)] = visited
            visited = padded
        return obstacles_pos.astype(np.float32), visited[: len(obstacles_pos)]

    def _insert_obstacle_detours(self, waypoints: np.ndarray, obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        if not self.use_obstacle_detours or len(waypoints) < 2:
            return waypoints.astype(np.float32), np.empty((0, 3), dtype=np.float32)

        obstacles, visited = self._obstacle_arrays(obs)
        if len(obstacles) == 0:
            return waypoints.astype(np.float32), np.empty((0, 3), dtype=np.float32)

        new_points: list[np.ndarray] = []
        detours_all: list[np.ndarray] = []
        seen_active_this_replan: set[int] = set()

        for seg_i in range(len(waypoints) - 1):
            p0 = waypoints[seg_i].astype(np.float32)
            p1 = waypoints[seg_i + 1].astype(np.float32)
            new_points.append(p0)

            d = p1 - p0
            d_xy = d[:2]
            seg_len_xy = float(np.linalg.norm(d_xy))
            seg_len_3d = float(np.linalg.norm(d))
            if seg_len_xy < self.detour_min_segment_len or seg_len_3d < self.detour_min_segment_len:
                continue

            perp = self._normalize(np.array([-d_xy[1], d_xy[0]], dtype=np.float32))
            candidates: list[tuple[float, np.ndarray, int]] = []

            for obs_i, o in enumerate(obstacles):
                obs_i_int = int(obs_i)
                if not self.avoid_nominal_obstacles and not bool(visited[obs_i]):
                    continue

                # Projection in XY, because the challenge sensor/racing layout is mostly horizontal.
                rel_xy = o[:2] - p0[:2]
                t = float(np.dot(rel_xy, d_xy) / max(float(np.dot(d_xy, d_xy)), 1e-8))
                if t <= 0.08 or t >= 0.92:
                    continue

                closest = p0 + t * d
                away_xy = closest[:2] - o[:2]
                dist_xy = float(np.linalg.norm(away_xy))
                clearance = self.obstacle_clearance_seen if bool(visited[obs_i]) else self.obstacle_clearance_nominal

                # Hysteresis: keep an active detour until the nominal segment is
                # comfortably outside the clearance. Without this, detours can
                # appear/disappear every replan near the threshold.
                was_active = bool(self.detour_active_memory.get(obs_i_int, False))
                enter_threshold = float(clearance)
                exit_threshold = float(clearance + self.detour_hysteresis_margin)
                threshold = exit_threshold if was_active else enter_threshold
                if dist_xy >= threshold:
                    continue

                # Pick and remember a side relative to the current segment. If the
                # line runs almost through the obstacle, the first deterministic side
                # choice is kept for this obstacle on future replans.
                remembered_side = self.detour_side_memory.get(obs_i_int)
                if remembered_side is not None:
                    side = float(remembered_side)
                else:
                    if dist_xy < 1e-5:
                        side = 1.0 if ((seg_i + obs_i_int) % 2 == 0) else -1.0
                    else:
                        side_raw = float(np.dot(away_xy / max(dist_xy, 1e-8), perp))
                        if abs(side_raw) < 0.10:
                            side = 1.0 if ((seg_i + obs_i_int) % 2 == 0) else -1.0
                        else:
                            side = 1.0 if side_raw >= 0.0 else -1.0
                    self.detour_side_memory[obs_i_int] = side

                detour = closest.astype(np.float32).copy()
                detour[:2] = o[:2] + side * perp.astype(np.float32) * clearance
                detour[2] = max(float(closest[2]), self.min_z)

                # Avoid detours too close to the segment endpoints; those create jitter.
                if float(np.linalg.norm(detour - p0)) < 0.15 or float(np.linalg.norm(detour - p1)) < 0.15:
                    continue

                candidates.append((t, detour.astype(np.float32), obs_i_int))

            if candidates:
                candidates.sort(key=lambda x: x[0])
                for _t, detour, obs_i_int in candidates[: self.max_detours_per_segment]:
                    new_points.append(detour)
                    detours_all.append(detour)
                    seen_active_this_replan.add(obs_i_int)

        for obs_i in range(len(obstacles)):
            self.detour_active_memory[int(obs_i)] = int(obs_i) in seen_active_this_replan

        new_points.append(waypoints[-1].astype(np.float32))
        out = self._remove_near_duplicate_points(np.vstack(new_points).astype(np.float32), min_dist=0.04)
        detours = np.vstack(detours_all).astype(np.float32) if detours_all else np.empty((0, 3), dtype=np.float32)
        return out, detours

    # -------------------------------------------------------------------------
    # Hermite sampling and arc-length parameterization
    # -------------------------------------------------------------------------
    @staticmethod
    def _hermite_segment(p0: np.ndarray, p1: np.ndarray, m0: np.ndarray, m1: np.ndarray, n: int) -> np.ndarray:
        s = np.linspace(0.0, 1.0, max(2, int(n)), endpoint=False, dtype=np.float32).reshape(-1, 1)
        s2 = s * s
        s3 = s2 * s
        h00 = 2.0 * s3 - 3.0 * s2 + 1.0
        h10 = s3 - 2.0 * s2 + s
        h01 = -2.0 * s3 + 3.0 * s2
        h11 = s3 - s2
        return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

    def _compute_tangents(self, points: np.ndarray) -> np.ndarray:
        n = len(points)
        tangents = np.zeros_like(points, dtype=np.float32)
        if n <= 1:
            return tangents

        for i in range(n):
            if i == 0:
                raw = points[1] - points[0]
                local_len = float(np.linalg.norm(raw))
            elif i == n - 1:
                raw = points[-1] - points[-2]
                local_len = float(np.linalg.norm(raw))
            else:
                prev_len = float(np.linalg.norm(points[i] - points[i - 1]))
                next_len = float(np.linalg.norm(points[i + 1] - points[i]))
                raw = points[i + 1] - points[i - 1]
                local_len = max(1e-6, min(prev_len, next_len))

            tangent = self.tangent_scale * raw
            tangent = self._clip_norm(tangent, self.max_tangent_fraction * max(local_len, 1e-6))
            tangents[i] = tangent.astype(np.float32)

        return tangents.astype(np.float32)

    def _sample_hermite_path(self, waypoints: np.ndarray) -> np.ndarray:
        waypoints = self._remove_near_duplicate_points(waypoints, min_dist=0.025)
        if len(waypoints) <= 1:
            return waypoints.astype(np.float32)

        tangents = self._compute_tangents(waypoints)
        chunks: list[np.ndarray] = []
        for i in range(len(waypoints) - 1):
            chunks.append(
                self._hermite_segment(
                    waypoints[i], waypoints[i + 1], tangents[i], tangents[i + 1], self.samples_per_segment
                )
            )
        chunks.append(waypoints[-1].reshape(1, 3))
        path = np.vstack(chunks).astype(np.float32)
        return self._remove_near_duplicate_points(path, min_dist=0.006)

    @staticmethod
    def _arc_lengths(path: np.ndarray) -> np.ndarray:
        path = np.asarray(path, dtype=np.float32)
        if len(path) <= 1:
            return np.zeros((len(path),), dtype=np.float32)
        ds = np.linalg.norm(np.diff(path, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(ds)]).astype(np.float32)

    @staticmethod
    def _interpolate_arrays(path: np.ndarray, path_s: np.ndarray, s_query: float) -> tuple[np.ndarray, np.ndarray]:
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
        point = (1.0 - alpha) * path[idx] + alpha * path[idx + 1]
        tangent = HermiteReplanMPCCStableController._normalize(path[idx + 1] - path[idx])
        return point.astype(np.float32), tangent.astype(np.float32)

    def _blend_new_path_with_old(
        self,
        new_path: np.ndarray,
        new_path_s: np.ndarray,
        old_path: np.ndarray,
        old_path_s: np.ndarray,
        old_progress_s: float,
        blend_distance: float,
    ) -> np.ndarray:
        """
        Blend only the first part of a new local path with the previous future path.

        This makes replanning continuous. The new long-horizon geometry still wins
        after blend_distance, but the near-term command does not instantly jump to
        a different obstacle side or Hermite lobe.
        """
        if len(new_path) < 2 or len(new_path_s) < 2 or len(old_path) < 2 or len(old_path_s) < 2:
            return new_path.astype(np.float32)
        if blend_distance <= 1e-6:
            return new_path.astype(np.float32)

        blended = np.asarray(new_path, dtype=np.float32).copy()
        max_old_s = float(old_path_s[-1])
        for i, s_new in enumerate(new_path_s):
            s_new_f = float(s_new)
            if s_new_f > blend_distance:
                break
            old_p, _old_t = self._interpolate_arrays(old_path, old_path_s, min(max_old_s, old_progress_s + s_new_f))
            a = float(np.clip(s_new_f / blend_distance, 0.0, 1.0))
            # Smoothstep: old path dominates immediately after replan, new path
            # fades in continuously.
            a = a * a * (3.0 - 2.0 * a)
            blended[i] = ((1.0 - a) * old_p + a * blended[i]).astype(np.float32)
        return blended.astype(np.float32)

    def _obs_signature(self, obs: dict[str, Any]) -> tuple[Any, ...]:
        target_gate = int(np.asarray(obs.get("target_gate", 0)).reshape(()))
        gates_pos = np.asarray(obs.get("gates_pos", []), dtype=np.float32)
        gates_quat = np.asarray(obs.get("gates_quat", []), dtype=np.float32)
        gates_visited = np.asarray(obs.get("gates_visited", []), dtype=bool)
        obstacles_pos = np.asarray(obs.get("obstacles_pos", []), dtype=np.float32)
        obstacles_visited = np.asarray(obs.get("obstacles_visited", []), dtype=bool)

        # Rounding prevents tiny float noise from constantly changing the signature.
        return (
            target_gate,
            tuple(np.round(gates_pos.reshape(-1), 3).tolist()),
            tuple(np.round(gates_quat.reshape(-1), 3).tolist()),
            tuple(gates_visited.reshape(-1).astype(int).tolist()),
            tuple(np.round(obstacles_pos.reshape(-1), 3).tolist()),
            tuple(obstacles_visited.reshape(-1).astype(int).tolist()),
        )

    def _rebuild_plan(self, obs: dict[str, Any], force: bool = False) -> None:
        old_sig = self.last_obs_signature
        sig = self._obs_signature(obs)
        sig_changed = sig != old_sig
        if not force and not sig_changed:
            return

        pos = self._arr(obs["pos"], (3,))
        target_gate = int(np.asarray(obs.get("target_gate", 0)).reshape(()))
        target_gate_changed = target_gate != self.last_target_gate

        old_path = self.path.copy()
        old_path_s = self.path_s.copy()
        old_progress_s = float(self.progress_s)
        old_debug_target = self.debug_target.copy() if self.debug_target is not None else pos.copy()

        base_waypoints, gate_normals, gate_centers = self._build_gate_waypoints(obs)
        waypoints, detours = self._insert_obstacle_detours(base_waypoints, obs)
        path = self._sample_hermite_path(waypoints)
        path_s = self._arc_lengths(path)

        # Blend only when we are still pursuing the same gate. On a real target
        # gate change, switching to the next gate should be immediate.
        if not target_gate_changed and len(path) >= 2 and len(old_path) >= 2:
            blend_distance = (
                self.path_blend_distance_sensor_event if sig_changed else self.path_blend_distance_periodic
            )
            path = self._blend_new_path_with_old(
                path, path_s, old_path, old_path_s, old_progress_s, blend_distance=blend_distance
            )
            path_s = self._arc_lengths(path)

        self.last_obs_signature = sig
        self.waypoints = waypoints.astype(np.float32)
        self.detour_points = detours.astype(np.float32)
        self.gate_normal_lines = gate_normals
        self.gate_centers = gate_centers.astype(np.float32)
        self.path = path.astype(np.float32)
        self.path_s = path_s.astype(np.float32)
        self.last_replan_pos = pos.copy()

        # Do NOT reset progress to zero. Re-project onto the new path and keep a
        # forward target near the previous target if possible. This is the main
        # anti-snap behavior.
        if len(self.path) >= 2:
            s_proj, _p_proj, _tan, _d2 = self._project_to_path(pos)
            self.progress_s = float(np.clip(s_proj, 0.0, float(self.path_s[-1])))

            old_target_s, _old_target_proj, _old_target_tan, old_target_d2 = self._project_to_path(old_debug_target)
            if not target_gate_changed and old_target_d2 < 0.45 * 0.45:
                lower = self.progress_s + self.lookahead_min
                upper = self.progress_s + self.lookahead_max
                self.last_target_s = float(np.clip(old_target_s, lower, min(upper, float(self.path_s[-1]))))
            else:
                self.last_target_s = float(min(self.progress_s + self.lookahead_min, float(self.path_s[-1])))
        else:
            self.progress_s = 0.0
            self.last_target_s = 0.0

        if target_gate_changed:
            # A gate transition is allowed to change the command more quickly, but
            # the next compute step will still apply normal target slew limiting.
            self.filtered_target_pos = None

    # -------------------------------------------------------------------------
    # MPCC-style path tracking
    # -------------------------------------------------------------------------
    def _project_to_path(self, pos: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
        """Return closest arc-length s, closest point, segment tangent, and squared distance."""
        if len(self.path) == 0:
            return 0.0, pos.astype(np.float32), np.array([1.0, 0.0, 0.0], dtype=np.float32), 0.0
        if len(self.path) == 1:
            tangent = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            return 0.0, self.path[0].copy(), tangent, float(np.sum((pos - self.path[0]) ** 2))

        best_d2 = float("inf")
        best_s = 0.0
        best_point = self.path[0].copy()
        best_tangent = self._normalize(self.path[1] - self.path[0])

        for i in range(len(self.path) - 1):
            a = self.path[i]
            b = self.path[i + 1]
            ab = b - a
            ab2 = float(np.dot(ab, ab))
            if ab2 < 1e-10:
                continue
            t = float(np.clip(np.dot(pos - a, ab) / ab2, 0.0, 1.0))
            p = a + t * ab
            d2 = float(np.sum((pos - p) ** 2))
            if d2 < best_d2:
                seg_len = float(np.sqrt(ab2))
                best_d2 = d2
                best_s = float(self.path_s[i] + t * seg_len)
                best_point = p.astype(np.float32)
                best_tangent = self._normalize(ab)

        return best_s, best_point.astype(np.float32), best_tangent.astype(np.float32), best_d2

    def _tube_metric_d2(self, pos: np.ndarray, point: np.ndarray) -> float:
        """
        Squared distance used for the path-progress tube.

        Z is down-weighted so a small altitude lag does not prevent progress from
        advancing when the drone is otherwise inside the lateral path corridor.
        """
        diff = np.asarray(pos - point, dtype=np.float32).copy()
        diff[2] *= float(self.path_tube_z_weight)
        return float(np.dot(diff, diff))

    def _project_to_path_with_tube(self, pos: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float, bool]:
        """
        Progress projection with tube-based advancement.

        Instead of choosing only the closest path point, find all path segments
        whose tube contains the drone and snap progress to the farthest such
        segment. That lets the controller skip a detour/control point once the
        drone has entered the path corridor beyond it.
        """
        if len(self.path) == 0:
            return 0.0, pos.astype(np.float32), np.array([1.0, 0.0, 0.0], dtype=np.float32), 0.0, False
        if len(self.path) == 1 or len(self.path_s) <= 1:
            tangent = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            d2 = self._tube_metric_d2(pos, self.path[0])
            return 0.0, self.path[0].copy(), tangent, d2, d2 <= self.path_tube_radius ** 2

        tube_r2 = float(self.path_tube_radius * self.path_tube_radius)
        min_allowed_s = max(0.0, float(self.progress_s) - float(self.progress_back_allowance))

        closest: tuple[float, np.ndarray, np.ndarray, float] | None = None
        farthest_inside: tuple[float, np.ndarray, np.ndarray, float] | None = None

        for i in range(len(self.path) - 1):
            a = self.path[i]
            b = self.path[i + 1]
            ab = b - a
            ab2 = float(np.dot(ab, ab))
            if ab2 < 1e-10:
                continue

            t = float(np.clip(np.dot(pos - a, ab) / ab2, 0.0, 1.0))
            p = (a + t * ab).astype(np.float32)
            seg_len = float(np.sqrt(ab2))
            s_here = float(self.path_s[i] + t * seg_len)
            tangent = self._normalize(ab)
            d2_tube = self._tube_metric_d2(pos, p)

            item = (s_here, p, tangent, d2_tube)
            if closest is None or d2_tube < closest[3]:
                closest = item

            if d2_tube <= tube_r2 and s_here >= min_allowed_s:
                if farthest_inside is None or s_here > farthest_inside[0]:
                    farthest_inside = item

        if closest is None:
            return 0.0, pos.astype(np.float32), np.array([1.0, 0.0, 0.0], dtype=np.float32), 0.0, False

        if farthest_inside is not None:
            s_here, p, tangent, d2 = farthest_inside
            self.progress_s = max(float(self.progress_s), float(s_here))
            return float(self.progress_s), p.astype(np.float32), tangent.astype(np.float32), float(d2), True

        # Outside the tube: do not let a nearby detour lobe pull progress backward.
        closest_s, closest_p, closest_tangent, closest_d2 = closest
        if closest_s > self.progress_s:
            self.progress_s = float(closest_s)
            return float(self.progress_s), closest_p.astype(np.float32), closest_tangent.astype(np.float32), float(closest_d2), False

        # Keep the monotonic progress state and interpolate back onto the current
        # path. This avoids oscillating around old detour points.
        p, tangent = self._interpolate_path(float(self.progress_s))
        return float(self.progress_s), p.astype(np.float32), tangent.astype(np.float32), float(closest_d2), False

    def _interpolate_path(self, s_query: float) -> tuple[np.ndarray, np.ndarray]:
        if len(self.path) == 0:
            return np.zeros(3, dtype=np.float32), np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if len(self.path) == 1 or len(self.path_s) <= 1:
            return self.path[0].copy(), np.array([1.0, 0.0, 0.0], dtype=np.float32)

        s_query = float(np.clip(s_query, 0.0, float(self.path_s[-1])))
        idx = int(np.searchsorted(self.path_s, s_query, side="right") - 1)
        idx = max(0, min(idx, len(self.path) - 2))

        s0 = float(self.path_s[idx])
        s1 = float(self.path_s[idx + 1])
        denom = max(s1 - s0, 1e-8)
        alpha = float(np.clip((s_query - s0) / denom, 0.0, 1.0))
        p = (1.0 - alpha) * self.path[idx] + alpha * self.path[idx + 1]
        tangent = self._normalize(self.path[idx + 1] - self.path[idx])
        return p.astype(np.float32), tangent.astype(np.float32)

    def _active_gate_distance(self, obs: dict[str, Any], pos: np.ndarray) -> float:
        target_gate = int(np.asarray(obs.get("target_gate", 0)).reshape(()))
        gates_pos = self._arr(obs.get("gates_pos", []))
        if target_gate < 0 or gates_pos.ndim != 2 or target_gate >= len(gates_pos):
            self.debug_active_gate_center = None
            return float("inf")
        center = gates_pos[target_gate].astype(np.float32).copy()
        self.debug_active_gate_center = center
        return float(np.linalg.norm(pos - center))

    def _tracking_speed_and_lookahead(self, obs: dict[str, Any], pos: np.ndarray, vel: np.ndarray, s_proj: float) -> tuple[float, float]:
        remaining = max(float(self.path_s[-1] - s_proj), 0.0) if len(self.path_s) else 0.0
        dist_gate = self._active_gate_distance(obs, pos)

        speed = self.cruise_speed
        lookahead = self.lookahead_nominal

        # Slow down near gates so the state controller has time to line up with the opening.
        if dist_gate < self.gate_slow_radius:
            speed = min(speed, self.gate_speed)
            lookahead = min(lookahead, 0.33)
        if dist_gate < self.gate_tight_radius:
            speed = min(speed, 0.38)
            lookahead = min(lookahead, 0.25)

        # Slow down near the end of the remaining path.
        if remaining < self.end_slow_distance:
            speed = min(speed, self.finish_speed)
            lookahead = min(lookahead, 0.28)

        # If the drone is already moving fast, do not place the target too close.
        vel_norm = float(np.linalg.norm(vel))
        lookahead = max(lookahead, 0.22 + 0.22 * min(vel_norm, 1.5))

        speed = float(np.clip(speed, self.min_speed, self.max_speed))
        lookahead = float(np.clip(lookahead, self.lookahead_min, self.lookahead_max))
        return speed, lookahead

    def compute_control(self, obs: dict[str, Any], info: dict | None = None) -> np.ndarray:
        pos = self._arr(obs["pos"], (3,))
        vel = self._arr(obs.get("vel", [0.0, 0.0, 0.0]), (3,))
        target_gate = int(np.asarray(obs.get("target_gate", 0)).reshape(()))

        # Replan immediately when the world model changes: a gate/obstacle becomes
        # exact, or the environment advances target_gate. Otherwise, replan only
        # after meaningful motion. This keeps online replanning but removes the
        # high-frequency "new path from current pose" snap.
        sig = self._obs_signature(obs)
        sig_changed = sig != self.last_obs_signature
        target_gate_changed = target_gate != self.last_target_gate
        moved_since_replan = float(np.linalg.norm(pos - self.last_replan_pos)) > self.replan_motion_trigger
        periodic_motion_replan = (self.tick % self.replan_every_steps == 0) and moved_since_replan
        force_replan = sig_changed or target_gate_changed or periodic_motion_replan
        self._rebuild_plan(obs, force=force_replan)

        cmd = np.zeros(13, dtype=np.float32)

        if target_gate < 0 or len(self.path) <= 1:
            # Finished or no usable path: hold current position gently.
            self.debug_target = pos.copy()
            self.debug_projection = pos.copy()
            cmd[0:3] = pos
            cmd[3:6] = 0.0
            cmd[6:9] = 0.0
            cmd[9] = self.last_yaw
            cmd[10:13] = 0.0
            self.last_des_vel[:] = 0.0
            self.filtered_target_pos = pos.copy()
            return cmd

        s_proj, projection, tangent_here, _d2, inside_tube = self._project_to_path_with_tube(pos)
        speed, lookahead = self._tracking_speed_and_lookahead(obs, pos, vel, s_proj)

        # Target progress is also monotonic within the current local plan. When
        # the tube projection jumps over a detour, the commanded lookahead jumps
        # with it instead of pulling back toward the skipped waypoint.
        s_target = min(max(float(self.last_target_s), s_proj + lookahead), float(self.path_s[-1]))
        self.last_target_s = s_target
        raw_target_pos, tangent_target = self._interpolate_path(s_target)
        raw_target_pos[2] = max(float(raw_target_pos[2]), self.min_z)

        # Slew-limit the actual commanded state setpoint. This prevents a replan
        # from teleporting the target across an obstacle detour and causing a
        # straight-line crash by the inner state controller.
        if self.filtered_target_pos is None or target_gate_changed:
            target_pos = raw_target_pos.astype(np.float32)
        else:
            target_pos = self.filtered_target_pos.astype(np.float32).copy()
            delta = raw_target_pos - target_pos
            dist = float(np.linalg.norm(delta))
            max_step = float(self.target_slew_rate * self.dt)
            if dist > max_step > 1e-6:
                target_pos = target_pos + delta * (max_step / dist)
            else:
                target_pos = raw_target_pos.astype(np.float32)

            # Do not allow the smoothed target to trail too far behind the raw
            # MPCC target, otherwise it can lag through a gate.
            lag = raw_target_pos - target_pos
            lag_norm = float(np.linalg.norm(lag))
            if lag_norm > self.target_filter_max_lag:
                target_pos = raw_target_pos - lag * (self.target_filter_max_lag / lag_norm)

        target_pos = target_pos.astype(np.float32)
        self.filtered_target_pos = target_pos.copy()

        des_vel_raw = speed * tangent_target
        des_vel_raw = self._clip_norm(des_vel_raw, self.max_speed)
        if not self.use_velocity_feedforward:
            des_vel = np.zeros(3, dtype=np.float32)
        else:
            des_vel_raw = self.velocity_ff_gain * des_vel_raw
            dv = des_vel_raw - self.last_des_vel
            max_dv = float(self.velocity_slew_rate * self.dt)
            des_vel = self.last_des_vel + self._clip_norm(dv, max_dv)

        if self.use_accel_feedforward:
            des_acc = (des_vel - self.last_des_vel) / max(self.dt, 1e-6)
            des_acc = self._clip_norm(des_acc, self.max_accel_ff)
        else:
            des_acc = np.zeros(3, dtype=np.float32)

        yaw_raw = self._yaw_from_vector(tangent_target, fallback=self.last_yaw)
        # Smooth only a little. Large yaw jumps are not useful for path planning here.
        yaw_err = self._angle_wrap(yaw_raw - self.last_yaw)
        yaw = self.last_yaw + float(np.clip(yaw_err, -0.10, 0.10))
        yaw = self._angle_wrap(yaw)

        cmd[0:3] = target_pos.astype(np.float32)
        cmd[3:6] = des_vel.astype(np.float32)
        cmd[6:9] = des_acc.astype(np.float32)
        cmd[9] = np.float32(yaw)
        cmd[10:13] = 0.0

        self.debug_target = target_pos.astype(np.float32)
        self.debug_projection = projection.astype(np.float32)
        self.debug_tube_projection = projection.astype(np.float32)
        line_len = 0.45 if inside_tube else 0.28
        self.debug_tangent_line = np.vstack([projection, projection + line_len * tangent_here]).astype(np.float32)
        self.last_des_vel = des_vel.astype(np.float32)
        self.last_yaw = yaw
        self.last_target_gate = target_gate

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
        # False means: let the environment decide when the episode is done.
        return False

    def episode_reset(self) -> None:
        self.tick = 0
        self.last_obs_signature = None
        self.path = np.empty((0, 3), dtype=np.float32)
        self.path_s = np.empty((0,), dtype=np.float32)
        self.progress_s = 0.0
        self.last_target_s = 0.0
        self.last_des_vel[:] = 0.0
        self.filtered_target_pos = None

    # -------------------------------------------------------------------------
    # Viewer debug drawing
    # -------------------------------------------------------------------------
    def render_callback(self, sim: Any) -> None:
        try:
            from crazyflow.sim.visualize import draw_line, draw_points
        except Exception:
            return

        try:
            # Main replanned Hermite path.
            if len(self.path) >= 2:
                draw_line(sim, np.asarray(self.path, dtype=np.float32), rgba=(0.0, 0.8, 1.0, 1.0))

            # Spline control waypoints, including gate triplets and obstacle detours.
            if len(self.waypoints) >= 1:
                draw_points(sim, np.asarray(self.waypoints, dtype=np.float32), rgba=(1.0, 0.75, 0.0, 1.0), size=0.027)

            # Detour points, if any.
            if len(self.detour_points) >= 1:
                draw_points(sim, np.asarray(self.detour_points, dtype=np.float32), rgba=(1.0, 0.0, 1.0, 1.0), size=0.045)

            # Gate centers.
            if len(self.gate_centers) >= 1:
                draw_points(sim, np.asarray(self.gate_centers, dtype=np.float32), rgba=(1.0, 0.0, 0.0, 1.0), size=0.04)

            # Current MPCC target and projection.
            draw_points(sim, np.asarray(self.debug_target, dtype=np.float32).reshape(1, 3), rgba=(0.0, 1.0, 0.0, 1.0), size=0.045)
            draw_points(sim, np.asarray(self.debug_projection, dtype=np.float32).reshape(1, 3), rgba=(1.0, 1.0, 1.0, 1.0), size=0.03)

            if len(self.debug_tangent_line) >= 2:
                draw_line(sim, np.asarray(self.debug_tangent_line, dtype=np.float32), rgba=(1.0, 1.0, 1.0, 1.0))

            # Gate crossing directions, local -x to +x.
            for normal_line in self.gate_normal_lines:
                if len(normal_line) >= 2:
                    draw_line(sim, np.asarray(normal_line, dtype=np.float32), rgba=(0.0, 1.0, 0.0, 1.0))
        except Exception:
            # Visualization should never crash the controller.
            return
