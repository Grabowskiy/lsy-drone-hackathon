"""MPCC-inspired gate/obstacle controller for the LSẎ Crazyflie hackathon.

Use with the normal *state* control mode. The controller returns the 13-D state
setpoint expected by the environment:

    [x, y, z, vx, vy, vz, ax, ay, az, yaw, roll_rate, pitch_rate, yaw_rate]

This is not the original acados MPCC backend. The uploaded acados code depends on
`multi_car_mujoco` objects and produces acceleration-command MPC controls, so it
cannot be used directly in the hackathon controller file. This file adapts the
same ideas into a dependency-free receding-horizon / MPCC-style tracker:

- gate-frame-aware pre/center/post path points,
- monotonic path progress instead of exact waypoint chasing,
- multiple lookahead points,
- lateral obstacle repulsion with inflated uncertainty margins,
- speed reduction near gates, large tracking error, and obstacles,
- smooth velocity/acceleration/yaw state setpoints.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

try:  # Works in the real repository.
    from lsy_drone_racing.control.controller import Controller as _BaseController
except Exception:  # Lets this file be imported/tested outside the repo.
    class _BaseController:  # type: ignore[too-many-ancestors]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs


_EPS = 1e-9


@dataclass(frozen=True)
class _PathPoint:
    pos: np.ndarray
    gate_idx: int
    kind: str  # "pre", "gate", "post", "finish"
    normal: np.ndarray


class MPCCGateObstacleController(_BaseController):
    """Single-file state-mode controller.

    The file intentionally contains only one Controller subclass so the course
    loader can discover it reliably.
    """

    def __init__(self, obs: dict[str, np.ndarray], info: dict | None = None, config: Any | None = None) -> None:
        try:
            super().__init__(obs, info, config)
        except TypeError:
            try:
                super().__init__()
            except Exception:
                pass

        self.config = config
        self.dt = 1.0 / float(self._cfg_get(config, ("env", "freq"), 50.0))
        if not np.isfinite(self.dt) or self.dt <= 0.0 or self.dt > 0.2:
            self.dt = 1.0 / 50.0

        # Geometry/tuning. These are deliberately conservative for Crazyflie.
        self.gate_approach_dist = 0.70
        self.gate_exit_dist = 0.82
        self.gate_pass_radius = 0.42       # precision allowance around gate center
        self.gate_funnel_radius = 1.05     # start centering before the gate
        self.gate_pin_sigma = 0.52         # keeps obstacle push from missing gates

        self.nominal_obstacle_radius = 0.30
        self.drone_radius = 0.12
        self.unknown_obstacle_margin = 0.42
        self.known_obstacle_margin = 0.26
        self.obstacle_influence_extra = 0.92
        self.max_lateral_push = 0.70
        self.max_vertical_push = 0.22

        self.max_speed = 1.45
        self.cruise_speed = 1.15
        self.gate_speed = 0.72
        self.obstacle_speed = 0.55
        self.finish_speed = 0.35
        self.max_setpoint_distance = 1.10

        self.max_vel_xy = 1.55
        self.max_vel_z = 0.75
        self.max_acc_xy = 1.95
        self.max_acc_z = 1.10
        self.max_yaw_rate = 1.40
        self.pos_smoothing = 0.38
        self.vel_smoothing = 0.45
        self.yaw_smoothing = 0.25

        # Internal state.
        self._last_target_gate: int | None = None
        self._path_progress_s: float | None = None
        self._last_des_pos: np.ndarray | None = None
        self._last_des_vel = np.zeros(3, dtype=float)
        self._last_yaw: float | None = None
        self._gate_normal_cache: dict[int, np.ndarray] = {}
        self._avoid_side_cache: dict[int, float] = {}
        self._step = 0

        # Debug data consumed by the optional render_callback if supported.
        self.debug_path: list[np.ndarray] = []
        self.debug_lookahead: list[np.ndarray] = []
        self.debug_target: np.ndarray | None = None

        self.reset()
        del obs, info

    # ------------------------------------------------------------------
    # Public controller API.
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._path_progress_s = None
        self._last_des_pos = None
        self._last_des_vel = np.zeros(3, dtype=float)
        self._last_yaw = None
        self._last_target_gate = None
        self._gate_normal_cache.clear()
        self._avoid_side_cache.clear()
        self._step = 0

    def episode_reset(self) -> None:
        self.reset()

    def episode_callback(self) -> None:  # Optional hook in the repo API.
        pass

    def step_callback(
        self,
        action: np.ndarray,
        obs: dict[str, np.ndarray],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict | None,
    ) -> bool:
        del action, obs, reward, terminated, truncated, info
        return False

    def compute_control(self, obs: dict[str, np.ndarray], info: dict | None = None) -> np.ndarray:
        del info
        self._step += 1

        pos = self._arr(obs.get("pos", np.zeros(3)), 3)
        vel = self._arr(obs.get("vel", np.zeros(3)), 3)
        gates_pos = np.asarray(obs.get("gates_pos", np.zeros((0, 3))), dtype=float)
        gates_quat = np.asarray(obs.get("gates_quat", np.zeros((len(gates_pos), 4))), dtype=float)
        target_gate = int(np.asarray(obs.get("target_gate", -1)).reshape(()))

        # Finished: hover smoothly at the current position.
        if target_gate < 0 or len(gates_pos) == 0:
            return self._hover_action(pos, vel)

        if target_gate != self._last_target_gate:
            self._last_target_gate = target_gate
            self._path_progress_s = None
            self._last_des_pos = None
            self._gate_normal_cache.clear()

        path = self._build_gate_path(pos, gates_pos, gates_quat, target_gate)
        points = np.asarray([p.pos for p in path], dtype=float)
        if len(points) < 2:
            return self._hover_action(pos, vel)

        projection_s, closest, tangent, lateral_error = self._project_to_polyline(pos, points)
        if self._path_progress_s is None:
            self._path_progress_s = projection_s
        else:
            # Monotonic MPCC-style progress, but allow tiny backwards correction
            # after a gate-pose update.
            self._path_progress_s = max(self._path_progress_s - 0.08, projection_s)
            self._path_progress_s = max(self._path_progress_s, projection_s)

        total_s = self._polyline_lengths(points)[-1]
        gate_center = gates_pos[target_gate]
        gate_normal = self._normal_for_gate(target_gate, pos, gates_pos, gates_quat)
        signed_gate_dist = float(np.dot(pos - gate_center, gate_normal))
        gate_lateral = float(np.linalg.norm((pos - gate_center) - signed_gate_dist * gate_normal))
        dist_to_gate = float(np.linalg.norm(pos - gate_center))

        obstacles = self._collect_obstacles(obs, points, self._path_progress_s)
        obstacle_danger = self._obstacle_danger(pos, obstacles)

        speed = self._choose_speed(
            lateral_error=lateral_error,
            dist_to_gate=dist_to_gate,
            signed_gate_dist=signed_gate_dist,
            gate_lateral=gate_lateral,
            obstacle_danger=obstacle_danger,
            target_gate=target_gate,
            n_gates=len(gates_pos),
        )

        lookahead = float(np.clip(0.48 + 0.42 * np.linalg.norm(vel), 0.42, 1.18))
        lookahead_s_values = [
            min(total_s, self._path_progress_s + 0.65 * lookahead),
            min(total_s, self._path_progress_s + 1.10 * lookahead),
            min(total_s, self._path_progress_s + 1.70 * lookahead),
        ]

        lookahead_points: list[np.ndarray] = []
        lookahead_tangents: list[np.ndarray] = []
        for s in lookahead_s_values:
            p, t = self._sample_polyline(points, s)
            p = self._push_point_away_from_obstacles(
                p,
                t,
                obstacles,
                gates_pos= gates_pos,
                current_gate=target_gate,
                gate_normal=gate_normal,
            )
            lookahead_points.append(p)
            lookahead_tangents.append(t)

        # Blend multiple pushed lookahead points. This acts like a short local
        # horizon and avoids chasing one jittery point.
        desired_pos = 0.62 * lookahead_points[0] + 0.26 * lookahead_points[1] + 0.12 * lookahead_points[2]

        # Gate funnel: when we are approaching the active gate, strongly remove
        # lateral error but do not require exact waypoint arrival.
        if abs(signed_gate_dist) < self.gate_funnel_radius and signed_gate_dist < self.gate_exit_dist:
            centerline_point = gate_center + gate_normal * float(np.clip(signed_gate_dist + 0.45, -0.35, 0.75))
            funnel = float(np.clip(1.0 - gate_lateral / max(self.gate_funnel_radius, _EPS), 0.0, 1.0))
            # Strongest before/in the gate, weaker after crossing.
            before_factor = 1.0 if signed_gate_dist < 0.25 else 0.55
            alpha = before_factor * (0.35 + 0.45 * funnel)
            desired_pos = (1.0 - alpha) * desired_pos + alpha * centerline_point

        desired_pos = self._clip_setpoint_distance(pos, desired_pos, self.max_setpoint_distance)
        desired_pos = self._bound_z(desired_pos, gates_pos)

        # Smooth the commanded position, but never lag too much close to a gate.
        smoothing = self.pos_smoothing
        if dist_to_gate < 0.9:
            smoothing = min(smoothing, 0.22)
        if self._last_des_pos is not None:
            desired_pos = (1.0 - smoothing) * desired_pos + smoothing * self._last_des_pos
            desired_pos = self._clip_setpoint_distance(pos, desired_pos, self.max_setpoint_distance)
            desired_pos = self._bound_z(desired_pos, gates_pos)

        # Velocity points along the locally pushed horizon, not directly at the
        # obstacle. Remove most sideways oscillation with a smoothing filter.
        direction = lookahead_points[2] - pos
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 0.05:
            direction = tangent
        else:
            direction = direction / direction_norm
        if np.linalg.norm(direction[:2]) < 0.15:
            direction = self._unit(tangent)

        desired_vel = speed * self._unit(direction)
        desired_vel = self._clip_vel(desired_vel)
        desired_vel = (1.0 - self.vel_smoothing) * desired_vel + self.vel_smoothing * self._last_des_vel
        desired_vel = self._clip_vel(desired_vel)

        # Desired acceleration feed-forward/feedback. It is a reference for the
        # Crazyflow state controller, not raw thrust, so keep it conservative.
        accel_track = 1.15 * (desired_vel - vel)
        accel_smooth = 0.55 * (desired_vel - self._last_des_vel) / max(self.dt, 1e-3)
        desired_acc = self._clip_accel(0.65 * accel_track + 0.35 * accel_smooth)

        desired_yaw = self._desired_yaw(pos, desired_pos, desired_vel, gate_normal)
        if self._last_yaw is None:
            yaw_cmd = desired_yaw
            yaw_rate_cmd = 0.0
        else:
            yaw_err = self._wrap_pi(desired_yaw - self._last_yaw)
            yaw_step = float(np.clip(yaw_err, -self.max_yaw_rate * self.dt, self.max_yaw_rate * self.dt))
            yaw_cmd = self._wrap_pi(self._last_yaw + (1.0 - self.yaw_smoothing) * yaw_step)
            yaw_rate_cmd = float(np.clip(yaw_step / max(self.dt, 1e-3), -self.max_yaw_rate, self.max_yaw_rate))

        self._last_des_pos = desired_pos.copy()
        self._last_des_vel = desired_vel.copy()
        self._last_yaw = yaw_cmd

        self.debug_path = [p.copy() for p in points]
        self.debug_lookahead = [p.copy() for p in lookahead_points]
        self.debug_target = desired_pos.copy()

        action = np.array(
            [
                desired_pos[0], desired_pos[1], desired_pos[2],
                desired_vel[0], desired_vel[1], desired_vel[2],
                desired_acc[0], desired_acc[1], desired_acc[2],
                yaw_cmd,
                0.0, 0.0, yaw_rate_cmd,
            ],
            dtype=np.float32,
        )
        return action

    def render_callback(self, sim: Any) -> None:
        """Best-effort visualization hook; safe no-op if unavailable.

        Some MuJoCo viewers expose marker APIs and some do not. This callback is
        intentionally defensive so it cannot break evaluation.
        """
        try:
            viewer = getattr(sim, "viewer", None)
            if viewer is None or not hasattr(viewer, "add_marker"):
                return
            for point in self.debug_path[:18]:
                viewer.add_marker(pos=point, size=np.array([0.035, 0.035, 0.035]), rgba=np.array([0.1, 0.7, 1.0, 0.35]), type=2)
            for point in self.debug_lookahead[:3]:
                viewer.add_marker(pos=point, size=np.array([0.055, 0.055, 0.055]), rgba=np.array([1.0, 0.7, 0.1, 0.55]), type=2)
            if self.debug_target is not None:
                viewer.add_marker(pos=self.debug_target, size=np.array([0.07, 0.07, 0.07]), rgba=np.array([0.1, 1.0, 0.1, 0.75]), type=2)
        except Exception:
            return

    # ------------------------------------------------------------------
    # Path construction and sampling.
    # ------------------------------------------------------------------
    def _build_gate_path(
        self,
        pos: np.ndarray,
        gates_pos: np.ndarray,
        gates_quat: np.ndarray,
        target_gate: int,
    ) -> list[_PathPoint]:
        n_gates = len(gates_pos)
        start = int(np.clip(target_gate, 0, max(n_gates - 1, 0)))
        path: list[_PathPoint] = []

        for i in range(start, n_gates):
            prev_context = pos if i == start else gates_pos[i - 1]
            if i + 1 < n_gates:
                next_context = gates_pos[i + 1]
            else:
                next_context = gates_pos[i] + self._unit(gates_pos[i] - prev_context)

            normal = self._normal_for_gate(i, prev_context, gates_pos, gates_quat, next_context=next_context)

            # Use shorter approach if two gates are close, but keep a real post
            # point so the drone exits instead of sitting in the gate.
            if i + 1 < n_gates:
                spacing = float(np.linalg.norm(gates_pos[i + 1] - gates_pos[i]))
                pre_d = min(self.gate_approach_dist, max(0.42, 0.30 * spacing))
                post_d = min(self.gate_exit_dist, max(0.50, 0.34 * spacing))
            else:
                pre_d = self.gate_approach_dist
                post_d = self.gate_exit_dist

            gate = np.asarray(gates_pos[i], dtype=float).reshape(3)
            path.append(_PathPoint(gate - normal * pre_d, i, "pre", normal))
            path.append(_PathPoint(gate, i, "gate", normal))
            path.append(_PathPoint(gate + normal * post_d, i, "post", normal))

        if path:
            finish_normal = path[-1].normal
            finish_pos = path[-1].pos + finish_normal * 0.50
            path.append(_PathPoint(finish_pos, path[-1].gate_idx, "finish", finish_normal))

        # Remove nearly duplicate consecutive points.
        compact: list[_PathPoint] = []
        for p in path:
            if not compact or np.linalg.norm(p.pos - compact[-1].pos) > 0.08:
                compact.append(p)
        return compact

    def _normal_for_gate(
        self,
        gate_idx: int,
        prev_context: np.ndarray,
        gates_pos: np.ndarray,
        gates_quat: np.ndarray,
        *,
        next_context: np.ndarray | None = None,
    ) -> np.ndarray:
        gate_idx = int(gate_idx)
        if gate_idx in self._gate_normal_cache:
            # Keep the sign fixed for this gate. Re-signing from the current
            # position after the drone has crossed the gate is exactly what can
            # make a controller turn around and get stuck inside the frame.
            return self._gate_normal_cache[gate_idx]

        q = gates_quat[gate_idx] if gate_idx < len(gates_quat) else np.array([0.0, 0.0, 0.0, 1.0])
        rot = self._quat_xyzw_to_rot(q)
        axes = [rot[:, 0], rot[:, 1], rot[:, 2], -rot[:, 0], -rot[:, 1], -rot[:, 2]]

        gate = gates_pos[gate_idx]
        if next_context is None:
            if gate_idx + 1 < len(gates_pos):
                next_context = gates_pos[gate_idx + 1]
            else:
                next_context = gate + self._unit(gate - prev_context)
        route = self._unit(np.asarray(next_context, dtype=float).reshape(3) - np.asarray(prev_context, dtype=float).reshape(3))
        if np.linalg.norm(route) < 0.5:
            route = self._unit(gate - np.asarray(prev_context, dtype=float).reshape(3))

        # Gate normals are expected to be mostly horizontal. Penalize vertical
        # axes unless the course itself is nearly vertical.
        best_axis = axes[0]
        best_score = -1e9
        for axis in axes:
            axis = self._unit(axis)
            vertical_penalty = 0.55 * abs(axis[2])
            score = float(np.dot(axis, route)) - vertical_penalty
            if score > best_score:
                best_score = score
                best_axis = axis
        cached = self._unit(best_axis)
        self._gate_normal_cache[gate_idx] = cached
        return cached

    def _project_to_polyline(self, pos: np.ndarray, points: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
        if len(points) == 1:
            return 0.0, points[0], np.array([1.0, 0.0, 0.0]), float(np.linalg.norm(pos - points[0]))
        lengths = self._polyline_lengths(points)
        best_dist2 = float("inf")
        best_s = 0.0
        best_point = points[0]
        best_tangent = self._unit(points[1] - points[0])
        for i in range(len(points) - 1):
            a = points[i]
            b = points[i + 1]
            ab = b - a
            seg_len2 = float(np.dot(ab, ab))
            if seg_len2 <= _EPS:
                continue
            u = float(np.clip(np.dot(pos - a, ab) / seg_len2, 0.0, 1.0))
            p = a + u * ab
            d2 = float(np.dot(pos - p, pos - p))
            if d2 < best_dist2:
                best_dist2 = d2
                best_s = lengths[i] + u * math.sqrt(seg_len2)
                best_point = p
                best_tangent = self._unit(ab)
        return best_s, best_point, best_tangent, math.sqrt(max(best_dist2, 0.0))

    def _sample_polyline(self, points: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray]:
        if len(points) == 1:
            return points[0].copy(), np.array([1.0, 0.0, 0.0])
        lengths = self._polyline_lengths(points)
        total = lengths[-1]
        s = float(np.clip(s, 0.0, total))
        idx = int(np.searchsorted(lengths, s, side="right") - 1)
        idx = int(np.clip(idx, 0, len(points) - 2))
        a = points[idx]
        b = points[idx + 1]
        seg = b - a
        seg_len = float(np.linalg.norm(seg))
        if seg_len <= _EPS:
            return a.copy(), np.array([1.0, 0.0, 0.0])
        u = (s - lengths[idx]) / seg_len
        return a + u * seg, self._unit(seg)

    def _polyline_lengths(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.zeros(1)
        lengths = np.zeros(len(points), dtype=float)
        for i in range(1, len(points)):
            lengths[i] = lengths[i - 1] + float(np.linalg.norm(points[i] - points[i - 1]))
        return lengths

    # ------------------------------------------------------------------
    # Obstacles and avoidance.
    # ------------------------------------------------------------------
    def _collect_obstacles(self, obs: dict[str, np.ndarray], points: np.ndarray, progress_s: float) -> list[dict[str, Any]]:
        raw = np.asarray(obs.get("obstacles_pos", np.zeros((0, 3))), dtype=float)
        if raw.ndim != 2 or raw.shape[1] != 3 or len(raw) == 0:
            return []
        visited = np.asarray(obs.get("obstacles_visited", np.zeros(len(raw), dtype=bool))).astype(bool).reshape(-1)
        if len(visited) < len(raw):
            visited = np.pad(visited, (0, len(raw) - len(visited)), constant_values=False)

        result: list[dict[str, Any]] = []
        for j, center in enumerate(raw):
            if not np.all(np.isfinite(center)):
                continue
            obs_s, _, _, obs_path_dist = self._project_to_polyline(center, points)
            # Obstacles far behind the current path progress do not matter.
            if obs_s < progress_s - 0.75 and obs_path_dist > 0.70:
                continue
            exact_or_seen = bool(visited[j])
            margin = self.known_obstacle_margin if exact_or_seen else self.unknown_obstacle_margin
            safe_radius = self.drone_radius + self.nominal_obstacle_radius + margin
            influence = safe_radius + self.obstacle_influence_extra
            result.append(
                {
                    "idx": j,
                    "center": center.astype(float),
                    "safe_radius": float(safe_radius),
                    "influence": float(influence),
                    "seen": exact_or_seen,
                    "path_s": float(obs_s),
                }
            )
        return result

    def _push_point_away_from_obstacles(
        self,
        point: np.ndarray,
        tangent: np.ndarray,
        obstacles: list[dict[str, Any]],
        *,
        gates_pos: np.ndarray,
        current_gate: int,
        gate_normal: np.ndarray,
    ) -> np.ndarray:
        if not obstacles:
            return point.copy()

        tangent = self._unit(tangent)
        total_offset = np.zeros(3, dtype=float)
        max_strength = 0.0

        for obstacle in obstacles:
            idx = int(obstacle["idx"])
            center = np.asarray(obstacle["center"], dtype=float)
            rel = point - center
            dist = float(np.linalg.norm(rel))
            influence = float(obstacle["influence"])
            if dist >= influence:
                continue

            safe_radius = float(obstacle["safe_radius"])
            clearance = dist - safe_radius
            strength = float(np.clip((influence - dist) / max(influence - safe_radius, 0.1), 0.0, 1.0)) ** 2
            if clearance < 0.0:
                strength = max(strength, 1.0)
            max_strength = max(max_strength, strength)

            away = self._unit(rel)
            # Avoid pushing backwards/forwards along the path; sidestep instead.
            lateral = away - tangent * float(np.dot(away, tangent))
            lateral[2] *= 0.35
            if np.linalg.norm(lateral[:2]) < 0.08:
                side = self._avoid_side_cache.get(idx)
                if side is None:
                    obs_left = float(np.cross(np.r_[tangent[:2], 0.0], np.r_[(center - point)[:2], 0.0])[2])
                    side = -1.0 if obs_left >= 0.0 else 1.0
                    self._avoid_side_cache[idx] = side
                lateral = np.array([-tangent[1] * side, tangent[0] * side, 0.0], dtype=float)
            lateral = self._unit(lateral)

            push = strength * self.max_lateral_push * lateral
            push[2] = float(np.clip(push[2], -self.max_vertical_push, self.max_vertical_push))
            total_offset += push

        if np.linalg.norm(total_offset) < 1e-6:
            return point.copy()

        # Do not allow obstacle repulsion to make us miss the active gate unless
        # a collision is imminent. The path itself already goes through center.
        gate = gates_pos[current_gate]
        signed = float(np.dot(point - gate, gate_normal))
        lateral_to_gate = float(np.linalg.norm((point - gate) - signed * gate_normal))
        near_gate = math.exp(-0.5 * (lateral_to_gate / max(self.gate_pin_sigma, 1e-3)) ** 2) * math.exp(
            -0.5 * (abs(signed) / 0.85) ** 2
        )
        pin = 0.78 * near_gate * (1.0 - min(max_strength, 1.0) * 0.35)
        offset_scale = float(np.clip(1.0 - pin, 0.20, 1.0))
        pushed = point + offset_scale * total_offset
        return self._bound_z(pushed, gates_pos)

    def _obstacle_danger(self, pos: np.ndarray, obstacles: list[dict[str, Any]]) -> float:
        danger = 0.0
        for obstacle in obstacles:
            center = np.asarray(obstacle["center"], dtype=float)
            dist = float(np.linalg.norm(pos - center))
            safe_radius = float(obstacle["safe_radius"])
            influence = max(float(obstacle["influence"]), safe_radius + 0.1)
            d = float(np.clip((influence - dist) / (influence - safe_radius), 0.0, 1.0))
            danger = max(danger, d)
        return danger

    # ------------------------------------------------------------------
    # Command generation.
    # ------------------------------------------------------------------
    def _choose_speed(
        self,
        *,
        lateral_error: float,
        dist_to_gate: float,
        signed_gate_dist: float,
        gate_lateral: float,
        obstacle_danger: float,
        target_gate: int,
        n_gates: int,
    ) -> float:
        speed = self.cruise_speed
        if dist_to_gate < 1.35 and signed_gate_dist < 0.85:
            speed = min(speed, self.gate_speed)
        if obstacle_danger > 0.05:
            speed = min(speed, self.obstacle_speed + 0.35 * (1.0 - obstacle_danger))
        if lateral_error > 0.55:
            speed *= float(np.clip(1.0 - 0.38 * (lateral_error - 0.55), 0.50, 1.0))
        if gate_lateral > self.gate_pass_radius and abs(signed_gate_dist) < 0.75:
            speed *= 0.78
        if target_gate >= n_gates - 1 and dist_to_gate < 0.95 and signed_gate_dist > 0.25:
            speed = min(speed, self.finish_speed)
        return float(np.clip(speed, 0.28, self.max_speed))

    def _desired_yaw(self, pos: np.ndarray, desired_pos: np.ndarray, desired_vel: np.ndarray, gate_normal: np.ndarray) -> float:
        del pos
        yaw_vec = desired_vel.copy()
        if np.linalg.norm(yaw_vec[:2]) < 0.25:
            yaw_vec = desired_pos - (self._last_des_pos if self._last_des_pos is not None else desired_pos - gate_normal)
        if np.linalg.norm(yaw_vec[:2]) < 0.08:
            yaw_vec = gate_normal
        return math.atan2(float(yaw_vec[1]), float(yaw_vec[0]))

    def _hover_action(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        des_pos = pos.copy()
        if self._last_des_pos is not None:
            des_pos = 0.72 * self._last_des_pos + 0.28 * pos
        des_vel = np.zeros(3, dtype=float)
        des_acc = self._clip_accel(-0.75 * vel)
        yaw = self._last_yaw if self._last_yaw is not None else 0.0
        self._last_des_pos = des_pos.copy()
        self._last_des_vel = des_vel.copy()
        self._last_yaw = yaw
        return np.array(
            [des_pos[0], des_pos[1], des_pos[2], 0.0, 0.0, 0.0, des_acc[0], des_acc[1], des_acc[2], yaw, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Math helpers.
    # ------------------------------------------------------------------
    def _clip_setpoint_distance(self, pos: np.ndarray, target: np.ndarray, max_dist: float) -> np.ndarray:
        delta = target - pos
        dist = float(np.linalg.norm(delta))
        if dist <= max_dist:
            return target.copy()
        return pos + delta * (max_dist / max(dist, _EPS))

    def _clip_vel(self, vel: np.ndarray) -> np.ndarray:
        out = vel.copy()
        xy = float(np.linalg.norm(out[:2]))
        if xy > self.max_vel_xy:
            out[:2] *= self.max_vel_xy / max(xy, _EPS)
        out[2] = float(np.clip(out[2], -self.max_vel_z, self.max_vel_z))
        return out

    def _clip_accel(self, acc: np.ndarray) -> np.ndarray:
        out = acc.copy()
        xy = float(np.linalg.norm(out[:2]))
        if xy > self.max_acc_xy:
            out[:2] *= self.max_acc_xy / max(xy, _EPS)
        out[2] = float(np.clip(out[2], -self.max_acc_z, self.max_acc_z))
        return out

    def _bound_z(self, point: np.ndarray, gates_pos: np.ndarray) -> np.ndarray:
        out = point.copy()
        if len(gates_pos):
            z_min = max(0.25, float(np.min(gates_pos[:, 2]) - 0.75))
            z_max = min(3.0, float(np.max(gates_pos[:, 2]) + 0.75))
        else:
            z_min, z_max = 0.25, 3.0
        out[2] = float(np.clip(out[2], z_min, z_max))
        return out

    def _arr(self, value: Any, size: int) -> np.ndarray:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if len(arr) < size:
            arr = np.pad(arr, (0, size - len(arr)), constant_values=0.0)
        return arr[:size].astype(float)

    def _unit(self, vec: Iterable[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=float).reshape(3)
        n = float(np.linalg.norm(arr))
        if n <= _EPS or not np.isfinite(n):
            return np.zeros(3, dtype=float)
        return arr / n

    def _wrap_pi(self, angle: float) -> float:
        return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi

    def _quat_xyzw_to_rot(self, quat: np.ndarray) -> np.ndarray:
        q = np.asarray(quat, dtype=float).reshape(-1)
        if len(q) < 4 or not np.all(np.isfinite(q)) or np.linalg.norm(q[:4]) < 1e-6:
            return np.eye(3)
        x, y, z, w = q[:4] / np.linalg.norm(q[:4])
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=float,
        )

    def _cfg_get(self, obj: Any, path: tuple[str, ...], default: Any) -> Any:
        cur = obj
        for key in path:
            if cur is None:
                return default
            if isinstance(cur, dict):
                cur = cur.get(key, default)
            else:
                cur = getattr(cur, key, default)
        return cur
