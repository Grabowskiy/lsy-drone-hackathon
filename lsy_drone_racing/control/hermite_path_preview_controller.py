from __future__ import annotations

from typing import Any

import numpy as np

from lsy_drone_racing.control import Controller


class HermitePathPreviewController(Controller):
    """
    Preview-only Hermite path planner for the drone racing task.

    What this controller does:
        - Reads gates from obs["gates_pos"] and obs["gates_quat"].
        - Builds approach / center / exit waypoints for every remaining gate.
        - Samples a cubic Hermite spline through those waypoints.
        - Draws the predicted path in the MuJoCo/Crazyflow viewer.
        - Commands the drone to hold its initial position; it does NOT track the path.

    Config requirement:
        [env]
        control_mode = "state"

    Run example:
        python scripts/sim.py --config level2.toml --controller hermite_path_preview_controller.py --render=True
    """

    def __init__(self, obs: dict[str, Any], info: dict | None, config: Any):
        super().__init__(obs, info, config)

        self.freq = int(getattr(config.env, "freq", 50))
        self.tick = 0

        # The actual command is only a hold command. The path is visualization-only.
        self.hold_pos = self._arr(obs["pos"], (3,)).copy()
        self.hold_yaw = self._yaw_from_quat(self._arr(obs.get("quat", [0.0, 0.0, 0.0, 1.0]), (4,)))

        # Path-planning/preview parameters.
        self.approach_dist = 0.55       # meters before each gate, along -gate_x
        self.exit_dist = 0.55           # meters after each gate, along +gate_x
        self.min_z = 0.25               # do not draw below this height
        self.samples_per_segment = 24   # more = smoother preview line
        self.tangent_scale = 0.50       # 0.0 = straight-ish, 0.5 = Catmull-Rom-like
        self.replan_every_steps = 10    # cheap, but no need to do it every render frame

        # Debug drawing buffers.
        self.debug_path: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.debug_waypoints: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.debug_gate_normals: list[np.ndarray] = []
        self.debug_gate_centers: np.ndarray = np.empty((0, 3), dtype=np.float32)

        self._last_signature: tuple[Any, ...] | None = None
        self._rebuild_preview(obs, force=True)

    @staticmethod
    def _arr(x: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if shape is not None:
            arr = arr.reshape(shape)
        return arr

    @staticmethod
    def _normalize(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n > 1e-8:
            return (v / n).astype(np.float32)
        if fallback is None:
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return np.asarray(fallback, dtype=np.float32)

    @staticmethod
    def _quat_rotate_xyzw(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate vector v by quaternion q in xyzw convention, without scipy."""
        q = np.asarray(q_xyzw, dtype=np.float32).copy()
        norm = float(np.linalg.norm(q))
        if norm < 1e-8:
            return np.asarray(v, dtype=np.float32).copy()
        q /= norm

        q_vec = q[:3]
        q_w = float(q[3])
        v = np.asarray(v, dtype=np.float32)

        # Equivalent to q * [v, 0] * q_conjugate.
        t = 2.0 * np.cross(q_vec, v)
        return (v + q_w * t + np.cross(q_vec, t)).astype(np.float32)

    @staticmethod
    def _yaw_from_quat(q_xyzw: np.ndarray) -> float:
        """Extract yaw from xyzw quaternion. Only used for the hold command."""
        x, y, z, w = [float(a) for a in q_xyzw]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    @staticmethod
    def _hermite_segment(p0: np.ndarray, p1: np.ndarray, m0: np.ndarray, m1: np.ndarray, n: int) -> np.ndarray:
        """Sample one cubic Hermite segment, excluding the final endpoint."""
        s = np.linspace(0.0, 1.0, max(2, int(n)), endpoint=False, dtype=np.float32).reshape(-1, 1)
        s2 = s * s
        s3 = s2 * s

        h00 = 2.0 * s3 - 3.0 * s2 + 1.0
        h10 = s3 - 2.0 * s2 + s
        h01 = -2.0 * s3 + 3.0 * s2
        h11 = s3 - s2

        return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

    def _gate_triplet(self, gate_pos: np.ndarray, gate_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build approach, center, exit points for one gate.

        Gate convention used by the challenge:
            cross from local -x to local +x.
        """
        center = np.asarray(gate_pos, dtype=np.float32).copy()
        center[2] = max(float(center[2]), self.min_z)

        gate_x = self._quat_rotate_xyzw(gate_quat, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        gate_x = self._normalize(gate_x)

        approach = center - self.approach_dist * gate_x
        exit_pt = center + self.exit_dist * gate_x
        approach[2] = max(float(approach[2]), self.min_z)
        exit_pt[2] = max(float(exit_pt[2]), self.min_z)

        return approach.astype(np.float32), center.astype(np.float32), exit_pt.astype(np.float32), gate_x.astype(np.float32)

    def _build_waypoints(self, obs: dict[str, Any]) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        pos = self._arr(obs["pos"], (3,))
        gates_pos = self._arr(obs["gates_pos"])
        gates_quat = self._arr(obs["gates_quat"])
        target_gate = int(np.asarray(obs.get("target_gate", 0)).reshape(()))

        if gates_pos.ndim != 2 or gates_pos.shape[1] != 3:
            return pos.reshape(1, 3).astype(np.float32), [], np.empty((0, 3), dtype=np.float32)
        if gates_quat.ndim != 2 or gates_quat.shape[1] != 4:
            return pos.reshape(1, 3).astype(np.float32), [], np.empty((0, 3), dtype=np.float32)

        n_gates = min(gates_pos.shape[0], gates_quat.shape[0])
        if target_gate < 0:
            start_idx = n_gates
        else:
            start_idx = max(0, min(target_gate, n_gates))

        waypoints: list[np.ndarray] = [pos.astype(np.float32)]
        gate_normals: list[np.ndarray] = []
        gate_centers: list[np.ndarray] = []

        for i in range(start_idx, n_gates):
            approach, center, exit_pt, gate_x = self._gate_triplet(gates_pos[i], gates_quat[i])
            waypoints.extend([approach, center, exit_pt])
            gate_centers.append(center)

            # Small line showing the intended crossing direction at this gate.
            normal_line = np.vstack([
                center - 0.25 * gate_x,
                center + 0.25 * gate_x,
            ]).astype(np.float32)
            gate_normals.append(normal_line)

        wp = np.vstack(waypoints).astype(np.float32)
        wp = self._remove_near_duplicate_points(wp, min_dist=0.025)
        centers = np.vstack(gate_centers).astype(np.float32) if gate_centers else np.empty((0, 3), dtype=np.float32)
        return wp, gate_normals, centers

    @staticmethod
    def _remove_near_duplicate_points(points: np.ndarray, min_dist: float) -> np.ndarray:
        if len(points) <= 1:
            return points.astype(np.float32)
        kept = [points[0].astype(np.float32)]
        for p in points[1:]:
            if float(np.linalg.norm(p - kept[-1])) >= min_dist:
                kept.append(p.astype(np.float32))
        return np.vstack(kept).astype(np.float32)

    def _compute_tangents(self, points: np.ndarray) -> np.ndarray:
        """
        Catmull-Rom-style tangents for Hermite segments.

        These are deliberately conservative. Larger tangent_scale gives a rounder but
        more overshooting path; smaller tangent_scale gives a tighter/polyline-like path.
        """
        n = len(points)
        tangents = np.zeros_like(points, dtype=np.float32)
        if n <= 1:
            return tangents

        for i in range(n):
            if i == 0:
                raw = points[1] - points[0]
            elif i == n - 1:
                raw = points[-1] - points[-2]
            else:
                raw = points[i + 1] - points[i - 1]
            tangents[i] = self.tangent_scale * raw

        return tangents.astype(np.float32)

    def _sample_hermite_path(self, waypoints: np.ndarray) -> np.ndarray:
        if len(waypoints) <= 1:
            return waypoints.astype(np.float32)

        tangents = self._compute_tangents(waypoints)
        chunks: list[np.ndarray] = []
        for i in range(len(waypoints) - 1):
            p0 = waypoints[i]
            p1 = waypoints[i + 1]
            m0 = tangents[i]
            m1 = tangents[i + 1]
            chunks.append(self._hermite_segment(p0, p1, m0, m1, self.samples_per_segment))

        chunks.append(waypoints[-1].reshape(1, 3))
        path = np.vstack(chunks).astype(np.float32)
        return self._remove_near_duplicate_points(path, min_dist=0.005)

    def _obs_signature(self, obs: dict[str, Any]) -> tuple[Any, ...]:
        """Compact signature to avoid rebuilding the exact same path every step."""
        target_gate = int(np.asarray(obs.get("target_gate", 0)).reshape(()))
        gates_pos = np.asarray(obs.get("gates_pos", []), dtype=np.float32)
        gates_quat = np.asarray(obs.get("gates_quat", []), dtype=np.float32)

        # Rounding avoids tiny float noise triggering constant replans.
        return (
            target_gate,
            tuple(np.round(gates_pos.reshape(-1), 3).tolist()),
            tuple(np.round(gates_quat.reshape(-1), 3).tolist()),
        )

    def _rebuild_preview(self, obs: dict[str, Any], force: bool = False) -> None:
        sig = self._obs_signature(obs)
        if not force and sig == self._last_signature:
            return

        self._last_signature = sig
        waypoints, normals, centers = self._build_waypoints(obs)
        self.debug_waypoints = waypoints
        self.debug_gate_normals = normals
        self.debug_gate_centers = centers
        self.debug_path = self._sample_hermite_path(waypoints)

    def compute_control(self, obs: dict[str, Any], info: dict | None = None) -> np.ndarray:
        # The planner can re-run when gate observations change, but the command stays a hold.
        if self.tick % self.replan_every_steps == 0:
            self._rebuild_preview(obs, force=False)

        cmd = np.zeros(13, dtype=np.float32)
        cmd[0:3] = self.hold_pos
        cmd[3:6] = 0.0
        cmd[6:9] = 0.0
        cmd[9] = self.hold_yaw
        cmd[10:13] = 0.0
        return cmd

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
        # False means: do not end the episode from the controller side.
        return False

    def episode_reset(self) -> None:
        self.tick = 0
        self._last_signature = None

    def render_callback(self, sim: Any) -> None:
        """Draw the preview path. This is skipped silently if visualization helpers are unavailable."""
        if self.debug_path is None or len(self.debug_path) < 1:
            return

        try:
            from crazyflow.sim.visualize import draw_line, draw_points
        except Exception:
            return

        try:
            # Full Hermite path.
            if len(self.debug_path) >= 2:
                draw_line(
                    sim,
                    np.asarray(self.debug_path, dtype=np.float32),
                    rgba=(0.0, 0.8, 1.0, 1.0),
                )

            # Raw spline waypoints: start, approach, centers, exits.
            if len(self.debug_waypoints) >= 1:
                draw_points(
                    sim,
                    np.asarray(self.debug_waypoints, dtype=np.float32),
                    rgba=(1.0, 0.75, 0.0, 1.0),
                    size=0.035,
                )

            # Gate centers separately, so you can verify that the curve goes through them.
            if len(self.debug_gate_centers) >= 1:
                draw_points(
                    sim,
                    np.asarray(self.debug_gate_centers, dtype=np.float32),
                    rgba=(1.0, 0.0, 0.0, 1.0),
                    size=0.045,
                )

            # Gate crossing directions, local -x to +x.
            for normal_line in self.debug_gate_normals:
                if len(normal_line) >= 2:
                    draw_line(
                        sim,
                        np.asarray(normal_line, dtype=np.float32),
                        rgba=(0.0, 1.0, 0.0, 1.0),
                    )
        except Exception:
            # Visualization must never crash the controller/simulation.
            return
