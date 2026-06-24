#!/usr/bin/env python3
"""Analyze literal-line MPC tracking logs.

This script post-processes the CSV written by
my_controller_acados_state_literal_line_logged.py.

Example:
    python analyze_mpc_tracking.py \
        --log debug_logs/literal_line_tracking.csv \
        --outdir debug_logs/tracking_report

Outputs:
    metrics.txt
    tracking_errors.png
    trajectory_xyz.png
    topdown_xy.png
    tracking_summary.npz
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np


def _find_header_row(path: Path) -> int:
    """Return the 0-based row index containing the CSV header."""
    with path.open("r", newline="") as f:
        for i, row in enumerate(csv.reader(f)):
            if row and row[0] == "tick":
                return i
    raise RuntimeError(f"Could not find header row starting with 'tick' in {path}")


def _load_csv(path: Path) -> np.ndarray:
    header_row = _find_header_row(path)
    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
        skip_header=header_row,
        comments=None,
    )
    if data.size == 0:
        raise RuntimeError(f"No samples found in {path}")
    if data.shape == ():
        data = np.array([data], dtype=data.dtype)
    return data


def _col(data: np.ndarray, name: str) -> np.ndarray:
    return np.asarray(data[name], dtype=float)


def _vec(data: np.ndarray, prefix: str) -> np.ndarray:
    return np.column_stack([_col(data, f"{prefix}_x"), _col(data, f"{prefix}_y"), _col(data, f"{prefix}_z")])


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x)))


def _pctl(x: np.ndarray, q: float) -> float:
    return float(np.percentile(np.asarray(x, dtype=float), q))


def _finite_diff(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return np.zeros_like(x)
    dt = np.gradient(t)
    dt = np.where(np.abs(dt) < 1e-9, np.nan, dt)
    return np.gradient(x, axis=0) / dt[:, None]


def _estimate_lag_seconds(s_actual: np.ndarray, s_cmd: np.ndarray, t: np.ndarray) -> float:
    """Crude lag estimate based on along-track progress cross-correlation.

    Positive lag means actual progress appears delayed relative to command progress.
    This is only meaningful while the line progress is changing.
    """
    if len(t) < 10:
        return float("nan")
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return float("nan")

    a = np.asarray(s_actual, dtype=float) - np.mean(s_actual)
    c = np.asarray(s_cmd, dtype=float) - np.mean(s_cmd)
    if np.std(a) < 1e-6 or np.std(c) < 1e-6:
        return float("nan")
    corr = np.correlate(c, a, mode="full")
    lags = np.arange(-len(a) + 1, len(a))
    lag_samples = int(lags[int(np.argmax(corr))])
    return lag_samples * dt


def analyze(log_path: Path, outdir: Path, trim_takeoff_s: float = 0.0) -> dict[str, float]:
    data = _load_csv(log_path)
    outdir.mkdir(parents=True, exist_ok=True)

    t = _col(data, "t")
    t = t - t[0]
    mask = t >= trim_takeoff_s
    if np.count_nonzero(mask) < 3:
        raise RuntimeError("Too few samples after trim; reduce --trim-takeoff-s")

    t = t[mask]
    pos = _vec(data, "pos")[mask]
    vel = _vec(data, "vel")[mask]
    cmd = _vec(data, "cmd")[mask]
    cmd_v = np.column_stack([_col(data, "cmd_vx"), _col(data, "cmd_vy"), _col(data, "cmd_vz")])[mask]
    ref0 = _vec(data, "ref0")[mask]
    ref0_v = np.column_stack([_col(data, "ref0_vx"), _col(data, "ref0_vy"), _col(data, "ref0_vz")])[mask]

    err_cmd = pos - cmd
    err_ref = pos - ref0
    vel_err_cmd = vel - cmd_v
    vel_err_ref = vel - ref0_v

    err_cmd_3d = np.linalg.norm(err_cmd, axis=1)
    err_cmd_xy = np.linalg.norm(err_cmd[:, :2], axis=1)
    err_ref_3d = np.linalg.norm(err_ref, axis=1)
    cross_track_3d = _col(data, "cross_track_3d")[mask]
    cross_track_xy = _col(data, "cross_track_xy")[mask]
    along_err_cmd = _col(data, "along_err_cmd")[mask]
    s_actual = _col(data, "s_actual")[mask]
    s_cmd = _col(data, "s_cmd")[mask]

    # Also compute acceleration magnitude from measured velocity, useful for diagnosing
    # whether the Crazyflie/state controller is being asked for unrealistic motion.
    acc_est = _finite_diff(vel, t)
    speed = np.linalg.norm(vel, axis=1)
    cmd_speed = np.linalg.norm(cmd_v, axis=1)

    metrics = {
        "samples": float(len(t)),
        "duration_s": float(t[-1] - t[0]),
        "mean_dt_s": float(np.mean(np.diff(t))) if len(t) > 1 else float("nan"),
        "cmd_pos_rms_3d_m": _rms(err_cmd_3d),
        "cmd_pos_p95_3d_m": _pctl(err_cmd_3d, 95),
        "cmd_pos_max_3d_m": float(np.max(err_cmd_3d)),
        "cmd_pos_rms_xy_m": _rms(err_cmd_xy),
        "cmd_z_rms_m": _rms(err_cmd[:, 2]),
        "ref0_pos_rms_3d_m": _rms(err_ref_3d),
        "line_cross_track_rms_3d_m": _rms(cross_track_3d),
        "line_cross_track_p95_3d_m": _pctl(cross_track_3d, 95),
        "line_cross_track_max_3d_m": float(np.max(cross_track_3d)),
        "line_cross_track_rms_xy_m": _rms(cross_track_xy),
        "along_error_cmd_rms_m": _rms(along_err_cmd),
        "along_error_cmd_mean_m": float(np.mean(along_err_cmd)),
        "vel_err_cmd_rms_mps": _rms(np.linalg.norm(vel_err_cmd, axis=1)),
        "vel_err_ref0_rms_mps": _rms(np.linalg.norm(vel_err_ref, axis=1)),
        "actual_speed_mean_mps": float(np.mean(speed)),
        "actual_speed_max_mps": float(np.max(speed)),
        "cmd_speed_mean_mps": float(np.mean(cmd_speed)),
        "cmd_speed_max_mps": float(np.max(cmd_speed)),
        "acc_est_p95_mps2": _pctl(np.linalg.norm(acc_est, axis=1), 95),
        "estimated_along_track_lag_s": _estimate_lag_seconds(s_actual, s_cmd, t),
    }

    metrics_txt = outdir / "metrics.txt"
    with metrics_txt.open("w") as f:
        f.write("MPC / Crazyflie state-mode tracking metrics\n")
        f.write(f"log: {log_path}\n")
        f.write(f"trim_takeoff_s: {trim_takeoff_s}\n\n")
        for k, v in metrics.items():
            if k == "samples":
                f.write(f"{k}: {int(v)}\n")
            else:
                f.write(f"{k}: {v:.6g}\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1) Tracking error time series.
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(t, err_cmd_3d, label="|actual - commanded setpoint| 3D")
        ax.plot(t, cross_track_3d, label="distance to ideal line 3D")
        ax.plot(t, np.abs(along_err_cmd), label="|along-track lag to command|")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Error [m]")
        ax.set_title("MPC/state-controller tracking errors")
        ax.grid(True)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(outdir / "tracking_errors.png", dpi=160)
        plt.close(fig)

        # 2) XYZ trajectory against command.
        fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        labels = ["x", "y", "z"]
        for i, ax in enumerate(axs):
            ax.plot(t, pos[:, i], label=f"actual {labels[i]}")
            ax.plot(t, cmd[:, i], "--", label=f"cmd {labels[i]}")
            ax.set_ylabel(f"{labels[i]} [m]")
            ax.grid(True)
            ax.legend(loc="best")
        axs[-1].set_xlabel("Time [s]")
        fig.suptitle("Actual vs commanded state setpoint")
        fig.tight_layout()
        fig.savefig(outdir / "trajectory_xyz.png", dpi=160)
        plt.close(fig)

        # 3) Top-down path.
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(pos[:, 0], pos[:, 1], label="actual XY")
        ax.plot(cmd[:, 0], cmd[:, 1], "--", label="commanded XY")
        ax.plot(ref0[:, 0], ref0[:, 1], ":", label="reference XY")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.axis("equal")
        ax.grid(True)
        ax.set_title("Top-down XY tracking")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(outdir / "topdown_xy.png", dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f"Plotting skipped: {exc}")

    np.savez(
        outdir / "tracking_summary.npz",
        t=t,
        pos=pos,
        vel=vel,
        cmd=cmd,
        cmd_v=cmd_v,
        ref0=ref0,
        err_cmd=err_cmd,
        err_ref=err_ref,
        cross_track_3d=cross_track_3d,
        cross_track_xy=cross_track_xy,
        along_err_cmd=along_err_cmd,
        metrics_keys=np.asarray(list(metrics.keys()), dtype=object),
        metrics_values=np.asarray(list(metrics.values()), dtype=float),
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=Path("debug_logs/literal_line_tracking.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("debug_logs/tracking_report"))
    parser.add_argument(
        "--trim-takeoff-s",
        type=float,
        default=0.0,
        help="Ignore early samples, useful if you want metrics only after takeoff/transient.",
    )
    args = parser.parse_args()

    metrics = analyze(args.log, args.outdir, trim_takeoff_s=args.trim_takeoff_s)
    print("\nTracking metrics")
    print("================")
    for k, v in metrics.items():
        if k == "samples":
            print(f"{k:32s}: {int(v)}")
        else:
            print(f"{k:32s}: {v:.6g}")
    print(f"\nWrote report to: {args.outdir}")


if __name__ == "__main__":
    main()
