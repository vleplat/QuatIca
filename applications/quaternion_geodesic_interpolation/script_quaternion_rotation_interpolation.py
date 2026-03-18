#!/usr/bin/env python3
"""
Quaternion rotation interpolation demo (SLERP vs SQUAD vs log–exp spline).

It:
- creates a small quaternion keyframe sequence on S^3,
- samples three interpolants (piecewise SLERP, SQUAD, log–exp cubic spline),
- computes simple smoothness diagnostics based on discrete angular velocity,
- plots angular speed / acceleration profiles,
- visualizes the induced trajectory on S^2 by rotating a fixed vector.

The script is designed to work both:
- inside the QuatIca repo (running directly), and
- after `pip install quatica` (importing as a package).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import quaternion


def _find_repo_root(start: str) -> str:
    """Find repo root by walking up until `pyproject.toml` is found."""
    cur = os.path.abspath(start)
    while True:
        if os.path.isfile(os.path.join(cur, "pyproject.toml")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(start)
        cur = parent


def _ensure_importable_quatica() -> None:
    try:
        import quatica  # noqa: F401
        return
    except Exception:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = _find_repo_root(script_dir)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)


def _save_png_and_pdf(fig, basepath_no_ext: str, *, dpi: int = 300) -> None:
    os.makedirs(os.path.dirname(basepath_no_ext), exist_ok=True)
    fig.savefig(basepath_no_ext + ".png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(basepath_no_ext + ".pdf", dpi=dpi, bbox_inches="tight", facecolor="white")


def _axis_angle_to_quat(axis: np.ndarray, theta: float) -> quaternion.quaternion:
    axis = np.asarray(axis, dtype=float).reshape(3)
    axis = axis / max(1e-15, np.linalg.norm(axis))
    # quaternion.from_rotation_vector uses magnitude = rotation angle (not half-angle).
    return quaternion.from_rotation_vector(axis * float(theta))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--K", type=int, default=6, help="Number of keyframes")
    parser.add_argument("--samples-per-seg", type=int, default=250)
    parser.add_argument("--no-display", action="store_true", help="Do not show figures (save only).")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: <repo>/output_figures/quaternion_geodesic_interpolation/)",
    )
    args = parser.parse_args()

    _ensure_importable_quatica()
    from quatica.qtraj import (
        enforce_sign_continuity,
        interpolate_piecewise_slerp,
        interpolate_squad,
        keyframe_errors_geodesic,
        log_euclidean_spline,
        smoothness_energy,
        velocity_jump_at_keyframes,
    )
    from quatica.visualization import Visualizer

    if args.no_display:
        plt.switch_backend("Agg")

    # Paper-ish defaults
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    rng = np.random.default_rng(args.seed)
    K = int(args.K)
    if K < 2:
        raise ValueError("Need K >= 2 keyframes.")
    ts = np.linspace(0.0, 1.0, K)

    # "Reasonably smooth" axis-angle sequence (matches the notebook idea).
    axes = rng.normal(size=(K, 3))
    angles = np.linspace(0.0, 1.6, K)  # total angle range ~ 1.6 rad
    qs = [_axis_angle_to_quat(axes[i], angles[i]) for i in range(K)]
    qs = enforce_sign_continuity(qs)

    # Sample three methods
    samples_per_seg = int(args.samples_per_seg)
    t_slerp, q_slerp = interpolate_piecewise_slerp(qs, ts, samples_per_seg=samples_per_seg)
    t_squad, q_squad = interpolate_squad(qs, ts, samples_per_seg=samples_per_seg)
    t_grid = np.linspace(0.0, 1.0, (K - 1) * samples_per_seg + 1)
    t_logexp, q_logexp = log_euclidean_spline(qs, ts, t_grid)

    # Diagnostics
    def _report_metrics(name: str, t_path: np.ndarray, q_path: Sequence[quaternion.quaternion]) -> Dict[str, float]:
        from quatica.qtraj import estimate_omega

        t_om, om = estimate_omega(q_path, t_path)
        return {
            "smoothness_energy": smoothness_energy(q_path, t_path),
            "omega_rms": float(np.sqrt(np.mean(np.sum(om**2, axis=1)))),
            "omega_max": float(np.max(np.linalg.norm(om, axis=1))),
        }

    m_slerp = _report_metrics("piecewise_slerp", t_slerp, q_slerp)
    m_squad = _report_metrics("squad", t_squad, q_squad)
    m_logexp = _report_metrics("logexp_cubic", t_logexp, q_logexp)

    jump_slerp = velocity_jump_at_keyframes(qs, ts, interpolate_piecewise_slerp, samples_per_seg=300)
    jump_squad = velocity_jump_at_keyframes(qs, ts, interpolate_squad, samples_per_seg=300)

    print("Metrics (lower smoothness_energy is better):")
    for name, m in [("piecewise_slerp", m_slerp), ("squad", m_squad), ("logexp_cubic", m_logexp)]:
        print(f"  {name:>14s}  E={m['smoothness_energy']:.3e}  omega_rms={m['omega_rms']:.3f}  omega_max={m['omega_max']:.3f}")
    print("Max omega jump at keyframes:")
    print("  piecewise_slerp:", jump_slerp)
    print("  squad         :", jump_squad)

    # Output directory
    if args.out_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = _find_repo_root(script_dir)
        out_dir = os.path.join(repo_root, "output_figures", "quaternion_geodesic_interpolation")
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Angular speed plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    Visualizer.plot_angular_speed(t_slerp, q_slerp, label="piecewise_slerp", ax=ax)
    Visualizer.plot_angular_speed(t_squad, q_squad, label="squad", ax=ax)
    Visualizer.plot_angular_speed(t_logexp, q_logexp, label="logexp_cubic", ax=ax)
    ax.set_title(r"Angular speed profile ($\|\omega(t)\|$)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_png_and_pdf(fig, os.path.join(out_dir, "angular_speed"))
    if not args.no_display:
        plt.show()
    else:
        plt.close(fig)

    # Angular acceleration plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    Visualizer.plot_angular_accel(t_slerp, q_slerp, label="piecewise_slerp", ax=ax)
    Visualizer.plot_angular_accel(t_squad, q_squad, label="squad", ax=ax)
    Visualizer.plot_angular_accel(t_logexp, q_logexp, label="logexp_cubic", ax=ax)
    ax.set_title(r"Angular acceleration (discrete) profile ($\|\dot\omega(t)\|$)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_png_and_pdf(fig, os.path.join(out_dir, "angular_accel"))
    if not args.no_display:
        plt.show()
    else:
        plt.close(fig)

    # Keyframe errors plot (nearest sample)
    errs_slerp, _ = keyframe_errors_geodesic(t_slerp, q_slerp, qs, ts)
    errs_squad, _ = keyframe_errors_geodesic(t_squad, q_squad, qs, ts)
    errs_logexp, _ = keyframe_errors_geodesic(t_logexp, q_logexp, qs, ts)

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.0))
    ax.plot(ts, np.degrees(errs_slerp), marker="o", label="piecewise_slerp")
    ax.plot(ts, np.degrees(errs_squad), marker="o", label="squad")
    ax.plot(ts, np.degrees(errs_logexp), marker="o", label="logexp_cubic")
    ax.set_xlabel(r"keyframe time $t_i$")
    ax.set_ylabel("geodesic error at keyframes (deg)")
    ax.set_title("Do sampled trajectories pass through the keyframes?")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_png_and_pdf(fig, os.path.join(out_dir, "keyframe_errors"))
    if not args.no_display:
        plt.show()
    else:
        plt.close(fig)

    # S^2 trajectory visualization (rotate a fixed vector)
    s2_paths = {
        "piecewise_slerp": q_slerp,
        "squad": q_squad,
        "logexp_cubic": q_logexp,
    }
    Visualizer.plot_quaternion_trajectories_on_s2(
        s2_paths,
        keyframes=qs,
        v0=(1.0, -0.9, 0.0),
        title=r"Trajectory of rotated unit vector on $S^2$ (with keyframes)",
        save_path=os.path.join(out_dir, "s2_trajectory.png"),
        show=not args.no_display,
    )
    Visualizer.plot_quaternion_trajectories_on_s2(
        s2_paths,
        keyframes=qs,
        v0=(1.0, -0.9, 0.0),
        title=r"Trajectory of rotated unit vector on $S^2$ (with keyframes)",
        save_path=os.path.join(out_dir, "s2_trajectory.pdf"),
        show=False,
    )

    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()

