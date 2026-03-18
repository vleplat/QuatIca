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
    parser.add_argument("--K", type=int, default=5, help="Number of keyframes (default: 5, smooth demo preset)")
    parser.add_argument("--samples-per-seg", type=int, default=250)
    parser.add_argument(
        "--keyframe-family",
        type=str,
        default="smooth_axes",
        choices=["random_axes", "smooth_axes", "fixed_axis"],
        help=(
            "How to generate the keyframe sequence. "
            "Default is 'smooth_axes' (smooth demo/visualization preset). "
            "'random_axes' is a stress-test mode (independent random axes; can be jerky). "
            "'smooth_axes' uses a random-walk axis (trajectory-like). "
            "'fixed_axis' rotates about one axis with increasing angle (very smooth)."
        ),
    )
    parser.add_argument(
        "--axis-noise",
        type=float,
        default=0.12,
        help="Axis random-walk step size for keyframe-family=smooth_axes (default: 0.12).",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help=(
            "Run a diagnosis sweep over K and samples_per_seg to understand whether "
            "angular-speed peaks are structural (spline shape / sparse keyframes) "
            "or artifacts. Writes CSV + summary figures."
        ),
    )
    parser.add_argument(
        "--diagnose-controlled",
        action="store_true",
        help=(
            "Run a minimal controlled diagnosis using one fixed smooth reference trajectory "
            "and nested keyframe subsets (K=5,9,17,...). This isolates the effect of K "
            "without changing the underlying motion."
        ),
    )
    parser.add_argument(
        "--ref-N",
        type=int,
        default=2001,
        help="Number of samples for the reference trajectory in --diagnose-controlled (default: 2001).",
    )
    parser.add_argument(
        "--subset-K-grid",
        type=str,
        default="5,9,17",
        help="Comma-separated K values for nested subsets in --diagnose-controlled (default: 5,9,17).",
    )
    parser.add_argument(
        "--window",
        type=str,
        default="0.35,0.65",
        help="Local time window [t0,t1] for the second controlled test (default: 0.35,0.65).",
    )
    parser.add_argument(
        "--window-K",
        type=int,
        default=5,
        help="Number of keyframes in the local-window controlled test (default: 5).",
    )
    parser.add_argument(
        "--diagnose-K-grid",
        type=str,
        default="5,9,17,33",
        help="Comma-separated K values for --diagnose (default: 5,9,17,33).",
    )
    parser.add_argument(
        "--diagnose-samples-grid",
        type=str,
        default="50,100,250",
        help="Comma-separated samples_per_seg values for --diagnose (default: 50,100,250).",
    )
    parser.add_argument(
        "--diagnose-trials",
        type=int,
        default=3,
        help="Number of random trials per (K, samples_per_seg) for --diagnose (default: 3).",
    )
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

    # Output directory (single source of truth; also used by --diagnose mode)
    if args.out_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = _find_repo_root(script_dir)
        out_dir = os.path.join(repo_root, "output_figures", "quaternion_geodesic_interpolation")
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    def _make_keyframes(rng_local: np.random.Generator, K_local: int):
        if K_local < 2:
            raise ValueError("Need K >= 2 keyframes.")
        ts_local = np.linspace(0.0, 1.0, K_local)
        # Keyframe construction choices:
        # - random_axes: independent random axes (can be quite "jerky")
        # - smooth_axes: axis is a random walk on S^2 (more trajectory-like)
        # - fixed_axis: constant axis, increasing angle (very smooth baseline)
        if args.keyframe_family == "fixed_axis":
            axis0 = rng_local.normal(size=(3,))
            axes_local = np.tile(axis0[None, :], (K_local, 1))
        elif args.keyframe_family == "smooth_axes":
            axes_local = np.zeros((K_local, 3), dtype=float)
            axes_local[0] = rng_local.normal(size=(3,))
            step = float(args.axis_noise)
            for i in range(1, K_local):
                axes_local[i] = axes_local[i - 1] + step * rng_local.normal(size=(3,))
        else:
            # "Random but plausible" axis-angle sequence (original demo behavior).
            axes_local = rng_local.normal(size=(K_local, 3))

        # Normalize axes
        axes_local /= np.linalg.norm(axes_local, axis=1, keepdims=True)
        angles_local = np.linspace(0.0, 1.6, K_local)  # total angle range ~ 1.6 rad
        qs_local = [_axis_angle_to_quat(axes_local[i], angles_local[i]) for i in range(K_local)]
        qs_local = enforce_sign_continuity(qs_local)
        return ts_local, qs_local

    K = int(args.K)
    ts, qs = _make_keyframes(rng, K)

    def _parse_int_list(s: str) -> list[int]:
        return [int(x.strip()) for x in s.split(",") if x.strip()]

    def _parse_float_pair(s: str) -> tuple[float, float]:
        parts = [x.strip() for x in s.split(",") if x.strip()]
        if len(parts) != 2:
            raise ValueError("Expected two comma-separated floats, e.g. '0.35,0.65'.")
        t0, t1 = float(parts[0]), float(parts[1])
        if not (t0 < t1):
            raise ValueError("Expected t0 < t1 for a window.")
        return t0, t1

    def _reference_trajectory(t_ref: np.ndarray) -> np.ndarray:
        """
        Smooth reference quaternion trajectory q_ref(t) on S^3, generated from a smooth
        rotation-vector curve r(t) in R^3 (then mapped via quaternion.from_rotation_vector).
        """
        t = np.asarray(t_ref, dtype=float)
        # Smooth, bounded rotation-vector curve with varying direction and speed.
        # This is deterministic and does not depend on K.
        w1 = 2.0 * np.pi
        w2 = 4.0 * np.pi
        r = np.zeros((t.size, 3), dtype=float)
        r[:, 0] = 0.8 * np.sin(w1 * t) + 0.15 * np.sin(w2 * t)
        r[:, 1] = 0.6 * np.cos(w1 * t) + 0.10 * np.cos(w2 * t + 0.3)
        r[:, 2] = 0.4 * np.sin(0.5 * w1 * t + 0.2)
        # Convert rotation vectors -> quaternions
        q_ref = [quaternion.from_rotation_vector(r[i]) for i in range(t.size)]
        return np.array(q_ref, dtype=object)

    def _metrics_from_path(t_path: np.ndarray, q_path: Sequence[quaternion.quaternion]) -> dict:
        from quatica.qtraj import estimate_omega

        t_om, om = estimate_omega(q_path, t_path)
        omega_norm = np.linalg.norm(om, axis=1) if len(om) else np.array([], dtype=float)
        return {
            "omega_rms": float(np.sqrt(np.mean(omega_norm**2))) if omega_norm.size else 0.0,
            "omega_max": float(np.max(omega_norm)) if omega_norm.size else 0.0,
            "smoothness_energy": smoothness_energy(q_path, t_path),
        }

    # ---------------------------------------------------------------------
    # Controlled diagnosis: fixed reference trajectory, nested subsets + one local window.
    # ---------------------------------------------------------------------
    if args.diagnose_controlled:
        import pandas as pd

        ref_N = int(args.ref_N)
        if ref_N < 101:
            raise ValueError("--ref-N must be reasonably large (>= 101).")
        t_ref = np.linspace(0.0, 1.0, ref_N)
        q_ref = _reference_trajectory(t_ref)
        q_ref = enforce_sign_continuity(list(q_ref))
        q_ref = np.array(q_ref, dtype=object)

        K_grid = _parse_int_list(args.subset_K_grid)
        if any(K0 < 2 for K0 in K_grid):
            raise ValueError("--subset-K-grid must contain only K >= 2.")

        def _subset_from_reference(t0: float, t1: float, K0: int) -> tuple[np.ndarray, np.ndarray]:
            # Use indices so subsets are nested by construction when t0=0,t1=1.
            mask = (t_ref >= t0) & (t_ref <= t1)
            idx_all = np.where(mask)[0]
            if idx_all.size < K0:
                raise ValueError("Window too small for requested K.")
            idx = np.round(np.linspace(idx_all[0], idx_all[-1], K0)).astype(int)
            idx = np.unique(idx)
            if idx.size < K0:
                # ensure exactly K0 by filling gaps (rare unless ref_N is tiny)
                extra = [j for j in idx_all.tolist() if j not in set(idx)]
                idx = np.sort(np.concatenate([idx, np.array(extra[: (K0 - idx.size)], dtype=int)]))
            ts_key = t_ref[idx]
            qs_key = q_ref[idx]
            qs_key = enforce_sign_continuity(list(qs_key))
            return ts_key, np.array(qs_key, dtype=object)

        rows = []

        # Experiment 1: nested subsets over full interval.
        for K0 in K_grid:
            ts_key, qs_key = _subset_from_reference(0.0, 1.0, K0)
            # Choose sampling so the interpolated path has approximately ref_N points.
            samples_per_seg0 = max(10, int(round((ref_N - 1) / max(1, (K0 - 1)))))

            t_slerp0, q_slerp0 = interpolate_piecewise_slerp(
                qs_key, ts_key, samples_per_seg=samples_per_seg0
            )
            t_squad0, q_squad0 = interpolate_squad(qs_key, ts_key, samples_per_seg=samples_per_seg0)
            t_logexp0, q_logexp0 = log_euclidean_spline(qs_key, ts_key, t_ref)

            for method, (tt, qq) in {
                "piecewise_slerp": (t_slerp0, q_slerp0),
                "squad": (t_squad0, q_squad0),
                "logexp_cubic": (t_logexp0, q_logexp0),
            }.items():
                m = _metrics_from_path(tt, qq)
                rows.append(
                    {
                        "experiment": "nested_subsets",
                        "K": K0,
                        "samples_per_seg": samples_per_seg0,
                        "method": method,
                        **m,
                    }
                )

        # Experiment 2: one local-window test on the same reference.
        w0, w1 = _parse_float_pair(args.window)
        K_w = int(args.window_K)
        ts_w, qs_w = _subset_from_reference(w0, w1, K_w)
        # build a dense query grid on the window
        mask_w = (t_ref >= w0) & (t_ref <= w1)
        t_w_grid = t_ref[mask_w]
        samples_per_seg_w = max(10, int(round((t_w_grid.size - 1) / max(1, (K_w - 1)))))

        t_slerp_w, q_slerp_w = interpolate_piecewise_slerp(qs_w, ts_w, samples_per_seg=samples_per_seg_w)
        t_squad_w, q_squad_w = interpolate_squad(qs_w, ts_w, samples_per_seg=samples_per_seg_w)
        t_logexp_w, q_logexp_w = log_euclidean_spline(qs_w, ts_w, t_w_grid)

        for method, (tt, qq) in {
            "piecewise_slerp": (t_slerp_w, q_slerp_w),
            "squad": (t_squad_w, q_squad_w),
            "logexp_cubic": (t_logexp_w, q_logexp_w),
        }.items():
            m = _metrics_from_path(tt, qq)
            rows.append(
                {
                    "experiment": "local_window",
                    "K": K_w,
                    "samples_per_seg": samples_per_seg_w,
                    "method": method,
                    **m,
                }
            )

        dfc = pd.DataFrame(rows)
        csv_path = os.path.join(out_dir, "controlled_diagnosis_metrics.csv")
        dfc.to_csv(csv_path, index=False)
        print(f"[diagnose-controlled] Wrote CSV: {csv_path}")

        # Figure 1: omega_max comparison for nested subsets (clustered bars by K).
        fig, ax = plt.subplots(1, 1, figsize=(8.8, 4.8))
        d = dfc[dfc["experiment"] == "nested_subsets"].copy()
        K_order = sorted(d["K"].unique())
        methods = ["piecewise_slerp", "squad", "logexp_cubic"]
        colors = {"piecewise_slerp": "#d62728", "squad": "#1f77b4", "logexp_cubic": "#2ca02c"}
        x = np.arange(len(K_order), dtype=float)
        width = 0.26
        for j, mth in enumerate(methods):
            y = np.array([float(d[(d["K"] == K0) & (d["method"] == mth)]["omega_max"].mean()) for K0 in K_order])
            ax.bar(
                x + (j - 1) * width,
                y,
                width=width,
                color=colors[mth],
                edgecolor="black",
                linewidth=0.8,
                alpha=0.85,
                label=mth,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"K={K0}" for K0 in K_order])
        ax.set_ylabel(r"Max angular speed $\max_t\|\omega(t)\|$")
        ax.set_title("Controlled diagnosis: nested keyframe subsets from one fixed reference trajectory")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False, ncol=3)
        _save_png_and_pdf(fig, os.path.join(out_dir, "controlled_diagnosis_omega_max"))
        if args.no_display:
            plt.close(fig)
        else:
            plt.show()

        # Figure 2: representative speed curves (nested subset + local window) as two panels.
        K_rep = int(sorted(K_grid)[len(K_grid) // 2])
        ts_rep, qs_rep = _subset_from_reference(0.0, 1.0, K_rep)
        samples_per_seg_rep = max(10, int(round((ref_N - 1) / max(1, (K_rep - 1)))))
        t_slerp_r, q_slerp_r = interpolate_piecewise_slerp(qs_rep, ts_rep, samples_per_seg=samples_per_seg_rep)
        t_squad_r, q_squad_r = interpolate_squad(qs_rep, ts_rep, samples_per_seg=samples_per_seg_rep)
        t_logexp_r, q_logexp_r = log_euclidean_spline(qs_rep, ts_rep, t_ref)

        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=False)
        ax0, ax1 = axes
        Visualizer.plot_angular_speed(t_slerp_r, q_slerp_r, label="piecewise_slerp", ax=ax0)
        Visualizer.plot_angular_speed(t_squad_r, q_squad_r, label="squad", ax=ax0)
        Visualizer.plot_angular_speed(t_logexp_r, q_logexp_r, label="logexp_cubic", ax=ax0)
        ax0.set_title(rf"Representative full path (nested subset, $K={K_rep}$)")
        ax0.grid(True, alpha=0.3)
        ax0.legend()

        Visualizer.plot_angular_speed(t_slerp_w, q_slerp_w, label="piecewise_slerp", ax=ax1)
        Visualizer.plot_angular_speed(t_squad_w, q_squad_w, label="squad", ax=ax1)
        Visualizer.plot_angular_speed(t_logexp_w, q_logexp_w, label="logexp_cubic", ax=ax1)
        ax1.set_title(rf"Local window $t\in[{w0:.2f},{w1:.2f}]$ ($K={K_w}$)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        fig.suptitle("Controlled diagnosis: representative angular speed curves", y=1.02)
        _save_png_and_pdf(fig, os.path.join(out_dir, "controlled_diagnosis_speed_curves"))
        if args.no_display:
            plt.close(fig)
        else:
            plt.show()

        print("[diagnose-controlled] Done.")

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

    # ---------------------------------------------------------------------
    # Diagnosis sweep: vary K and sampling density, compare methods.
    # ---------------------------------------------------------------------
    if args.diagnose:
        import pandas as pd
        from quatica.qtraj import estimate_omega, geodesic_distance_s3

        K_grid = _parse_int_list(args.diagnose_K_grid)
        samples_grid = _parse_int_list(args.diagnose_samples_grid)
        trials = int(args.diagnose_trials)
        if trials < 1:
            raise ValueError("--diagnose-trials must be >= 1.")

        rows = []
        for K0 in K_grid:
            for sp0 in samples_grid:
                for trial in range(trials):
                    # deterministic per-run RNG for reproducibility
                    seed_run = int(args.seed) * 10_000 + K0 * 100 + sp0 * 10 + trial
                    rng_run = np.random.default_rng(seed_run)
                    ts0, qs0 = _make_keyframes(rng_run, K0)

                    # Basic geometry diagnostic: gaps between neighboring keyframes.
                    gaps = [geodesic_distance_s3(qs0[i], qs0[i + 1]) for i in range(K0 - 1)]
                    gap_max = float(np.max(gaps)) if gaps else 0.0
                    gap_mean = float(np.mean(gaps)) if gaps else 0.0

                    # Sample methods
                    t_slerp0, q_slerp0 = interpolate_piecewise_slerp(qs0, ts0, samples_per_seg=sp0)
                    t_squad0, q_squad0 = interpolate_squad(qs0, ts0, samples_per_seg=sp0)
                    t_grid0 = np.linspace(0.0, 1.0, (K0 - 1) * sp0 + 1)
                    t_logexp0, q_logexp0 = log_euclidean_spline(qs0, ts0, t_grid0)

                    for method, (tt, qq) in {
                        "piecewise_slerp": (t_slerp0, q_slerp0),
                        "squad": (t_squad0, q_squad0),
                        "logexp_cubic": (t_logexp0, q_logexp0),
                    }.items():
                        t_om, om = estimate_omega(qq, tt)
                        omega_norm = np.linalg.norm(om, axis=1) if len(om) else np.array([], dtype=float)
                        rows.append(
                            {
                                "seed_base": int(args.seed),
                                "seed_run": seed_run,
                                "trial": trial,
                                "K": K0,
                                "samples_per_seg": sp0,
                                "method": method,
                                "gap_mean_rad": gap_mean,
                                "gap_max_rad": gap_max,
                                "omega_rms": float(np.sqrt(np.mean(omega_norm**2))) if omega_norm.size else 0.0,
                                "omega_max": float(np.max(omega_norm)) if omega_norm.size else 0.0,
                                "smoothness_energy": smoothness_energy(qq, tt),
                            }
                        )

        df = pd.DataFrame(rows)
        csv_path = os.path.join(out_dir, "diagnosis_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"[diagnose] Wrote CSV: {csv_path}")

        # Summary figure: omega_max vs K (mean±std across trials), faceted by samples_per_seg.
        fig, axes = plt.subplots(1, len(samples_grid), figsize=(5.8 * len(samples_grid), 4.2), sharey=True)
        if len(samples_grid) == 1:
            axes = [axes]
        colors = {"piecewise_slerp": "#d62728", "squad": "#1f77b4", "logexp_cubic": "#2ca02c"}
        for ax, sp0 in zip(axes, samples_grid):
            d0 = df[df["samples_per_seg"] == sp0].copy()
            for method in ["piecewise_slerp", "squad", "logexp_cubic"]:
                dm = d0[d0["method"] == method]
                g = dm.groupby("K", observed=True)["omega_max"]
                x = np.array(sorted(g.mean().index.values), dtype=int)
                y = np.array([float(g.mean().loc[k]) for k in x], dtype=float)
                e = np.array([float(g.std().loc[k]) if k in g.std().index else 0.0 for k in x], dtype=float)
                ax.errorbar(
                    x,
                    y,
                    yerr=e,
                    marker="o",
                    linewidth=2.2,
                    capsize=4,
                    label=method,
                    color=colors[method],
                    markerfacecolor="white",
                    markeredgewidth=1.6,
                )
            ax.set_title(rf"$\mathrm{{samples\_per\_seg}}={sp0}$")
            ax.set_xlabel(r"Keyframes $K$")
            ax.grid(True, alpha=0.3)
            if ax is axes[0]:
                ax.set_ylabel(r"Max angular speed $\max_t \|\omega(t)\|$")
            ax.legend(frameon=False)
        fig.suptitle("Diagnosis sweep: do max-speed peaks shrink with denser keyframes?", y=1.02)
        _save_png_and_pdf(fig, os.path.join(out_dir, "diagnosis_omega_max_vs_K"))
        if args.no_display:
            plt.close(fig)
        else:
            plt.show()

        # Representative plot: omega(t) curves for one configuration (median K, largest sampling).
        K_rep = int(sorted(K_grid)[len(K_grid) // 2])
        sp_rep = int(max(samples_grid))
        seed_run = int(args.seed) * 10_000 + K_rep * 100 + sp_rep * 10 + 0
        rng_run = np.random.default_rng(seed_run)
        ts_rep, qs_rep = _make_keyframes(rng_run, K_rep)
        t_slerp_r, q_slerp_r = interpolate_piecewise_slerp(qs_rep, ts_rep, samples_per_seg=sp_rep)
        t_squad_r, q_squad_r = interpolate_squad(qs_rep, ts_rep, samples_per_seg=sp_rep)
        t_grid_r = np.linspace(0.0, 1.0, (K_rep - 1) * sp_rep + 1)
        t_logexp_r, q_logexp_r = log_euclidean_spline(qs_rep, ts_rep, t_grid_r)

        fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.6))
        Visualizer.plot_angular_speed(t_slerp_r, q_slerp_r, label="piecewise_slerp", ax=ax)
        Visualizer.plot_angular_speed(t_squad_r, q_squad_r, label="squad", ax=ax)
        Visualizer.plot_angular_speed(t_logexp_r, q_logexp_r, label="logexp_cubic", ax=ax)
        ax.set_title(rf"Representative angular speed ($K={K_rep}$, samples/seg={sp_rep}$)$")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _save_png_and_pdf(fig, os.path.join(out_dir, "diagnosis_speed_curves_rep"))
        if args.no_display:
            plt.close(fig)
        else:
            plt.show()

        print("[diagnose] Done.")

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
        v0=(1.0, -2.0, 0.0),
        title=r"Trajectory of rotated unit vector on $S^2$ (with keyframes)",
        save_path=os.path.join(out_dir, "s2_trajectory.png"),
        show=not args.no_display,
    )
    Visualizer.plot_quaternion_trajectories_on_s2(
        s2_paths,
        keyframes=qs,
        v0=(1.0, -2.0, 0.0),
        title=r"Trajectory of rotated unit vector on $S^2$ (with keyframes)",
        save_path=os.path.join(out_dir, "s2_trajectory.pdf"),
        show=False,
    )

    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()

