#!/usr/bin/env python3
r"""
Paper-ready Q-SVD benchmark for QuatIca.

This script compares:
- classical_qsvd   -> "Truncated Q-SVD"
- rand_qsvd        -> "Randomized Q-SVD"
- pass_eff_qsvd    -> "Pass-efficient Q-SVD"

Benchmark parts:
A) Exact rank-10 recovery sanity check
B) Relative approximation error vs target rank (smooth spectral decay)
C) Runtime vs matrix size

Outputs (default: validation_output/qsvd_benchmark/):
- qsvd_benchmark_metrics.csv
- qsvd_benchmark_dashboard.png
- qsvd_benchmark__exact_rank10_recovery.pdf
- qsvd_benchmark__error_vs_rank.pdf
- qsvd_benchmark__runtime_vs_size.pdf
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _find_repo_root(start: str) -> str:
    here = os.path.abspath(start)
    while True:
        if os.path.exists(os.path.join(here, "pyproject.toml")) and os.path.isdir(
            os.path.join(here, "quatica")
        ):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        here = parent


def _ensure_quatica_importable() -> None:
    root = _find_repo_root(os.path.dirname(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_quatica_importable()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import quaternion  # type: ignore

from quatica.decomp.qsvd import classical_qsvd, pass_eff_qsvd, rand_qsvd
from quatica.utils import quat_frobenius_norm, quat_hermitian, quat_matmat


def _setup_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "lines.linewidth": 2.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_axis_pdf(fig: plt.Figure, ax: plt.Axes, path: Path, *, pad: float = 0.02) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_tightbbox(renderer).expanded(1.0 + pad, 1.0 + pad)
    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(path, bbox_inches=bbox_inches, facecolor="white")


def _random_quat(m: int, n: int) -> np.ndarray:
    data = np.random.randn(m, n, 4)
    return quaternion.as_quat_array(data)


def _diag_real_quat(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=float).reshape(-1)
    r = s.size
    D = np.zeros((r, r, 4), dtype=float)
    D[np.arange(r), np.arange(r), 0] = s
    return quaternion.as_quat_array(D)


def _recon(U: np.ndarray, s: np.ndarray, V: np.ndarray) -> np.ndarray:
    return quat_matmat(quat_matmat(U, _diag_real_quat(s)), quat_hermitian(V))


def _rel_err(X: np.ndarray, U: np.ndarray, s: np.ndarray, V: np.ndarray) -> float:
    Xhat = _recon(U, s, V)
    return float(quat_frobenius_norm(X - Xhat) / (quat_frobenius_norm(X) + 1e-30))


def _exact_low_rank(m: int, n: int, r: int) -> np.ndarray:
    A = _random_quat(m, r)
    B = _random_quat(r, n)
    return quat_matmat(A, B)


def _real_to_quat(M: np.ndarray) -> np.ndarray:
    Z = np.zeros_like(M)
    return quaternion.as_quat_array(np.stack([M, Z, Z, Z], axis=-1))


def _decay_matrix(m: int, n: int, decay: float = 3.0) -> np.ndarray:
    """
    Build a rectangular quaternion matrix with controlled singular-value decay.

    We use a stable real-orthogonal lift:
      X = U Σ V^H,  with U,V real-orthogonal (lifted to quaternions), Σ real diagonal.
    """
    r = min(m, n)
    U, _ = np.linalg.qr(np.random.randn(m, m))
    V, _ = np.linalg.qr(np.random.randn(n, n))
    s = np.logspace(0.0, -decay, r)

    Uq = _real_to_quat(U)[:, :r]
    Vq = _real_to_quat(V)[:, :r]
    Sigq = _diag_real_quat(s)

    return quat_matmat(quat_matmat(Uq, Sigq), quat_hermitian(Vq))


METHODS: dict[str, str] = {
    "classical": "Truncated Q-SVD",
    "rand": "Randomized Q-SVD",
    "pass_eff": "Pass-efficient Q-SVD",
}


def _run_method(
    X: np.ndarray,
    method_key: str,
    R: int,
    *,
    oversample: int,
    n_iter: int,
    n_passes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    t0 = time.time()
    if method_key == "classical":
        U, s, V = classical_qsvd(X, R)
    elif method_key == "rand":
        U, s, V = rand_qsvd(X, R, oversample=oversample, n_iter=n_iter)
    elif method_key == "pass_eff":
        U, s, V = pass_eff_qsvd(X, R, oversample=oversample, n_passes=n_passes)
    else:
        raise ValueError(f"Unknown method: {method_key}")
    dt = time.time() - t0
    return U, s, V, dt


def _safe_run(
    f: Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray, float]]
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, float]:
    try:
        return f()
    except Exception:
        return None, None, None, float("nan")


def main(argv: list[str] | None = None) -> None:
    _setup_style()

    p = argparse.ArgumentParser(description="Paper-ready Q-SVD benchmark for QuatIca.")
    p.add_argument(
        "--out-dir",
        default="validation_output/qsvd_benchmark",
        help="Output directory (relative to repo root).",
    )
    p.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds.")
    p.add_argument("--oversample", type=int, default=10, help="Oversampling parameter.")
    p.add_argument("--n-iter", type=int, default=2, help="Power iterations for rand_qsvd.")
    p.add_argument("--n-passes", type=int, default=2, help="Pass budget for pass_eff_qsvd.")
    args = p.parse_args(argv)

    root = Path(_find_repo_root(os.path.dirname(__file__)))
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    oversample = int(args.oversample)
    n_iter = int(args.n_iter)
    n_passes = int(args.n_passes)

    rows: list[dict[str, Any]] = []

    # -------------------------
    # Part A: exact rank-10 recovery
    # -------------------------
    exact_m, exact_n, exact_r = 100, 200, 10
    for seed in seeds:
        np.random.seed(seed)
        X = _exact_low_rank(exact_m, exact_n, exact_r)
        for method_key, method_label in METHODS.items():
            U, s, V, dt = _run_method(
                X,
                method_key,
                exact_r,
                oversample=oversample,
                n_iter=n_iter,
                n_passes=n_passes,
            )
            err = _rel_err(X, U, s, V)
            rows.append(
                {
                    "experiment": "exact_rank10_recovery",
                    "method_key": method_key,
                    "method": method_label,
                    "seed": seed,
                    "m": exact_m,
                    "n": exact_n,
                    "R": exact_r,
                    "oversample": oversample,
                    "n_iter": n_iter,
                    "n_passes": n_passes,
                    "time_s": dt,
                    "rel_err": err,
                }
            )

    # -------------------------
    # Part B: approximation error vs rank (smooth decay)
    # -------------------------
    rank_shape = (300, 200)
    rank_values = [5, 10, 20, 40, 60, 80, 100]
    for seed in seeds:
        np.random.seed(seed)
        X = _decay_matrix(*rank_shape, decay=3.0)
        for R in rank_values:
            for method_key, method_label in METHODS.items():
                U, s, V, dt = _run_method(
                    X, method_key, R, oversample=oversample, n_iter=n_iter, n_passes=n_passes
                )
                err = _rel_err(X, U, s, V)
                rows.append(
                    {
                        "experiment": "error_vs_rank",
                        "method_key": method_key,
                        "method": method_label,
                        "seed": seed,
                        "m": rank_shape[0],
                        "n": rank_shape[1],
                        "R": R,
                        "oversample": oversample,
                        "n_iter": n_iter,
                        "n_passes": n_passes,
                        "time_s": dt,
                        "rel_err": err,
                    }
                )

    # -------------------------
    # Part C: runtime vs size
    # -------------------------
    runtime_sizes = [(100, 80), (200, 150), (300, 200), (400, 300), (800, 600), (1200, 900)]
    runtime_rank = 20

    for (m, n) in runtime_sizes:
        for seed in seeds:
            np.random.seed(seed)
            X = _random_quat(m, n)

            for method_key, method_label in METHODS.items():
                U, s, V, dt = _safe_run(
                    lambda mk=method_key: _run_method(
                        X,
                        mk,
                        runtime_rank,
                        oversample=oversample,
                        n_iter=n_iter,
                        n_passes=n_passes,
                    )
                )
                if U is None:
                    err = float("nan")
                else:
                    err = _rel_err(X, U, s, V)

                rows.append(
                    {
                        "experiment": "runtime_vs_size",
                        "method_key": method_key,
                        "method": method_label,
                        "seed": seed,
                        "m": m,
                        "n": n,
                        "size_label": f"{m}×{n}",
                        "R": runtime_rank,
                        "oversample": oversample,
                        "n_iter": n_iter,
                        "n_passes": n_passes,
                        "time_s": dt,
                        "rel_err": err,
                    }
                )

    # -------------------------
    # Export CSV
    # -------------------------
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "qsvd_benchmark_metrics.csv", index=False)

    # -------------------------
    # Build dashboard (3 panels)
    # -------------------------
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    axA, axB, axC = axs

    colors = {
        "classical": "#d62728",
        "rand": "#1f77b4",
        "pass_eff": "#2ca02c",
    }

    keys = ["classical", "rand", "pass_eff"]

    # A) exact rank-10 recovery
    d = df[df["experiment"] == "exact_rank10_recovery"].copy()
    means = d.groupby("method_key", observed=True)["rel_err"].mean()
    stds = d.groupby("method_key", observed=True)["rel_err"].std()
    x = np.arange(len(keys))
    y = [float(means.get(k, np.nan)) for k in keys]
    e = [float(stds.get(k, np.nan)) for k in keys]

    axA.bar(
        x,
        y,
        yerr=e,
        color=[colors[k] for k in keys],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
        capsize=5,
    )
    axA.set_yscale("log")
    axA.set_xticks(x)
    axA.set_xticklabels([METHODS[k] for k in keys], rotation=15, ha="right")
    axA.set_ylabel(r"Relative reconstruction error $\|X-\hat X\|_F/\|X\|_F$")
    axA.set_title("Exact rank-10 recovery")
    axA.grid(True, ls=":", axis="y")

    # B) error vs rank
    d = df[df["experiment"] == "error_vs_rank"].copy()
    for k in keys:
        dd = d[d["method_key"] == k]
        g = dd.groupby("R", observed=True)["rel_err"].mean()
        axB.plot(g.index.values, g.values, marker="o", color=colors[k], label=METHODS[k])
    axB.set_yscale("log")
    axB.set_xlabel(r"Target rank $R$")
    axB.set_ylabel(r"Relative reconstruction error $\|X-\hat X_R\|_F/\|X\|_F$")
    axB.set_title("Approximation error vs rank")
    axB.grid(True, ls=":")
    axB.legend(frameon=False)

    # C) runtime vs size
    d = df[df["experiment"] == "runtime_vs_size"].copy()
    size_order = [f"{m}×{n}" for (m, n) in runtime_sizes]
    x = np.arange(len(size_order))
    for k in keys:
        dd = d[d["method_key"] == k]
        # Ignore failed runs (NaN)
        means = dd.groupby("size_label", observed=True)["time_s"].mean()
        stds = dd.groupby("size_label", observed=True)["time_s"].std()
        y = np.array([means.get(lbl, np.nan) for lbl in size_order], dtype=float)
        e = np.array([stds.get(lbl, np.nan) for lbl in size_order], dtype=float)
        axC.errorbar(
            x,
            y,
            yerr=e,
            marker="o",
            markersize=6,
            capsize=4,
            color=colors[k],
            label=METHODS[k],
        )
    axC.set_yscale("log")
    axC.set_xticks(x)
    axC.set_xticklabels([rf"${lbl}$" for lbl in size_order])
    axC.set_xlabel("Size")
    axC.set_ylabel("Runtime (s)")
    axC.set_title(rf"Runtime vs matrix size ($R={runtime_rank}$)")
    axC.grid(True, ls=":")
    axC.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_dir / "qsvd_benchmark_dashboard.png", dpi=300, bbox_inches="tight", facecolor="white")
    _save_axis_pdf(fig, axA, out_dir / "qsvd_benchmark__exact_rank10_recovery.pdf")
    _save_axis_pdf(fig, axB, out_dir / "qsvd_benchmark__error_vs_rank.pdf")
    _save_axis_pdf(fig, axC, out_dir / "qsvd_benchmark__runtime_vs_size.pdf")
    plt.close(fig)

    print(f"Wrote benchmark outputs to: {out_dir}")


if __name__ == "__main__":
    main()

