#!/usr/bin/env python3
r"""
Targeted audit for pass-efficient Q-SVD (`pass_eff_qsvd`) before paper benchmarks.

This script validates:
- Reconstruction / approximation quality
- Orthogonality of returned factors
- Behavior across pass counts and target ranks
- Comparison against `classical_qsvd` (baseline) and `rand_qsvd`

Outputs (default: validation_output/qsvd_audit/):
- `qsvd_audit_metrics.csv`
- `qsvd_audit_dashboard.png`
- `qsvd_audit__error_vs_rank.pdf`
- `qsvd_audit__error_vs_passes.pdf`
- `qsvd_audit__runtime_vs_shape.pdf`
- `qsvd_audit__orthogonality_boxplot.pdf`
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    qmod = os.path.join(root, "quatica")
    if qmod not in sys.path:
        sys.path.insert(0, qmod)


_ensure_quatica_importable()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import quaternion  # type: ignore

from quatica.decomp.qsvd import classical_qsvd, classical_qsvd_full, pass_eff_qsvd, rand_qsvd
from quatica.utils import quat_eye, quat_frobenius_norm, quat_hermitian, quat_matmat


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


def _low_rank(m: int, n: int, r: int) -> np.ndarray:
    A = _random_quat(m, r)
    B = _random_quat(r, n)
    return quat_matmat(A, B)


def _ill_rect(m: int, n: int, *, cond: float = 1e4) -> np.ndarray:
    r = min(m, n)
    # Real-orthogonal lift for stability and reproducibility
    U, _ = np.linalg.qr(np.random.randn(m, m))
    V, _ = np.linalg.qr(np.random.randn(n, n))
    s = np.logspace(0.0, -np.log10(cond), r)
    # Build quaternion lift helpers
    def real_to_quat(M: np.ndarray) -> np.ndarray:
        Z = np.zeros_like(M)
        return quaternion.as_quat_array(np.stack([M, Z, Z, Z], axis=-1))

    Uq = real_to_quat(U)[:, :r]
    Vq = real_to_quat(V)[:, :r]
    Sigma = np.zeros((r, r, 4), dtype=float)
    Sigma[np.arange(r), np.arange(r), 0] = s
    Sigq = quaternion.as_quat_array(Sigma)
    return quat_matmat(quat_matmat(Uq, Sigq), quat_hermitian(Vq))


def _recon(U: np.ndarray, s: np.ndarray, V: np.ndarray) -> np.ndarray:
    return quat_matmat(quat_matmat(U, np.diag(s)), quat_hermitian(V))


def _orth_err(Q: np.ndarray) -> float:
    r = Q.shape[1]
    I = quat_eye(r)
    return float(quat_frobenius_norm(quat_matmat(quat_hermitian(Q), Q) - I))


@dataclass(frozen=True)
class RunCfg:
    family: str
    m: int
    n: int
    seed: int
    R: int
    n_passes: int


def main(argv: list[str] | None = None) -> None:
    _setup_style()
    p = argparse.ArgumentParser(description="Audit pass_eff_qsvd before paper benchmarks.")
    p.add_argument("--out-dir", default="validation_output/qsvd_audit", help="Output directory (relative to repo root).")
    p.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds.")
    p.add_argument("--passes", default="2,3,4,5", help="Comma-separated n_passes values.")
    p.add_argument("--ranks", default="5,10,20", help="Comma-separated target ranks.")
    args = p.parse_args(argv)

    root = Path(_find_repo_root(os.path.dirname(__file__)))
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    passes = [int(x) for x in args.passes.split(",") if x.strip()]
    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]

    shapes = [(120, 80), (80, 120), (100, 100)]
    families = ["dense_random", "low_rank", "ill_conditioned"]

    rows: list[dict[str, Any]] = []

    for family in families:
        for (m, n) in shapes:
            for seed in seeds:
                np.random.seed(seed)
                if family == "dense_random":
                    X = _random_quat(m, n)
                elif family == "low_rank":
                    X = _low_rank(m, n, r=min(20, min(m, n)))
                else:
                    X = _ill_rect(m, n, cond=1e4)

                Xn = float(quat_frobenius_norm(X)) + 1e-30

                for R in ranks:
                    if R > min(m, n):
                        continue

                    # Baselines: truncated classical and randomized
                    t0 = time.time()
                    Uc, sc, Vc = classical_qsvd(X, R)
                    tc = time.time() - t0
                    err_c = float(quat_frobenius_norm(X - _recon(Uc, sc, Vc)) / Xn)

                    t0 = time.time()
                    Ur, sr, Vr = rand_qsvd(X, R, oversample=10, n_iter=2)
                    tr = time.time() - t0
                    err_r = float(quat_frobenius_norm(X - _recon(Ur, sr, Vr)) / Xn)

                    for v in passes:
                        cfg = RunCfg(family, m, n, seed, R, v)
                        t0 = time.time()
                        Up, sp, Vp = pass_eff_qsvd(X, R, oversample=10, n_passes=v)
                        tp = time.time() - t0

                        err_p = float(quat_frobenius_norm(X - _recon(Up, sp, Vp)) / Xn)
                        uo = _orth_err(Up)
                        vo = _orth_err(Vp)

                        rows.append(
                            {
                                "family": cfg.family,
                                "m": cfg.m,
                                "n": cfg.n,
                                "shape": f"{cfg.m}×{cfg.n}",
                                "seed": cfg.seed,
                                "R": cfg.R,
                                "n_passes": cfg.n_passes,
                                "err_pass_eff": err_p,
                                "err_classical": err_c,
                                "err_rand": err_r,
                                "orth_U_pass_eff": uo,
                                "orth_V_pass_eff": vo,
                                "time_pass_eff_s": tp,
                                "time_classical_s": tc,
                                "time_rand_s": tr,
                            }
                        )

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "qsvd_audit_metrics.csv", index=False)

    # Dashboard + panel PDFs
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    ax1, ax2, ax3, ax4 = axs.ravel()

    # Error vs rank (aggregate, dense_random)
    d = df[df["family"] == "dense_random"].copy()
    for v in sorted(d["n_passes"].unique()):
        dd = d[d["n_passes"] == v]
        g = dd.groupby("R")["err_pass_eff"].mean()
        ax1.plot(g.index.values, g.values, marker="o", label=rf"$v={int(v)}$")
    ax1.set_yscale("log")
    ax1.set_title("pass_eff_qsvd: error vs rank (dense random)")
    ax1.set_xlabel(r"Target rank $R$")
    ax1.set_ylabel(r"Relative error $\|X-\hat X\|_F/\|X\|_F$")
    ax1.grid(True, ls=":")
    ax1.legend(frameon=False)

    # Error vs passes (aggregate, ill_conditioned)
    d = df[df["family"] == "ill_conditioned"].copy()
    for R in sorted(d["R"].unique()):
        dd = d[d["R"] == R]
        g = dd.groupby("n_passes")["err_pass_eff"].mean()
        ax2.plot(g.index.values, g.values, marker="o", label=rf"$R={int(R)}$")
    ax2.set_yscale("log")
    ax2.set_title("pass_eff_qsvd: error vs passes (ill-conditioned)")
    ax2.set_xlabel(r"Pass budget $v$")
    ax2.set_ylabel(r"Relative error $\|X-\hat X\|_F/\|X\|_F$")
    ax2.grid(True, ls=":")
    ax2.legend(frameon=False)

    # Runtime vs shape (R=10, v=2) comparing methods
    d = df[(df["R"] == 10) & (df["n_passes"] == min(passes))].copy()
    shapes_sorted = sorted(d["shape"].unique())
    x = np.arange(len(shapes_sorted))
    width = 0.27
    ax3.bar(x - width, [d[d["shape"] == s]["time_classical_s"].mean() for s in shapes_sorted], width, label="classical_qsvd")
    ax3.bar(x, [d[d["shape"] == s]["time_rand_s"].mean() for s in shapes_sorted], width, label="rand_qsvd")
    ax3.bar(x + width, [d[d["shape"] == s]["time_pass_eff_s"].mean() for s in shapes_sorted], width, label=rf"pass_eff_qsvd ($v={min(passes)}$)")
    ax3.set_yscale("log")
    ax3.set_title("Runtime vs shape (log scale)")
    ax3.set_xlabel("Shape")
    ax3.set_ylabel("Time (s)")
    ax3.set_xticks(x)
    ax3.set_xticklabels([rf"${s}$" for s in shapes_sorted])
    ax3.grid(True, ls=":", axis="y")
    ax3.legend(frameon=False)

    # Orthogonality residuals boxplot (pass_eff)
    vals = [
        df["orth_U_pass_eff"].values,
        df["orth_V_pass_eff"].values,
    ]
    ax4.boxplot(vals, tick_labels=[r"$\|U^HU-I\|_F$", r"$\|V^HV-I\|_F$"], showfliers=True)
    ax4.set_yscale("log")
    ax4.set_title("Orthogonality residuals (pass_eff_qsvd)")
    ax4.grid(True, ls=":", axis="y")

    fig.tight_layout()
    fig.savefig(out_dir / "qsvd_audit_dashboard.png", dpi=300, bbox_inches="tight", facecolor="white")
    _save_axis_pdf(fig, ax1, out_dir / "qsvd_audit__error_vs_rank.pdf")
    _save_axis_pdf(fig, ax2, out_dir / "qsvd_audit__error_vs_passes.pdf")
    _save_axis_pdf(fig, ax3, out_dir / "qsvd_audit__runtime_vs_shape.pdf")
    _save_axis_pdf(fig, ax4, out_dir / "qsvd_audit__orthogonality_boxplot.pdf")
    plt.close(fig)

    print(f"Wrote audit outputs to: {out_dir}")


if __name__ == "__main__":
    main()

