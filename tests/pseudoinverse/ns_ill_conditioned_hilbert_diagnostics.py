#!/usr/bin/env python3
r"""
Newton–Schulz pseudoinverse on ill-conditioned quaternion matrices (Hilbert test).

This script builds a *square invertible* quaternion matrix whose inverse is known
analytically, runs QuatIca's Newton–Schulz Moore–Penrose pseudoinverse routine,
and evaluates accuracy/residual metrics across increasing ill-conditioning.

Construction
------------
Let H_n be the (real) Hilbert matrix:
    (H_n)_{ij} = 1/(i+j-1),  1<=i,j<=n.

Let q be a fixed nonzero quaternion scalar:
    q = 1 + 2 i + 3 j + 4 k.

Define the quaternion matrix:
    Q = H_n * q,
so every quaternion entry Q_ij equals (H_n)_{ij} multiplied by the same scalar q.

Since H_n is real, it commutes with quaternion scalars, hence:
    Q^{-1} = H_n^{-1} * q^{-1} = q^{-1} * H_n^{-1}.

Because Q is square and invertible, we expect:
    Q^\dagger = Q^{-1}.

Outputs
-------
- Printed table of errors/residuals and runtime per n
- Paper-ready PNG+PDF plots under:
    validation_output/pseudoinverse/ns_hilbert_ill_conditioned/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import quaternion


def _find_project_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.isfile(os.path.join(cur, "pyproject.toml")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(start_dir)
        cur = parent


def _ensure_quatica_importable() -> None:
    try:
        import quatica  # noqa: F401
        return
    except Exception:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root = _find_project_root(script_dir)
        if root not in sys.path:
            sys.path.insert(0, root)


def _save_png_and_pdf(fig, path_png: str, *, dpi: int = 300) -> None:
    base, ext = os.path.splitext(path_png)
    if ext.lower() != ".png":
        path_png = base + ".png"
    path_pdf = base + ".pdf"
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    fig.savefig(path_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(path_pdf, dpi=dpi, bbox_inches="tight", facecolor="white")


def hilbert(n: int) -> np.ndarray:
    """Return H_n as float64 array."""
    i = np.arange(1, n + 1, dtype=float)
    j = np.arange(1, n + 1, dtype=float)
    return 1.0 / (i[:, None] + j[None, :] - 1.0)


def quat_scalar(a: float, b: float, c: float, d: float) -> quaternion.quaternion:
    return quaternion.quaternion(float(a), float(b), float(c), float(d))


def build_Q_from_H_and_q(H: np.ndarray, q: quaternion.quaternion) -> np.ndarray:
    """Build Q = H*q as a quaternion array, using explicit components."""
    w = H * float(q.w)
    x = H * float(q.x)
    y = H * float(q.y)
    z = H * float(q.z)
    return quaternion.as_quat_array(np.stack([w, x, y, z], axis=-1))


def build_theoretical_inverse(H_inv: np.ndarray, q_inv: quaternion.quaternion) -> np.ndarray:
    """Build Q^{-1}_theory = H^{-1} * q^{-1} as a quaternion array."""
    w = H_inv * float(q_inv.w)
    x = H_inv * float(q_inv.x)
    y = H_inv * float(q_inv.y)
    z = H_inv * float(q_inv.z)
    return quaternion.as_quat_array(np.stack([w, x, y, z], axis=-1))


def quat_eye(n: int) -> np.ndarray:
    I = np.zeros((n, n), dtype=np.quaternion)
    one = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
    for i in range(n):
        I[i, i] = one
    return I


def _latex_sci(x: float, *, sig: int = 2) -> str:
    """Format a float in LaTeX scientific notation."""
    if not np.isfinite(x):
        return r"\text{nan}"
    if x == 0.0:
        return "0"
    s = f"{x:.{sig}e}"
    mant, exp = s.split("e")
    exp_i = int(exp)
    return rf"{mant}\times 10^{{{exp_i}}}"


def _print_latex_table(
    metrics: List["Metrics"],
    *,
    caption: str,
    label: str,
    sig: int = 2,
) -> None:
    """Print a booktabs-style LaTeX table for the metrics."""
    print("\nLaTeX table (copy-paste):")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(rf"\caption{{{caption}}}")
    print(rf"\label{{{label}}}")
    print(r"\begin{tabular}{rrrrrrrr}")
    print(r"\toprule")
    print(
        r"$n$ & $\kappa_2(H_n)$ & iters & time (s) & rel.\ err & $\|QQ^\dagger-I\|_F$ & $\|Q^\dagger Q-I\|_F$ & $\|QQ^\dagger Q-Q\|_F$ \\"
    )
    print(r"\midrule")
    for m in metrics:
        print(
            rf"{m.n:d} & ${_latex_sci(m.cond_H2, sig=sig)}$ & {m.iters:d} & {m.time_s:.3f} & "
            rf"${_latex_sci(m.rel_frob_err, sig=sig)}$ & ${_latex_sci(m.inv_res_left, sig=sig)}$ & "
            rf"${_latex_sci(m.inv_res_right, sig=sig)}$ & ${_latex_sci(m.mp_res_1, sig=sig)}$ \\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


@dataclass
class Metrics:
    n: int
    cond_H2: float
    time_s: float
    iters: int
    rel_frob_err: float
    inv_res_left: float
    inv_res_right: float
    mp_res_1: float
    mp_res_2: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_min", type=int, default=3)
    parser.add_argument("--n_max", type=int, default=10)
    parser.add_argument("--n_list", type=str, default="", help="Comma list overriding n_min/n_max (e.g. 3,4,5).")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--no-latex",
        action="store_true",
        help="Do not print the LaTeX table (plain-text output only).",
    )
    parser.add_argument(
        "--latex_caption",
        type=str,
        default="Newton--Schulz pseudoinverse on quaternion-scaled Hilbert matrices $Q=H_n q$ ($q=1+2\\mathbf{i}+3\\mathbf{j}+4\\mathbf{k}$).",
    )
    parser.add_argument("--latex_label", type=str, default="tab:ns_hilbert")
    parser.add_argument(
        "--outdir",
        type=str,
        default="validation_output/pseudoinverse/ns_hilbert_ill_conditioned",
    )
    args = parser.parse_args()

    if args.no_display:
        plt.switch_backend("Agg")

    _ensure_quatica_importable()
    from quatica.solver import NewtonSchulzPseudoinverse
    from quatica.utils import quat_frobenius_norm, quat_matmat

    # Paper-ish defaults
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    if args.n_list.strip():
        ns = [int(x.strip()) for x in args.n_list.split(",") if x.strip()]
    else:
        ns = list(range(int(args.n_min), int(args.n_max) + 1))

    q = quat_scalar(1.0, 2.0, 3.0, 4.0)
    q_inv = (1.0 / q)  # scalar inverse

    out_dir = os.path.abspath(args.outdir)
    os.makedirs(out_dir, exist_ok=True)

    solver = NewtonSchulzPseudoinverse(
        gamma=float(args.gamma),
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        verbose=False,
        compute_residuals=True,
    )

    metrics: List[Metrics] = []
    histories: Dict[int, Dict[str, List[float]]] = {}

    print("Hilbert NS--Q ill-conditioning test")
    print(f"q = {q}")
    print(f"gamma={args.gamma}  tol={args.tol}  max_iter={args.max_iter}")
    print("")
    header = (
        " n | cond2(H)      | time(s) | it | rel_err(inv) | ||QQ†-I|| | ||Q†Q-I|| | ||QQ†Q-Q|| | ||Q†QQ†-Q†||"
    )
    print(header)
    print("-" * len(header))

    for n in ns:
        H = hilbert(n)
        cond_H2 = float(np.linalg.cond(H, 2))
        H_inv = np.linalg.inv(H)
        Q = build_Q_from_H_and_q(H, q)
        Q_inv_theory = build_theoretical_inverse(H_inv, q_inv)

        t0 = time.perf_counter()
        Q_pinv, residuals, covariances = solver.compute(Q)
        elapsed = time.perf_counter() - t0

        # Relative error vs theoretical inverse
        rel_err = float(
            quat_frobenius_norm(Q_pinv - Q_inv_theory)
            / max(quat_frobenius_norm(Q_inv_theory), 1e-30)
        )

        I = quat_eye(n)
        inv_res_left = float(quat_frobenius_norm(quat_matmat(Q, Q_pinv) - I))
        inv_res_right = float(quat_frobenius_norm(quat_matmat(Q_pinv, Q) - I))
        mp_res_1 = float(quat_frobenius_norm(quat_matmat(quat_matmat(Q, Q_pinv), Q) - Q))
        mp_res_2 = float(quat_frobenius_norm(quat_matmat(quat_matmat(Q_pinv, Q), Q_pinv) - Q_pinv))

        m = Metrics(
            n=n,
            cond_H2=cond_H2,
            time_s=float(elapsed),
            iters=len(covariances),
            rel_frob_err=rel_err,
            inv_res_left=inv_res_left,
            inv_res_right=inv_res_right,
            mp_res_1=mp_res_1,
            mp_res_2=mp_res_2,
        )
        metrics.append(m)
        histories[n] = {"cov": list(covariances), "mp_max": []}
        if residuals:
            # Track max MP residual per iteration for one curve.
            keys = list(residuals.keys())
            for k in range(len(covariances)):
                vals = []
                for kk in keys:
                    if k < len(residuals[kk]):
                        vals.append(residuals[kk][k])
                histories[n]["mp_max"].append(float(max(vals)) if vals else float("nan"))

        print(
            f"{n:2d} | {cond_H2:12.3e} | {elapsed:7.3f} | {len(covariances):2d} | {rel_err:12.3e} |"
            f" {inv_res_left:9.2e} | {inv_res_right:9.2e} | {mp_res_1:11.2e} | {mp_res_2:13.2e}"
        )

    if not args.no_latex:
        _print_latex_table(
            metrics,
            caption=str(args.latex_caption),
            label=str(args.latex_label),
            sig=2,
        )

    # Plot: errors/residuals/time vs conditioning
    x = np.array([m.cond_H2 for m in metrics], dtype=float)
    nvals = np.array([m.n for m in metrics], dtype=int)

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.2))
    ax.semilogy(x, [m.rel_frob_err for m in metrics], "o-", label=r"rel. error $\|Q^\dagger-Q^{-1}\|_F/\|Q^{-1}\|_F$")
    ax.semilogy(x, [m.inv_res_left for m in metrics], "s-", label=r"$\|QQ^\dagger-I\|_F$")
    ax.semilogy(x, [m.inv_res_right for m in metrics], "^-", label=r"$\|Q^\dagger Q-I\|_F$")
    ax.set_xlabel(r"$\kappa_2(H_n)$")
    ax.set_ylabel("Error / residual (log scale)")
    ax.set_title("Newton–Schulz on quaternion-scaled Hilbert matrices")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    for xi, ni in zip(x, nvals):
        ax.annotate(f"n={ni}", (xi, max(1e-300, metrics[list(nvals).index(ni)].rel_frob_err)), xytext=(6, 4), textcoords="offset points", fontsize=9)
    _save_png_and_pdf(fig, os.path.join(out_dir, "ns_hilbert_errors_vs_cond.png"), dpi=300)
    if not args.no_display:
        plt.show()
    else:
        plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.2))
    ax.plot(x, [m.time_s for m in metrics], "o-")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa_2(H_n)$")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime vs conditioning")
    ax.grid(True, which="both", alpha=0.3)
    _save_png_and_pdf(fig, os.path.join(out_dir, "ns_hilbert_time_vs_cond.png"), dpi=300)
    if not args.no_display:
        plt.show()
    else:
        plt.close(fig)

    # Plot: iteration histories (covariance deviation and max MP residual)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for n in ns:
        cov = histories[n]["cov"]
        if cov:
            ax1.semilogy(cov, label=f"n={n}")
        mp = histories[n]["mp_max"]
        if mp:
            ax2.semilogy(mp, label=f"n={n}")
    ax1.set_title("Covariance deviation per iteration")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel(r"$\|XA-I\|_F$ or $\|AX-I\|_F$")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    ax2.set_title("Max MP residual per iteration")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("max residual (log scale)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    fig.suptitle("Newton–Schulz convergence diagnostics (Hilbert test)", fontsize=14)
    plt.tight_layout()
    _save_png_and_pdf(fig, os.path.join(out_dir, "ns_hilbert_iteration_histories.png"), dpi=300)
    if not args.no_display:
        plt.show()
    else:
        plt.close(fig)

    print(f"\nSaved figures under: {out_dir}")


if __name__ == "__main__":
    main()

