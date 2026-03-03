#!/usr/bin/env python3
"""
Visual diagnostics for quaternion Cholesky (dense + optional sparse).

This is a *plotting* test script (not a unit test). It builds Hermitian PD
quaternion matrices, computes Cholesky factors, and saves figures illustrating:
  - components (w/x/y/z) of A and L
  - magnitude |A|, |L|
  - lower-triangular structure of L
  - reconstruction error |A - L L^*|
  - (optional) sparse embedding sparsity patterns (χ(A), CHOLMOD L)
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import quaternion  # type: ignore
from scipy import sparse

def _find_project_root(start: str) -> str:
    cur = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(cur, "quatica")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(start)
        cur = parent


# Ensure `import quatica` works when running this script directly.
PROJECT_ROOT = _find_project_root(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from quatica.decomp.chol import chol_quat_dense, solve_chol_quat_dense, chol_quat_sparse
from quatica.utils import (
    SparseQuaternionMatrix,
    complex_expand_sparse,
    quat_eye,
    quat_frobenius_norm,
    quat_hermitian,
    quat_matmat,
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _quat_abs(Q: np.ndarray) -> np.ndarray:
    comp = quaternion.as_float_array(Q)
    return np.sqrt(np.sum(comp * comp, axis=-1))


def _plot_quat_components_heatmaps(Q: np.ndarray, title: str, outpath: str) -> None:
    comp = quaternion.as_float_array(Q)
    labels = ["w (real)", "x (i)", "y (j)", "z (k)"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    vmax = np.max(np.abs(comp)) + 1e-30
    for k, ax in enumerate(axes):
        im = ax.imshow(comp[..., k], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(labels[k])
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(title, fontsize=12)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def _plot_abs_heatmap(M: np.ndarray, title: str, outpath: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, label="|·|")
    ax.set_title(title)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def _plot_L_structure(L: np.ndarray, title: str, outpath: str) -> None:
    A = _quat_abs(L)
    upper = np.triu(A, k=1)
    lower = np.tril(A, k=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    im0 = axes[0].imshow(A, cmap="viridis", aspect="auto")
    axes[0].set_title("|L| (all)")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)

    im1 = axes[1].imshow(lower, cmap="viridis", aspect="auto")
    axes[1].set_title("|L| (lower incl diag)")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)

    im2 = axes[2].imshow(upper, cmap="magma", aspect="auto")
    axes[2].set_title("|L| (upper; should be ~0)")
    fig.colorbar(im2, ax=axes[2], shrink=0.85)

    for ax in axes:
        ax.set_xlabel("col")
        ax.set_ylabel("row")
    fig.suptitle(title, fontsize=12)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def _rand_quat_matrix(m: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Qf = rng.standard_normal((m, n, 4))
    return quaternion.as_quat_array(Qf)


@dataclass(frozen=True)
class DenseCase:
    n: int
    seed: int
    alpha: float
    hermitianize: bool = True
    tol: float = 1e-12
    jitter: float = 0.0


def run_dense_case(case: DenseCase, outdir: str) -> None:
    _ensure_dir(outdir)
    n = case.n
    B = _rand_quat_matrix(n, n, seed=case.seed)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        A = quat_matmat(B, quat_hermitian(B)) + float(case.alpha) * quat_eye(n)

    L = chol_quat_dense(A, tol=case.tol, hermitianize=case.hermitianize, jitter=case.jitter)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        A_rec = quat_matmat(L, quat_hermitian(L))
        E = A - A_rec

    rel = float(quat_frobenius_norm(E) / (quat_frobenius_norm(A) + 1e-30))

    # Save plots
    tag = f"n{n}_seed{case.seed}_alpha{case.alpha:g}"
    _plot_quat_components_heatmaps(A, f"Dense SPD A components ({tag})", os.path.join(outdir, f"{tag}_A_components.png"))
    _plot_quat_components_heatmaps(L, f"Cholesky factor L components ({tag})", os.path.join(outdir, f"{tag}_L_components.png"))

    _plot_abs_heatmap(_quat_abs(A), f"|A| ({tag})", os.path.join(outdir, f"{tag}_A_abs.png"))
    _plot_abs_heatmap(_quat_abs(L), f"|L| ({tag})", os.path.join(outdir, f"{tag}_L_abs.png"))
    _plot_L_structure(L, f"L triangular structure ({tag})", os.path.join(outdir, f"{tag}_L_structure.png"))
    _plot_abs_heatmap(_quat_abs(E), f"|A - L L^*| (rel={rel:.2e}) ({tag})", os.path.join(outdir, f"{tag}_recon_abs.png"))

    # Diagonal sanity (real positive)
    diag = quaternion.as_float_array(np.diag(L))
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)
    ax.plot(diag[:, 0], label="Re(diag(L))")
    ax.plot(np.linalg.norm(diag[:, 1:], axis=1), label="||Im(diag(L))||", linestyle="--")
    ax.set_title(f"Diagonal sanity ({tag})")
    ax.set_xlabel("k")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.savefig(os.path.join(outdir, f"{tag}_diag_sanity.png"), dpi=200)
    plt.close(fig)

    # Solve check
    rng = np.random.default_rng(case.seed + 123)
    b = quaternion.as_quat_array(rng.standard_normal((n, 4)))
    x = solve_chol_quat_dense(L, b)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        r = quat_matmat(A, x.reshape(n, 1)).reshape(n) - b
    res = float(quat_frobenius_norm(r) / (quat_frobenius_norm(b) + 1e-30))

    # Small text summary
    with open(os.path.join(outdir, f"{tag}_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Dense quaternion Cholesky diagnostics ({tag})\n")
        f.write(f"  rel_reconstruction = {rel:.3e}\n")
        f.write(f"  rel_solve_residual = {res:.3e}\n")


def run_sparse_case(n: int, outdir: str, seed: int = 0, density: float = 0.05, jitter: float = 1e-12) -> None:
    _ensure_dir(outdir)

    try:
        import sksparse.cholmod  # type: ignore  # noqa: F401
    except Exception:
        # Optional dependency; just write a note.
        with open(os.path.join(outdir, "sparse_skipped.txt"), "w", encoding="utf-8") as f:
            f.write("Sparse case skipped: sksparse.cholmod not available.\n")
        return

    rng = np.random.default_rng(seed)
    Lr = sparse.random(n, n, density=density, format="csr", random_state=rng)
    Lr = sparse.tril(Lr, k=0).tocsr()
    Lr = Lr + sparse.diags(np.full(n, 1.0), format="csr")

    zero = sparse.csr_matrix((n, n))
    Lq = SparseQuaternionMatrix(Lr, zero, zero, zero, (n, n))
    Aq = Lq @ Lq.conjugate().transpose()

    # Embedding sparsity plots
    chiA = complex_expand_sparse(Aq).tocsr()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
    ax.spy(chiA, markersize=1)
    ax.set_title(f"Sparsity of χ(A) (n={n}, density={density:g})")
    fig.savefig(os.path.join(outdir, f"sparse_chiA_spy_n{n}.png"), dpi=200)
    plt.close(fig)

    F = chol_quat_sparse(Aq, jitter=jitter, ordering="cholmod")

    # If CHOLMOD exposes its L factor, show sparsity of (complex) L.
    Lc: Optional[sparse.csc_matrix] = None
    if hasattr(F.factor, "L"):
        try:
            Lc = F.factor.L()
        except Exception:
            Lc = None

    if Lc is not None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
        ax.spy(Lc, markersize=1)
        ax.set_title(f"CHOLMOD sparsity of L (complex, 2n×2n), n={n}")
        fig.savefig(os.path.join(outdir, f"sparse_cholmod_L_spy_n{n}.png"), dpi=200)
        plt.close(fig)

    # Solve sanity (quaternion RHS, solved via embedding)
    b = quaternion.as_quat_array(rng.standard_normal((n, 4)))
    x = F.solve(b)
    Ax = Aq @ x
    num = np.linalg.norm(quaternion.as_float_array(Ax) - quaternion.as_float_array(b))
    den = np.linalg.norm(quaternion.as_float_array(b)) + 1e-30
    res = float(num / den)

    with open(os.path.join(outdir, f"sparse_summary_n{n}.txt"), "w", encoding="utf-8") as f:
        f.write("Sparse quaternion Cholesky via complex embedding (CHOLMOD)\n")
        f.write(f"  n = {n}\n")
        f.write(f"  density = {density:g}\n")
        f.write(f"  jitter = {jitter:.3e}\n")
        f.write(f"  rel_solve_residual = {res:.3e}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="validation_output/decomp/chol", help="Output folder for plots.")
    ap.add_argument("--n", type=int, default=12, help="Dense case size.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.0, help="SPD shift: A = B B^H + alpha I.")
    ap.add_argument("--also-sparse", action="store_true", help="Run optional sparse (requires scikit-sparse).")
    ap.add_argument("--sparse-n", type=int, default=80)
    ap.add_argument("--sparse-density", type=float, default=0.04)
    args = ap.parse_args()

    outdir = args.outdir
    _ensure_dir(outdir)

    run_dense_case(
        DenseCase(n=int(args.n), seed=int(args.seed), alpha=float(args.alpha)),
        outdir=os.path.join(outdir, "dense"),
    )

    if args.also_sparse:
        run_sparse_case(
            n=int(args.sparse_n),
            outdir=os.path.join(outdir, "sparse"),
            seed=int(args.seed),
            density=float(args.sparse_density),
        )

    print(f"Saved outputs under: {outdir}")


if __name__ == "__main__":
    main()

