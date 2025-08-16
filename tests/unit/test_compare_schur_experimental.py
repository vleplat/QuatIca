"""
Compare experimental quaternion Schur variants (windowed AED and Francis-like two-shift).

Outputs:
- Convergence plots and |T| visualizations saved into validation_output/.

Usage (after venv activation):
  python tests/validation/compare_schur_experimental.py --sizes 30 50 --iters 500 --tol 1e-10 --tag exp
  python tests/validation/compare_schur_experimental.py --sizes 50 --iters 500 --tol 1e-10 --hermitian --tag herm --window 12
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib
import numpy as np

# Non-interactive backend for saving figures
matplotlib.use("Agg")
# Robust sys.path setup when run directly
import sys

import matplotlib.pyplot as plt
import quaternion  # type: ignore

root = os.getcwd()
if root not in sys.path:
    sys.path.insert(0, root)
core_path = os.path.join(root, "quatica")
if core_path not in sys.path:
    sys.path.insert(0, core_path)

from quatica.data_gen import create_sparse_quat_matrix, create_test_matrix
from quatica.decomp.schur import quaternion_schur_experimental
from quatica.utils import quat_frobenius_norm, quat_hermitian, quat_matmat


def _extract_curve(diag: dict) -> List[float]:
    return [d.get("max_subdiag", float("nan")) for d in diag.get("iterations", [])]


def _quat_abs(q: quaternion.quaternion) -> float:
    return float((q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z) ** 0.5)


def _max_first_subdiag(T: np.ndarray) -> float:
    n = T.shape[0]
    m = 0.0
    for i in range(1, n):
        m = max(m, _quat_abs(T[i, i - 1]))
    return m


def _max_below_diagonal(T: np.ndarray) -> float:
    n = T.shape[0]
    m = 0.0
    for i in range(n):
        for j in range(0, i):
            m = max(m, _quat_abs(T[i, j]))
    return m


def run_variants(
    A: np.ndarray,
    n: int,
    iters: int,
    tol: float,
    window: int,
    out_dir: Path,
    tag: str = "",
) -> None:
    variants: List[Tuple[str, str]] = [
        ("AED-windowed", "aed_windowed"),
        ("Francis-DS", "francis_ds"),
    ]

    curves = {}
    Ts = {}
    for name, vkey in variants:
        t0 = os.times()[4]
        Q, T, diag = quaternion_schur_experimental(
            A,
            variant=vkey,
            max_iter=iters,
            tol=tol,
            window=window,
            return_diagnostics=True,
        )
        t1 = os.times()[4]
        cpu = t1 - t0
        curve = _extract_curve(diag)
        sim = quat_frobenius_norm(quat_matmat(quat_hermitian(Q), quat_matmat(A, Q)) - T)
        # Unitarity check of Q: Q^H Q - I
        unit = quat_frobenius_norm(
            quat_matmat(quat_hermitian(Q), Q) - np.eye(n, dtype=np.quaternion)
        )
        sub1 = _max_first_subdiag(T)
        below = _max_below_diagonal(T)
        curves[name] = (curve, cpu, sim, unit, sub1, below)
        Ts[name] = T
        print(
            f"n={n:3d} | {name:12s} | ran={len(curve):4d}/{iters} | cpu={cpu:6.2f}s | "
            f"sub1={sub1:.3e} | below_diag_max={below:.3e} | sim={sim:.3e} | unit={unit:.3e}"
        )

    # Plot convergence
    out_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(7, 4))
    for name, (curve, *_rest) in curves.items():
        if curve:
            plt.semilogy(curve, label=name)
    plt.title(f"Quaternion Schur (Experimental) — n={n}")
    plt.xlabel("iteration")
    plt.ylabel("max |subdiag|")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    suffix = f"_{tag}" if tag else ""
    fname = out_dir / f"schur_experimental{suffix}_n{n}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    print(f"saved plot: {fname}")

    # Visualize |T| for each variant
    import quaternion as _q

    for name, T in Ts.items():
        Tf = _q.as_float_array(T)  # (n,n,4)
        Tarr = np.sqrt(np.sum(Tf * Tf, axis=2))
        plt.figure(figsize=(5, 4))
        im = plt.imshow(Tarr, cmap="viridis", aspect="auto")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(f"Experimental |T| — {name}, n={n}")
        plt.xlabel("column")
        plt.ylabel("row")
        tname = out_dir / f"schur_T{suffix}_n{n}_{name}_abs.png"
        plt.tight_layout()
        plt.savefig(tname, dpi=300)
        plt.close()
        print(f"saved T visualization: {tname}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare experimental quaternion Schur variants"
    )
    parser.add_argument("--sizes", nargs="*", type=int, default=[10, 20, 30, 50])
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument(
        "--window", type=int, default=12, help="Trailing window size for AED/DS sweeps"
    )
    parser.add_argument(
        "--hermitian", action="store_true", help="Use A = B^H @ B (Hermitian)"
    )
    parser.add_argument(
        "--sparse", action="store_true", help="Use a random sparse quaternion matrix"
    )
    parser.add_argument(
        "--tag", type=str, default="", help="Optional tag to append to output filenames"
    )
    args = parser.parse_args()

    out_dir = Path("validation_output")
    for n in args.sizes:
        if args.hermitian:
            B = create_test_matrix(n, n)
            A = quat_matmat(quat_hermitian(B), B)
            tag = args.tag or "herm"
        elif args.sparse:
            S = create_sparse_quat_matrix(n, n, density=0.05)
            A = quaternion.as_quat_array(
                np.stack(
                    [S.real.toarray(), S.i.toarray(), S.j.toarray(), S.k.toarray()],
                    axis=-1,
                )
            )
            tag = args.tag or "sparse"
        else:
            A = create_test_matrix(n, n)
            tag = args.tag or "rand"

        run_variants(
            A,
            n=n,
            iters=args.iters,
            tol=args.tol,
            window=args.window,
            out_dir=out_dir,
            tag=tag,
        )

    print("Done.")


if __name__ == "__main__":
    main()
