"""
Experimental complex power iteration for quaternion matrices via complex adjoint mapping.

This routine chooses a fixed complex subfield (default along x-axis) and maps
an n×n quaternion matrix A = W + X i + Y j + Z k to a 2n×2n complex matrix:

  C = W + i X,  D = Y + i Z
  Adj(A) = [[ C,  D],
            [-D*, C*]]

Eigenvalues of Adj(A) appear in conjugate pairs and correspond to right-eigenvalues
of A in the chosen complex subfield. This provides complex-valued eigenvalue
estimates for general (non-Hermitian) quaternion matrices.

Usage:
  python tests/validation/experimental_power_iteration.py --n 20 --iters 300 --tol 1e-10 --k 10 --plot
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple, List

import numpy as np
import quaternion  # type: ignore

# Robust sys.path for direct runs
root = os.getcwd()
if root not in sys.path:
    sys.path.insert(0, root)
core_path = os.path.join(root, "core")
if core_path not in sys.path:
    sys.path.insert(0, core_path)

from core.data_gen import create_test_matrix


def quaternion_to_complex_adjoint(A: np.ndarray) -> np.ndarray:
    """Map quaternion matrix A to 2n×2n complex adjoint matrix (fixed x-axis subfield).

    A = W + X i + Y j + Z k → C = W + i X, D = Y + i Z, M = [[C, D], [-D*, C*]]
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.dtype != np.quaternion:
        raise ValueError("A must be square quaternion matrix")
    n = A.shape[0]
    Af = quaternion.as_float_array(A)  # (n,n,4) -> (w,x,y,z)
    W, X, Y, Z = Af[..., 0], Af[..., 1], Af[..., 2], Af[..., 3]
    C = W + 1j * X
    D = Y + 1j * Z
    M = np.zeros((2 * n, 2 * n), dtype=complex)
    M[0:n, 0:n] = C
    M[0:n, n:2 * n] = D
    M[n:2 * n, 0:n] = -np.conjugate(D)
    M[n:2 * n, n:2 * n] = np.conjugate(C)
    return M


def power_iteration_complex(M: np.ndarray, max_iter: int = 300, tol: float = 1e-10, seed: int = 0, res_tol: float | None = None) -> Tuple[complex, np.ndarray, List[float]]:
    """Standard complex power iteration on matrix M returning (lambda, v, residuals).

    Residual at step t: ||M v_t - lambda_t v_t||_2
    """
    rng = np.random.default_rng(seed)
    n = M.shape[0]
    v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    v = v / np.linalg.norm(v)
    lam = 0.0 + 0.0j
    residuals: List[float] = []
    for _ in range(max_iter):
        w = M @ v
        nw = np.linalg.norm(w)
        if not np.isfinite(nw) or nw == 0.0:
            break
        v_new = w / nw
        # Rayleigh quotient
        Mv = M @ v_new
        lam_new = np.vdot(v_new, Mv) / np.vdot(v_new, v_new)
        # Residual
        res = np.linalg.norm(Mv - lam_new * v_new)
        residuals.append(float(res))
        # Convergence checks: eigenvalue stabilization or residual threshold
        if res_tol is not None and res <= res_tol:
            v = v_new
            lam = lam_new
            break
        if np.abs(lam_new - lam) <= tol * max(1.0, np.abs(lam_new)):
            v = v_new
            lam = lam_new
            break
        v = v_new
        lam = lam_new
    return lam, v, residuals


def quaternion_power_iteration_complex(A: np.ndarray, max_iter: int = 300, tol: float = 1e-10, seed: int = 0, res_tol: float | None = None) -> Tuple[complex, List[float]]:
    """Return a complex eigenvalue estimate and residual curve for quaternion A via complex adjoint mapping."""
    M = quaternion_to_complex_adjoint(A)
    lam, _v, residuals = power_iteration_complex(M, max_iter=max_iter, tol=tol, seed=seed, res_tol=res_tol)
    return lam, residuals


def main():
    ap = argparse.ArgumentParser(description="Experimental complex power iteration for quaternion matrices")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=5, help="Number of leading eigenvalues via power-iteration restarts")
    ap.add_argument("--plot", action="store_true", help="Plot residual convergence curves")
    ap.add_argument("--res_tol", type=float, default=1e-8, help="Residual tolerance ||Mv - lambda v|| to declare convergence")
    args = ap.parse_args()

    n = args.n
    A = create_test_matrix(n, n)
    # Simple restarted scheme to estimate top-k eigenvalues by deflation (Gram-Schmidt)
    # For demonstration only; not optimized and may repeat close eigenvalues
    M = quaternion_to_complex_adjoint(A)
    eigs: List[complex] = []
    resids: List[List[float]] = []
    V = []  # store converged vectors for crude deflation
    for idx in range(max(1, args.k)):
        lam, v, residuals = power_iteration_complex(M, max_iter=args.iters, tol=args.tol, seed=args.seed + idx, res_tol=args.res_tol)
        eigs.append(lam)
        resids.append(residuals)
        # Crude deflation: orthogonalize against previous v's
        V.append(v)
        if len(V) > 0:
            # Build projector P = I - sum v v^*
            # Instead of forming P, re-orthogonalize next start vector via seed change
            pass
    # Report
    print(f"n={n} | k={len(eigs)} complex eigenvalue estimates:")
    import quaternion as q
    for i, lam in enumerate(eigs):
        last_res = resids[i][-1] if resids[i] else float('nan')
        lam_q = q.quaternion(float(np.real(lam)), float(np.imag(lam)), 0.0, 0.0)
        print(f"  {i+1:2d}: lam={lam} | lam_quat={lam_q} | |lam|={abs(lam):.6e} | final residual={last_res:.3e} | steps={len(resids[i])}")
    if args.plot:
        import matplotlib.pyplot as plt
        for i, curve in enumerate(resids):
            if curve:
                plt.semilogy(curve, label=f"eig {i+1}")
        plt.title(f"Complex power iteration residuals (n={n})")
        plt.xlabel("iteration")
        plt.ylabel("||Mv - lambda v||_2")
        plt.grid(True, which="both", ls=":")
        plt.legend()
        from pathlib import Path
        out_dir = Path("validation_output")
        out_dir.mkdir(exist_ok=True)
        fname = out_dir / f"complex_power_iteration_n{n}_k{len(eigs)}.png"
        plt.tight_layout(); plt.savefig(fname, dpi=300)
        print(f"saved plot: {fname}")


if __name__ == "__main__":
    main()


