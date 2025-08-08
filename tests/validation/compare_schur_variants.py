"""
Compare quaternion Schur (Lead 2) variants across sizes.

Variants:
 - none:        pure QR iteration without shift
 - rayleigh:    pure QR iteration with Rayleigh shift
 - implicit:    quaternion-only implicit QR with simple shift and bulge chasing
 - implicit+AED:implicit + aggressive early deflation (script-local)
 - implicit+DS: implicit + double-shift surrogate from trailing 2x2 (script-local)

Saves semilogy convergence plots to validation_output/ and prints metrics.

Run:
  PYTHONPATH=$PWD:$PWD/core python tests/validation/compare_schur_variants.py

Optional args:
  --sizes 10 20 30 50
  --iters 1000
  --tol 1e-10
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import List

import numpy as np
import matplotlib

# Non-interactive backend for saving figures
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import quaternion  # type: ignore


# Robust sys.path setup when run directly
import sys
root = os.getcwd()
if root not in sys.path:
    sys.path.insert(0, root)
core_path = os.path.join(root, "core")
if core_path not in sys.path:
    sys.path.insert(0, core_path)

from core.decomp.schur import (
    quaternion_schur_pure,
    quaternion_schur_pure_implicit,
)
from core.decomp.hessenberg import hessenbergize, check_hessenberg
from core.decomp.tridiagonalize import householder_matrix
from core.data_gen import create_test_matrix
from core.utils import (
    quat_matmat,
    quat_hermitian,
    quat_frobenius_norm,
    quat_eye,
)


def max_subdiag(H: np.ndarray) -> float:
    n = H.shape[0]
    maximum = 0.0
    for i in range(1, n):
        h = H[i, i - 1]
        s = math.sqrt(h.w * h.w + h.x * h.x + h.y * h.y + h.z * h.z)
        maximum = max(maximum, s)
    return maximum


def pure_implicit_custom(
    A: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-10,
    shift_mode: str = "rayleigh",
    aed_factor: float = 1.0,
    double_shift: bool = False,
    return_curve: bool = False,
):
    # Fast helpers: apply 2x2 quaternion block from left/right to the Hessenberg
    def apply_left_rows(M: np.ndarray, s_idx: int, B: np.ndarray) -> None:
        # B is 2x2 quaternion; updates rows s and s+1: [r0;r1] <- B @ [r0;r1]
        a, b = B[0, 0], B[0, 1]
        c, d = B[1, 0], B[1, 1]
        r0 = M[s_idx, :].copy()
        r1 = M[s_idx + 1, :].copy()
        M[s_idx, :] = a * r0 + b * r1
        M[s_idx + 1, :] = c * r0 + d * r1

    def apply_right_cols(M: np.ndarray, s_idx: int, B: np.ndarray) -> None:
        # Right-apply B^H to columns s and s+1: [c0,c1] <- [c0,c1] @ B^H
        BH = quat_hermitian(B)
        a, b = BH[0, 0], BH[0, 1]
        c, d = BH[1, 0], BH[1, 1]
        c0 = M[:, s_idx].copy()
        c1 = M[:, s_idx + 1].copy()
        M[:, s_idx] = c0 * a + c1 * c
        M[:, s_idx + 1] = c0 * b + c1 * d

    n = A.shape[0]
    P0, H = hessenbergize(A)
    H = check_hessenberg(H)
    Q_accum = quat_eye(n)
    curve: List[float] = []

    for _ in range(max_iter):
        # Choose shift(s)
        if double_shift and n >= 2:
            w11 = float(H[n - 2, n - 2].w)
            w12 = float(H[n - 2, n - 1].w)
            w21 = float(H[n - 1, n - 2].w)
            w22 = float(H[n - 1, n - 1].w)
            B = np.array([[w11, w12], [w21, w22]], dtype=float)
            evals = np.linalg.eigvals(B)
            sigmas = [float(np.real(evals[0])), float(np.real(evals[1]))]
        else:
            if shift_mode == "rayleigh":
                sigmas = [float(H[n - 1, n - 1].w)]
            else:
                sigmas = [0.0]

        for sigma in sigmas:
            qs = quaternion.quaternion(float(sigma), 0.0, 0.0, 0.0)
            # Bulge init + chase (structured updates)
            for s in range(0, n - 1):
                v0 = H[s, s] - qs
                v1 = H[s + 1, s]
                # If subdiagonal is already zero, skip
                if (v1.w == 0.0) and (v1.x == 0.0) and (v1.y == 0.0) and (v1.z == 0.0):
                    continue
                local = np.array([v0, v1], dtype=np.quaternion)
                e1 = np.zeros(2)
                e1[0] = 1.0
                Hj_sub = householder_matrix(local, e1)  # 2x2
                # Left-apply to rows s,s+1 and right-apply to cols s,s+1 only
                apply_left_rows(H, s, Hj_sub)
                apply_right_cols(H, s, Hj_sub)
                # Accumulate Q: Q_accum <- Q_accum @ Hj^H (right-apply to columns)
                apply_right_cols(Q_accum, s, Hj_sub)

        # AED sweep
        for i in range(1, n):
            h = H[i, i - 1]
            sv = math.sqrt(h.w * h.w + h.x * h.x + h.y * h.y + h.z * h.z)
            dscale = (
                math.sqrt(
                    H[i - 1, i - 1].w ** 2
                    + H[i - 1, i - 1].x ** 2
                    + H[i - 1, i - 1].y ** 2
                    + H[i - 1, i - 1].z ** 2
                )
                + math.sqrt(
                    H[i, i].w ** 2 + H[i, i].x ** 2 + H[i, i].y ** 2 + H[i, i].z ** 2
                )
                + 1e-30
            )
            if sv <= aed_factor * tol * max(1.0, dscale):
                H[i, i - 1] = quaternion.quaternion(0.0, 0.0, 0.0, 0.0)

        ms = max_subdiag(H)
        curve.append(ms)
        if ms <= tol:
            break

    Q_total = quat_matmat(quat_hermitian(P0), Q_accum)
    T = H
    if return_curve:
        return Q_total, T, curve
    return Q_total, T


def run_variants(A: np.ndarray, n: int, iters: int, tol: float, out_dir: Path, tag: str = "") -> None:
    # Heuristic to tune AED factor based on size n
    def select_aed_factor(n_val: int) -> float:
        if n_val <= 20:
            return 3.0
        if n_val <= 50:
            return 5.0
        if n_val <= 75:
            return 6.0
        return 8.0

    aed = select_aed_factor(n)

    # Retain only the two preferred variants for tests
    variants = [
        (
            "rayleigh",
            lambda: quaternion_schur_pure(
                A, max_iter=iters, tol=tol, return_diagnostics=True, shift_mode="rayleigh"
            ),
        ),
        (
            "implicit+AED",
            lambda: pure_implicit_custom(
                A,
                max_iter=iters,
                tol=tol,
                shift_mode="rayleigh",
                aed_factor=aed,
                double_shift=False,
                return_curve=True,
            ),
        ),
    ]

    curves = {}
    Ts = {}
    for name, fn in variants:
        t0 = os.times()[4]
        res = fn()
        t1 = os.times()[4]
        cpu = t1 - t0
        if name.startswith("implicit+"):
            Q, T, curve = res
        else:
            Q, T, diag = res
            curve = [d["max_subdiag"] for d in diag["iterations"]]
        sim = quat_frobenius_norm(quat_matmat(quat_hermitian(Q), quat_matmat(A, Q)) - T)
        unit = quat_frobenius_norm(quat_matmat(quat_hermitian(Q), Q) - np.eye(n, dtype=np.quaternion))
        final = curve[-1] if curve else float("nan")
        curves[name] = (curve, cpu, sim, unit, final)
        Ts[name] = T
        print(
            f"n={n:3d} | {name:13s} | ran={len(curve):4d}/{iters} | cpu={cpu:6.2f}s | final_sub={final:.3e} | sim={sim:.3e} | unit={unit:.3e}"
        )

    # Plot
    out_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(7, 4))
    for name, (curve, *_rest) in curves.items():
        if curve:
            plt.semilogy(curve, label=name)
    plt.title(f"Quaternion Schur (Lead 2) variants — n={n}")
    plt.xlabel("iteration")
    plt.ylabel("max |subdiag|")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    suffix = f"_{tag}" if tag else ""
    fname = out_dir / f"schur_lead2_variants{suffix}_n{n}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    print(f"saved plot: {fname}")

    # Visualize Schur matrices (real component) for each variant
    import quaternion as _q
    for name, T in Ts.items():
        Tarr = _q.as_float_array(T)[:, :, 0]
        plt.figure(figsize=(5, 4))
        im = plt.imshow(Tarr, cmap='viridis', aspect='auto')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(f"Schur T (real) — {name}, n={n}")
        plt.xlabel("column"); plt.ylabel("row")
        tname = out_dir / f"schur_T{suffix}_n{n}_{name}_real.png"
        plt.tight_layout(); plt.savefig(tname, dpi=300); plt.close()
        print(f"saved T visualization: {tname}")


def main():
    parser = argparse.ArgumentParser(description="Compare quaternion Schur variants")
    parser.add_argument("--sizes", nargs="*", type=int, default=[10, 20, 30, 50])
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--hermitian", action="store_true", help="Use A = B^H @ B (Hermitian)")
    parser.add_argument("--tag", type=str, default="", help="Optional tag to append to output filenames")
    parser.add_argument("--balance", type=str, default="none", choices=["none", "diag"], help="Optional diagonal equilibration before QR")
    args = parser.parse_args()

    out_dir = Path("validation_output")
    for n in args.sizes:
        if args.hermitian:
            B = create_test_matrix(n, n)
            A = quat_matmat(quat_hermitian(B), B)
            base_tag = args.tag or "herm"
        else:
            A = create_test_matrix(n, n)
            base_tag = args.tag or "rand"

        if args.balance == "diag":
            # Simple diagonal equilibration (Parlett–Reinsch style, 3 sweeps)
            import quaternion as _q
            Aq = A.copy()
            for _ in range(3):
                Af = _q.as_float_array(Aq)  # (n,n,4)
                # Element-wise quaternion modulus matrix M (n,n)
                M = np.sqrt(np.sum(Af * Af, axis=2))
                # Row/column 2-norms of modulus matrix
                R = np.linalg.norm(M, axis=1) + 1e-30
                C = np.linalg.norm(M, axis=0) + 1e-30
                # Compute scaling factors per index i
                for i in range(n):
                    f = np.sqrt(C[i] / R[i])
                    # Limit growth to avoid over/underflow
                    if f < 0.5:
                        f = 0.5
                    elif f > 2.0:
                        f = 2.0
                    if abs(f - 1.0) < 1e-3:
                        continue
                    fi = quaternion.quaternion(float(f), 0.0, 0.0, 0.0)
                    fi_inv = quaternion.quaternion(float(1.0 / f), 0.0, 0.0, 0.0)
                    # Scale row i by f^{-1} on the left, column i by f on the right
                    Aq[i, :] = fi_inv * Aq[i, :]
                    Aq[:, i] = Aq[:, i] * fi
            A_used = Aq
            tag = base_tag + "_bal"
        else:
            A_used = A
            tag = base_tag

        run_variants(A_used, n=n, iters=args.iters, tol=args.tol, out_dir=out_dir, tag=tag)
    print("Done.")


if __name__ == "__main__":
    main()


