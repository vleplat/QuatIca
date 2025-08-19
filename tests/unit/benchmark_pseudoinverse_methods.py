#!/usr/bin/env python3
"""
Benchmark: Compare quaternion pseudoinverse solvers on tall matrices m = n + 50.

Methods:
  - Newton–Schulz (γ=1)
  - Higher-order Newton–Schulz (3rd order)
  - RSP-Q (column variant)
  - Hybrid RSP-Q + NS (column)
  - CGNE–Q (column) — new

Outputs:
  - Summary table (stdout)
  - CSV file with metrics (validation_output/pinv_benchmark_summary.csv)
  - Plots (validation_output/pinv_benchmark_{time}.png)

Note: Uses only quaternion-native operations and saves artifacts to validation_output/.
"""

import os
import time
import csv
from datetime import datetime
import numpy as np
import quaternion
import matplotlib.pyplot as plt

from quatica.solver import (
    NewtonSchulzPseudoinverse,
    HigherOrderNewtonSchulzPseudoinverse,
    RandomizedSketchProjectPseudoinverse,
    HybridRSPNewtonSchulz,
    CGNEQSolver,
)
from quatica.utils import quat_matmat, quat_frobenius_norm, quat_eye


def rand_quat(m: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A_real = rng.standard_normal((m, n))
    A_i = rng.standard_normal((m, n))
    A_j = rng.standard_normal((m, n))
    A_k = rng.standard_normal((m, n))
    return quaternion.as_quat_array(np.stack([A_real, A_i, A_j, A_k], axis=-1))


def rel_residuals(A: np.ndarray, X: np.ndarray) -> tuple[float, float]:
    m, n = A.shape
    I_m = quat_eye(m)
    I_n = quat_eye(n)
    AX = quat_matmat(A, X)
    XA = quat_matmat(X, A)
    rel_AX = quat_frobenius_norm(AX - I_m) / max(quat_frobenius_norm(I_m), 1e-30)
    rel_XA = quat_frobenius_norm(XA - I_n) / max(quat_frobenius_norm(I_n), 1e-30)
    return float(rel_AX), float(rel_XA)


def pick_rsp_block(n: int) -> int:
    # Heuristic block size for RSP-Q
    return max(8, min(32, n // 8))


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main():
    outdir = os.path.join("validation_output")
    ensure_outdir(outdir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    sizes = [20, 50, 100, 150]
    seed = 123
    tol_ns = 1e-8
    tol_rsp = 1e-2
    tol_cg = 1e-8

    rows = []

    for n in sizes:
        m = n + 50  # tall to support all column-oriented methods
        A = rand_quat(m, n, seed + n)

        # Newton–Schulz (γ=1)
        ns = NewtonSchulzPseudoinverse(gamma=1.0, max_iter=300, tol=tol_ns, verbose=False)
        t0 = time.time(); X_ns, _, _ = ns.compute(A); t_ns = time.time() - t0
        rAX_ns, rXA_ns = rel_residuals(A, X_ns)

        # Higher-order NS (3rd order)
        hon = HigherOrderNewtonSchulzPseudoinverse(max_iter=120, tol=0.0, verbose=False)
        t0 = time.time(); X_hon, _, _ = hon.compute(A); t_hon = time.time() - t0
        rAX_hon, rXA_hon = rel_residuals(A, X_hon)

        # RSP-Q (column)
        r = pick_rsp_block(n)
        # Use QR variant for column update (faster); keep SPD available as opt-in
        r_eff = r
        rsp = RandomizedSketchProjectPseudoinverse(block_size=r_eff, max_iter=600, tol=tol_rsp, test_sketch_size=6, verbose=False, seed=seed, column_solver="qr")
        t0 = time.time(); X_rsp, info_rsp = rsp.compute_column_variant(A); t_rsp = time.time() - t0
        rAX_rsp, rXA_rsp = rel_residuals(A, X_rsp)

        # Hybrid RSP-Q + NS (column)
        hyb = HybridRSPNewtonSchulz(r=r_eff, p=4, T=5, tol=tol_rsp, max_iter=600, verbose=False, seed=seed, column_solver="qr")
        t0 = time.time(); X_hyb, info_hyb = hyb.compute(A); t_hyb = time.time() - t0
        rAX_hyb, rXA_hyb = rel_residuals(A, X_hyb)

        # CGNE–Q (column)
        cg = CGNEQSolver(tol=tol_cg, max_iter=1000, verbose=False, preconditioner_rank=0, seed=seed)
        t0 = time.time(); X_cg, info_cg = cg.compute(A); t_cg = time.time() - t0
        rAX_cg, rXA_cg = rel_residuals(A, X_cg)

        rows.append({
            'n': n, 'm': m,
            'r_rsp': r_eff,
            'time_ns': t_ns, 'rel_AX_ns': rAX_ns, 'rel_XA_ns': rXA_ns,
            'time_hon': t_hon, 'rel_AX_hon': rAX_hon, 'rel_XA_hon': rXA_hon,
            'time_rsp': t_rsp, 'rel_AX_rsp': rAX_rsp, 'rel_XA_rsp': rXA_rsp,
            'time_hyb': t_hyb, 'rel_AX_hyb': rAX_hyb, 'rel_XA_hyb': rXA_hyb,
            'time_cg': t_cg, 'rel_AX_cg': rAX_cg, 'rel_XA_cg': rXA_cg,
        })

    # Write CSV
    csv_path = os.path.join(outdir, f"pinv_benchmark_summary_{ts}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Print summary table (compact)
    print("\nSummary (n, times [s], rel_XA):")
    for r in rows:
        print(
            f"n={r['n']:>3} | ns {r['time_ns']:.3f}/{r['rel_XA_ns']:.2e} | hon {r['time_hon']:.3f}/{r['rel_XA_hon']:.2e} | "
            f"rsp {r['time_rsp']:.3f}/{r['rel_XA_rsp']:.2e} | hyb {r['time_hyb']:.3f}/{r['rel_XA_hyb']:.2e} | cg {r['time_cg']:.3f}/{r['rel_XA_cg']:.2e}"
        )

    # Plots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    ns = rows
    xs = [r['n'] for r in rows]
    # Time
    axs[0].plot(xs, [r['time_ns'] for r in rows], '-o', label='NS (γ=1)')
    axs[0].plot(xs, [r['time_hon'] for r in rows], '-o', label='HON (3rd)')
    axs[0].plot(xs, [r['time_rsp'] for r in rows], '-o', label='RSP-Q col')
    axs[0].plot(xs, [r['time_hyb'] for r in rows], '-o', label='Hybrid RSP+NS')
    axs[0].plot(xs, [r['time_cg'] for r in rows], '-o', label='CGNE–Q')
    axs[0].set_xlabel('n')
    axs[0].set_ylabel('time [s]')
    axs[0].set_title('Runtime vs n (m=n+50)')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()
    # Residual (rel_XA)
    axs[1].semilogy(xs, [max(1e-16, r['rel_XA_ns']) for r in rows], '-o', label='NS (γ=1)')
    axs[1].semilogy(xs, [max(1e-16, r['rel_XA_hon']) for r in rows], '-o', label='HON (3rd)')
    axs[1].semilogy(xs, [max(1e-16, r['rel_XA_rsp']) for r in rows], '-o', label='RSP-Q col')
    axs[1].semilogy(xs, [max(1e-16, r['rel_XA_hyb']) for r in rows], '-o', label='Hybrid RSP+NS')
    axs[1].semilogy(xs, [max(1e-16, r['rel_XA_cg']) for r in rows], '-o', label='CGNE–Q')
    axs[1].set_xlabel('n')
    axs[1].set_ylabel('rel ||XA - I||_F')
    axs[1].set_title('Accuracy (rel_XA) vs n')
    axs[1].grid(True, which='both', alpha=0.3)
    axs[1].legend()

    fig.tight_layout()
    png_path = os.path.join(outdir, f"pinv_benchmark_{ts}.png")
    fig.savefig(png_path, dpi=150)
    print(f"\nSaved CSV: {csv_path}\nSaved plot: {png_path}")


if __name__ == '__main__':
    main()


