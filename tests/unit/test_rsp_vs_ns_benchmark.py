#!/usr/bin/env python3
"""
Quick benchmark: Newton–Schulz (gamma=1) vs RSP–Q (column variant) with varying r.
- Few sizes, few r values
- Reports CPU time, iterations, final proxy residual, and MP max error
- Tests BOTH:
  (a) Solver class RSP-Q column: RandomizedSketchProjectPseudoinverse.compute_column_variant
  (b) Reference RSP-Q column step: qr_qua + triangular solve
"""

import time
import numpy as np
import quaternion
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core.utils import quat_matmat, quat_frobenius_norm, quat_hermitian
from core.decomp.qsvd import qr_qua
from core.solver import NewtonSchulzPseudoinverse, RandomizedSketchProjectPseudoinverse, _solve_upper_triangular_quat


def random_quat_matrix(m: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A_real = rng.standard_normal((m, n))
    A_i = rng.standard_normal((m, n))
    A_j = rng.standard_normal((m, n))
    A_k = rng.standard_normal((m, n))
    return quaternion.as_quat_array(np.stack([A_real, A_i, A_j, A_k], axis=-1))


def rspq_column_solver_once(A: np.ndarray, r: int, tol: float, maxit: int, seed: int):
    # Use the solver class implementation (column variant)
    solver = RandomizedSketchProjectPseudoinverse(block_size=r, max_iter=maxit, tol=tol, verbose=False, seed=seed)
    t0 = time.time()
    X, info = solver.compute_column_variant(A)
    elapsed = time.time() - t0
    # MP quick check
    AX = quat_matmat(A, X)
    XA = quat_matmat(X, A)
    e1 = quat_frobenius_norm(quat_matmat(AX, A) - A)
    e2 = quat_frobenius_norm(quat_matmat(XA, X) - X)
    e3 = quat_frobenius_norm(AX - quat_hermitian(AX))
    e4 = quat_frobenius_norm(XA - quat_hermitian(XA))
    mp_max = max(e1, e2, e3, e4)
    final_resid = info['residual_norms'][-1] if info['residual_norms'] else np.nan
    return {
        'time': elapsed,
        'iterations': info.get('iterations', np.nan),
        'proxy_residual': final_resid,
        'mp_max_error': mp_max,
        'converged': info.get('converged', False),
    }


def rspq_column_ref_once(A: np.ndarray, r: int, s: int, tol: float, maxit: int, seed: int):
    m, n = A.shape
    # Init X0 = alpha A^H with alpha = 1 / ||A||_F^2
    A_H = quat_hermitian(A)
    alpha = 1.0 / max(quat_frobenius_norm(A) ** 2, 1e-16)
    X = alpha * A_H
    # Test sketch
    Pi = random_quat_matrix(n, s, seed + 1)
    A_Pi = quat_matmat(A, Pi)
    Pi_norm = max(quat_frobenius_norm(Pi), 1e-30)
    it = 0
    t0 = time.time()
    for k in range(maxit):
        it = k + 1
        Omega = random_quat_matrix(n, r, seed + 1000 + k)
        Y = quat_matmat(A, Omega)                # (m x r)
        R_id = Omega - quat_matmat(X, Y)         # (n x r)
        # Thin QR via real-embedded routine
        U, RY = qr_qua(Y)                        # U: (m x r), RY: (r x r)
        U_H = quat_hermitian(U)                  # (r x m)
        # Solve RY Z = U^H for Z
        Z = _solve_upper_triangular_quat(RY, U_H)  # (r x m)
        # Update
        X = X + quat_matmat(R_id, Z)             # (n x m)
        # Proxy residual
        resid = quat_frobenius_norm(Pi - quat_matmat(X, A_Pi)) / Pi_norm
        if resid <= tol:
            break
    elapsed = time.time() - t0
    # MP quick check
    AX = quat_matmat(A, X)
    XA = quat_matmat(X, A)
    e1 = quat_frobenius_norm(quat_matmat(AX, A) - A)
    e2 = quat_frobenius_norm(quat_matmat(XA, X) - X)
    e3 = quat_frobenius_norm(AX - quat_hermitian(AX))
    e4 = quat_frobenius_norm(XA - quat_hermitian(XA))
    mp_max = max(e1, e2, e3, e4)
    return {
        'time': elapsed,
        'iterations': it,
        'proxy_residual': resid,
        'mp_max_error': mp_max,
        'converged': resid <= tol,
    }


def run_quick_benchmark():
    # Config: a couple of sizes, and r grid
    cases = [
        (80, 40),  # tall
        (40, 30),  # smaller tall
    ]
    r_values = [4, 8, 12, 16]
    s = 6
    tol = 1e-6
    maxit = 300
    base_seed = 42

    print("Quick Benchmark: NS (gamma=1) vs RSP-Q (column)")
    print("=" * 60)

    for idx, (m, n) in enumerate(cases):
        print(f"\nCase {idx+1}: {m}x{n}")
        A = random_quat_matrix(m, n, base_seed + idx)

        # Baseline: Newton–Schulz gamma=1
        ns = NewtonSchulzPseudoinverse(gamma=1.0, max_iter=200, tol=1e-10, verbose=False)
        t0 = time.time()
        X_ns, metrics_ns, residuals_ns = ns.compute(A)
        ns_time = time.time() - t0
        ns_final_cov = residuals_ns[-1] if residuals_ns else np.nan
        # MP check
        AX = quat_matmat(A, X_ns)
        XA = quat_matmat(X_ns, A)
        e1 = quat_frobenius_norm(quat_matmat(AX, A) - A)
        e2 = quat_frobenius_norm(quat_matmat(XA, X_ns) - X_ns)
        e3 = quat_frobenius_norm(AX - quat_hermitian(AX))
        e4 = quat_frobenius_norm(XA - quat_hermitian(XA))
        ns_mp = max(e1, e2, e3, e4)
        print(f"NS (γ=1):    time={ns_time:.3f}s, iters={len(residuals_ns):3d}, cov_res={ns_final_cov:.2e}, mp_max={ns_mp:.2e}")

        # RSP-Q across r: solver version and reference version
        for r in r_values:
            out_solver = rspq_column_solver_once(A, r=r, tol=tol, maxit=maxit, seed=base_seed + 200 + r)
            out_ref = rspq_column_ref_once(A, r=r, s=s, tol=tol, maxit=maxit, seed=base_seed + 100 + r)
            speedup_solver = ns_time / out_solver['time'] if out_solver['time'] > 0 else float('inf')
            speedup_ref = ns_time / out_ref['time'] if out_ref['time'] > 0 else float('inf')
            print(
                f"r={r:2d} | SOLVER: time={out_solver['time']:.3f}s, iters={int(out_solver['iterations']) if out_solver['iterations']==out_solver['iterations'] else -1:3d}, "
                f"proxy_res={out_solver['proxy_residual']:.2e}, mp_max={out_solver['mp_max_error']:.2e}, conv={out_solver['converged']}, speedup={speedup_solver:.2f}x"
            )
            print(
                f"     |   REF: time={out_ref['time']:.3f}s, iters={out_ref['iterations']:3d}, proxy_res={out_ref['proxy_residual']:.2e}, "
                f"mp_max={out_ref['mp_max_error']:.2e}, conv={out_ref['converged']}, speedup={speedup_ref:.2f}x"
            )


if __name__ == "__main__":
    run_quick_benchmark()
