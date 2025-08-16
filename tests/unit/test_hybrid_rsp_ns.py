#!/usr/bin/env python3
"""
Quick test for HybridRSPNewtonSchulz (column variant)
"""

import time
import numpy as np
import quaternion

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core.utils import quat_matmat, quat_frobenius_norm, quat_hermitian
from core.solver import HybridRSPNewtonSchulz, RandomizedSketchProjectPseudoinverse, NewtonSchulzPseudoinverse


def random_quat_matrix(m: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A_real = rng.standard_normal((m, n))
    A_i = rng.standard_normal((m, n))
    A_j = rng.standard_normal((m, n))
    A_k = rng.standard_normal((m, n))
    return quaternion.as_quat_array(np.stack([A_real, A_i, A_j, A_k], axis=-1))


def mp_errors(A: np.ndarray, X: np.ndarray) -> dict:
    AX = quat_matmat(A, X)
    XA = quat_matmat(X, A)
    return {
        'e1_AXA_A': quat_frobenius_norm(quat_matmat(AX, A) - A),
        'e2_XAX_X': quat_frobenius_norm(quat_matmat(XA, X) - X),
        'e3_AX_herm': quat_frobenius_norm(AX - quat_hermitian(AX)),
        'e4_XA_herm': quat_frobenius_norm(XA - quat_hermitian(XA)),
    }


def main():
    m, n = 500, 200
    tol = 1e-6
    maxit = 500
    seed = 123
    sketch_block_size = 30

    print(f"Testing HybridRSPNewtonSchulz on {m}x{n} (column variant)")
    A = random_quat_matrix(m, n, seed)

    # Baseline: NS gamma=1
    ns = NewtonSchulzPseudoinverse(gamma=1.0, max_iter=maxit, tol=1e-10, verbose=True)
    t0 = time.time()
    X_ns, metrics_ns, residuals_ns = ns.compute(A)
    ns_time = time.time() - t0
    ns_mp = mp_errors(A, X_ns)
    print(f"NS(γ=1): time={ns_time:.3f}s, iters={len(residuals_ns):3d}, mp_max={max(ns_mp.values()):.2e}")

    # RSP-Q column (solver) as reference
    rsp = RandomizedSketchProjectPseudoinverse(block_size=sketch_block_size, max_iter=maxit, tol=tol, verbose=True, seed=seed)
    t0 = time.time()
    X_rsp, info_rsp = rsp.compute_column_variant(A)
    rsp_time = time.time() - t0
    rsp_mp = mp_errors(A, X_rsp)
    print(f"RSP-Q(r=12): time={rsp_time:.3f}s, iters={info_rsp.get('iterations', 0):3d}, proxy={info_rsp['residual_norms'][-1] if info_rsp['residual_norms'] else float('nan'):.2e}, mp_max={max(rsp_mp.values()):.2e}")

    # Hybrid settings
    hybrid = HybridRSPNewtonSchulz(r=sketch_block_size, p=3, T=5, tol=tol, max_iter=maxit, verbose=True, seed=seed)
    t0 = time.time()
    X_h, info_h = hybrid.compute(A)
    hyb_time = time.time() - t0
    hyb_mp = mp_errors(A, X_h)
    last_proxy = info_h['residual_norms'][-1] if info_h['residual_norms'] else float('nan')
    print(f"Hybrid(r=12,p=4,T=5): time={hyb_time:.3f}s, RSP-steps={info_h['iterations_rsp']:3d}, proxy={last_proxy:.2e}, mp_max={max(hyb_mp.values()):.2e}")


if __name__ == "__main__":
    main()
