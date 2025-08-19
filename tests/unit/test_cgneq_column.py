#!/usr/bin/env python3
"""
Quick test for CGNE–Q (column variant): solves XA = I_n for m >= n
"""

import os
import sys
import time

import numpy as np
import quaternion

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from quatica.solver import CGNEQSolver
from quatica.utils import quat_eye, quat_frobenius_norm, quat_matmat


def random_quat_matrix(m: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A_real = rng.standard_normal((m, n))
    A_i = rng.standard_normal((m, n))
    A_j = rng.standard_normal((m, n))
    A_k = rng.standard_normal((m, n))
    return quaternion.as_quat_array(np.stack([A_real, A_i, A_j, A_k], axis=-1))


def main():
    m, n = 60, 30  # tall matrix (full column rank generically)
    tol = 1e-3
    maxit = 300
    seed = 42

    print(f"Testing CGNE–Q on {m}x{n} (column variant: XA=I_n)")
    A = random_quat_matrix(m, n, seed)

    solver = CGNEQSolver(tol=tol, max_iter=maxit, verbose=True, preconditioner_rank=0, seed=seed)
    t0 = time.time()
    X, info = solver.compute(A)
    t_cg = time.time() - t0

    # Check residuals
    XA = quat_matmat(X, A)
    I_n = quat_eye(n)
    rel_res = quat_frobenius_norm(XA - I_n) / max(quat_frobenius_norm(I_n), 1e-30)

    print(
        {
            'iters': info['iterations'],
            'proxy_last': info['residual_norms'][-1] if info['residual_norms'] else None,
            'rel_res_XA_I': float(rel_res),
            'time_s': t_cg,
        }
    )


if __name__ == "__main__":
    main()
