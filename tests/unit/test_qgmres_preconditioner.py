#!/usr/bin/env python3
"""
Unit test: Q-GMRES with and without LU preconditioning (left).
Compares iterations for a random quaternion system; prints residuals for reference.
"""

import os
import sys
import numpy as np
import quaternion  # type: ignore
import pytest

# Path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core'))
from utils import quat_matmat, quat_hermitian
from data_gen import create_test_matrix
from solver import QGMRESSolver

@pytest.mark.parametrize("n,seed", [(20, 0), (30, 1)])
def test_qgmres_lu_preconditioner(n: int, seed: int):
    np.random.seed(seed)
    # Construct a moderately ill-conditioned A = B^H B + alpha*I
    B = create_test_matrix(n, n)
    A = quat_matmat(quat_hermitian(B), B)
    # Add small identity to improve diagonals
    I = np.zeros((n, n), dtype=np.quaternion)
    for i in range(n):
        I[i, i] = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
    A = A + 1e-3 * I
    # Right-hand side
    b = create_test_matrix(n, 1)

    # Baseline GMRES
    gmres = QGMRESSolver(tol=1e-6, max_iter=2*n, verbose=False, preconditioner='none')
    x0, info0 = gmres.solve(A, b)

    # LU-preconditioned GMRES (left, L only)
    gmres_lu = QGMRESSolver(tol=1e-6, max_iter=2*n, verbose=False, preconditioner='left_lu')
    x1, info1 = gmres_lu.solve(A, b)

    # Assertions: preconditioner should not be worse than baseline in iterations
    assert info1['iterations'] <= info0['iterations']

    # Log residuals (true residuals if available)
    r0 = info0.get('residual_true', info0['residual'])
    r1 = info1.get('residual_true', info1['residual'])
    print(f"n={n} seed={seed} | iters: none={info0['iterations']} lu={info1['iterations']} | res: none={r0:.3e} lu={r1:.3e}")
