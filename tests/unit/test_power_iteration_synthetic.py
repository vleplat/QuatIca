#!/usr/bin/env python3
"""
Synthetic test for non-Hermitian power iteration on a diagonalizable quaternion matrix:
- Build P unitary (complex unitary embedded in the x-axis subfield) of size n
- Build diagonal S with complex entries in the x-axis subfield (j=k=0)
- Form A = P @ S @ P^H
- Verify power_iteration_nonhermitian recovers one of the eigenvalues and has a small adjoint residual
"""

import os
import sys
import numpy as np
import quaternion  # type: ignore
import pytest

# Robust import path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core'))
from utils import (
    quat_matmat,
    quat_hermitian,
    power_iteration_nonhermitian,
)

def complex_to_quaternion_matrix(C: np.ndarray) -> np.ndarray:
    m, n = C.shape
    Q = np.empty((m, n), dtype=np.quaternion)
    for i in range(m):
        for j in range(n):
            a = float(np.real(C[i, j]))
            b = float(np.imag(C[i, j]))
            Q[i, j] = quaternion.quaternion(a, b, 0.0, 0.0)
    return Q


def build_diagonal_complex_quat(values: np.ndarray) -> np.ndarray:
    n = values.shape[0]
    S = np.zeros((n, n), dtype=np.quaternion)
    for i, lam in enumerate(values):
        S[i, i] = quaternion.quaternion(float(np.real(lam)), float(np.imag(lam)), 0.0, 0.0)
    return S


def random_complex_unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    X = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(X)
    return Q

@pytest.mark.parametrize("n,seed", [(8, 0), (12, 1)])
def test_power_iteration_synthetic_unitary_similarity(n: int, seed: int):
    rng = np.random.default_rng(seed)
    # Complex unitary embedded as quaternion (x-axis subfield)
    Uc = random_complex_unitary(n, rng)
    P = complex_to_quaternion_matrix(Uc)
    # Complex eigenvalues in x-axis subfield
    vals = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    S = build_diagonal_complex_quat(vals)
    # A = P S P^H
    A = quat_matmat(quat_matmat(P, S), quat_hermitian(P))

    # Run non-Hermitian power iteration
    q_vec, lam_c, residuals = power_iteration_nonhermitian(
        A, max_iterations=8000, eig_tol=1e-14, res_tol=1e-12, seed=seed, return_vector=True
    )

    # Metric: eigenvalue close to spectrum (allowing conjugate)
    dists = [abs(lam_c - ev) for ev in vals] + [abs(lam_c - np.conjugate(ev)) for ev in vals]
    min_dist = min(dists)
    scale = max(1e-12, max(abs(ev) for ev in vals))
    assert min_dist / scale < 1e-10

    # Residual small (adjoint residual)
    assert residuals and residuals[-1] <= 5e-7
