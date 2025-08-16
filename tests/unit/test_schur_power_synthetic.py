#!/usr/bin/env python3
"""
Transversal validation: Schur vs Power-Iteration on synthetic diagonalizable matrix.
- Build A = P S P^H with P a complex unitary (embedded in x-axis subfield)
  and S diagonal with complex entries (j=k=0).
- Run quaternion Schur (rayleigh) and extract a complex eigenvalue from diag(T).
- Run power_iteration_nonhermitian and compare eigenvalues (up to conjugation) and residual.
"""

import os
import sys

import numpy as np
import pytest
import quaternion  # type: ignore

# Robust path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "quatica"))
from decomp.schur import quaternion_schur_unified
from utils import power_iteration_nonhermitian, quat_hermitian, quat_matmat


def complex_to_quaternion_matrix(C: np.ndarray) -> np.ndarray:
    m, n = C.shape
    Q = np.empty((m, n), dtype=np.quaternion)
    for i in range(m):
        for j in range(n):
            a = float(np.real(C[i, j]))
            b = float(np.imag(C[i, j]))
            Q[i, j] = quaternion.quaternion(a, b, 0.0, 0.0)
    return Q


def random_complex_unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    X = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Qc, _ = np.linalg.qr(X)
    return Qc


def build_diagonal_complex_quat(values: np.ndarray) -> np.ndarray:
    n = values.shape[0]
    S = np.zeros((n, n), dtype=np.quaternion)
    for i, lam in enumerate(values):
        S[i, i] = quaternion.quaternion(
            float(np.real(lam)), float(np.imag(lam)), 0.0, 0.0
        )
    return S


@pytest.mark.parametrize("n,seed", [(16, 7), (12, 11)])
def test_schur_power_synthetic_compare(n: int, seed: int):
    rng = np.random.default_rng(seed)
    Uc = random_complex_unitary(n, rng)
    P = complex_to_quaternion_matrix(Uc)
    vals = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    S = build_diagonal_complex_quat(vals)
    A = quat_matmat(quat_matmat(P, S), quat_hermitian(P))

    # Schur (rayleigh variant)
    Q, T, _ = quaternion_schur_unified(
        A, variant="rayleigh", max_iter=2000, tol=1e-10, return_diagnostics=True
    )
    Tdiag = [complex(T[i, i].w, T[i, i].x) for i in range(n)]
    lam_schur = Tdiag[int(np.argmax([abs(z) for z in Tdiag]))]

    # Power iteration (non-Hermitian)
    _qv, lam_pi, res = power_iteration_nonhermitian(
        A, max_iterations=8000, eig_tol=1e-14, res_tol=1e-12, seed=1, return_vector=True
    )

    # Compare eigenvalues up to conjugation
    rel_err = min(abs(lam_pi - lam_schur), abs(lam_pi - np.conjugate(lam_schur))) / (
        max(1e-12, abs(lam_schur))
    )

    # Assertions
    assert rel_err < 1e-10
    assert res and res[-1] <= 5e-7
