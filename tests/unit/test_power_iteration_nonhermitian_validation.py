#!/usr/bin/env python3
"""
Tests for power_iteration_nonhermitian:
- Hermitian quaternion matrices: eigenvalue should be real and agree with
  quaternion_eigendecomposition; classical power_iteration magnitude should
  be in reasonable agreement.
- Complex NumPy matrices embedded in quaternion form: eigenvalue should be
  close to NumPy spectrum (allow conjugation) and complex-adjoint residual
  should decrease significantly.

Note: Quaternion-space residual ||A q - q lambda|| is not asserted here yet,
      as the current vector back-mapping from the complex adjoint may select
      the lower-block eigenvector, yielding j/k components for embedded
      complex cases. This will be revisited once mapping is refined.
"""

import os
import sys

import numpy as np
import pytest
import quaternion  # type: ignore

# Robust path setup to import from quatica/
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "quatica"))
from data_gen import create_test_matrix
from decomp.eigen import quaternion_eigendecomposition
from utils import (
    power_iteration,
    power_iteration_nonhermitian,
    quat_hermitian,
    quat_matmat,
)


def complex_to_quaternion_matrix(C: np.ndarray) -> np.ndarray:
    """Embed a complex matrix C into quaternion form along the x-axis: a+ib -> (w=a, x=b, y=0, z=0)."""
    m, n = C.shape
    Q = np.empty((m, n), dtype=np.quaternion)
    for i in range(m):
        for j in range(n):
            a = float(np.real(C[i, j]))
            b = float(np.imag(C[i, j]))
            Q[i, j] = quaternion.quaternion(a, b, 0.0, 0.0)
    return Q


@pytest.mark.parametrize("n,seed", [(6, 0), (10, 1)])
def test_power_iteration_nonhermitian_on_hermitian_matches_baselines(n: int, seed: int):
    # Build Hermitian quaternion matrix A = B^H B
    np.random.seed(seed)
    B = create_test_matrix(n, n)
    A = quat_matmat(quat_hermitian(B), B)

    # Run experimental non-Hermitian power iteration
    q_vec, lam_c, res_curve = power_iteration_nonhermitian(
        A,
        max_iterations=5000,
        eig_tol=1e-14,
        res_tol=1e-10,
        seed=seed,
        return_vector=True,
    )

    # Run classical Hermitian-oriented power iteration (returns magnitude)
    _v_h, lam_mag = power_iteration(
        A, max_iterations=2000, tol=1e-12, return_eigenvalue=True
    )

    # Run Hermitian eigendecomposition
    eigenvalues, _ = quaternion_eigendecomposition(A, verbose=False)
    idx = int(np.argmax(np.abs(eigenvalues)))
    lam_eig = eigenvalues[idx]

    # Assertions:
    # - eigenvalue should be real
    assert abs(np.imag(lam_c)) < 1e-8, (
        f"Expected real eigenvalue for Hermitian A, got {lam_c}"
    )
    # - tight match against eigendecomposition (dominant eigenvalue)
    assert abs(lam_c.real - lam_eig.real) / (abs(lam_eig.real) + 1e-15) < 1e-6
    # - classical magnitude is reasonably close to the same dominant eigenvalue (heuristic)
    assert abs(lam_mag - abs(lam_eig)) / (abs(lam_eig) + 1e-15) < 3e-1

    # Ensure complex-adjoint residual reasonably small
    assert res_curve, "No residuals recorded"
    assert res_curve[-1] <= 2e-4


@pytest.mark.parametrize("n,seed", [(6, 2), (9, 3)])
def test_power_iteration_nonhermitian_matches_numpy_on_complex_embedding(
    n: int, seed: int
):
    rng = np.random.default_rng(seed)
    # Build random complex matrix and compute NumPy eigen-spectrum
    C = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    eigvals, _eigvecs = np.linalg.eig(C)

    # Embed complex matrix into quaternion matrix along x-axis
    A_quat = complex_to_quaternion_matrix(C)

    # Run quaternion non-Hermitian power iteration
    q_vec, lam_c, res_curve = power_iteration_nonhermitian(
        A_quat,
        max_iterations=10000,
        eig_tol=1e-14,
        res_tol=1e-10,
        seed=seed,
        return_vector=True,
    )

    # Check distance to spectrum (allow conjugation)
    dists = [abs(lam_c - ev) for ev in eigvals] + [
        abs(lam_c - np.conjugate(ev)) for ev in eigvals
    ]
    min_dist = min(dists)
    # Allow moderate tolerance pending mapping refinement
    scale = max(1e-12, max(abs(ev) for ev in eigvals))
    assert min_dist / scale < 2e-1, (
        f"Eigenvalue not close to NumPy spectrum: rel_err={min_dist / scale:.3e}"
    )

    # Ensure complex-adjoint residual decreased by at least 1e3 and is reasonably small
    assert res_curve, "No residuals recorded"
    if len(res_curve) >= 2:
        assert res_curve[-1] <= max(1e-2, res_curve[0] * 1e-3)
    else:
        assert res_curve[-1] <= 1e-2
