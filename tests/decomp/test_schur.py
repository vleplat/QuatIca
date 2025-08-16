#!/usr/bin/env python3
"""
Unit tests for quaternion Schur decomposition.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import unittest

import numpy as np
import quaternion  # type: ignore

from quatica.decomp.schur import quaternion_schur
from quatica.utils import quat_frobenius_norm, quat_hermitian, quat_matmat


def random_quaternion_matrix(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, n, 4))
    return quaternion.as_quat_array(data)


class TestSchur(unittest.TestCase):
    def test_small_identity(self):
        A = np.eye(3, dtype=np.quaternion)
        Q, T = quaternion_schur(A, max_iter=50, tol=1e-12)
        self.assertTrue(np.allclose(T, A))
        I = quat_matmat(quat_hermitian(Q), Q)
        self.assertTrue(np.allclose(I, np.eye(3, dtype=np.quaternion), atol=1e-10))

    def test_random_matrix_properties(self):
        A = random_quaternion_matrix(5, seed=7)
        Q, T = quaternion_schur(A, max_iter=200, tol=1e-10)
        # Q unitary
        QT = quat_hermitian(Q)
        I = quat_matmat(QT, Q)
        self.assertTrue(np.allclose(I, np.eye(5, dtype=np.quaternion), atol=1e-8))
        # T should be almost upper-triangular (small subdiagonal)
        subdiag = []
        for i in range(1, T.shape[0]):
            subdiag.append(
                np.linalg.norm(
                    [T[i, i - 1].w, T[i, i - 1].x, T[i, i - 1].y, T[i, i - 1].z]
                )
            )
        self.assertLess(max(subdiag), 1e-6)

        # Temporarily removing larger stress test until QR convergence improvements
        # Similarity: A ≈ Q T Q^H
        A_recon = quat_matmat(quat_matmat(Q, T), QT)
        err = quat_frobenius_norm(A - A_recon) / (1e-12 + quat_frobenius_norm(A))
        self.assertLess(err, 1e-6)


if __name__ == "__main__":
    unittest.main()
