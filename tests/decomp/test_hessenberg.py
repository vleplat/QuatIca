#!/usr/bin/env python3
"""
Unit tests for quaternion Hessenberg reduction.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import unittest
import numpy as np
import quaternion  # type: ignore

from core.decomp.hessenberg import hessenbergize, is_hessenberg
from core.utils import quat_matmat, quat_frobenius_norm, quat_hermitian, real_expand, quat_eye


def random_quaternion_matrix(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, n, 4))
    return quaternion.as_quat_array(data)


class TestHessenberg(unittest.TestCase):
    def assert_hessenberg_pattern(self, H: np.ndarray, atol: float = 1e-12):
        m, n = H.shape
        for i in range(m):
            for j in range(n):
                if i > j + 1:
                    q = H[i, j]
                    self.assertLessEqual(np.linalg.norm([q.w, q.x, q.y, q.z]), atol)

    def assert_block_hessenberg_real(self, H: np.ndarray, atol: float = 1e-10):
        """Verify the real-expanded matrix is block upper-Hessenberg (4x4 blocks)."""
        m, n = H.shape
        R = real_expand(H)
        for i in range(m):
            for j in range(n):
                if i > j + 1:
                    bi, bj = 4 * i, 4 * j
                    block = R[bi:bi+4, bj:bj+4]
                    self.assertLessEqual(np.linalg.norm(block, ord='fro'), atol)

    def test_small_identities(self):
        for n in [1, 2]:
            A = np.eye(n, dtype=np.quaternion)
            P, H = hessenbergize(A)
            self.assertTrue(np.allclose(P, np.eye(n, dtype=np.quaternion)))
            self.assertTrue(np.allclose(H, A))
            self.assertTrue(is_hessenberg(H))
            self.assert_hessenberg_pattern(H)
            self.assert_block_hessenberg_real(H)

    def test_random_3x3(self):
        A = random_quaternion_matrix(3, seed=42)
        P, H = hessenbergize(A)
        self.assertTrue(is_hessenberg(H))
        self.assert_hessenberg_pattern(H)
        self.assert_block_hessenberg_real(H)
        # Verify similarity transform: H ≈ P A P^H
        PA = quat_matmat(P, A)
        PH = quat_hermitian(P)
        recon = quat_matmat(PA, PH)
        err = quat_frobenius_norm(recon - H)
        self.assertLess(err, 1e-10)

    def test_random_5x5(self):
        A = random_quaternion_matrix(5, seed=123)
        P, H = hessenbergize(A)
        self.assertTrue(is_hessenberg(H))
        self.assert_hessenberg_pattern(H)
        self.assert_block_hessenberg_real(H)
        # Verify similarity transform
        PA = quat_matmat(P, A)
        PH = quat_hermitian(P)
        recon = quat_matmat(PA, PH)
        err = quat_frobenius_norm(recon - H)
        self.assertLess(err, 1e-10)

    def test_unitarity_of_P(self):
        A = random_quaternion_matrix(6, seed=7)
        P, H = hessenbergize(A)
        PH = quat_hermitian(P)
        I = quat_matmat(PH, P)
        self.assertTrue(np.allclose(I, quat_eye(P.shape[0]), atol=1e-10))
        self.assertTrue(is_hessenberg(H))

    def test_idempotence(self):
        A = random_quaternion_matrix(6, seed=11)
        P1, H1 = hessenbergize(A)
        P2, H2 = hessenbergize(H1)
        # Both should be Hessenberg
        self.assertTrue(is_hessenberg(H1))
        self.assertTrue(is_hessenberg(H2))
        self.assert_hessenberg_pattern(H1)
        self.assert_hessenberg_pattern(H2)
        # Similarity relation must hold by construction
        recon2 = quat_matmat(quat_matmat(P2, H1), quat_hermitian(P2))
        self.assertLess(quat_frobenius_norm(recon2 - H2), 1e-10)

    def test_constructed_matrix(self):
        # Create a 6x6 matrix with deliberate non-zeros well below the subdiagonal
        n = 6
        A = random_quaternion_matrix(n, seed=21)
        # Inject larger values to stress elimination
        for i in range(n):
            for j in range(n):
                if i >= j + 2:
                    A[i, j] = quaternion.quaternion(0.5*(i-j), 0.1, -0.2, 0.3)
        P, H = hessenbergize(A)
        self.assertTrue(is_hessenberg(H))
        self.assert_hessenberg_pattern(H, atol=1e-11)
        self.assert_block_hessenberg_real(H, atol=1e-9)


if __name__ == '__main__':
    unittest.main()


