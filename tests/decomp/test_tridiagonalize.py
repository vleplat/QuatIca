#!/usr/bin/env python3
"""
Unit tests for quaternion matrix tridiagonalization.

Tests the tridiagonalization of Hermitian quaternion matrices using
Householder transformations.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import unittest

import numpy as np
import quaternion

from quatica.decomp.tridiagonalize import (
    check_tridiagonal,
    householder_matrix,
    householder_vector,
    internal_tridiagonalizer,
    tridiagonalize,
)
from quatica.utils import quat_frobenius_norm, quat_hermitian, quat_matmat


class TestTridiagonalization(unittest.TestCase):
    """Test cases for quaternion matrix tridiagonalization."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)  # For reproducible tests

    def create_hermitian_matrix(self, size):
        """Create a random Hermitian quaternion matrix."""
        A_data = np.random.randn(size, size, 4)
        A_quat = quaternion.as_quat_array(A_data)

        # Make it Hermitian: A = (A + A^H) / 2
        A_hermitian = quat_hermitian(A_quat)
        A_symmetric = (A_quat + A_hermitian) / 2.0

        return A_symmetric

    def test_householder_vector_basic(self):
        """Test basic Householder vector computation."""
        # Test with simple vectors
        a = np.array([quaternion.quaternion(1, 2, 3, 4)])
        v = np.array([1.0])

        u, zeta = householder_vector(a, v)

        # Check that u has norm sqrt(2)
        u_norm = quat_frobenius_norm(u)
        self.assertAlmostEqual(u_norm, np.sqrt(2), places=10)

        # Check that zeta is a quaternion
        self.assertIsInstance(zeta, np.quaternion)

    def test_householder_vector_zero_input(self):
        """Test Householder vector with zero input."""
        a = np.array([quaternion.quaternion(0, 0, 0, 0)])
        v = np.array([0.0])

        u, zeta = householder_vector(a, v)

        # Should return zero vector and zeta = 1
        self.assertAlmostEqual(quat_frobenius_norm(u), 0, places=10)
        self.assertAlmostEqual(zeta.w, 1.0, places=10)

    def test_householder_matrix_basic(self):
        """Test basic Householder matrix computation."""
        a = np.array([quaternion.quaternion(1, 2, 3, 4)])
        v = np.array([1.0])

        H = householder_matrix(a, v)

        # Check that H is square
        self.assertEqual(H.shape, (1, 1))

        # Check that H is unitary (H * H^H = I)
        H_H = quat_hermitian(H)
        HH_H = quat_matmat(H, H_H)
        I = np.eye(1, dtype=np.quaternion)

        error = quat_frobenius_norm(HH_H - I)
        self.assertLess(error, 1e-10)

    def test_householder_matrix_zero_norm(self):
        """Test Householder matrix with zero norm vector."""
        a = np.array([quaternion.quaternion(1, 2, 3, 4)])
        v = np.array([0.0])  # Zero norm

        H = householder_matrix(a, v)

        # Should return identity matrix
        I = np.eye(1, dtype=np.quaternion)
        error = quat_frobenius_norm(H - I)
        self.assertLess(error, 1e-10)

    def test_tridiagonalize_2x2(self):
        """Test tridiagonalization of 2x2 Hermitian matrix."""
        A = self.create_hermitian_matrix(2)

        P, B = tridiagonalize(A)

        # Check shapes
        self.assertEqual(P.shape, (2, 2))
        self.assertEqual(B.shape, (2, 2))

        # Check that P is unitary
        P_H = quat_hermitian(P)
        PP_H = quat_matmat(P, P_H)
        I = np.eye(2, dtype=np.quaternion)
        unitarity_error = quat_frobenius_norm(PP_H - I)
        self.assertLess(unitarity_error, 1e-10)

        # Check transformation: P * A * P^H = B
        PAP_H = quat_matmat(quat_matmat(P, A), P_H)
        transformation_error = quat_frobenius_norm(PAP_H - B)
        self.assertLess(transformation_error, 1e-10)

        # Check that B is tridiagonal
        for i in range(2):
            for j in range(2):
                if abs(i - j) > 1:  # Off-tridiagonal elements
                    element = B[i, j]
                    magnitude = np.sqrt(
                        element.w**2 + element.x**2 + element.y**2 + element.z**2
                    )
                    self.assertLess(magnitude, 1e-10)

    def test_tridiagonalize_3x3(self):
        """Test tridiagonalization of 3x3 Hermitian matrix."""
        A = self.create_hermitian_matrix(3)

        P, B = tridiagonalize(A)

        # Check shapes
        self.assertEqual(P.shape, (3, 3))
        self.assertEqual(B.shape, (3, 3))

        # Check that P is unitary
        P_H = quat_hermitian(P)
        PP_H = quat_matmat(P, P_H)
        I = np.eye(3, dtype=np.quaternion)
        unitarity_error = quat_frobenius_norm(PP_H - I)
        self.assertLess(unitarity_error, 1e-10)

        # Check transformation: P * A * P^H = B
        PAP_H = quat_matmat(quat_matmat(P, A), P_H)
        transformation_error = quat_frobenius_norm(PAP_H - B)
        self.assertLess(transformation_error, 1e-10)

        # Check that B is tridiagonal
        for i in range(3):
            for j in range(3):
                if abs(i - j) > 1:  # Off-tridiagonal elements
                    element = B[i, j]
                    magnitude = np.sqrt(
                        element.w**2 + element.x**2 + element.y**2 + element.z**2
                    )
                    self.assertLess(magnitude, 1e-10)

    def test_tridiagonalize_4x4(self):
        """Test tridiagonalization of 4x4 Hermitian matrix."""
        A = self.create_hermitian_matrix(4)

        P, B = tridiagonalize(A)

        # Check shapes
        self.assertEqual(P.shape, (4, 4))
        self.assertEqual(B.shape, (4, 4))

        # Check that P is unitary
        P_H = quat_hermitian(P)
        PP_H = quat_matmat(P, P_H)
        I = np.eye(4, dtype=np.quaternion)
        unitarity_error = quat_frobenius_norm(PP_H - I)
        self.assertLess(unitarity_error, 1e-10)

        # Check transformation: P * A * P^H = B
        PAP_H = quat_matmat(quat_matmat(P, A), P_H)
        transformation_error = quat_frobenius_norm(PAP_H - B)
        self.assertLess(transformation_error, 1e-10)

        # Check that B is tridiagonal
        for i in range(4):
            for j in range(4):
                if abs(i - j) > 1:  # Off-tridiagonal elements
                    element = B[i, j]
                    magnitude = np.sqrt(
                        element.w**2 + element.x**2 + element.y**2 + element.z**2
                    )
                    self.assertLess(magnitude, 1e-10)

    def test_tridiagonalize_5x5(self):
        """Test tridiagonalization of 5x5 Hermitian matrix."""
        A = self.create_hermitian_matrix(5)

        P, B = tridiagonalize(A)

        # Check shapes
        self.assertEqual(P.shape, (5, 5))
        self.assertEqual(B.shape, (5, 5))

        # Check that P is unitary
        P_H = quat_hermitian(P)
        PP_H = quat_matmat(P, P_H)
        I = np.eye(5, dtype=np.quaternion)
        unitarity_error = quat_frobenius_norm(PP_H - I)
        self.assertLess(unitarity_error, 1e-10)

        # Check transformation: P * A * P^H = B
        PAP_H = quat_matmat(quat_matmat(P, A), P_H)
        transformation_error = quat_frobenius_norm(PAP_H - B)
        self.assertLess(transformation_error, 1e-10)

        # Check that B is tridiagonal
        for i in range(5):
            for j in range(5):
                if abs(i - j) > 1:  # Off-tridiagonal elements
                    element = B[i, j]
                    magnitude = np.sqrt(
                        element.w**2 + element.x**2 + element.y**2 + element.z**2
                    )
                    self.assertLess(magnitude, 1e-10)

    def test_tridiagonalize_non_square(self):
        """Test that non-square matrices raise ValueError."""
        A = np.random.randn(3, 4, 4)  # Non-square
        A_quat = quaternion.as_quat_array(A)

        with self.assertRaises(ValueError):
            tridiagonalize(A_quat)

    def test_tridiagonalize_too_small(self):
        """Test that matrices smaller than 2x2 raise ValueError."""
        A = np.array([[quaternion.quaternion(1, 0, 0, 0)]])  # 1x1

        with self.assertRaises(ValueError):
            tridiagonalize(A)

    def test_tridiagonalize_non_hermitian(self):
        """Test that non-Hermitian matrices raise ValueError."""
        # Create a non-Hermitian matrix
        A_data = np.random.randn(3, 3, 4)
        A_quat = quaternion.as_quat_array(A_data)

        with self.assertRaises(ValueError):
            tridiagonalize(A_quat)

    def test_check_tridiagonal(self):
        """Test the check_tridiagonal function."""
        # Create a tridiagonal matrix
        B = np.zeros((3, 3), dtype=np.quaternion)
        B[0, 0] = quaternion.quaternion(1, 0, 0, 0)
        B[0, 1] = quaternion.quaternion(2, 0, 0, 0)
        B[1, 0] = quaternion.quaternion(2, 0, 0, 0)
        B[1, 1] = quaternion.quaternion(3, 0, 0, 0)
        B[1, 2] = quaternion.quaternion(4, 0, 0, 0)
        B[2, 1] = quaternion.quaternion(4, 0, 0, 0)
        B[2, 2] = quaternion.quaternion(5, 0, 0, 0)

        B_clean = check_tridiagonal(B)

        # Check that B_clean is the same as B (since B is already tridiagonal)
        error = quat_frobenius_norm(B_clean - B)
        self.assertLess(error, 1e-10)

    def test_internal_tridiagonalizer(self):
        """Test the internal_tridiagonalizer function."""
        A = self.create_hermitian_matrix(3)

        P, B = internal_tridiagonalizer(A)

        # Check shapes
        self.assertEqual(P.shape, (3, 3))
        self.assertEqual(B.shape, (3, 3))

        # Check that P is unitary
        P_H = quat_hermitian(P)
        PP_H = quat_matmat(P, P_H)
        I = np.eye(3, dtype=np.quaternion)
        unitarity_error = quat_frobenius_norm(PP_H - I)
        self.assertLess(unitarity_error, 1e-10)

        # Check transformation: P * A * P^H = B
        PAP_H = quat_matmat(quat_matmat(P, A), P_H)
        transformation_error = quat_frobenius_norm(PAP_H - B)
        self.assertLess(transformation_error, 1e-10)


if __name__ == "__main__":
    unittest.main()
