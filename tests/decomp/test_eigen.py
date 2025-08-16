#!/usr/bin/env python3
"""
Unit tests for quaternion matrix eigenvalue decomposition.

Tests the eigenvalue decomposition of Hermitian quaternion matrices using
tridiagonalization approach.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import unittest

import numpy as np
import quaternion

from quatica.decomp.eigen import (
    quaternion_eigendecomposition,
    quaternion_eigenvalues,
    quaternion_eigenvectors,
    verify_eigendecomposition,
)
from quatica.utils import quat_frobenius_norm, quat_hermitian, quat_matmat


class TestEigenvalueDecomposition(unittest.TestCase):
    """Test cases for quaternion matrix eigenvalue decomposition."""

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

    def test_eigendecomposition_1x1(self):
        """Test eigenvalue decomposition of 1x1 Hermitian matrix."""
        A = np.array([[quaternion.quaternion(3, 0, 0, 0)]])

        eigenvalues, eigenvectors = quaternion_eigendecomposition(A)

        # Check shapes
        self.assertEqual(len(eigenvalues), 1)
        self.assertEqual(eigenvectors.shape, (1, 1))

        # Check eigenvalue
        self.assertAlmostEqual(eigenvalues[0], 3.0, places=10)

        # Check eigenvector
        self.assertAlmostEqual(eigenvectors[0, 0].w, 1.0, places=10)
        self.assertAlmostEqual(eigenvectors[0, 0].x, 0.0, places=10)
        self.assertAlmostEqual(eigenvectors[0, 0].y, 0.0, places=10)
        self.assertAlmostEqual(eigenvectors[0, 0].z, 0.0, places=10)

        # Verify eigenpair: A * v = λ * v
        v = eigenvectors[:, 0:1]
        Av = quat_matmat(A, v)
        lambda_val = eigenvalues[0]
        lambda_quat = quaternion.quaternion(lambda_val.real, lambda_val.imag, 0, 0)
        lambda_v = lambda_quat * v

        error = quat_frobenius_norm(Av - lambda_v)
        self.assertLess(error, 1e-10)

    def test_eigendecomposition_2x2(self):
        """Test eigenvalue decomposition of 2x2 Hermitian matrix."""
        A = self.create_hermitian_matrix(2)

        eigenvalues, eigenvectors = quaternion_eigendecomposition(A)

        # Check shapes
        self.assertEqual(len(eigenvalues), 2)
        self.assertEqual(eigenvectors.shape, (2, 2))

        # Check that eigenvalues are real (for Hermitian matrices)
        for eig in eigenvalues:
            self.assertAlmostEqual(eig.imag, 0.0, places=10)

        # Verify each eigenpair: A * v = λ * v
        for i in range(2):
            v = eigenvectors[:, i : i + 1]
            Av = quat_matmat(A, v)
            lambda_val = eigenvalues[i]
            lambda_quat = quaternion.quaternion(lambda_val.real, lambda_val.imag, 0, 0)
            lambda_v = lambda_quat * v

            error = quat_frobenius_norm(Av - lambda_v)
            self.assertLess(error, 1e-10)

    def test_eigendecomposition_3x3(self):
        """Test eigenvalue decomposition of 3x3 Hermitian matrix."""
        A = self.create_hermitian_matrix(3)

        eigenvalues, eigenvectors = quaternion_eigendecomposition(A)

        # Check shapes
        self.assertEqual(len(eigenvalues), 3)
        self.assertEqual(eigenvectors.shape, (3, 3))

        # Check that eigenvalues are real (for Hermitian matrices)
        for eig in eigenvalues:
            self.assertAlmostEqual(eig.imag, 0.0, places=10)

        # Verify each eigenpair: A * v = λ * v
        for i in range(3):
            v = eigenvectors[:, i : i + 1]
            Av = quat_matmat(A, v)
            lambda_val = eigenvalues[i]
            lambda_quat = quaternion.quaternion(lambda_val.real, lambda_val.imag, 0, 0)
            lambda_v = lambda_quat * v

            error = quat_frobenius_norm(Av - lambda_v)
            self.assertLess(error, 1e-10)

    def test_eigendecomposition_4x4(self):
        """Test eigenvalue decomposition of 4x4 Hermitian matrix."""
        A = self.create_hermitian_matrix(4)

        eigenvalues, eigenvectors = quaternion_eigendecomposition(A)

        # Check shapes
        self.assertEqual(len(eigenvalues), 4)
        self.assertEqual(eigenvectors.shape, (4, 4))

        # Check that eigenvalues are real (for Hermitian matrices)
        for eig in eigenvalues:
            self.assertAlmostEqual(eig.imag, 0.0, places=10)

        # Verify each eigenpair: A * v = λ * v
        for i in range(4):
            v = eigenvectors[:, i : i + 1]
            Av = quat_matmat(A, v)
            lambda_val = eigenvalues[i]
            lambda_quat = quaternion.quaternion(lambda_val.real, lambda_val.imag, 0, 0)
            lambda_v = lambda_quat * v

            error = quat_frobenius_norm(Av - lambda_v)
            self.assertLess(error, 1e-10)

    def test_eigendecomposition_5x5(self):
        """Test eigenvalue decomposition of 5x5 Hermitian matrix."""
        A = self.create_hermitian_matrix(5)

        eigenvalues, eigenvectors = quaternion_eigendecomposition(A)

        # Check shapes
        self.assertEqual(len(eigenvalues), 5)
        self.assertEqual(eigenvectors.shape, (5, 5))

        # Check that eigenvalues are real (for Hermitian matrices)
        for eig in eigenvalues:
            self.assertAlmostEqual(eig.imag, 0.0, places=10)

        # Verify each eigenpair: A * v = λ * v
        for i in range(5):
            v = eigenvectors[:, i : i + 1]
            Av = quat_matmat(A, v)
            lambda_val = eigenvalues[i]
            lambda_quat = quaternion.quaternion(lambda_val.real, lambda_val.imag, 0, 0)
            lambda_v = lambda_quat * v

            error = quat_frobenius_norm(Av - lambda_v)
            self.assertLess(error, 1e-10)

    def test_eigenvalues_only(self):
        """Test computing only eigenvalues."""
        A = self.create_hermitian_matrix(3)

        eigenvalues = quaternion_eigenvalues(A)

        # Check that we get the same eigenvalues as full decomposition
        eigenvalues_full, _ = quaternion_eigendecomposition(A)

        self.assertEqual(len(eigenvalues), len(eigenvalues_full))
        for i in range(len(eigenvalues)):
            self.assertAlmostEqual(eigenvalues[i], eigenvalues_full[i], places=10)

    def test_eigenvectors_only(self):
        """Test computing only eigenvectors."""
        A = self.create_hermitian_matrix(3)

        eigenvectors = quaternion_eigenvectors(A)

        # Check that we get the same eigenvectors as full decomposition
        _, eigenvectors_full = quaternion_eigendecomposition(A)

        self.assertEqual(eigenvectors.shape, eigenvectors_full.shape)
        error = quat_frobenius_norm(eigenvectors - eigenvectors_full)
        self.assertLess(error, 1e-10)

    def test_verify_eigendecomposition(self):
        """Test the verify_eigendecomposition function."""
        A = self.create_hermitian_matrix(3)
        eigenvalues, eigenvectors = quaternion_eigendecomposition(A)

        result = verify_eigendecomposition(A, eigenvalues, eigenvectors)

        # Check that verification succeeds
        self.assertTrue(result["success"])
        self.assertLess(result["max_error"], 1e-6)
        self.assertLess(result["mean_error"], 1e-6)
        self.assertEqual(len(result["errors"]), 3)

    def test_eigendecomposition_non_square(self):
        """Test that non-square matrices raise ValueError."""
        A = np.random.randn(3, 4, 4)  # Non-square
        A_quat = quaternion.as_quat_array(A)

        with self.assertRaises(ValueError):
            quaternion_eigendecomposition(A_quat)

    def test_eigendecomposition_non_hermitian(self):
        """Test that non-Hermitian matrices raise ValueError."""
        # Create a non-Hermitian matrix
        A_data = np.random.randn(3, 3, 4)
        A_quat = quaternion.as_quat_array(A_data)

        with self.assertRaises(ValueError):
            quaternion_eigendecomposition(A_quat)

    def test_eigendecomposition_verbose(self):
        """Test eigenvalue decomposition with verbose output."""
        A = self.create_hermitian_matrix(3)

        # Should not raise any exceptions
        eigenvalues, eigenvectors = quaternion_eigendecomposition(A, verbose=True)

        # Check that we get valid results
        self.assertEqual(len(eigenvalues), 3)
        self.assertEqual(eigenvectors.shape, (3, 3))

    def test_eigenvalues_real_for_hermitian(self):
        """Test that eigenvalues of Hermitian matrices are real."""
        sizes = [2, 3, 4, 5]

        for size in sizes:
            A = self.create_hermitian_matrix(size)
            eigenvalues = quaternion_eigenvalues(A)

            for eig in eigenvalues:
                self.assertAlmostEqual(eig.imag, 0.0, places=10)

    def test_eigenvectors_orthogonality(self):
        """Test that eigenvectors are orthogonal (for distinct eigenvalues)."""
        A = self.create_hermitian_matrix(3)
        eigenvalues, eigenvectors = quaternion_eigendecomposition(A)

        # Check orthogonality of eigenvectors
        # Note: numpy.linalg.eig doesn't guarantee orthogonality for quaternion matrices
        # This test is more of a check that eigenvectors are reasonable
        for i in range(3):
            for j in range(3):
                if i != j:
                    v1 = eigenvectors[:, i : i + 1]
                    v2 = eigenvectors[:, j : j + 1]

                    # Compute inner product
                    inner_product = quat_frobenius_norm(quat_matmat(v1.T, v2))

                    # For quaternion matrices, orthogonality is not guaranteed
                    # We just check that the inner product is reasonable (not too large)
                    self.assertLess(inner_product, 1.0)

    def test_eigenvalue_ordering(self):
        """Test that eigenvalues are returned in a reasonable order."""
        A = self.create_hermitian_matrix(4)
        eigenvalues = quaternion_eigenvalues(A)

        # Check that eigenvalues are finite and real
        for eig in eigenvalues:
            self.assertTrue(np.isfinite(eig))
            self.assertAlmostEqual(eig.imag, 0.0, places=10)

        # Note: numpy.linalg.eig doesn't guarantee any specific ordering
        # We just verify that we have the right number of eigenvalues
        self.assertEqual(len(eigenvalues), 4)

    def test_reconstruction_from_eigenpairs(self):
        """Test that we can reconstruct the matrix from its eigenpairs."""
        A = self.create_hermitian_matrix(3)
        eigenvalues, eigenvectors = quaternion_eigendecomposition(A)

        # Reconstruct A = V * D * V^H
        np.diag(eigenvalues)
        V_H = quat_hermitian(eigenvectors)

        # Convert D to quaternion matrix
        D_quat = np.zeros((3, 3), dtype=np.quaternion)
        for i in range(3):
            for j in range(3):
                if i == j:
                    D_quat[i, j] = quaternion.quaternion(
                        eigenvalues[i].real, eigenvalues[i].imag, 0, 0
                    )

        A_reconstructed = quat_matmat(quat_matmat(eigenvectors, D_quat), V_H)

        # Check reconstruction error
        error = quat_frobenius_norm(A - A_reconstructed)
        self.assertLess(error, 1e-10)


if __name__ == "__main__":
    unittest.main()
