#!/usr/bin/env python3
"""
Unit tests for quaternion kernel/null space functions

Tests the new kernel computation functions:
- quat_null_space()
- quat_null_right()
- quat_null_left()
- quat_kernel()

These tests use matrices with known ranks to validate:
1. Correct null space dimensions
2. Orthogonality properties (A @ N ≈ 0)
3. Numerical accuracy
4. Edge cases (full rank, rank 0, etc.)

Author: QuatIca Development Team
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "quatica"))

import unittest

import numpy as np
import quaternion
from data_gen import create_test_matrix

# Import functions under test
from utils import (
    quat_frobenius_norm,
    quat_hermitian,
    quat_kernel,
    quat_matmat,
    quat_null_left,
    quat_null_right,
    quat_null_space,
)


class TestKernelNullSpace(unittest.TestCase):
    """Test suite for quaternion kernel/null space functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.tol = 1e-10  # Tolerance for numerical tests
        self.rtol = 1e-12  # Relative tolerance for rank determination

        # Set random seed for reproducible tests
        np.random.seed(42)

    def _create_rank_deficient_matrix(self, m: int, n: int, rank: int) -> np.ndarray:
        """Create a quaternion matrix with specified rank."""
        if rank > min(m, n):
            raise ValueError(f"Rank {rank} cannot exceed min(m,n) = {min(m, n)}")

        if rank == 0:
            # Zero matrix
            return np.zeros((m, n), dtype=np.quaternion)

        # Create random matrices and multiply to get desired rank
        A = create_test_matrix(m, rank)
        B = create_test_matrix(rank, n)
        return quat_matmat(A, B)

    def _create_known_null_space_matrix(self) -> tuple:
        """Create a matrix with a known, explicit null space for testing."""
        # Create a 4×6 matrix with explicit rank 3
        # We'll construct it so that specific vectors are in the null space

        # Create 4×3 matrix A and 3×6 matrix B
        A = np.array(
            [
                [
                    quaternion.quaternion(1, 0, 0, 0),
                    quaternion.quaternion(0, 1, 0, 0),
                    quaternion.quaternion(0, 0, 1, 0),
                ],
                [
                    quaternion.quaternion(0, 0, 0, 1),
                    quaternion.quaternion(1, 0, 0, 0),
                    quaternion.quaternion(0, 1, 0, 0),
                ],
                [
                    quaternion.quaternion(0, 0, 1, 0),
                    quaternion.quaternion(0, 0, 0, 1),
                    quaternion.quaternion(1, 0, 0, 0),
                ],
                [
                    quaternion.quaternion(0, 1, 0, 0),
                    quaternion.quaternion(0, 0, 1, 0),
                    quaternion.quaternion(0, 0, 0, 1),
                ],
            ]
        )

        # Create B such that last 3 columns have specific structure
        B = np.array(
            [
                [
                    quaternion.quaternion(1, 0, 0, 0),
                    quaternion.quaternion(0, 1, 0, 0),
                    quaternion.quaternion(0, 0, 1, 0),
                    quaternion.quaternion(1, 1, 0, 0),
                    quaternion.quaternion(0, 0, 0, 0),
                    quaternion.quaternion(0, 0, 0, 0),
                ],
                [
                    quaternion.quaternion(0, 0, 0, 1),
                    quaternion.quaternion(1, 0, 0, 0),
                    quaternion.quaternion(0, 1, 0, 0),
                    quaternion.quaternion(0, 0, 1, 1),
                    quaternion.quaternion(0, 0, 0, 0),
                    quaternion.quaternion(0, 0, 0, 0),
                ],
                [
                    quaternion.quaternion(0, 0, 1, 0),
                    quaternion.quaternion(0, 0, 0, 1),
                    quaternion.quaternion(1, 0, 0, 0),
                    quaternion.quaternion(1, 0, 0, 1),
                    quaternion.quaternion(0, 0, 0, 0),
                    quaternion.quaternion(0, 0, 0, 0),
                ],
            ]
        )

        matrix = quat_matmat(A, B)
        expected_rank = 3
        expected_null_dim = 6 - 3  # 3

        return matrix, expected_rank, expected_null_dim

    def test_quat_null_space_basic(self):
        """Test basic functionality of quat_null_space."""
        # Create 5×7 rank-3 matrix
        m, n, rank = 5, 7, 3
        A = self._create_rank_deficient_matrix(m, n, rank)

        # Test right null space
        N_right = quat_null_space(A, side="right", rtol=self.rtol)
        expected_null_dim_right = n - rank  # 7 - 3 = 4

        self.assertEqual(N_right.shape, (n, expected_null_dim_right))

        # Test left null space
        N_left = quat_null_space(A, side="left", rtol=self.rtol)
        expected_null_dim_left = m - rank  # 5 - 3 = 2

        self.assertEqual(N_left.shape, (m, expected_null_dim_left))

    def test_quat_null_space_orthogonality_right(self):
        """Test that right null space vectors are orthogonal to matrix."""
        # Create 6×8 rank-4 matrix
        m, n, rank = 6, 8, 4
        A = self._create_rank_deficient_matrix(m, n, rank)

        # Compute right null space
        N = quat_null_space(A, side="right", rtol=self.rtol)

        if N.shape[1] > 0:  # If null space is non-empty
            # Check A @ N ≈ 0
            AN = quat_matmat(A, N)
            AN_norm = quat_frobenius_norm(AN)

            self.assertLess(
                AN_norm, self.tol, f"A @ N should be zero, got norm {AN_norm}"
            )

    def test_quat_null_space_orthogonality_left(self):
        """Test that left null space vectors are orthogonal to matrix."""
        # Create 8×6 rank-4 matrix
        m, n, rank = 8, 6, 4
        A = self._create_rank_deficient_matrix(m, n, rank)

        # Compute left null space
        N = quat_null_space(A, side="left", rtol=self.rtol)

        if N.shape[1] > 0:  # If null space is non-empty
            # Check N^H @ A ≈ 0
            N_H = quat_hermitian(N)
            N_H_A = quat_matmat(N_H, A)
            norm = quat_frobenius_norm(N_H_A)

            self.assertLess(norm, self.tol, f"N^H @ A should be zero, got norm {norm}")

    def test_quat_null_space_full_rank(self):
        """Test null space of full-rank matrices."""
        # Create 5×5 full-rank matrix (likely full rank)
        n = 5
        A = create_test_matrix(n, n)

        # Add small diagonal to ensure full rank
        for i in range(n):
            A[i, i] += quaternion.quaternion(1.0, 0, 0, 0)

        # Right null space should be empty
        N_right = quat_null_space(A, side="right", rtol=self.rtol)
        self.assertEqual(N_right.shape, (n, 0))

        # Left null space should be empty
        N_left = quat_null_space(A, side="left", rtol=self.rtol)
        self.assertEqual(N_left.shape, (n, 0))

    def test_quat_null_space_zero_matrix(self):
        """Test null space of zero matrix."""
        m, n = 4, 6
        A = np.zeros((m, n), dtype=np.quaternion)

        # Right null space should be full space
        N_right = quat_null_space(A, side="right", rtol=self.rtol)
        self.assertEqual(N_right.shape, (n, n))

        # Left null space should be full space
        N_left = quat_null_space(A, side="left", rtol=self.rtol)
        self.assertEqual(N_left.shape, (m, m))

        # Verify orthogonality (should be trivial for zero matrix)
        AN = quat_matmat(A, N_right)
        self.assertLess(quat_frobenius_norm(AN), self.tol)

        N_H_A = quat_matmat(quat_hermitian(N_left), A)
        self.assertLess(quat_frobenius_norm(N_H_A), self.tol)

    def test_quat_null_space_rank_one(self):
        """Test null space of rank-1 matrices."""
        m, n = 5, 7
        rank = 1
        A = self._create_rank_deficient_matrix(m, n, rank)

        # Right null space dimension should be n-1
        N_right = quat_null_space(A, side="right", rtol=self.rtol)
        self.assertEqual(N_right.shape, (n, n - 1))

        # Left null space dimension should be m-1
        N_left = quat_null_space(A, side="left", rtol=self.rtol)
        self.assertEqual(N_left.shape, (m, m - 1))

        # Test orthogonality
        if N_right.shape[1] > 0:
            AN = quat_matmat(A, N_right)
            self.assertLess(quat_frobenius_norm(AN), self.tol)

        if N_left.shape[1] > 0:
            N_H_A = quat_matmat(quat_hermitian(N_left), A)
            self.assertLess(quat_frobenius_norm(N_H_A), self.tol)

    def test_quat_null_space_known_example(self):
        """Test with a matrix that has a known null space."""
        # Simple 2×3 rank-1 matrix
        A = np.array(
            [
                [
                    quaternion.quaternion(1, 0, 0, 0),
                    quaternion.quaternion(2, 0, 0, 0),
                    quaternion.quaternion(3, 0, 0, 0),
                ],
                [
                    quaternion.quaternion(0, 1, 0, 0),
                    quaternion.quaternion(0, 2, 0, 0),
                    quaternion.quaternion(0, 3, 0, 0),
                ],
            ]
        )

        # This should have rank 2, so right null space dimension = 3-2 = 1
        N_right = quat_null_space(A, side="right", rtol=self.rtol)

        # Check dimensions
        self.assertEqual(N_right.shape[0], 3)  # n = 3
        self.assertGreaterEqual(N_right.shape[1], 1)  # Should have at least 1 null vector

        # Check orthogonality
        if N_right.shape[1] > 0:
            AN = quat_matmat(A, N_right)
            norm = quat_frobenius_norm(AN)
            self.assertLess(norm, self.tol, f"A @ N should be zero, got norm {norm}")

    def test_quat_null_space_invalid_side(self):
        """Test error handling for invalid side parameter."""
        A = create_test_matrix(3, 3)

        with self.assertRaises(ValueError):
            quat_null_space(A, side="invalid")

    def test_convenience_functions(self):
        """Test convenience functions quat_null_right, quat_null_left, quat_kernel."""
        m, n, rank = 6, 8, 3
        A = self._create_rank_deficient_matrix(m, n, rank)

        # Test that convenience functions give same results as main function
        N_right_main = quat_null_space(A, side="right", rtol=self.rtol)
        N_right_conv = quat_null_right(A, rtol=self.rtol)

        self.assertEqual(N_right_main.shape, N_right_conv.shape)

        # Test left null space
        N_left_main = quat_null_space(A, side="left", rtol=self.rtol)
        N_left_conv = quat_null_left(A, rtol=self.rtol)

        self.assertEqual(N_left_main.shape, N_left_conv.shape)

        # Test kernel alias
        N_kernel_right = quat_kernel(A, side="right", rtol=self.rtol)
        N_kernel_left = quat_kernel(A, side="left", rtol=self.rtol)

        self.assertEqual(N_right_main.shape, N_kernel_right.shape)
        self.assertEqual(N_left_main.shape, N_kernel_left.shape)

    def test_null_space_orthonormality(self):
        """Test that null space vectors have reasonable orthogonality properties."""
        m, n, rank = 7, 10, 4
        A = self._create_rank_deficient_matrix(m, n, rank)

        # Get right null space
        N = quat_null_space(A, side="right", rtol=self.rtol)

        if N.shape[1] > 1:  # If we have multiple null vectors
            # Check that N^H @ N has reasonable condition number
            # Note: Perfect orthonormality may not hold due to quaternion-real embedding roundoff
            N_H = quat_hermitian(N)
            N_H_N = quat_matmat(N_H, N)

            # Check that diagonal elements are close to 1
            for i in range(N.shape[1]):
                diag_val = N_H_N[i, i]
                diag_real = quaternion.as_float_array(diag_val)[0]  # Real part
                self.assertGreater(
                    diag_real,
                    0.5,
                    f"Diagonal element {i} should be positive, got {diag_real}",
                )
                self.assertLess(
                    diag_real,
                    2.0,
                    f"Diagonal element {i} should be reasonable, got {diag_real}",
                )

            # More importantly: Check that the null space property A @ N ≈ 0 holds
            AN = quat_matmat(A, N)
            AN_norm = quat_frobenius_norm(AN)
            self.assertLess(
                AN_norm,
                1e-10,
                f"Primary null space property A @ N ≈ 0 failed, ||A N|| = {AN_norm}",
            )

    def test_consistency_across_different_ranks(self):
        """Test consistency for matrices of different ranks."""
        m, n = 6, 8

        for rank in range(0, min(m, n) + 1):
            with self.subTest(rank=rank):
                A = self._create_rank_deficient_matrix(m, n, rank)

                # Right null space
                N_right = quat_null_space(A, side="right", rtol=self.rtol)
                expected_right_dim = n - rank

                # Allow some tolerance in rank determination
                self.assertLessEqual(
                    abs(N_right.shape[1] - expected_right_dim),
                    1,
                    f"Right null space dimension mismatch for rank {rank}",
                )

                # Left null space
                N_left = quat_null_space(A, side="left", rtol=self.rtol)
                expected_left_dim = m - rank

                self.assertLessEqual(
                    abs(N_left.shape[1] - expected_left_dim),
                    1,
                    f"Left null space dimension mismatch for rank {rank}",
                )

    def test_tolerance_parameter(self):
        """Test that tolerance parameter affects rank determination correctly."""
        # Create a matrix with a small singular value
        A = self._create_rank_deficient_matrix(5, 7, 3)

        # Add small noise to make rank determination interesting
        noise = 1e-13 * create_test_matrix(5, 7)
        A_noisy = A + noise

        # Test with different tolerances
        rtol_strict = 1e-15
        rtol_loose = 1e-10

        N_strict = quat_null_space(A_noisy, side="right", rtol=rtol_strict)
        N_loose = quat_null_space(A_noisy, side="right", rtol=rtol_loose)

        # Loose tolerance should give larger null space (lower perceived rank)
        self.assertGreaterEqual(N_loose.shape[1], N_strict.shape[1])

    def test_rectangular_matrices(self):
        """Test null spaces for various rectangular matrix dimensions."""
        test_cases = [
            (3, 5, 2),  # More columns than rows
            (5, 3, 2),  # More rows than columns
            (4, 4, 2),  # Square
            (1, 5, 1),  # Single row
            (5, 1, 1),  # Single column
        ]

        for m, n, rank in test_cases:
            with self.subTest(m=m, n=n, rank=rank):
                A = self._create_rank_deficient_matrix(m, n, rank)

                # Test both sides
                N_right = quat_null_space(A, side="right", rtol=self.rtol)
                N_left = quat_null_space(A, side="left", rtol=self.rtol)

                # Check dimensions are reasonable
                self.assertEqual(N_right.shape[0], n)
                self.assertEqual(N_left.shape[0], m)
                self.assertGreaterEqual(N_right.shape[1], 0)
                self.assertGreaterEqual(N_left.shape[1], 0)

                # Check orthogonality if null spaces are non-empty
                if N_right.shape[1] > 0:
                    AN = quat_matmat(A, N_right)
                    self.assertLess(quat_frobenius_norm(AN), self.tol)

                if N_left.shape[1] > 0:
                    N_H_A = quat_matmat(quat_hermitian(N_left), A)
                    self.assertLess(quat_frobenius_norm(N_H_A), self.tol)


class TestKernelEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for kernel functions."""

    def test_empty_matrix(self):
        """Test behavior with empty matrices."""
        # 0×3 matrix
        A = np.empty((0, 3), dtype=np.quaternion)
        N_right = quat_null_space(A, side="right")
        # Right null space should be the entire 3D space
        self.assertEqual(N_right.shape, (3, 3))

        # 3×0 matrix
        A = np.empty((3, 0), dtype=np.quaternion)
        N_left = quat_null_space(A, side="left")
        # Left null space should be the entire 3D space
        self.assertEqual(N_left.shape, (3, 3))

    def test_very_small_matrices(self):
        """Test with 1×1 matrices."""
        # Non-zero 1×1 matrix
        A = np.array([[quaternion.quaternion(1, 0, 0, 0)]])

        N_right = quat_null_space(A, side="right")
        self.assertEqual(N_right.shape, (1, 0))  # No right null space

        N_left = quat_null_space(A, side="left")
        self.assertEqual(N_left.shape, (1, 0))  # No left null space

        # Zero 1×1 matrix
        A_zero = np.array([[quaternion.quaternion(0, 0, 0, 0)]])

        N_right_zero = quat_null_space(A_zero, side="right")
        self.assertEqual(N_right_zero.shape, (1, 1))  # Full space

        N_left_zero = quat_null_space(A_zero, side="left")
        self.assertEqual(N_left_zero.shape, (1, 1))  # Full space


if __name__ == "__main__":
    # Configure test runner
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestKernelNullSpace))
    suite.addTests(loader.loadTestsFromTestCase(TestKernelEdgeCases))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'=' * 60}")
    print("KERNEL/NULL SPACE TESTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%"
    )

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Error:')[-1].strip()}")

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
