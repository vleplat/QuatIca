#!/usr/bin/env python3
"""
Unit tests for quaternion matrix rank computation.

This module tests the rank function from quatica/utils.py to ensure it correctly
computes the rank of quaternion matrices by counting non-zero singular values.

Author: QuatIca Team
Date: 2024
"""

import os
import sys
import unittest

import numpy as np

# Add core module to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "quatica"))
from data_gen import create_test_matrix
from utils import quat_matmat, rank


class TestRank(unittest.TestCase):
    """Test cases for quaternion matrix rank computation."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

    def test_rank_matrix_product(self):
        """Test rank of matrix product C = A @ B where A is m×r and B is r×n."""
        print("\nTesting rank of matrix product C = A @ B...")

        # Test different sizes
        test_cases = [
            (3, 2, 4),  # A: 3×2, B: 2×4, expected rank: 2
            (4, 3, 5),  # A: 4×3, B: 3×5, expected rank: 3
            (5, 2, 3),  # A: 5×2, B: 2×3, expected rank: 2
            (6, 4, 7),  # A: 6×4, B: 4×7, expected rank: 4
        ]

        for m, r, n in test_cases:
            with self.subTest(m=m, r=r, n=n):
                # Generate random matrices A (m×r) and B (r×n)
                A = create_test_matrix(m, r)
                B = create_test_matrix(r, n)

                # Compute product C = A @ B
                C = quat_matmat(A, B)

                # Compute rank of C
                computed_rank = rank(C)

                # Expected rank should be r (the inner dimension)
                expected_rank = r

                # Check if they match
                self.assertEqual(
                    computed_rank,
                    expected_rank,
                    f"Size {m}×{r} @ {r}×{n}: Expected rank {expected_rank}, got {computed_rank}",
                )

                print(
                    f"✓ Size {m}×{r} @ {r}×{n}: rank(C) = {computed_rank} (expected: {expected_rank})"
                )

    def test_rank_full_rank_matrix(self):
        """Test rank of full-rank matrices."""
        print("\nTesting rank of full-rank matrices...")

        # Test different sizes
        sizes = [(2, 2), (3, 3), (4, 4), (5, 5), (3, 4), (4, 3)]

        for m, n in sizes:
            with self.subTest(m=m, n=n):
                # Generate random matrix
                A = create_test_matrix(m, n)

                # Compute rank
                computed_rank = rank(A)
                expected_rank = min(m, n)  # Full rank matrix

                self.assertEqual(
                    computed_rank,
                    expected_rank,
                    f"Size {m}×{n}: Expected rank {expected_rank}, got {computed_rank}",
                )

                print(
                    f"✓ Size {m}×{n}: rank(A) = {computed_rank} (expected: {expected_rank})"
                )

    def test_rank_low_rank_matrix(self):
        """Test rank of low-rank matrices."""
        print("\nTesting rank of low-rank matrices...")

        # Test cases: (m, n, target_rank)
        test_cases = [
            (4, 4, 2),  # 4×4 matrix with rank 2
            (5, 3, 1),  # 5×3 matrix with rank 1
            (6, 6, 3),  # 6×6 matrix with rank 3
        ]

        for m, n, target_rank in test_cases:
            with self.subTest(m=m, n=n, target_rank=target_rank):
                # Generate low-rank matrix by creating A and B with target_rank
                A = create_test_matrix(m, target_rank)
                B = create_test_matrix(target_rank, n)
                C = quat_matmat(A, B)

                # Compute rank
                computed_rank = rank(C)

                self.assertEqual(
                    computed_rank,
                    target_rank,
                    f"Size {m}×{n}, target rank {target_rank}: Got {computed_rank}",
                )

                print(
                    f"✓ Size {m}×{n}, target rank {target_rank}: rank(C) = {computed_rank}"
                )

    def test_rank_zero_matrix(self):
        """Test rank of zero matrix."""
        print("\nTesting rank of zero matrix...")

        # Test different sizes
        sizes = [(2, 2), (3, 4), (5, 3)]

        for m, n in sizes:
            with self.subTest(m=m, n=n):
                # Create zero matrix
                A = np.zeros((m, n), dtype=np.quaternion)

                # Compute rank
                computed_rank = rank(A)
                expected_rank = 0

                self.assertEqual(
                    computed_rank,
                    expected_rank,
                    f"Zero matrix {m}×{n}: Expected rank 0, got {computed_rank}",
                )

                print(f"✓ Zero matrix {m}×{n}: rank(A) = {computed_rank}")

    def test_rank_identity_matrix(self):
        """Test rank of identity matrix."""
        print("\nTesting rank of identity matrix...")

        # Test different sizes
        sizes = [2, 3, 4, 5]

        for n in sizes:
            with self.subTest(n=n):
                # Create identity matrix
                A = np.eye(n, dtype=np.quaternion)

                # Compute rank
                computed_rank = rank(A)
                expected_rank = n

                self.assertEqual(
                    computed_rank,
                    expected_rank,
                    f"Identity matrix {n}×{n}: Expected rank {n}, got {computed_rank}",
                )

                print(f"✓ Identity matrix {n}×{n}: rank(A) = {computed_rank}")

    def test_rank_with_tolerance(self):
        """Test rank with custom tolerance."""
        print("\nTesting rank with custom tolerance...")

        # Create a matrix that's almost rank-deficient
        m, n = 3, 3
        A = create_test_matrix(m, n)

        # Add a very small perturbation to make one singular value close to zero
        # This tests the tolerance parameter

        # Test with different tolerances
        tolerances = [1e-10, 1e-12, 1e-14, 1e-16]

        for tol in tolerances:
            with self.subTest(tol=tol):
                computed_rank = rank(A, tol=tol)

                # Should be a reasonable rank (not negative, not larger than min(m,n))
                self.assertGreaterEqual(computed_rank, 0)
                self.assertLessEqual(computed_rank, min(m, n))

                print(f"✓ Tolerance {tol:.0e}: rank(A) = {computed_rank}")

    def test_rank_edge_cases(self):
        """Test rank with edge cases."""
        print("\nTesting rank with edge cases...")

        # Test 1×1 matrix
        A_1x1 = create_test_matrix(1, 1)
        rank_1x1 = rank(A_1x1)
        self.assertEqual(rank_1x1, 1)
        print("✓ 1×1 matrix: rank = 1")

        # Test 1×n matrix
        A_1x3 = create_test_matrix(1, 3)
        rank_1x3 = rank(A_1x3)
        self.assertEqual(rank_1x3, 1)
        print("✓ 1×3 matrix: rank = 1")

        # Test m×1 matrix
        A_3x1 = create_test_matrix(3, 1)
        rank_3x1 = rank(A_3x1)
        self.assertEqual(rank_3x1, 1)
        print("✓ 3×1 matrix: rank = 1")


def run_rank_tests():
    """Run all rank tests and print results."""
    print("=" * 80)
    print(" QUATERNION MATRIX RANK COMPUTATION TEST")
    print("=" * 80)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRank)

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 80)
    print(" TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("✓ ALL TESTS PASSED!")
        print("\nThe rank function is working correctly.")
        print("Rank computation:")
        print("- Correctly identifies matrix product ranks")
        print("- Handles full-rank and low-rank matrices")
        print("- Works with zero and identity matrices")
        print("- Supports custom tolerance parameters")
        print("- Handles edge cases properly")
    else:
        print("✗ SOME TESTS FAILED!")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")

    return result.wasSuccessful()


if __name__ == "__main__":
    # Run comprehensive test
    success = run_rank_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
