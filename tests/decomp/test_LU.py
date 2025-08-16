"""
Unit tests for quaternion LU decomposition functions.
"""

import os
import sys

import numpy as np
import pytest
import quaternion

# Add core module to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from quatica.decomp.LU import (
    quaternion_lu,
    quaternion_modulus,
    quaternion_tril,
    quaternion_triu,
    verify_lu_decomposition,
)
from quatica.utils import quat_eye, quat_frobenius_norm, quat_matmat


class TestQuaternionModulus:
    """Test cases for quaternion modulus computation."""

    def test_quaternion_modulus_basic(self):
        """Test modulus computation on simple quaternions."""
        # Create a simple quaternion matrix
        A = quaternion.as_quat_array(
            [[[1, 0, 0, 0], [0, 1, 0, 0]], [[0, 0, 1, 0], [1, 1, 0, 0]]]
        )

        moduli = quaternion_modulus(A)

        # Expected moduli: sqrt(1^2 + 0^2 + 0^2 + 0^2) = 1, etc.
        expected = np.array([[1.0, 1.0], [1.0, np.sqrt(2.0)]])
        assert np.allclose(moduli, expected, atol=1e-12)

    def test_quaternion_modulus_random(self):
        """Test modulus computation on random quaternion matrix."""
        np.random.seed(42)
        m, n = 3, 4

        # Create random quaternion matrix
        A_components = np.random.randn(m, n, 4)
        A = quaternion.as_quat_array(A_components)

        moduli = quaternion_modulus(A)

        # Check that moduli are non-negative
        assert np.all(moduli >= 0)

        # Check that moduli match expected values
        expected_moduli = np.sqrt(np.sum(A_components**2, axis=-1))
        assert np.allclose(moduli, expected_moduli, atol=1e-12)

    def test_quaternion_modulus_complex(self):
        """Test modulus computation on complex quaternions."""
        # Create quaternions with non-zero imaginary parts
        A = quaternion.as_quat_array(
            [[[1, 2, 3, 4], [0, 1, 0, 0]], [[1, 1, 1, 1], [2, 0, 0, 0]]]
        )

        moduli = quaternion_modulus(A)

        # Expected moduli: sqrt(1^2 + 2^2 + 3^2 + 4^2) = sqrt(30), etc.
        expected = np.array([[np.sqrt(30), 1.0], [2.0, 2.0]])
        assert np.allclose(moduli, expected, atol=1e-12)


class TestTriangularExtraction:
    """Test cases for triangular matrix extraction."""

    def test_quaternion_triu_basic(self):
        """Test upper triangular extraction."""
        # Create a 3x3 quaternion matrix
        A = quaternion.as_quat_array(
            [
                [[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]],
                [[4, 0, 0, 0], [5, 0, 0, 0], [6, 0, 0, 0]],
                [[7, 0, 0, 0], [8, 0, 0, 0], [9, 0, 0, 0]],
            ]
        )

        U = quaternion_triu(A)

        # Check that lower triangular part is zero
        U_float = quaternion.as_float_array(U)
        assert np.allclose(U_float[1, 0], 0)  # (1,0) should be zero
        assert np.allclose(U_float[2, 0], 0)  # (2,0) should be zero
        assert np.allclose(U_float[2, 1], 0)  # (2,1) should be zero

        # Check that upper triangular part is preserved
        assert U[0, 0] == A[0, 0]
        assert U[0, 1] == A[0, 1]
        assert U[0, 2] == A[0, 2]
        assert U[1, 1] == A[1, 1]
        assert U[1, 2] == A[1, 2]
        assert U[2, 2] == A[2, 2]

    def test_quaternion_triu_with_offset(self):
        """Test upper triangular extraction with offset."""
        A = quaternion.as_quat_array(
            [
                [[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]],
                [[4, 0, 0, 0], [5, 0, 0, 0], [6, 0, 0, 0]],
                [[7, 0, 0, 0], [8, 0, 0, 0], [9, 0, 0, 0]],
            ]
        )

        U = quaternion_triu(A, k=1)  # Offset by 1

        # Check that elements below k=1 diagonal are zero
        U_float = quaternion.as_float_array(U)
        assert np.allclose(U_float[0, 0], 0)  # (0,0) should be zero
        assert np.allclose(U_float[1, 0], 0)  # (1,0) should be zero
        assert np.allclose(U_float[2, 0], 0)  # (2,0) should be zero
        assert np.allclose(U_float[2, 1], 0)  # (2,1) should be zero

    def test_quaternion_tril_basic(self):
        """Test lower triangular extraction."""
        # Create a 3x3 quaternion matrix
        A = quaternion.as_quat_array(
            [
                [[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]],
                [[4, 0, 0, 0], [5, 0, 0, 0], [6, 0, 0, 0]],
                [[7, 0, 0, 0], [8, 0, 0, 0], [9, 0, 0, 0]],
            ]
        )

        L = quaternion_tril(A)

        # Check that upper triangular part is zero
        L_float = quaternion.as_float_array(L)
        assert np.allclose(L_float[0, 1], 0)  # (0,1) should be zero
        assert np.allclose(L_float[0, 2], 0)  # (0,2) should be zero
        assert np.allclose(L_float[1, 2], 0)  # (1,2) should be zero

        # Check that lower triangular part is preserved
        assert L[0, 0] == A[0, 0]
        assert L[1, 0] == A[1, 0]
        assert L[1, 1] == A[1, 1]
        assert L[2, 0] == A[2, 0]
        assert L[2, 1] == A[2, 1]
        assert L[2, 2] == A[2, 2]

    def test_quaternion_tril_with_offset(self):
        """Test lower triangular extraction with offset."""
        A = quaternion.as_quat_array(
            [
                [[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]],
                [[4, 0, 0, 0], [5, 0, 0, 0], [6, 0, 0, 0]],
                [[7, 0, 0, 0], [8, 0, 0, 0], [9, 0, 0, 0]],
            ]
        )

        L = quaternion_tril(A, k=-1)  # Offset by -1 (exclude diagonal)

        # Check that diagonal and upper triangular part are zero
        L_float = quaternion.as_float_array(L)
        assert np.allclose(L_float[0, 0], 0)  # (0,0) should be zero
        assert np.allclose(L_float[0, 1], 0)  # (0,1) should be zero
        assert np.allclose(L_float[0, 2], 0)  # (0,2) should be zero
        assert np.allclose(L_float[1, 1], 0)  # (1,1) should be zero
        assert np.allclose(L_float[1, 2], 0)  # (1,2) should be zero
        assert np.allclose(L_float[2, 2], 0)  # (2,2) should be zero


class TestLUDecomposition:
    """Test cases for quaternion LU decomposition."""

    def test_lu_basic_square(self):
        """Test LU decomposition on basic square quaternion matrix."""
        # Create a well-conditioned 3x3 quaternion matrix
        A = quaternion.as_quat_array(
            [
                [[2, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
                [[1, 0, 0, 0], [3, 0, 0, 0], [1, 0, 0, 0]],
                [[0, 0, 0, 0], [1, 0, 0, 0], [4, 0, 0, 0]],
            ]
        )

        L, U = quaternion_lu(A)

        # Check shapes
        assert L.shape == (3, 3)
        assert U.shape == (3, 3)

        # Check reconstruction accuracy
        LU = quat_matmat(L, U)
        reconstruction_error = quat_frobenius_norm(A - LU)
        original_norm = quat_frobenius_norm(A)
        relative_error = reconstruction_error / original_norm
        assert relative_error < 1e-10, f"Relative reconstruction error: {relative_error}"

        # Check L is lower triangular with unit diagonal
        L_float = quaternion.as_float_array(L)
        L_real = L_float[:, :, 0]  # Real part
        assert np.allclose(L_real, np.tril(L_real), atol=1e-12)
        assert np.allclose(np.diag(L_real), np.ones(3), atol=1e-12)

        # Check U is upper triangular
        U_float = quaternion.as_float_array(U)
        U_real = U_float[:, :, 0]  # Real part
        assert np.allclose(U_real, np.triu(U_real), atol=1e-12)

    def test_lu_with_permutation(self):
        """Test LU decomposition with permutation matrix."""
        # Create a matrix that requires pivoting
        A = quaternion.as_quat_array(
            [
                [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]],
                [[3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]],
                [[6, 0, 0, 0], [7, 0, 0, 0], [8, 0, 0, 0]],
            ]
        )

        L, U, P = quaternion_lu(A, return_p=True)

        # Check shapes
        assert L.shape == (3, 3)
        assert U.shape == (3, 3)
        assert P.shape == (3, 3)

        # Check P * A = L * U
        PA = quat_matmat(P, A)
        LU = quat_matmat(L, U)
        reconstruction_error = quat_frobenius_norm(PA - LU)
        original_norm = quat_frobenius_norm(PA)
        relative_error = reconstruction_error / original_norm
        assert relative_error < 1e-10, f"Relative reconstruction error: {relative_error}"

        # Check P is a permutation matrix
        P_float = quaternion.as_float_array(P)
        P_real = P_float[:, :, 0]  # Real part
        assert np.allclose(P_real @ P_real.T, np.eye(3), atol=1e-12)
        assert np.allclose(P_real.T @ P_real, np.eye(3), atol=1e-12)

        # Check P has exactly one 1 per row and column
        assert np.allclose(np.sum(P_real, axis=0), np.ones(3), atol=1e-12)
        assert np.allclose(np.sum(P_real, axis=1), np.ones(3), atol=1e-12)

    def test_lu_rectangular_tall(self):
        """Test LU decomposition on tall rectangular matrix (m > n)."""
        # Create a well-conditioned 4x3 quaternion matrix
        A = quaternion.as_quat_array(
            [
                [[2, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
                [[1, 0, 0, 0], [3, 0, 0, 0], [1, 0, 0, 0]],
                [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]],
                [[1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]],
            ]
        )

        L, U = quaternion_lu(A)

        # Check shapes: L is m×N, U is N×n where N = min(m,n)
        assert L.shape == (4, 3)
        assert U.shape == (3, 3)

        # Check reconstruction accuracy
        LU = quat_matmat(L, U)
        reconstruction_error = quat_frobenius_norm(A - LU)
        original_norm = quat_frobenius_norm(A)
        relative_error = reconstruction_error / original_norm
        assert relative_error < 1e-10, f"Relative reconstruction error: {relative_error}"

    def test_lu_rectangular_wide(self):
        """Test LU decomposition on wide rectangular matrix (m < n)."""
        # Create a well-conditioned 3x4 quaternion matrix
        A = quaternion.as_quat_array(
            [
                [[2, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]],
                [[1, 0, 0, 0], [3, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [1, 0, 0, 0]],
            ]
        )

        L, U = quaternion_lu(A)

        # Check shapes: L is m×N, U is N×n where N = min(m,n)
        assert L.shape == (3, 3)
        assert U.shape == (3, 4)

        # Check reconstruction accuracy
        LU = quat_matmat(L, U)
        reconstruction_error = quat_frobenius_norm(A - LU)
        original_norm = quat_frobenius_norm(A)
        relative_error = reconstruction_error / original_norm
        assert relative_error < 1e-10, f"Relative reconstruction error: {relative_error}"

    def test_lu_random_matrix(self):
        """Test LU decomposition on random quaternion matrix."""
        np.random.seed(42)
        m, n = 5, 4

        # Create a better conditioned random quaternion matrix
        # Add identity matrix to improve conditioning
        A_components = np.random.randn(m, n, 4) * 0.1  # Smaller random values
        A_components[:, :, 0] += np.eye(m, n)  # Add identity to diagonal
        A = quaternion.as_quat_array(A_components)

        L, U = quaternion_lu(A)

        # Check shapes
        assert L.shape == (m, min(m, n))
        assert U.shape == (min(m, n), n)

        # Check reconstruction accuracy
        LU = quat_matmat(L, U)
        reconstruction_error = quat_frobenius_norm(A - LU)
        original_norm = quat_frobenius_norm(A)
        relative_error = reconstruction_error / original_norm
        assert relative_error < 1e-8, f"Relative reconstruction error: {relative_error}"

        # Check L is lower triangular with unit diagonal
        L_float = quaternion.as_float_array(L)
        L_real = L_float[:, :, 0]  # Real part
        assert np.allclose(L_real, np.tril(L_real), atol=1e-12)
        assert np.allclose(np.diag(L_real), np.ones(min(m, n)), atol=1e-12)

        # Check U is upper triangular
        U_float = quaternion.as_float_array(U)
        U_real = U_float[:, :, 0]  # Real part
        assert np.allclose(U_real, np.triu(U_real), atol=1e-12)

    def test_lu_complex_quaternions(self):
        """Test LU decomposition on complex quaternion matrix."""
        # Create a matrix with non-zero imaginary parts
        A = quaternion.as_quat_array(
            [[[1, 1, 0, 0], [0, 1, 1, 0]], [[1, 0, 0, 1], [1, 1, 1, 1]]]
        )

        L, U = quaternion_lu(A)

        # Check shapes
        assert L.shape == (2, 2)
        assert U.shape == (2, 2)

        # Check reconstruction accuracy
        LU = quat_matmat(L, U)
        reconstruction_error = quat_frobenius_norm(A - LU)
        original_norm = quat_frobenius_norm(A)
        relative_error = reconstruction_error / original_norm
        assert relative_error < 1e-10, f"Relative reconstruction error: {relative_error}"

    def test_lu_zero_pivot_error(self):
        """If the matrix is truly singular by column (no nonzero pivot available), LU should raise;
        otherwise pivoting may succeed and reconstruction must be accurate."""
        # Case 1: Column of all zeros (true singular for LU even with pivoting)
        A_true_singular = quaternion.as_quat_array(
            [[[0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [1, 0, 0, 0]]]
        )
        with pytest.raises(ValueError):
            quaternion_lu(A_true_singular)

        # Case 2: First row zero but pivoting can recover; then no error and reconstruction must hold
        A_pivotable = quaternion.as_quat_array(
            [[[0, 0, 0, 0], [0, 0, 0, 0]], [[1, 0, 0, 0], [1, 0, 0, 0]]]
        )
        L, U = quaternion_lu(A_pivotable)
        LU = quat_matmat(L, U)
        relative_error = quat_frobenius_norm(A_pivotable - LU) / (
            quat_frobenius_norm(A_pivotable) + 1e-30
        )
        assert relative_error < 1e-10

    def test_lu_identity_matrix(self):
        """Test LU decomposition on identity matrix."""
        # Create 3x3 identity matrix
        A = quat_eye(3)

        L, U = quaternion_lu(A)

        # Check shapes
        assert L.shape == (3, 3)
        assert U.shape == (3, 3)

        # Check reconstruction accuracy
        LU = quat_matmat(L, U)
        reconstruction_error = quat_frobenius_norm(A - LU)
        original_norm = quat_frobenius_norm(A)
        relative_error = reconstruction_error / original_norm
        assert relative_error < 1e-12, f"Relative reconstruction error: {relative_error}"

        # For identity matrix, L should be identity and U should be identity
        L_float = quaternion.as_float_array(L)
        U_float = quaternion.as_float_array(U)
        assert np.allclose(L_float[:, :, 0], np.eye(3), atol=1e-12)
        assert np.allclose(U_float[:, :, 0], np.eye(3), atol=1e-12)

    def test_lu_diagonal_matrix(self):
        """Test LU decomposition on diagonal matrix."""
        # Create diagonal matrix
        A = quaternion.as_quat_array(
            [
                [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [3, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [4, 0, 0, 0]],
            ]
        )

        L, U = quaternion_lu(A)

        # Check reconstruction accuracy
        LU = quat_matmat(L, U)
        reconstruction_error = quat_frobenius_norm(A - LU)
        original_norm = quat_frobenius_norm(A)
        relative_error = reconstruction_error / original_norm
        assert relative_error < 1e-12, f"Relative reconstruction error: {relative_error}"

        # For diagonal matrix, L should be identity and U should be the diagonal matrix
        L_float = quaternion.as_float_array(L)
        U_float = quaternion.as_float_array(U)
        assert np.allclose(L_float[:, :, 0], np.eye(3), atol=1e-12)
        assert np.allclose(U_float[:, :, 0], np.diag([2, 3, 4]), atol=1e-12)

    def test_lu_permutation_consistency(self):
        """Test that permutation handling is consistent."""
        # Create a matrix that requires pivoting
        A = quaternion.as_quat_array(
            [[[0, 0, 0, 0], [1, 0, 0, 0]], [[2, 0, 0, 0], [3, 0, 0, 0]]]
        )

        # Test both output modes
        L1, U1 = quaternion_lu(A, return_p=False)
        L2, U2, P = quaternion_lu(A, return_p=True)

        # Check that L1*U1 = A
        LU1 = quat_matmat(L1, U1)
        error1 = quat_frobenius_norm(A - LU1)

        # Check that L2*U2 = P*A
        LU2 = quat_matmat(L2, U2)
        PA = quat_matmat(P, A)
        error2 = quat_frobenius_norm(PA - LU2)

        assert error1 < 1e-12, f"Error in 2-output mode: {error1}"
        assert error2 < 1e-12, f"Error in 3-output mode: {error2}"

    def test_lu_small_matrices(self):
        """Test LU decomposition on very small matrices."""
        # Test 1x1 matrix
        A1 = quaternion.as_quat_array([[[1, 0, 0, 0]]])
        L1, U1 = quaternion_lu(A1)
        assert L1.shape == (1, 1)
        assert U1.shape == (1, 1)

        # Test 1x2 matrix
        A2 = quaternion.as_quat_array([[[1, 0, 0, 0], [2, 0, 0, 0]]])
        L2, U2 = quaternion_lu(A2)
        assert L2.shape == (1, 1)
        assert U2.shape == (1, 2)

        # Test 2x1 matrix
        A3 = quaternion.as_quat_array([[[1, 0, 0, 0]], [[2, 0, 0, 0]]])
        L3, U3 = quaternion_lu(A3)
        assert L3.shape == (2, 1)
        assert U3.shape == (1, 1)


class TestLUVerification:
    """Test cases for LU decomposition verification."""

    def test_verify_lu_decomposition_basic(self):
        """Test verification function on basic case."""
        # Create a simple matrix
        A = quaternion.as_quat_array(
            [[[1, 0, 0, 0], [2, 0, 0, 0]], [[3, 0, 0, 0], [4, 0, 0, 0]]]
        )

        L, U = quaternion_lu(A)
        result = verify_lu_decomposition(A, L, U, verbose=True)

        assert result["passed"]
        assert result["relative_error"] < 1e-12

    def test_verify_lu_decomposition_with_permutation(self):
        """Test verification function with permutation matrix."""
        # Create a matrix that requires pivoting
        A = quaternion.as_quat_array(
            [[[0, 0, 0, 0], [1, 0, 0, 0]], [[2, 0, 0, 0], [3, 0, 0, 0]]]
        )

        L, U, P = quaternion_lu(A, return_p=True)
        result = verify_lu_decomposition(A, L, U, P=P, verbose=True)

        assert result["passed"]
        assert result["relative_error"] < 1e-12

    def test_verify_lu_decomposition_edge_cases(self):
        """Test verification function on edge cases."""
        # Test with identity matrix
        A = quat_eye(2)
        L, U = quaternion_lu(A)
        result = verify_lu_decomposition(A, L, U)
        assert result["passed"]

        # Test with zero matrix (should fail due to zero pivot)
        A_zero = np.zeros((2, 2), dtype=np.quaternion)
        with pytest.raises(ValueError):
            quaternion_lu(A_zero)


def test_lu_zero_first_pivot_uses_permutation():
    """When the first pivot is zero, LU with permutation should row-swap and reconstruct P @ A = L @ U."""
    A = quaternion.as_quat_array(
        [[[0, 0, 0, 0], [1, 0, 0, 0]], [[2, 0, 0, 0], [3, 0, 0, 0]]]
    )

    L, U, P = quaternion_lu(A, return_p=True)

    # Check reconstruction: P * A = L * U
    PA = quat_matmat(P, A)
    LU = quat_matmat(L, U)
    error = quat_frobenius_norm(PA - LU) / (quat_frobenius_norm(PA) + 1e-30)
    assert error < 1e-12

    # Check P is the swap matrix [[0,1],[1,0]] in real part
    P_real = quaternion.as_float_array(P)[:, :, 0]
    assert np.allclose(P_real, np.array([[0.0, 1.0], [1.0, 0.0]]), atol=1e-12)


class TestLargeRandomMatrices:
    """Test cases for large random matrices."""

    def test_lu_large_square_matrix(self):
        """Test LU decomposition on large square matrix."""
        np.random.seed(42)
        n = 10

        # Create a better conditioned random quaternion matrix
        A_components = np.random.randn(n, n, 4) * 0.1  # Smaller random values
        A_components[:, :, 0] += np.eye(n)  # Add identity to diagonal
        A = quaternion.as_quat_array(A_components)

        L, U = quaternion_lu(A)

        # Check shapes
        assert L.shape == (n, n)
        assert U.shape == (n, n)

        # Check reconstruction accuracy
        LU = quat_matmat(L, U)
        reconstruction_error = quat_frobenius_norm(A - LU)
        original_norm = quat_frobenius_norm(A)
        relative_error = reconstruction_error / original_norm
        assert relative_error < 1e-8, f"Relative reconstruction error: {relative_error}"

    def test_lu_large_rectangular_matrix(self):
        """Test LU decomposition on large rectangular matrix."""
        np.random.seed(42)
        m, n = 8, 12

        # Create a better conditioned random quaternion matrix
        A_components = np.random.randn(m, n, 4) * 0.1  # Smaller random values
        A_components[:, :, 0] += np.eye(m, n)  # Add identity to diagonal
        A = quaternion.as_quat_array(A_components)

        L, U = quaternion_lu(A)

        # Check shapes
        assert L.shape == (m, min(m, n))
        assert U.shape == (min(m, n), n)

        # Check reconstruction accuracy
        LU = quat_matmat(L, U)
        reconstruction_error = quat_frobenius_norm(A - LU)
        original_norm = quat_frobenius_norm(A)
        relative_error = reconstruction_error / original_norm
        assert relative_error < 1e-8, f"Relative reconstruction error: {relative_error}"


if __name__ == "__main__":
    # Run basic tests
    print("Running LU decomposition tests...")

    # Test modulus computation
    test_modulus = TestQuaternionModulus()
    test_modulus.test_quaternion_modulus_basic()
    test_modulus.test_quaternion_modulus_random()
    test_modulus.test_quaternion_modulus_complex()
    print("✅ Modulus tests passed")

    # Test triangular extraction
    test_triangular = TestTriangularExtraction()
    test_triangular.test_quaternion_triu_basic()
    test_triangular.test_quaternion_triu_with_offset()
    test_triangular.test_quaternion_tril_basic()
    test_triangular.test_quaternion_tril_with_offset()
    print("✅ Triangular extraction tests passed")

    # Test LU decomposition
    test_lu = TestLUDecomposition()
    test_lu.test_lu_basic_square()
    test_lu.test_lu_with_permutation()
    test_lu.test_lu_rectangular_tall()
    test_lu.test_lu_rectangular_wide()
    test_lu.test_lu_random_matrix()
    test_lu.test_lu_complex_quaternions()
    test_lu.test_lu_identity_matrix()
    test_lu.test_lu_diagonal_matrix()
    test_lu.test_lu_permutation_consistency()
    test_lu.test_lu_small_matrices()
    print("✅ LU decomposition tests passed")

    # Test verification
    test_verify = TestLUVerification()
    test_verify.test_verify_lu_decomposition_basic()
    test_verify.test_verify_lu_decomposition_with_permutation()
    test_verify.test_verify_lu_decomposition_edge_cases()
    print("✅ Verification tests passed")

    # Test large matrices
    test_large = TestLargeRandomMatrices()
    test_large.test_lu_large_square_matrix()
    test_large.test_lu_large_rectangular_matrix()
    print("✅ Large matrix tests passed")

    print("\n🎉 All LU decomposition tests passed successfully!")
