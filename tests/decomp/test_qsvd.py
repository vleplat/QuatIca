"""
Unit tests for quaternion matrix decomposition functions.
"""

import os
import sys

import numpy as np
import quaternion

# Add core module to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from quatica.decomp.qsvd import classical_qsvd, classical_qsvd_full, qr_qua
from quatica.utils import quat_frobenius_norm, quat_hermitian, quat_matmat


class TestQRQuaternion:
    """Test cases for quaternion QR decomposition."""

    def test_qr_qua_basic(self):
        """Test basic QR decomposition on small quaternion matrix."""
        # Create a simple 3x2 quaternion matrix
        X = quaternion.as_quat_array(
            [
                [[1, 0, 0, 0], [0, 1, 0, 0]],
                [[0, 0, 1, 0], [1, 1, 0, 0]],
                [[0, 0, 0, 1], [0, 1, 1, 0]],
            ]
        )

        Q, R = qr_qua(X)

        # Check shapes
        assert Q.shape == (3, 2)
        assert R.shape == (2, 2)

        # Check reconstruction accuracy
        X_recon = quat_matmat(Q, R)
        reconstruction_error = quat_frobenius_norm(X - X_recon)
        assert reconstruction_error < 1e-12, (
            f"Reconstruction error: {reconstruction_error}"
        )

        # Check orthonormality of Q (Q^H @ Q ≈ I)
        QH_Q = quat_matmat(quat_hermitian(Q), Q)
        QH_Q_float = quaternion.as_float_array(QH_Q)

        # Check diagonal elements are 1
        diag_elements = np.diag(QH_Q_float[:, :, 0])  # Real part of diagonal
        assert np.allclose(diag_elements, np.ones(len(diag_elements)), atol=1e-12)

        # Check off-diagonal elements are 0
        off_diag_mask = ~np.eye(QH_Q_float.shape[0], dtype=bool)
        off_diag_elements = QH_Q_float[off_diag_mask]
        assert np.max(np.abs(off_diag_elements)) < 1e-12

        # Check R is upper triangular
        R_float = quaternion.as_float_array(R)
        R_real = R_float[:, :, 0]  # Real part
        assert np.allclose(R_real, np.triu(R_real), atol=1e-12)

    def test_qr_qua_random(self):
        """Test QR decomposition on random quaternion matrix."""
        np.random.seed(42)
        m, n = 6, 4

        # Create random quaternion matrix
        X_components = np.random.randn(m, n, 4)
        X = quaternion.as_quat_array(X_components)

        Q, R = qr_qua(X)

        # Check shapes
        assert Q.shape == (m, n)
        assert R.shape == (n, n)

        # Check reconstruction accuracy
        X_recon = quat_matmat(Q, R)
        reconstruction_error = quat_frobenius_norm(X - X_recon)
        original_norm = quat_frobenius_norm(X)
        relative_error = reconstruction_error / original_norm
        assert relative_error < 1e-12, f"Relative reconstruction error: {relative_error}"

        # Check orthonormality of Q
        QH_Q = quat_matmat(quat_hermitian(Q), Q)
        QH_Q_float = quaternion.as_float_array(QH_Q)

        # Check diagonal elements are 1
        diag_elements = np.diag(QH_Q_float[:, :, 0])
        assert np.allclose(diag_elements, np.ones(len(diag_elements)), atol=1e-12)

        # Check off-diagonal elements are small
        off_diag_mask = ~np.eye(QH_Q_float.shape[0], dtype=bool)
        off_diag_elements = QH_Q_float[off_diag_mask]
        assert np.max(np.abs(off_diag_elements)) < 1e-12


class TestClassicalQSVD:
    """Test cases for classical Q-SVD."""

    def test_classical_qsvd_basic(self):
        """Test basic classical Q-SVD on small matrix."""
        # Create a 4x3 quaternion matrix
        X = quaternion.as_quat_array(
            [
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                [[0, 0, 0, 1], [1, 1, 0, 0], [0, 1, 1, 0]],
                [[1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 1]],
                [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 0]],
            ]
        )

        R = 2  # Target rank
        U, s, V = classical_qsvd(X, R)

        # Check shapes
        assert U.shape == (4, R)
        assert len(s) == R
        assert V.shape == (3, R)

        # Check singular values are non-negative and decreasing
        assert np.all(s >= 0)
        assert np.all(np.diff(s) <= 0)

        # Check reconstruction accuracy
        S = np.diag(s)
        X_recon = quat_matmat(quat_matmat(U, S), quat_hermitian(V))
        reconstruction_error = quat_frobenius_norm(X - X_recon)
        original_norm = quat_frobenius_norm(X)
        relative_error = reconstruction_error / original_norm

        # Check reconstruction accuracy (relaxed tolerance for now)
        assert relative_error < 0.5, f"Reconstruction error too large: {relative_error}"

        # Check orthonormality of U (U^H @ U ≈ I)
        UH_U = quat_matmat(quat_hermitian(U), U)
        UH_U_float = quaternion.as_float_array(UH_U)

        # Check diagonal elements are 1
        diag_elements_U = np.diag(UH_U_float[:, :, 0])
        assert np.allclose(diag_elements_U, np.ones(len(diag_elements_U)), atol=1e-6)

        # Check off-diagonal elements are small
        off_diag_mask_U = ~np.eye(UH_U_float.shape[0], dtype=bool)
        off_diag_elements_U = UH_U_float[off_diag_mask_U]
        assert np.max(np.abs(off_diag_elements_U)) < 1e-6

        # Check orthonormality of V (V^H @ V ≈ I)
        VH_V = quat_matmat(quat_hermitian(V), V)
        VH_V_float = quaternion.as_float_array(VH_V)

        # Check diagonal elements are 1
        diag_elements_V = np.diag(VH_V_float[:, :, 0])
        assert np.allclose(diag_elements_V, np.ones(len(diag_elements_V)), atol=1e-6)

        # Check off-diagonal elements are small
        off_diag_mask_V = ~np.eye(VH_V_float.shape[0], dtype=bool)
        off_diag_elements_V = VH_V_float[off_diag_mask_V]
        assert np.max(np.abs(off_diag_elements_V)) < 1e-6


class TestQSVDReconstructionMonotonicity:
    """Test cases for Q-SVD reconstruction error monotonicity."""

    def test_reconstruction_error_monotonicity(self):
        """
        Test that reconstruction error decreases monotonically as rank increases.
        This validates our singular value mapping for truncated Q-SVD.
        """
        np.random.seed(42)

        # Test with different matrix sizes
        test_cases = [
            (4, 3),  # Rectangular matrix
            (5, 5),  # Square matrix
            (6, 4),  # Another rectangular matrix
        ]

        for m, n in test_cases:
            print(f"\n--- Testing matrix size {m}x{n} ---")
            min_dim = min(m, n)

            # Create random quaternion matrix
            X_components = np.random.randn(m, n, 4)
            X = quaternion.as_quat_array(X_components)
            original_norm = quat_frobenius_norm(X)

            print(f"Original matrix norm: {original_norm:.6f}")
            print(f"Testing ranks: 1 to {min_dim}")

            # Test all ranks from 1 to min_dim
            reconstruction_errors = []
            singular_values_list = []

            for r in range(1, min_dim + 1):
                # Compute truncated Q-SVD
                U, s, V = classical_qsvd(X, r)

                # Check shapes
                assert U.shape == (m, r), f"U shape should be ({m}, {r}), got {U.shape}"
                assert V.shape == (n, r), f"V shape should be ({n}, {r}), got {V.shape}"
                assert len(s) == r, f"s length should be {r}, got {len(s)}"

                # Check singular values are non-negative and decreasing
                assert np.all(s >= 0), f"Singular values should be non-negative: {s}"
                if r > 1:
                    assert np.all(np.diff(s) <= 0), (
                        f"Singular values should be decreasing: {s}"
                    )

                # Compute reconstruction
                S_diag = np.diag(s)
                X_recon = quat_matmat(quat_matmat(U, S_diag), quat_hermitian(V))

                # Calculate reconstruction error
                reconstruction_error = quat_frobenius_norm(X - X_recon)
                relative_error = reconstruction_error / original_norm

                reconstruction_errors.append(reconstruction_error)
                singular_values_list.append(s.copy())

                print(
                    f"  Rank {r}: error = {reconstruction_error:.6f} ({relative_error:.6f})"
                )
                print(f"    Singular values: {s}")

            # Validate monotonicity: error should decrease as rank increases
            print(
                f"  Reconstruction errors: {[f'{e:.6f}' for e in reconstruction_errors]}"
            )

            # Check that errors are monotonically decreasing
            for i in range(1, len(reconstruction_errors)):
                assert reconstruction_errors[i] <= reconstruction_errors[i - 1], (
                    f"Error should decrease: {reconstruction_errors[i - 1]:.6f} -> {reconstruction_errors[i]:.6f}"
                )

            # Check that full rank gives perfect reconstruction
            assert reconstruction_errors[-1] < 1e-10, (
                f"Full rank should give perfect reconstruction, error: {reconstruction_errors[-1]:.6f}"
            )

            # Check that rank 1 gives the largest error
            assert reconstruction_errors[0] > 1e-10, (
                f"Rank 1 should have some error, got: {reconstruction_errors[0]:.6f}"
            )

            print(f"  ✅ Monotonicity validated for {m}x{n} matrix")

            # Additional validation: check singular values consistency
            print("  Validating singular value consistency...")
            for r in range(1, min_dim):
                # Singular values for rank r should be prefix of singular values for rank r+1
                assert np.allclose(
                    singular_values_list[r - 1], singular_values_list[r][:r]
                ), f"Singular values for rank {r} should be prefix of rank {r + 1}"

            print("  ✅ Singular value consistency validated")

    def test_singular_value_mapping_validation(self):
        """
        Specifically test that our singular value mapping works correctly
        by comparing with full Q-SVD results.
        """
        np.random.seed(42)

        # Test with a larger matrix to see more singular values
        m, n = 8, 6
        min_dim = min(m, n)

        # Create random quaternion matrix
        X_components = np.random.randn(m, n, 4)
        X = quaternion.as_quat_array(X_components)

        print(f"\n--- Testing singular value mapping validation ({m}x{n}) ---")

        # Get full Q-SVD
        U_full, s_full, V_full = classical_qsvd_full(X)
        print(f"Full Q-SVD singular values: {s_full}")

        # Test each rank and verify singular values match
        for r in range(1, min_dim + 1):
            U_trunc, s_trunc, V_trunc = classical_qsvd(X, r)

            # Check that truncated singular values match the first r values from full SVD
            s_full_prefix = s_full[:r]
            diff = np.linalg.norm(s_trunc - s_full_prefix)

            print(
                f"  Rank {r}: truncated = {s_trunc}, full_prefix = {s_full_prefix}, diff = {diff:.2e}"
            )

            assert diff < 1e-12, (
                f"Singular values for rank {r} don't match full SVD prefix"
            )

        print("  ✅ Singular value mapping validated for all ranks")


class TestQSVDTruncation:
    """Test cases for Q-SVD truncation behavior."""

    def test_truncated_vs_full_qsvd(self):
        """Test truncated vs full Q-SVD reconstruction."""
        np.random.seed(42)
        m, n = 6, 4

        # Create random quaternion matrix
        X_components = np.random.randn(m, n, 4)
        X = quaternion.as_quat_array(X_components)

        # Get full Q-SVD
        U_full, s_full, V_full = classical_qsvd_full(X)

        # Test different truncation ranks
        for R in [1, 2, 3, 4]:
            print(f"\n--- Testing rank R = {R} ---")

            # Get truncated Q-SVD
            U_trunc, s_trunc, V_trunc = classical_qsvd(X, R)

            # Check shapes
            assert U_trunc.shape == (m, R)
            assert V_trunc.shape == (n, R)
            assert len(s_trunc) == R

            # Method 1: Truncated reconstruction (U_trunc @ diag(s) @ V_trunc^H)
            S_diag = np.diag(s_trunc)
            X_recon_trunc = quat_matmat(
                quat_matmat(U_trunc, S_diag), quat_hermitian(V_trunc)
            )

            # Method 2: Full reconstruction with truncation (U_full @ Σ @ V_full^H)
            Sigma_full = np.zeros((m, n))
            Sigma_full[:R, :R] = np.diag(s_trunc)
            X_recon_full = quat_matmat(
                quat_matmat(U_full, Sigma_full), quat_hermitian(V_full)
            )

            # Calculate errors
            error_trunc = quat_frobenius_norm(X - X_recon_trunc)
            error_full = quat_frobenius_norm(X - X_recon_full)
            original_norm = quat_frobenius_norm(X)

            print(f"Truncated method error: {error_trunc:.6f}")
            print(f"Full method error: {error_full:.6f}")
            print(f"Relative error (truncated): {error_trunc / original_norm:.6f}")
            print(f"Relative error (full): {error_full / original_norm:.6f}")

            # Both methods should give the same result
            assert np.abs(error_trunc - error_full) < 1e-12, (
                "Methods should give same result"
            )

            # Check that higher rank gives better reconstruction
            if R == n:  # Full rank
                assert error_trunc < 1e-10, "Full rank should have perfect reconstruction"
            else:  # Low rank
                assert error_trunc > 1e-10, "Low rank should have some error"


class TestLargeRandomMatrices:
    """Test cases for larger random quaternion matrices."""

    def test_qr_large_random(self):
        """Test QR decomposition on large random quaternion matrix."""
        np.random.seed(42)
        m, n = 12, 8

        # Create large random quaternion matrix
        X_components = np.random.randn(m, n, 4)
        X = quaternion.as_quat_array(X_components)

        Q, R = qr_qua(X)

        # Check shapes
        assert Q.shape == (m, n)
        assert R.shape == (n, n)

        # Check reconstruction accuracy
        X_recon = quat_matmat(Q, R)
        reconstruction_error = quat_frobenius_norm(X - X_recon)
        original_norm = quat_frobenius_norm(X)
        relative_error = reconstruction_error / original_norm
        assert relative_error < 1e-12, f"Relative reconstruction error: {relative_error}"

        # Check orthonormality of Q
        QH_Q = quat_matmat(quat_hermitian(Q), Q)
        QH_Q_float = quaternion.as_float_array(QH_Q)

        # Check diagonal elements are 1
        diag_elements = np.diag(QH_Q_float[:, :, 0])
        assert np.allclose(diag_elements, np.ones(len(diag_elements)), atol=1e-12)

        # Check off-diagonal elements are small
        off_diag_mask = ~np.eye(QH_Q_float.shape[0], dtype=bool)
        off_diag_elements = QH_Q_float[off_diag_mask]
        assert np.max(np.abs(off_diag_elements)) < 1e-12

        # Check R is upper triangular
        R_float = quaternion.as_float_array(R)
        R_real = R_float[:, :, 0]  # Real part
        assert np.allclose(R_real, np.triu(R_real), atol=1e-12)

    def test_qsvd_large_random_full_rank(self):
        """Test Q-SVD on large random quaternion matrix with full rank."""
        np.random.seed(42)
        m, n = 10, 6

        # Create large random quaternion matrix
        X_components = np.random.randn(m, n, 4)
        X = quaternion.as_quat_array(X_components)

        # Test full rank
        R = min(m, n)
        U, s, V = classical_qsvd(X, R)

        # Check shapes
        assert U.shape == (m, R)
        assert len(s) == R
        assert V.shape == (n, R)

        # Check singular values are non-negative and decreasing
        assert np.all(s >= 0)
        assert np.all(np.diff(s) <= 0)

        # Check reconstruction accuracy
        S = np.diag(s)
        X_recon = quat_matmat(quat_matmat(U, S), quat_hermitian(V))
        reconstruction_error = quat_frobenius_norm(X - X_recon)
        original_norm = quat_frobenius_norm(X)
        relative_error = reconstruction_error / original_norm

        # Should have good reconstruction for full rank
        assert relative_error < 0.1, f"Reconstruction error too large: {relative_error}"

        # Check orthonormality of U and V
        UH_U = quat_matmat(quat_hermitian(U), U)
        VH_V = quat_matmat(quat_hermitian(V), V)

        UH_U_float = quaternion.as_float_array(UH_U)
        VH_V_float = quaternion.as_float_array(VH_V)

        # Check diagonal elements are 1
        assert np.allclose(np.diag(UH_U_float[:, :, 0]), np.ones(R), atol=1e-6)
        assert np.allclose(np.diag(VH_V_float[:, :, 0]), np.ones(R), atol=1e-6)

        # Check off-diagonal elements are small
        off_diag_U = UH_U_float[~np.eye(R, dtype=bool)]
        off_diag_V = VH_V_float[~np.eye(R, dtype=bool)]
        assert np.max(np.abs(off_diag_U)) < 1e-6
        assert np.max(np.abs(off_diag_V)) < 1e-6

    def test_qsvd_large_random_low_rank(self):
        """Test Q-SVD on large random quaternion matrix with low rank approximation."""
        np.random.seed(42)
        m, n = 15, 10

        # Create large random quaternion matrix
        X_components = np.random.randn(m, n, 4)
        X = quaternion.as_quat_array(X_components)

        # Test low rank approximation
        R = 3  # Low rank
        U, s, V = classical_qsvd(X, R)

        # Check shapes
        assert U.shape == (m, R)
        assert len(s) == R
        assert V.shape == (n, R)

        # Check singular values are non-negative and decreasing
        assert np.all(s >= 0)
        assert np.all(np.diff(s) <= 0)

        # Check reconstruction accuracy (should be worse for low rank)
        S = np.diag(s)
        X_recon = quat_matmat(quat_matmat(U, S), quat_hermitian(V))
        reconstruction_error = quat_frobenius_norm(X - X_recon)
        original_norm = quat_frobenius_norm(X)
        relative_error = reconstruction_error / original_norm

        # Low rank should have higher error but still reasonable
        assert relative_error < 0.8, f"Reconstruction error too large: {relative_error}"

        # Check orthonormality of U and V
        UH_U = quat_matmat(quat_hermitian(U), U)
        VH_V = quat_matmat(quat_hermitian(V), V)

        UH_U_float = quaternion.as_float_array(UH_U)
        VH_V_float = quaternion.as_float_array(VH_V)

        # Check diagonal elements are 1
        assert np.allclose(np.diag(UH_U_float[:, :, 0]), np.ones(R), atol=1e-6)
        assert np.allclose(np.diag(VH_V_float[:, :, 0]), np.ones(R), atol=1e-6)

        # Check off-diagonal elements are small
        off_diag_U = UH_U_float[~np.eye(R, dtype=bool)]
        off_diag_V = VH_V_float[~np.eye(R, dtype=bool)]
        assert np.max(np.abs(off_diag_U)) < 1e-6
        assert np.max(np.abs(off_diag_V)) < 1e-6

    def test_qsvd_square_matrix(self):
        """Test Q-SVD on square quaternion matrix."""
        np.random.seed(42)
        n = 8

        # Create square random quaternion matrix
        X_components = np.random.randn(n, n, 4)
        X = quaternion.as_quat_array(X_components)

        # Test different ranks
        for R in [2, 4, 6, 8]:
            U, s, V = classical_qsvd(X, R)

            # Check shapes
            assert U.shape == (n, R)
            assert len(s) == R
            assert V.shape == (n, R)

            # Check singular values are non-negative and decreasing
            assert np.all(s >= 0)
            assert np.all(np.diff(s) <= 0)

            # Check reconstruction accuracy
            S = np.diag(s)
            X_recon = quat_matmat(quat_matmat(U, S), quat_hermitian(V))
            reconstruction_error = quat_frobenius_norm(X - X_recon)
            original_norm = quat_frobenius_norm(X)
            relative_error = reconstruction_error / original_norm

            # Higher rank should have better reconstruction
            if R == n:  # Full rank
                assert relative_error < 0.1, (
                    f"Full rank reconstruction error: {relative_error}"
                )
            else:  # Low rank
                assert relative_error < 0.8, (
                    f"Rank {R} reconstruction error: {relative_error}"
                )

            # Check orthonormality
            UH_U = quat_matmat(quat_hermitian(U), U)
            VH_V = quat_matmat(quat_hermitian(V), V)

            UH_U_float = quaternion.as_float_array(UH_U)
            VH_V_float = quaternion.as_float_array(VH_V)

            assert np.allclose(np.diag(UH_U_float[:, :, 0]), np.ones(R), atol=1e-6)
            assert np.allclose(np.diag(VH_V_float[:, :, 0]), np.ones(R), atol=1e-6)
