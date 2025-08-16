#!/usr/bin/env python3
"""
Test basic Q-GMRES functions
"""

import os
import sys

import numpy as np
import quaternion
from scipy import sparse

# Add core to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "quatica"))

from utils import (
    GRSGivens,
    Hess_QR_ggivens,
    Realp,
    UtriangleQsparse,
    absQsparse,
    dotinvQsparse,
    ggivens,
    normQ,
    normQsparse,
    quat_frobenius_norm,
    quat_matmat,
    timesQsparse,
)


def test_normQsparse_basic():
    """Test basic normQsparse functionality"""
    print("Testing normQsparse basic functionality...")

    # Create simple test matrices
    A0 = np.array([[1, 0], [0, 1]])
    A1 = np.array([[0, 1], [-1, 0]])
    A2 = np.array([[0, 0], [0, 0]])
    A3 = np.array([[0, 0], [0, 0]])

    # Test default (Frobenius norm)
    norm_fro = normQsparse(A0, A1, A2, A3)
    expected_fro = 2.0  # sqrt(1^2 + 0^2 + 0^2 + 0^2 + 0^2 + 1^2 + 0^2 + 0^2 + 0^2 + 1^2 + 0^2 + 0^2 + (-1)^2 + 0^2 + 0^2 + 0^2) = sqrt(4) = 2.0

    assert np.abs(norm_fro - expected_fro) < 1e-10
    print(f"✓ Frobenius norm: {norm_fro:.6f} (expected: {expected_fro:.6f})")

    # Test 2-norm
    norm_2 = normQsparse(A0, A1, A2, A3, "2")
    assert norm_2 > 0
    print(f"✓ 2-norm: {norm_2:.6f}")

    # Test 1-norm
    norm_1 = normQsparse(A0, A1, A2, A3, "1")
    assert norm_1 > 0
    print(f"✓ 1-norm: {norm_1:.6f}")

    # Test dual norm
    norm_d = normQsparse(A0, A1, A2, A3, "d")
    assert norm_d > 0
    print(f"✓ Dual norm: {norm_d:.6f}")


def test_normQ_basic():
    """Test basic normQ functionality"""
    print("\nTesting normQ basic functionality...")

    # Create quaternion matrix
    A = np.array(
        [
            [quaternion.quaternion(1, 0, 0, 0), quaternion.quaternion(0, 1, 0, 0)],
            [quaternion.quaternion(0, 0, 1, 0), quaternion.quaternion(0, 0, 0, 1)],
        ],
        dtype=np.quaternion,
    )

    # Test default (Frobenius norm)
    norm_fro = normQ(A)
    expected_fro = 2.0  # sqrt(1^2 + 1^2 + 1^2 + 1^2)
    assert np.abs(norm_fro - expected_fro) < 1e-10
    print(f"✓ Frobenius norm: {norm_fro:.6f} (expected: {expected_fro:.6f})")

    # Test that it matches quat_frobenius_norm
    norm_fro_alt = quat_frobenius_norm(A)
    assert np.abs(norm_fro - norm_fro_alt) < 1e-10
    print(f"✓ Matches quat_frobenius_norm: {norm_fro_alt:.6f}")


def test_normQsparse_sparse():
    """Test normQsparse with sparse matrices"""
    print("\nTesting normQsparse with sparse matrices...")

    # Create sparse test matrices
    A0 = sparse.csr_matrix(np.array([[1, 0], [0, 1]]))
    A1 = sparse.csr_matrix(np.array([[0, 1], [-1, 0]]))
    A2 = sparse.csr_matrix(np.array([[0, 0], [0, 0]]))
    A3 = sparse.csr_matrix(np.array([[0, 0], [0, 0]]))

    # Test default (Frobenius norm)
    norm_fro = normQsparse(A0, A1, A2, A3)
    expected_fro = 2.0  # Same as dense case

    assert np.abs(norm_fro - expected_fro) < 1e-10
    print(f"✓ Sparse Frobenius norm: {norm_fro:.6f} (expected: {expected_fro:.6f})")


def test_normQsparse_vector():
    """Test normQsparse with vectors (1D arrays)"""
    print("\nTesting normQsparse with vectors...")

    # Create test vectors
    v0 = np.array([1, 0, 0])
    v1 = np.array([0, 1, 0])
    v2 = np.array([0, 0, 1])
    v3 = np.array([0, 0, 0])

    # Test default (Frobenius norm)
    norm_fro = normQsparse(v0, v1, v2, v3)
    expected_fro = np.sqrt(
        3
    )  # sqrt(1^2 + 1^2 + 1^2) for stacked vector [1 0 0 0 0 1 0 1 0 0 0 0]

    assert np.abs(norm_fro - expected_fro) < 1e-10
    print(f"✓ Vector Frobenius norm: {norm_fro:.6f} (expected: {expected_fro:.6f})")


def test_timesQsparse_basic():
    """Test basic timesQsparse functionality"""
    print("\nTesting timesQsparse basic functionality...")

    # Create simple test matrices (2x2)
    B0 = np.array([[1, 0], [0, 1]])
    B1 = np.array([[0, 1], [-1, 0]])
    B2 = np.array([[0, 0], [0, 0]])
    B3 = np.array([[0, 0], [0, 0]])

    C0 = np.array([[1, 0], [0, 1]])
    C1 = np.array([[0, 0], [0, 0]])
    C2 = np.array([[0, 0], [0, 0]])
    C3 = np.array([[0, 0], [0, 0]])

    # Test multiplication: B * C
    A0, A1, A2, A3 = timesQsparse(B0, B1, B2, B3, C0, C1, C2, C3)

    # Expected result: A should be the same as B since C is identity-like
    expected_A0 = np.array([[1, 0], [0, 1]])
    expected_A1 = np.array([[0, 1], [-1, 0]])
    expected_A2 = np.array([[0, 0], [0, 0]])
    expected_A3 = np.array([[0, 0], [0, 0]])

    assert np.allclose(A0, expected_A0)
    assert np.allclose(A1, expected_A1)
    assert np.allclose(A2, expected_A2)
    assert np.allclose(A3, expected_A3)

    print(f"✓ Basic multiplication: A0 shape {A0.shape}, A1 shape {A1.shape}")


def test_timesQsparse_identity():
    """Test timesQsparse with identity matrices"""
    print("\nTesting timesQsparse with identity matrices...")

    # Create identity matrices
    I0 = np.array([[1, 0], [0, 1]])
    I1 = np.array([[0, 0], [0, 0]])
    I2 = np.array([[0, 0], [0, 0]])
    I3 = np.array([[0, 0], [0, 0]])

    # Test: I * I = I
    A0, A1, A2, A3 = timesQsparse(I0, I1, I2, I3, I0, I1, I2, I3)

    assert np.allclose(A0, I0)
    assert np.allclose(A1, I1)
    assert np.allclose(A2, I2)
    assert np.allclose(A3, I3)

    print("✓ Identity multiplication: I * I = I")


def test_timesQsparse_sparse():
    """Test timesQsparse with sparse matrices"""
    print("\nTesting timesQsparse with sparse matrices...")

    # Create sparse test matrices
    B0 = sparse.csr_matrix(np.array([[1, 0], [0, 1]]))
    B1 = sparse.csr_matrix(np.array([[0, 1], [-1, 0]]))
    B2 = sparse.csr_matrix(np.array([[0, 0], [0, 0]]))
    B3 = sparse.csr_matrix(np.array([[0, 0], [0, 0]]))

    C0 = sparse.csr_matrix(np.array([[1, 0], [0, 1]]))
    C1 = sparse.csr_matrix(np.array([[0, 0], [0, 0]]))
    C2 = sparse.csr_matrix(np.array([[0, 0], [0, 0]]))
    C3 = sparse.csr_matrix(np.array([[0, 0], [0, 0]]))

    # Test multiplication
    A0, A1, A2, A3 = timesQsparse(B0, B1, B2, B3, C0, C1, C2, C3)

    # Results should be dense numpy arrays
    assert isinstance(A0, np.ndarray)
    assert isinstance(A1, np.ndarray)
    assert isinstance(A2, np.ndarray)
    assert isinstance(A3, np.ndarray)

    # Check results
    expected_A0 = np.array([[1, 0], [0, 1]])
    expected_A1 = np.array([[0, 1], [-1, 0]])

    assert np.allclose(A0, expected_A0)
    assert np.allclose(A1, expected_A1)
    assert np.allclose(A2, np.zeros((2, 2)))
    assert np.allclose(A3, np.zeros((2, 2)))

    print("✓ Sparse matrix multiplication")


def test_timesQsparse_quaternion_consistency():
    """Test that timesQsparse gives same results as quat_matmat"""
    print("\nTesting timesQsparse consistency with quat_matmat...")

    # Create quaternion matrices
    B = np.array(
        [
            [quaternion.quaternion(1, 0, 0, 0), quaternion.quaternion(0, 1, 0, 0)],
            [quaternion.quaternion(0, 0, 1, 0), quaternion.quaternion(0, 0, 0, 1)],
        ],
        dtype=np.quaternion,
    )

    C = np.array(
        [
            [quaternion.quaternion(1, 0, 0, 0), quaternion.quaternion(0, 0, 0, 0)],
            [quaternion.quaternion(0, 0, 0, 0), quaternion.quaternion(1, 0, 0, 0)],
        ],
        dtype=np.quaternion,
    )

    # Convert to component format
    B_comp = quaternion.as_float_array(B)
    C_comp = quaternion.as_float_array(C)
    B0, B1, B2, B3 = B_comp[..., 0], B_comp[..., 1], B_comp[..., 2], B_comp[..., 3]
    C0, C1, C2, C3 = C_comp[..., 0], C_comp[..., 1], C_comp[..., 2], C_comp[..., 3]

    # Use timesQsparse
    A0, A1, A2, A3 = timesQsparse(B0, B1, B2, B3, C0, C1, C2, C3)

    # Use quat_matmat
    A_quat = quat_matmat(B, C)
    A_quat_comp = quaternion.as_float_array(A_quat)
    A_quat0, A_quat1, A_quat2, A_quat3 = (
        A_quat_comp[..., 0],
        A_quat_comp[..., 1],
        A_quat_comp[..., 2],
        A_quat_comp[..., 3],
    )

    # Compare results
    assert np.allclose(A0, A_quat0)
    assert np.allclose(A1, A_quat1)
    assert np.allclose(A2, A_quat2)
    assert np.allclose(A3, A_quat3)

    print("✓ Consistency with quat_matmat verified")


def test_Realp_basic():
    """Test basic Realp functionality"""
    print("\nTesting Realp basic functionality...")

    # Create simple 1x1 test matrices
    A1 = np.array([[1]])
    A2 = np.array([[0]])
    A3 = np.array([[0]])
    A4 = np.array([[0]])

    AR = Realp(A1, A2, A3, A4)

    # Expected result: 4x4 identity-like matrix
    expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    assert np.allclose(AR, expected)
    print(f"✓ Realp basic: AR shape {AR.shape}")


def test_ggivens_basic():
    """Test basic ggivens functionality"""
    print("\nTesting ggivens basic functionality...")

    # Test with simple vectors
    x1 = np.array([1, 0, 0, 0])
    x2 = np.array([0, 1, 0, 0])

    G = ggivens(x1, x2)

    # G should be an 8x8 matrix (quaternion Givens rotation)
    assert G.shape == (8, 8)

    # For the simplified version, just check that it's a valid matrix
    # The full quaternion Givens rotation would require more complex testing
    assert np.allclose(G @ G.T, np.eye(8))  # Should be orthogonal

    print(f"✓ ggivens basic: G shape {G.shape}, orthogonal matrix verified")


def test_Hess_QR_ggivens_basic():
    """Test basic Hess_QR_ggivens functionality"""
    print("\nTesting Hess_QR_ggivens basic functionality...")

    # Create a simple 2x2 Hessenberg matrix in real block format
    # Hess = [A0; A1; A2; A3] where each component is 2x2
    A0 = np.array([[1, 0], [0, 1]])
    A1 = np.array([[0, 0], [0, 0]])
    A2 = np.array([[0, 0], [0, 0]])
    A3 = np.array([[0, 0], [0, 0]])

    # Create the real block matrix Hess = [A0; A1; A2; A3]
    Hess = np.vstack([A0, A1, A2, A3])  # 8x2 matrix

    W, Hess_new = Hess_QR_ggivens(Hess)

    # Check shapes
    assert W.shape == (2, 8)  # m x 4m
    assert Hess_new.shape == (2, 8)  # m x 4n

    print(f"✓ Hess_QR_ggivens basic: W shape {W.shape}, Hess_new shape {Hess_new.shape}")


def test_absQsparse_basic():
    """Test basic absQsparse functionality"""
    print("\nTesting absQsparse basic functionality...")

    # Test with simple quaternion
    A0, A1, A2, A3 = 1.0, 0.0, 0.0, 0.0
    r, s0, s1, s2, s3 = absQsparse(A0, A1, A2, A3)

    assert np.abs(r - 1.0) < 1e-10
    assert np.abs(s0 - 1.0) < 1e-10
    assert np.abs(s1) < 1e-10
    assert np.abs(s2) < 1e-10
    assert np.abs(s3) < 1e-10
    print("✓ absQsparse with unit quaternion")

    # Test with non-unit quaternion
    A0, A1, A2, A3 = 3.0, 4.0, 0.0, 0.0
    r, s0, s1, s2, s3 = absQsparse(A0, A1, A2, A3)

    assert np.abs(r - 5.0) < 1e-10  # sqrt(3^2 + 4^2) = 5
    assert np.abs(s0 - 0.6) < 1e-10  # 3/5 = 0.6
    assert np.abs(s1 - 0.8) < 1e-10  # 4/5 = 0.8
    print("✓ absQsparse with non-unit quaternion")


def test_dotinvQsparse_basic():
    """Test basic dotinvQsparse functionality"""
    print("\nTesting dotinvQsparse basic functionality...")

    # Test with unit quaternion
    A0, A1, A2, A3 = 1.0, 0.0, 0.0, 0.0
    inv0, inv1, inv2, inv3 = dotinvQsparse(A0, A1, A2, A3)

    assert np.abs(inv0 - 1.0) < 1e-10
    assert np.abs(inv1) < 1e-10
    assert np.abs(inv2) < 1e-10
    assert np.abs(inv3) < 1e-10
    print("✓ dotinvQsparse with unit quaternion")

    # Test with non-unit quaternion
    A0, A1, A2, A3 = 2.0, 0.0, 0.0, 0.0
    inv0, inv1, inv2, inv3 = dotinvQsparse(A0, A1, A2, A3)

    assert np.abs(inv0 - 0.5) < 1e-10  # 2/4 = 0.5
    assert np.abs(inv1) < 1e-10
    assert np.abs(inv2) < 1e-10
    assert np.abs(inv3) < 1e-10
    print("✓ dotinvQsparse with non-unit quaternion")


def test_UtriangleQsparse_basic():
    """Test basic UtriangleQsparse functionality"""
    print("\nTesting UtriangleQsparse basic functionality...")

    # Create a simple 2x2 upper triangular system
    R0 = np.array([[1.0, 1.0], [0.0, 1.0]])
    R1 = np.array([[0.0, 0.0], [0.0, 0.0]])
    R2 = np.array([[0.0, 0.0], [0.0, 0.0]])
    R3 = np.array([[0.0, 0.0], [0.0, 0.0]])

    b0 = np.array([[2.0], [1.0]])  # Right-hand side
    b1 = np.array([[0.0], [0.0]])
    b2 = np.array([[0.0], [0.0]])
    b3 = np.array([[0.0], [0.0]])

    # Solve R * x = b
    x0, x1, x2, x3 = UtriangleQsparse(R0, R1, R2, R3, b0, b1, b2, b3)

    # Expected solution: x = [1, 1] (since R is identity-like)
    assert np.abs(x0[0, 0] - 1.0) < 1e-10
    assert np.abs(x0[1, 0] - 1.0) < 1e-10
    assert np.allclose(x1, 0)
    assert np.allclose(x2, 0)
    assert np.allclose(x3, 0)

    print("✓ UtriangleQsparse solves simple upper triangular system")


def test_GRSGivens_basic():
    """Test basic GRSGivens functionality"""
    print("\nTesting GRSGivens basic functionality...")

    # Test case 1: GRSGivens(g1, g2, g3, g4) with zeros
    G1 = GRSGivens(1, 0, 0, 0)
    assert np.allclose(G1, np.eye(4))
    print("✓ GRSGivens with zeros returns identity")

    # Test case 2: GRSGivens(g1, g2, g3, g4) with non-zeros
    G2 = GRSGivens(1, 1, 0, 0)
    assert G2.shape == (4, 4)
    # Check that it's a rotation matrix (orthogonal)
    assert np.allclose(G2 @ G2.T, np.eye(4))
    print("✓ GRSGivens with non-zeros returns valid rotation matrix")

    # Test case 3: GRSGivens(g1) with vector input
    g1 = np.array([1, 0, 0, 0])
    G3 = GRSGivens(g1)
    assert np.allclose(G3, np.eye(4))
    print("✓ GRSGivens with vector input works correctly")


def main():
    """Run all tests"""
    print("=" * 60)
    print("TESTING BASIC Q-GMRES FUNCTIONS")
    print("=" * 60)

    try:
        test_normQsparse_basic()
        test_normQ_basic()
        test_normQsparse_sparse()
        test_normQsparse_vector()
        test_timesQsparse_basic()
        test_timesQsparse_identity()
        test_timesQsparse_sparse()
        test_timesQsparse_quaternion_consistency()
        test_Realp_basic()
        test_ggivens_basic()
        test_GRSGivens_basic()
        test_Hess_QR_ggivens_basic()
        test_absQsparse_basic()
        test_dotinvQsparse_basic()
        test_UtriangleQsparse_basic()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()
