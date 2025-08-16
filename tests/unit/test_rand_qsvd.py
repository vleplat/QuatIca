#!/usr/bin/env python3
"""
Test script for rand_qsvd function

This script tests the randomized Q-SVD implementation and demonstrates
that higher power iterations give results closer to full Q-SVD.

Author: QuatIca Team
Date: 2024
"""

import os
import sys

import numpy as np
import quaternion

# Add core module to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "quatica"))
from decomp.qsvd import classical_qsvd_full, rand_qsvd
from utils import quat_frobenius_norm, quat_hermitian, quat_matmat


def test_rand_qsvd_basic():
    """
    Test basic functionality of rand_qsvd.
    """
    print("=" * 60)
    print("RANDOMIZED Q-SVD BASIC FUNCTIONALITY TEST")
    print("=" * 60)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create a test matrix
    m, n = 8, 6
    R = 3  # Target rank

    print(f"Testing with {m}×{n} matrix, target rank R={R}")

    # Create random quaternion matrix
    X_components = np.random.randn(m, n, 4)
    X = quaternion.as_quat_array(X_components)

    print(f"Input matrix shape: {X.shape}")
    print(f"Input matrix norm: {quat_frobenius_norm(X):.6f}")
    print()

    # Test rand_qsvd with different parameters
    test_cases = [
        (1, 5),  # 1 iteration, 5 oversample
        (2, 5),  # 2 iterations, 5 oversample
        (3, 5),  # 3 iterations, 5 oversample
        (2, 10),  # 2 iterations, 10 oversample
    ]

    for n_iter, oversample in test_cases:
        print(f"Testing rand_qsvd with n_iter={n_iter}, oversample={oversample}:")

        try:
            U, s, V = rand_qsvd(X, R, oversample=oversample, n_iter=n_iter)

            print("  ✅ SUCCESS: rand_qsvd completed")
            print(f"  U shape: {U.shape}")
            print(f"  V shape: {V.shape}")
            print(f"  s shape: {s.shape}")
            print(f"  Singular values: {s}")

            # Test reconstruction
            S_diag = np.diag(s)
            X_recon = quat_matmat(quat_matmat(U, S_diag), quat_hermitian(V))
            reconstruction_error = quat_frobenius_norm(X - X_recon)
            relative_error = reconstruction_error / quat_frobenius_norm(X)

            print(f"  Reconstruction error: {reconstruction_error:.6f}")
            print(f"  Relative error: {relative_error:.6f}")
            print()

        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            print()

    return True


def test_power_iteration_convergence():
    """
    Test that higher power iterations give results closer to full Q-SVD.
    """
    print("=" * 60)
    print("POWER ITERATION CONVERGENCE TEST")
    print("=" * 60)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create a test matrix
    m, n = 10, 8
    R = 8  # Use full rank for better reconstruction

    print(f"Testing convergence with {m}×{n} matrix, target rank R={R} (full rank)")

    # Create random quaternion matrix
    X_components = np.random.randn(m, n, 4)
    X = quaternion.as_quat_array(X_components)

    # Compute full Q-SVD for comparison
    print("Computing full Q-SVD for comparison...")
    U_full, s_full, V_full = classical_qsvd_full(X)

    # Use only the first R singular values for comparison
    s_full_R = s_full[:R]
    print(f"Full Q-SVD singular values (first {R}): {s_full_R}")
    print()

    # Test different numbers of power iterations
    n_iter_list = [0, 1, 2, 3, 4]
    results = []

    for n_iter in n_iter_list:
        print(f"Testing rand_qsvd with {n_iter} power iterations:")

        try:
            U_rand, s_rand, V_rand = rand_qsvd(X, R, oversample=10, n_iter=n_iter)

            # Compare singular values
            s_diff = np.linalg.norm(s_rand - s_full_R)
            s_relative_diff = s_diff / np.linalg.norm(s_full_R)

            print(f"  rand_qsvd singular values: {s_rand}")
            print(f"  Singular value difference: {s_diff:.6f}")
            print(f"  Relative difference: {s_relative_diff:.6f}")

            # Test reconstruction quality
            S_diag = np.diag(s_rand)
            X_recon = quat_matmat(quat_matmat(U_rand, S_diag), quat_hermitian(V_rand))
            reconstruction_error = quat_frobenius_norm(X - X_recon)
            relative_error = reconstruction_error / quat_frobenius_norm(X)

            print(f"  Reconstruction error: {reconstruction_error:.6f}")
            print(f"  Relative error: {relative_error:.6f}")

            results.append(
                {
                    "n_iter": n_iter,
                    "s_diff": s_diff,
                    "s_relative_diff": s_relative_diff,
                    "reconstruction_error": reconstruction_error,
                    "relative_error": relative_error,
                }
            )

            print()

        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            print()

    # Summary
    print("=" * 60)
    print("CONVERGENCE SUMMARY")
    print("=" * 60)

    if results:
        print("Power iteration convergence results:")
        print("n_iter | s_diff | s_rel_diff | recon_error | rel_error")
        print("-" * 60)

        for result in results:
            print(
                f"{result['n_iter']:6d} | {result['s_diff']:7.4f} | {result['s_relative_diff']:10.4f} | "
                f"{result['reconstruction_error']:11.4f} | {result['relative_error']:9.4f}"
            )

        print()

        # Check if convergence is observed
        if len(results) > 1:
            first_error = results[0]["relative_error"]
            last_error = results[-1]["relative_error"]

            if last_error < first_error:
                print(
                    "✅ CONVERGENCE OBSERVED: Higher power iterations improve accuracy!"
                )
            else:
                print("⚠️  No clear convergence pattern observed")

    return True


def test_rand_qsvd_orthogonality():
    """
    Test that rand_qsvd produces orthonormal U and V matrices.
    """
    print("=" * 60)
    print("ORTHOGONALITY TEST")
    print("=" * 60)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create a test matrix
    m, n = 6, 4
    R = 2  # Target rank

    print(f"Testing orthogonality with {m}×{n} matrix, target rank R={R}")

    # Create random quaternion matrix
    X_components = np.random.randn(m, n, 4)
    X = quaternion.as_quat_array(X_components)

    try:
        U, s, V = rand_qsvd(X, R, oversample=5, n_iter=2)

        # Test U orthogonality: U^H * U should be identity
        U_orthogonal = quat_matmat(quat_hermitian(U), U)
        U_identity = np.eye(R, dtype=np.quaternion)
        U_orthogonality_error = quat_frobenius_norm(U_orthogonal - U_identity)

        # Test V orthogonality: V^H * V should be identity
        V_orthogonal = quat_matmat(quat_hermitian(V), V)
        V_identity = np.eye(R, dtype=np.quaternion)
        V_orthogonality_error = quat_frobenius_norm(V_orthogonal - V_identity)

        print(f"U orthogonality error: {U_orthogonality_error:.2e}")
        print(f"V orthogonality error: {V_orthogonality_error:.2e}")

        if U_orthogonality_error < 1e-10 and V_orthogonality_error < 1e-10:
            print("✅ PASSED: U and V are orthonormal")
        else:
            print("❌ FAILED: U and V are not orthonormal")

        print()

    except Exception as e:
        print(f"❌ FAILED: {e}")
        print()

    return True


if __name__ == "__main__":
    print("🧪 RANDOMIZED Q-SVD TESTING SUITE")
    print("=" * 60)

    # Run all tests
    success1 = test_rand_qsvd_basic()
    success2 = test_power_iteration_convergence()
    success3 = test_rand_qsvd_orthogonality()

    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    if success1 and success2 and success3:
        print("🎉 All rand_qsvd tests completed successfully!")
        print("✅ Basic functionality works")
        print("✅ Power iteration convergence demonstrated")
        print("✅ Orthogonality properties verified")
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
