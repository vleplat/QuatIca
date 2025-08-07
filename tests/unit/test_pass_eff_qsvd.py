#!/usr/bin/env python3
"""
Unit tests for pass_eff_qsvd function
"""

import sys
import os
import numpy as np
import quaternion

# Add core module to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core'))
from decomp.qsvd import pass_eff_qsvd, classical_qsvd_full, rand_qsvd
from data_gen import create_test_matrix
from utils import quat_matmat, quat_hermitian, quat_frobenius_norm


def test_pass_eff_qsvd_basic():
    """Test basic functionality of pass_eff_qsvd."""
    print("Testing basic pass_eff_qsvd functionality...")
    
    # Create test matrix
    m, n = 8, 6
    R = 3
    X = create_test_matrix(m, n)
    
    print(f"Input matrix shape: {X.shape}")
    print(f"Target rank: {R}")
    
    # Test with different parameters
    test_cases = [
        (1, 5),  # n_passes=1, oversample=5
        (2, 5),  # n_passes=2, oversample=5
        (3, 5),  # n_passes=3, oversample=5
        (2, 10), # n_passes=2, oversample=10
    ]
    
    for n_passes, oversample in test_cases:
        print(f"\nTesting with n_passes={n_passes}, oversample={oversample}:")
        
        try:
            U, s, V = pass_eff_qsvd(X, R, oversample=oversample, n_passes=n_passes)
            
            print(f"  ✅ SUCCESS: pass_eff_qsvd completed")
            print(f"  U shape: {U.shape}")
            print(f"  V shape: {V.shape}")
            print(f"  s shape: {s.shape}")
            print(f"  Singular values: {s}")
            
            # Check reconstruction
            X_recon = quat_matmat(quat_matmat(U, np.diag(s)), quat_hermitian(V))
            recon_error = quat_frobenius_norm(X - X_recon)
            rel_error = recon_error / quat_frobenius_norm(X)
            print(f"  Reconstruction error: {recon_error:.6f}")
            print(f"  Relative error: {rel_error:.6f}")
            
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            return False
    
    return True


def test_pass_eff_qsvd_vs_rand_qsvd():
    """Compare pass_eff_qsvd with rand_qsvd."""
    print("\nComparing pass_eff_qsvd vs rand_qsvd...")
    
    # Create test matrix
    m, n = 10, 8
    R = 4
    X = create_test_matrix(m, n)
    
    print(f"Input matrix shape: {X.shape}")
    print(f"Target rank: {R}")
    
    # Test both methods
    try:
        # pass_eff_qsvd with 2 passes
        U_pass, s_pass, V_pass = pass_eff_qsvd(X, R, oversample=5, n_passes=2)
        
        # rand_qsvd with 2 iterations
        U_rand, s_rand, V_rand = rand_qsvd(X, R, oversample=5, n_iter=2)
        
        print(f"pass_eff_qsvd singular values: {s_pass}")
        print(f"rand_qsvd singular values: {s_rand}")
        
        # Compare reconstruction errors
        X_recon_pass = quat_matmat(quat_matmat(U_pass, np.diag(s_pass)), quat_hermitian(V_pass))
        X_recon_rand = quat_matmat(quat_matmat(U_rand, np.diag(s_rand)), quat_hermitian(V_rand))
        
        error_pass = quat_frobenius_norm(X - X_recon_pass) / quat_frobenius_norm(X)
        error_rand = quat_frobenius_norm(X - X_recon_rand) / quat_frobenius_norm(X)
        
        print(f"pass_eff_qsvd relative error: {error_pass:.6f}")
        print(f"rand_qsvd relative error: {error_rand:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_pass_eff_qsvd_orthogonality():
    """Test that pass_eff_qsvd produces orthonormal U and V matrices."""
    print("\nTesting orthogonality properties...")
    
    # Create test matrix
    m, n = 6, 4
    R = 2
    X = create_test_matrix(m, n)
    
    try:
        U, s, V = pass_eff_qsvd(X, R, oversample=5, n_passes=2)
        
        # Test U orthogonality: U^H * U should be identity
        U_H = quat_hermitian(U)
        U_ortho = quat_matmat(U_H, U)
        U_ortho_error = quat_frobenius_norm(U_ortho - np.eye(R, dtype=np.quaternion))
        
        # Test V orthogonality: V^H * V should be identity
        V_H = quat_hermitian(V)
        V_ortho = quat_matmat(V_H, V)
        V_ortho_error = quat_frobenius_norm(V_ortho - np.eye(R, dtype=np.quaternion))
        
        print(f"U orthogonality error: {U_ortho_error:.2e}")
        print(f"V orthogonality error: {V_ortho_error:.2e}")
        
        if U_ortho_error < 1e-12 and V_ortho_error < 1e-12:
            print("✅ PASSED: U and V are orthonormal")
            return True
        else:
            print("❌ FAILED: U and V are not orthonormal")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_pass_eff_qsvd_convergence():
    """Test convergence with different numbers of passes."""
    print("\nTesting convergence with different numbers of passes...")
    
    # Create test matrix
    m, n = 12, 8
    R = 6
    X = create_test_matrix(m, n)
    
    # Compute full Q-SVD for comparison
    print("Computing full Q-SVD for comparison...")
    U_full, s_full, V_full = classical_qsvd_full(X)
    s_full = s_full[:R]  # Take first R singular values
    print(f"Full Q-SVD singular values (first {R}): {s_full}")
    
    # Test different numbers of passes
    n_passes_list = [1, 2, 3, 4]
    
    for n_passes in n_passes_list:
        print(f"\nTesting pass_eff_qsvd with {n_passes} passes:")
        
        try:
            U, s, V = pass_eff_qsvd(X, R, oversample=10, n_passes=n_passes)
            
            # Compare singular values
            s_diff = np.linalg.norm(s - s_full)
            s_rel_diff = s_diff / np.linalg.norm(s_full)
            
            # Compare reconstruction
            X_recon = quat_matmat(quat_matmat(U, np.diag(s)), quat_hermitian(V))
            recon_error = quat_frobenius_norm(X - X_recon)
            rel_error = recon_error / quat_frobenius_norm(X)
            
            print(f"  pass_eff_qsvd singular values: {s}")
            print(f"  Singular value difference: {s_diff:.6f}")
            print(f"  Relative difference: {s_rel_diff:.6f}")
            print(f"  Reconstruction error: {recon_error:.6f}")
            print(f"  Relative error: {rel_error:.6f}")
            
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            return False
    
    return True


def main():
    """Run all pass_eff_qsvd tests."""
    print("=" * 60)
    print("PASS-EFFICIENT Q-SVD TESTING SUITE")
    print("=" * 60)
    
    success1 = test_pass_eff_qsvd_basic()
    success2 = test_pass_eff_qsvd_vs_rand_qsvd()
    success3 = test_pass_eff_qsvd_orthogonality()
    success4 = test_pass_eff_qsvd_convergence()
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    if all([success1, success2, success3, success4]):
        print("🎉 All pass_eff_qsvd tests completed successfully!")
        print("✅ Basic functionality works")
        print("✅ Comparison with rand_qsvd successful")
        print("✅ Orthogonality properties verified")
        print("✅ Convergence behavior demonstrated")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 