#!/usr/bin/env python3
"""
Simple test for power_iteration function comparing with quaternion_eigendecomposition.
"""

import sys
import os
import numpy as np
import quaternion

# Add core module to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core'))
from utils import power_iteration, quat_matmat, quat_frobenius_norm, quat_hermitian
from data_gen import create_test_matrix
from decomp.eigen import quaternion_eigendecomposition


def test_power_iteration_vs_eigendecomposition():
    """Test power iteration by comparing with quaternion_eigendecomposition."""
    print("Testing power iteration vs eigendecomposition...")
    
    # Test cases: (size, description)
    test_cases = [
        (4, "Small matrix"),
        (8, "Medium matrix"), 
        (12, "Large matrix")
    ]
    
    all_passed = True
    
    for size, description in test_cases:
        print(f"\n{'='*60}")
        print(f"TESTING {description.upper()}: {size}×{size}")
        print(f"{'='*60}")
        
        # Create Hermitian matrix: A = B^H @ B (positive definite)
        B = create_test_matrix(size, size)
        A = quat_matmat(quat_hermitian(B), B)
        
        print(f"Matrix A shape: {A.shape}")
        
        # Run power iteration
        print(f"\n--- Power Iteration ---")
        power_eigenvector, power_eigenvalue = power_iteration(A, return_eigenvalue=True, verbose=True)
        
        # Run quaternion eigendecomposition
        print(f"\n--- Eigendecomposition ---")
        eigenvalues, eigenvectors = quaternion_eigendecomposition(A, verbose=True)
        
        # Find the dominant eigenvalue (largest magnitude)
        dominant_idx = np.argmax(np.abs(eigenvalues))
        dominant_eigenvalue = eigenvalues[dominant_idx]
        dominant_eigenvector = eigenvectors[:, dominant_idx:dominant_idx+1]  # Make it column vector
        
        print(f"\n--- Comparison Results ---")
        print(f"Power iteration eigenvalue: {power_eigenvalue:.6f}")
        print(f"Eigendecomposition dominant eigenvalue: {dominant_eigenvalue:.6f}")
        print(f"Eigenvalue difference: {abs(power_eigenvalue - abs(dominant_eigenvalue)):.2e}")
        
        # Compare eigenvalues (power iteration gives magnitude, eigendecomposition gives complex)
        eigenvalue_error = abs(power_eigenvalue - abs(dominant_eigenvalue))
        print(f"Eigenvalue error: {eigenvalue_error:.2e}")
        
        # Compare eigenvectors (up to sign/phase)
        # Normalize both eigenvectors
        power_norm = quat_frobenius_norm(power_eigenvector)
        decomp_norm = quat_frobenius_norm(dominant_eigenvector)
        
        power_normalized = power_eigenvector / power_norm
        decomp_normalized = dominant_eigenvector / decomp_norm
        
        # Check if eigenvectors are parallel (up to sign)
        dot_product = quat_matmat(quat_hermitian(power_normalized), decomp_normalized)
        dot_product_norm = quat_frobenius_norm(dot_product)
        
        print(f"Eigenvector dot product norm: {dot_product_norm:.6f}")
        
        # Results for this test case
        eigenvalue_pass = eigenvalue_error < 1e-6
        eigenvector_pass = abs(dot_product_norm - 1.0) < 1e-6
        case_passed = eigenvalue_pass and eigenvector_pass
        
        print(f"\n{description} Results:")
        print(f"✅ Eigenvalue comparison: {'PASS' if eigenvalue_pass else 'FAIL'}")
        print(f"✅ Eigenvector comparison: {'PASS' if eigenvector_pass else 'FAIL'}")
        print(f"✅ Overall: {'PASS' if case_passed else 'FAIL'}")
        
        all_passed = all_passed and case_passed
    
    return all_passed


if __name__ == "__main__":
    success = test_power_iteration_vs_eigendecomposition()
    if success:
        print("\n🎉 Power iteration test PASSED!")
    else:
        print("\n❌ Power iteration test FAILED!")
    sys.exit(0 if success else 1) 