#!/usr/bin/env python3
"""
Test script for normQsparse function from utils.py

This script tests the normQsparse function with different opt values:
- opt=1: 1-norm of element-wise square root
- opt=2: 2-norm (largest singular value of horizontally stacked matrix)
- opt='d': Dual norm
- opt=None: Frobenius norm (should match quat_frobenius_norm)

Author: QuatIca Team
Date: 2024
"""

import sys
import os
import numpy as np
import quaternion

# Add core module to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core'))
from utils import normQsparse, quat_frobenius_norm

def test_normQsparse_comparison():
    """
    Test normQsparse with different opt values and compare with quat_frobenius_norm.
    """
    print("=" * 60)
    print("NORMQSPARSE FUNCTION TESTING")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create a random quaternion matrix
    m, n = 5, 4
    print(f"Creating random quaternion matrix of size {m}×{n}")
    
    # Generate random components
    A_components = np.random.randn(m, n, 4)
    A = quaternion.as_quat_array(A_components)
    
    # Extract individual components for normQsparse
    A0, A1, A2, A3 = A_components[..., 0], A_components[..., 1], A_components[..., 2], A_components[..., 3]
    
    print(f"Matrix A shape: {A.shape}")
    print(f"Component shapes: A0={A0.shape}, A1={A1.shape}, A2={A2.shape}, A3={A3.shape}")
    print()
    
    # Test 1: normQsparse with opt=1 (1-norm of element-wise square root)
    print("1. Testing normQsparse with opt=1:")
    norm_opt1 = normQsparse(A0, A1, A2, A3, opt=1)
    print(f"   normQsparse(A, opt=1) = {norm_opt1:.8f}")
    print()
    
    # Test 2: normQsparse with opt=2 (2-norm - largest singular value of horizontally stacked matrix)
    print("2. Testing normQsparse with opt=2:")
    norm_opt2 = normQsparse(A0, A1, A2, A3, opt=2)
    print(f"   normQsparse(A, opt=2) = {norm_opt2:.8f}")
    print("   Note: This is NOT the quaternion operator norm, but the largest singular value")
    print("   of the horizontally stacked matrix [A0, A2, A1, A3]")
    print()
    
    # Test 3: normQsparse with opt='d' (dual norm)
    print("3. Testing normQsparse with opt='d' (dual norm):")
    norm_opt_d = normQsparse(A0, A1, A2, A3, opt='d')
    print(f"   normQsparse(A, opt='d') = {norm_opt_d:.8f}")
    print()
    
    # Test 4: normQsparse with opt=None (Frobenius norm)
    print("4. Testing normQsparse with opt=None (Frobenius norm):")
    norm_opt_none = normQsparse(A0, A1, A2, A3, opt=None)
    print(f"   normQsparse(A, opt=None) = {norm_opt_none:.8f}")
    print()
    
    # Test 5: quat_frobenius_norm for comparison
    print("5. Testing quat_frobenius_norm for comparison:")
    norm_frobenius = quat_frobenius_norm(A)
    print(f"   quat_frobenius_norm(A) = {norm_frobenius:.8f}")
    print()
    
    # Comparison and verification
    print("=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    # Check if opt=None matches quat_frobenius_norm
    diff_frobenius = abs(norm_opt_none - norm_frobenius)
    print(f"normQsparse(A, opt=None) vs quat_frobenius_norm(A):")
    print(f"   normQsparse(A, opt=None) = {norm_opt_none:.8f}")
    print(f"   quat_frobenius_norm(A)   = {norm_frobenius:.8f}")
    print(f"   Difference               = {diff_frobenius:.2e}")
    
    if diff_frobenius < 1e-12:
        print("   ✅ PASSED: opt=None matches quat_frobenius_norm")
    else:
        print("   ❌ FAILED: opt=None does not match quat_frobenius_norm")
    
    print()
    
    # Show all norm values for comparison
    print("All norm values:")
    print(f"   opt=1 (1-norm of sqrt):     {norm_opt1:.8f}")
    print(f"   opt=2 (2-norm):             {norm_opt2:.8f}")
    print(f"   opt='d' (dual norm):        {norm_opt_d:.8f}")
    print(f"   opt=None (Frobenius):       {norm_opt_none:.8f}")
    print(f"   quat_frobenius_norm:        {norm_frobenius:.8f}")
    print()
    
    # Additional verification: test with different matrix sizes
    print("=" * 60)
    print("ADDITIONAL VERIFICATION WITH DIFFERENT SIZES")
    print("=" * 60)
    
    test_sizes = [(3, 3), (4, 2), (2, 4), (6, 6)]
    
    for m, n in test_sizes:
        print(f"\nTesting matrix size {m}×{n}:")
        
        # Create random matrix
        A_components = np.random.randn(m, n, 4)
        A = quaternion.as_quat_array(A_components)
        A0, A1, A2, A3 = A_components[..., 0], A_components[..., 1], A_components[..., 2], A_components[..., 3]
        
        # Compute norms
        norm_opt_none = normQsparse(A0, A1, A2, A3, opt=None)
        norm_frobenius = quat_frobenius_norm(A)
        
        # Check agreement
        diff = abs(norm_opt_none - norm_frobenius)
        status = "✅ PASSED" if diff < 1e-12 else "❌ FAILED"
        print(f"   {status}: diff = {diff:.2e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_normQsparse_comparison()
    if success:
        print("🎉 All normQsparse tests completed successfully!")
    else:
        print("❌ Some tests failed!")
        sys.exit(1) 