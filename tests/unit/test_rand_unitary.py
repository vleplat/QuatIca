#!/usr/bin/env python3
"""
Unit tests for random unitary quaternion matrix generation.

This module tests the generate_random_unitary_matrix function from core/data_gen.py
to ensure it produces valid unitary matrices of the correct size and properties.

Author: QuatIca Team
Date: 2024
"""

import numpy as np
import quaternion
import sys
import os
import unittest

# Add core directory to path to import our framework functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core'))
from data_gen import generate_random_unitary_matrix
from utils import quat_matmat, quat_hermitian, quat_frobenius_norm, quat_eye


class TestRandomUnitaryMatrix(unittest.TestCase):
    """Test cases for random unitary matrix generation."""
    
    def test_matrix_size(self):
        """Test that the generated matrix has the correct size."""
        print("\nTesting matrix size...")
        
        # Test different sizes
        sizes = [2, 3, 4, 5, 8, 10]
        
        for n in sizes:
            with self.subTest(n=n):
                Q = generate_random_unitary_matrix(n)
                
                # Check shape
                self.assertEqual(Q.shape, (n, n), 
                                f"Matrix should be {n}×{n}, got {Q.shape}")
                
                # Check data type
                self.assertEqual(Q.dtype, np.quaternion,
                                f"Matrix should be quaternion type, got {Q.dtype}")
                
                print(f"✓ Size {n}×{n}: OK")
    
    def test_unitarity_property(self):
        """Test that Q^H * Q = I (unitarity property)."""
        print("\nTesting unitarity property (Q^H * Q = I)...")
        
        # Test different sizes
        sizes = [2, 3, 4, 5, 8]
        tolerance = 1e-12  # Numerical tolerance for floating point errors
        
        for n in sizes:
            with self.subTest(n=n):
                Q = generate_random_unitary_matrix(n)
                
                # Compute Q^H * Q
                Q_H = quat_hermitian(Q)
                Q_H_Q = quat_matmat(Q_H, Q)
                
                # Expected identity matrix
                I = quat_eye(n)
                
                # Compute the difference
                diff = Q_H_Q - I
                diff_norm = quat_frobenius_norm(diff)
                
                # Check if difference is within tolerance
                self.assertLess(diff_norm, tolerance,
                               f"Q^H * Q should equal I, but ||Q^H * Q - I||_F = {diff_norm:.2e}")
                
                print(f"✓ Size {n}×{n}: ||Q^H * Q - I||_F = {diff_norm:.2e}")
    
    def test_unitarity_property_alternative(self):
        """Test unitarity using Q * Q^H = I (alternative formulation)."""
        print("\nTesting alternative unitarity property (Q * Q^H = I)...")
        
        # Test different sizes
        sizes = [2, 3, 4, 5, 8]
        tolerance = 1e-12
        
        for n in sizes:
            with self.subTest(n=n):
                Q = generate_random_unitary_matrix(n)
                
                # Compute Q * Q^H
                Q_H = quat_hermitian(Q)
                Q_Q_H = quat_matmat(Q, Q_H)
                
                # Expected identity matrix
                I = quat_eye(n)
                
                # Compute the difference
                diff = Q_Q_H - I
                diff_norm = quat_frobenius_norm(diff)
                
                # Check if difference is within tolerance
                self.assertLess(diff_norm, tolerance,
                               f"Q * Q^H should equal I, but ||Q * Q^H - I||_F = {diff_norm:.2e}")
                
                print(f"✓ Size {n}×{n}: ||Q * Q^H - I||_F = {diff_norm:.2e}")
    
    def test_determinant_property(self):
        """Test that the determinant of a unitary matrix has unit magnitude."""
        print("\nTesting determinant property (|det(Q)| = 1)...")
        
        # Test different sizes
        sizes = [2, 3, 4, 5]
        tolerance = 1e-12
        
        for n in sizes:
            with self.subTest(n=n):
                Q = generate_random_unitary_matrix(n)
                
                # Convert to real block matrix for determinant computation
                Q_real = np.zeros((4*n, 4*n))
                Q_float = quaternion.as_float_array(Q)
                
                # Build the real block matrix
                for i in range(n):
                    for j in range(n):
                        w, x, y, z = Q_float[i, j]
                        bi, bj = 4*i, 4*j
                        Q_real[bi:bi+4, bj:bj+4] = np.array([
                            [ w, -x, -y, -z],
                            [ x,  w, -z,  y],
                            [ y,  z,  w, -x],
                            [ z, -y,  x,  w]
                        ])
                
                # Compute determinant
                det = np.linalg.det(Q_real)
                det_magnitude = abs(det)
                
                # Check if magnitude is close to 1
                self.assertLess(abs(det_magnitude - 1.0), tolerance,
                               f"|det(Q)| should be 1, but got {det_magnitude:.6f}")
                
                print(f"✓ Size {n}×{n}: |det(Q)| = {det_magnitude:.6f}")
    
    def test_randomness(self):
        """Test that different calls produce different matrices."""
        print("\nTesting randomness (different matrices generated)...")
        
        n = 4
        Q1 = generate_random_unitary_matrix(n)
        Q2 = generate_random_unitary_matrix(n)
        
        # Check that matrices are different
        diff = Q1 - Q2
        diff_norm = quat_frobenius_norm(diff)
        
        self.assertGreater(diff_norm, 1e-10,
                          "Different calls should produce different matrices")
        
        print(f"✓ Randomness: ||Q1 - Q2||_F = {diff_norm:.6f}")
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("\nTesting edge cases...")
        
        # Test size 1
        Q1 = generate_random_unitary_matrix(1)
        self.assertEqual(Q1.shape, (1, 1))
        self.assertEqual(Q1.dtype, np.quaternion)
        
        # Test that it's unitary
        Q1_H = quat_hermitian(Q1)
        Q1_H_Q1 = quat_matmat(Q1_H, Q1)
        I1 = quat_eye(1)
        diff_norm = quat_frobenius_norm(Q1_H_Q1 - I1)
        self.assertLess(diff_norm, 1e-12)
        
        print("✓ Edge case 1×1: OK")
        
        # Test larger size
        Q10 = generate_random_unitary_matrix(10)
        self.assertEqual(Q10.shape, (10, 10))
        
        # Test unitarity for larger matrix
        Q10_H = quat_hermitian(Q10)
        Q10_H_Q10 = quat_matmat(Q10_H, Q10)
        I10 = quat_eye(10)
        diff_norm = quat_frobenius_norm(Q10_H_Q10 - I10)
        self.assertLess(diff_norm, 1e-12)
        
        print("✓ Edge case 10×10: OK")


def run_comprehensive_test():
    """Run a comprehensive test with detailed output."""
    print("="*80)
    print(" RANDOM UNITARY MATRIX GENERATION TEST")
    print("="*80)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRandomUnitaryMatrix)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*80)
    print(" TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✓ ALL TESTS PASSED!")
        print("\nThe generate_random_unitary_matrix function is working correctly.")
        print("Generated matrices are:")
        print("- Correct size (n×n)")
        print("- Unitary (Q^H * Q = I)")
        print("- Random (different each time)")
        print("- Proper quaternion type")
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
    success = run_comprehensive_test()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 