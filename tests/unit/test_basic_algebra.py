#!/usr/bin/env python3
"""
Unit tests for basic quaternion algebra functions.

Tests the ishermitian and det functions from core.utils.

Author: QuatIca Team
Date: 2024
"""

import sys
import os
import numpy as np
import quaternion
import unittest

# Add core module to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core'))
from utils import ishermitian, det, quat_hermitian, induced_matrix_norm_1, induced_matrix_norm_inf, matrix_norm
from data_gen import create_test_matrix


class TestBasicAlgebra(unittest.TestCase):
    """Test cases for basic quaternion algebra functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
    
    def test_ishermitian_square_hermitian(self):
        """Test ishermitian with square Hermitian matrix."""
        # Create a Hermitian matrix
        A = create_test_matrix(3, 3)
        A_hermitian = (A + quat_hermitian(A)) / 2  # Make it Hermitian
        
        result = ishermitian(A_hermitian)
        self.assertTrue(result, "Hermitian matrix should return True")
    
    def test_ishermitian_square_non_hermitian(self):
        """Test ishermitian with square non-Hermitian matrix."""
        # Create a non-Hermitian matrix
        A = create_test_matrix(3, 3)
        
        result = ishermitian(A)
        self.assertFalse(result, "Non-Hermitian matrix should return False")
    
    def test_ishermitian_non_square(self):
        """Test ishermitian with non-square matrix."""
        A = create_test_matrix(3, 4)
        
        with self.assertRaises(ValueError):
            ishermitian(A)
    
    def test_ishermitian_zero_matrix(self):
        """Test ishermitian with zero matrix."""
        A = np.zeros((3, 3), dtype=np.quaternion)
        
        result = ishermitian(A)
        self.assertTrue(result, "Zero matrix should be Hermitian")
    
    def test_ishermitian_identity_matrix(self):
        """Test ishermitian with identity matrix."""
        A = np.eye(3, dtype=np.quaternion)
        
        result = ishermitian(A)
        self.assertTrue(result, "Identity matrix should be Hermitian")
    
    def test_ishermitian_with_tolerance(self):
        """Test ishermitian with custom tolerance."""
        # Create a matrix that's almost Hermitian
        A = create_test_matrix(2, 2)
        A_almost_hermitian = (A + quat_hermitian(A)) / 2
        
        # Should be Hermitian with default tolerance
        result_default = ishermitian(A_almost_hermitian)
        self.assertTrue(result_default, "Should be Hermitian with default tolerance")
        
        # Test with very small tolerance - should still be Hermitian for exact Hermitian matrix
        result_small_tol = ishermitian(A_almost_hermitian, tol=1e-20)
        self.assertTrue(result_small_tol, "Exact Hermitian matrix should be Hermitian with any tolerance")
    
    def test_det_dieudonne_square_matrix(self):
        """Test det with Dieudonné determinant on square matrix."""
        A = create_test_matrix(3, 3)
        
        result = det(A, 'Dieudonne')
        self.assertIsInstance(result, (float, np.floating))
        self.assertGreater(result, 0, "Dieudonné determinant should be positive")
    
    def test_det_dieudonne_unicode(self):
        """Test det with Dieudonné determinant using unicode character."""
        A = create_test_matrix(3, 3)
        
        result1 = det(A, 'Dieudonne')
        result2 = det(A, 'Dieudonné')
        
        self.assertEqual(result1, result2, "Both spellings should give same result")
    
    def test_det_moore_hermitian(self):
        """Test det with Moore determinant on Hermitian matrix."""
        # Create a Hermitian matrix
        A = create_test_matrix(3, 3)
        A_hermitian = (A + quat_hermitian(A)) / 2
        
        result = det(A_hermitian, 'Moore')
        self.assertIsInstance(result, (complex, np.complexfloating))
    
    def test_det_moore_non_hermitian(self):
        """Test det with Moore determinant on non-Hermitian matrix."""
        A = create_test_matrix(3, 3)
        
        with self.assertRaises(ValueError):
            det(A, 'Moore')
    
    def test_det_study_not_implemented(self):
        """Test det with Study determinant (not implemented)."""
        A = create_test_matrix(3, 3)
        
        with self.assertRaises(NotImplementedError):
            det(A, 'Study')
    
    def test_det_non_square_matrix(self):
        """Test det with non-square matrix."""
        A = create_test_matrix(3, 4)
        
        with self.assertRaises(ValueError):
            det(A, 'Dieudonne')

    def test_induced_matrix_norms(self):
        """Test induced 1- and infinity-norms and wrapper matrix_norm."""
        A = np.zeros((2, 3), dtype=np.quaternion)
        # |(1 + 2i)| = sqrt(5), |(3j + 4k)| = 5
        A[0, 0] = quaternion.quaternion(1, 2, 0, 0)
        A[1, 2] = quaternion.quaternion(0, 0, 3, 4)
        n1 = induced_matrix_norm_1(A)
        ninf = induced_matrix_norm_inf(A)
        self.assertAlmostEqual(n1, 5.0, places=12)
        self.assertAlmostEqual(ninf, 5.0, places=12)
        self.assertAlmostEqual(matrix_norm(A, 1), 5.0, places=12)
        self.assertAlmostEqual(matrix_norm(A, np.inf), 5.0, places=12)
    
    def test_det_invalid_type(self):
        """Test det with invalid determinant type."""
        A = create_test_matrix(3, 3)
        
        with self.assertRaises(ValueError):
            det(A, 'InvalidType')
    
    def test_det_consistency(self):
        """Test consistency between different determinant types."""
        # Create a Hermitian matrix
        A = create_test_matrix(3, 3)
        A_hermitian = (A + quat_hermitian(A)) / 2
        
        # Dieudonné determinant should be real and positive
        det_dieudonne = det(A_hermitian, 'Dieudonne')
        self.assertIsInstance(det_dieudonne, (float, np.floating))
        self.assertGreater(det_dieudonne, 0)
        
        # Moore determinant can be complex
        det_moore = det(A_hermitian, 'Moore')
        self.assertIsInstance(det_moore, (complex, np.complexfloating))
    
    def test_det_scaling_property(self):
        """Test determinant scaling property."""
        A = create_test_matrix(3, 3)
        scalar = 2.0
        
        # Scale the matrix
        A_scaled = scalar * A
        
        # Determinant should scale by scalar^n where n is matrix size
        det_original = det(A, 'Dieudonne')
        det_scaled = det(A_scaled, 'Dieudonne')
        
        expected_scaling = scalar ** A.shape[0]
        self.assertAlmostEqual(det_scaled, det_original * expected_scaling, places=10)


def run_basic_algebra_tests():
    """Run all basic algebra tests and print results."""
    print("=" * 60)
    print("BASIC QUATERNION ALGEBRA FUNCTION TESTING")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBasicAlgebra)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED!")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_basic_algebra_tests()
    sys.exit(0 if success else 1) 