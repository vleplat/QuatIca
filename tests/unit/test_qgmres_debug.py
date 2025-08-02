#!/usr/bin/env python3
"""
Debug test for QGMRES algorithm issues
"""

import sys
import os
import numpy as np
import quaternion

# Add core to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core'))

from solver import QGMRESSolver
from utils import quat_matmat, quat_frobenius_norm


def test_qgmres_debug_simple():
    """Test QGMRES with a very simple case to debug the algorithm."""
    print("Testing QGMRES with simple case...")
    
    # Create a simple 2x2 diagonal matrix
    A = np.zeros((2, 2), dtype=np.quaternion)
    A[0, 0] = quaternion.quaternion(2.0, 0.0, 0.0, 0.0)
    A[1, 1] = quaternion.quaternion(3.0, 0.0, 0.0, 0.0)
    
    # Create right-hand side
    b = np.array([[quaternion.quaternion(4.0, 0.0, 0.0, 0.0)],
                  [quaternion.quaternion(6.0, 0.0, 0.0, 0.0)]], dtype=np.quaternion)
    
    print(f"A:\n{A}")
    print(f"b:\n{b}")
    
    # Expected solution: x = [2, 2]
    expected_solution = np.array([[quaternion.quaternion(2.0, 0.0, 0.0, 0.0)],
                                 [quaternion.quaternion(2.0, 0.0, 0.0, 0.0)]], dtype=np.quaternion)
    
    # Solve with QGMRES
    qgmres_solver = QGMRESSolver(tol=1e-12, max_iter=10, verbose=True)
    x_qgmres, info_qgmres = qgmres_solver.solve(A, b)
    
    print(f"Q-GMRES solution:\n{x_qgmres}")
    print(f"Expected solution:\n{expected_solution}")
    print(f"Q-GMRES iterations: {info_qgmres['iterations']}")
    print(f"Q-GMRES residual: {info_qgmres['residual']:.2e}")
    
    # Check residuals
    residual_qgmres = quat_frobenius_norm(quat_matmat(A, x_qgmres) - b)
    residual_expected = quat_frobenius_norm(quat_matmat(A, expected_solution) - b)
    
    print(f"Q-GMRES residual: {residual_qgmres:.2e}")
    print(f"Expected residual: {residual_expected:.2e}")
    
    # Check solution difference
    solution_diff = quat_frobenius_norm(x_qgmres - expected_solution)
    print(f"Solution difference: {solution_diff:.2e}")
    
    # This should be very small for a simple diagonal case
    assert solution_diff < 1e-10, f"Solution difference too large: {solution_diff}"
    assert residual_qgmres < 1e-10, f"Q-GMRES residual too large: {residual_qgmres}"
    
    print("✓ Simple diagonal case test passed!")


def test_qgmres_debug_identity():
    """Test QGMRES with identity matrix (should be exact in 1 iteration)."""
    print("\nTesting QGMRES with identity matrix...")
    
    # Create identity matrix
    A = np.zeros((2, 2), dtype=np.quaternion)
    A[0, 0] = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
    A[1, 1] = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
    
    # Create right-hand side
    b = np.array([[quaternion.quaternion(1.0, 0.0, 0.0, 0.0)],
                  [quaternion.quaternion(2.0, 0.0, 0.0, 0.0)]], dtype=np.quaternion)
    
    print(f"A:\n{A}")
    print(f"b:\n{b}")
    
    # Expected solution: x = b (for identity matrix)
    expected_solution = b.copy()
    
    # Solve with QGMRES
    qgmres_solver = QGMRESSolver(tol=1e-12, max_iter=10, verbose=True)
    x_qgmres, info_qgmres = qgmres_solver.solve(A, b)
    
    print(f"Q-GMRES solution:\n{x_qgmres}")
    print(f"Expected solution:\n{expected_solution}")
    print(f"Q-GMRES iterations: {info_qgmres['iterations']}")
    print(f"Q-GMRES residual: {info_qgmres['residual']:.2e}")
    
    # Check residuals
    residual_qgmres = quat_frobenius_norm(quat_matmat(A, x_qgmres) - b)
    residual_expected = quat_frobenius_norm(quat_matmat(A, expected_solution) - b)
    
    print(f"Q-GMRES residual: {residual_qgmres:.2e}")
    print(f"Expected residual: {residual_expected:.2e}")
    
    # Check solution difference
    solution_diff = quat_frobenius_norm(x_qgmres - expected_solution)
    print(f"Solution difference: {solution_diff:.2e}")
    
    # This should be exact for identity matrix
    assert solution_diff < 1e-10, f"Solution difference too large: {solution_diff}"
    assert residual_qgmres < 1e-10, f"Q-GMRES residual too large: {residual_qgmres}"
    
    print("✓ Identity matrix test passed!")


def main():
    """Run debug tests."""
    print("=" * 60)
    print("QGMRES DEBUG TESTS")
    print("=" * 60)
    
    try:
        test_qgmres_debug_simple()
        test_qgmres_debug_identity()
        
        print("\n" + "=" * 60)
        print("✓ ALL DEBUG TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main() 