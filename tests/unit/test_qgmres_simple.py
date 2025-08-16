#!/usr/bin/env python3
"""
Simple test to debug QGMRES convergence issues
"""

import os
import sys

import numpy as np
import quaternion

# Add core to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "quatica"))

from solver import QGMRESSolver
from utils import quat_frobenius_norm, quat_matmat


def test_simple_identity():
    """Test QGMRES on identity matrix (should be exact in 1 iteration)"""
    print("Testing QGMRES on identity matrix...")

    # Create identity matrix
    n = 2
    A = np.zeros((n, n), dtype=np.quaternion)
    for i in range(n):
        A[i, i] = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)

    # Create right-hand side
    b = np.array(
        [
            [quaternion.quaternion(1.0, 0.0, 0.0, 0.0)],
            [quaternion.quaternion(0.0, 1.0, 0.0, 0.0)],
        ],
        dtype=np.quaternion,
    )

    print(f"A:\n{A}")
    print(f"b:\n{b}")

    # Solve with QGMRES
    qgmres_solver = QGMRESSolver(tol=1e-12, max_iter=10, verbose=True)
    x_qgmres, info_qgmres = qgmres_solver.solve(A, b)

    print(f"Q-GMRES solution:\n{x_qgmres}")
    print(f"Q-GMRES iterations: {info_qgmres['iterations']}")
    print(f"Q-GMRES residual: {info_qgmres['residual']:.2e}")

    # Check residual
    residual = quat_frobenius_norm(quat_matmat(A, x_qgmres) - b)
    print(f"Computed residual: {residual:.2e}")

    # For identity matrix, solution should be exact
    expected_solution = b.copy()
    print(f"Expected solution:\n{expected_solution}")

    # Check if solution is correct
    solution_diff = quat_frobenius_norm(x_qgmres - expected_solution)
    print(f"Solution difference: {solution_diff:.2e}")

    assert residual < 1e-10, f"Residual too large: {residual}"
    assert solution_diff < 1e-10, f"Solution difference too large: {solution_diff}"
    print("✓ Identity matrix test passed!")


def test_simple_diagonal():
    """Test QGMRES on diagonal matrix"""
    print("\nTesting QGMRES on diagonal matrix...")

    # Create diagonal matrix
    n = 2
    A = np.zeros((n, n), dtype=np.quaternion)
    A[0, 0] = quaternion.quaternion(2.0, 0.0, 0.0, 0.0)
    A[1, 1] = quaternion.quaternion(3.0, 0.0, 0.0, 0.0)

    # Create right-hand side
    b = np.array(
        [
            [quaternion.quaternion(2.0, 0.0, 0.0, 0.0)],
            [quaternion.quaternion(3.0, 0.0, 0.0, 0.0)],
        ],
        dtype=np.quaternion,
    )

    print(f"A:\n{A}")
    print(f"b:\n{b}")

    # Solve with QGMRES
    qgmres_solver = QGMRESSolver(tol=1e-12, max_iter=10, verbose=True)
    x_qgmres, info_qgmres = qgmres_solver.solve(A, b)

    print(f"Q-GMRES solution:\n{x_qgmres}")
    print(f"Q-GMRES iterations: {info_qgmres['iterations']}")
    print(f"Q-GMRES residual: {info_qgmres['residual']:.2e}")

    # Check residual
    residual = quat_frobenius_norm(quat_matmat(A, x_qgmres) - b)
    print(f"Computed residual: {residual:.2e}")

    # Expected solution: [1, 1]
    expected_solution = np.array(
        [
            [quaternion.quaternion(1.0, 0.0, 0.0, 0.0)],
            [quaternion.quaternion(1.0, 0.0, 0.0, 0.0)],
        ],
        dtype=np.quaternion,
    )
    print(f"Expected solution:\n{expected_solution}")

    # Check if solution is correct
    solution_diff = quat_frobenius_norm(x_qgmres - expected_solution)
    print(f"Solution difference: {solution_diff:.2e}")

    assert residual < 1e-10, f"Residual too large: {residual}"
    assert solution_diff < 1e-10, f"Solution difference too large: {solution_diff}"
    print("✓ Diagonal matrix test passed!")


def main():
    """Run simple tests"""
    print("=" * 60)
    print("SIMPLE QGMRES TESTS")
    print("=" * 60)

    try:
        test_simple_identity()
        test_simple_diagonal()

        print("\n" + "=" * 60)
        print("✓ ALL SIMPLE TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()
