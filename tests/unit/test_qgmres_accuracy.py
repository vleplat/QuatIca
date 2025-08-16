#!/usr/bin/env python3
"""
Test QGMRES solver accuracy with well-conditioned square systems
"""

import os
import sys
import time

import numpy as np
import quaternion

# Add core to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "quatica"))

from solver import NewtonSchulzPseudoinverse, QGMRESSolver
from utils import quat_frobenius_norm, quat_matmat


def create_well_conditioned_matrix(n):
    """Create a well-conditioned quaternion matrix for testing"""
    # Create a diagonal matrix with reasonable condition number
    A = np.zeros((n, n), dtype=np.quaternion)

    # Create diagonal elements with condition number around 10
    for i in range(n):
        if i == 0:
            A[i, i] = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
        else:
            A[i, i] = quaternion.quaternion(0.1 + 0.1 * i, 0.0, 0.0, 0.0)

    # Add small off-diagonal elements to make it non-diagonal
    for i in range(n):
        for j in range(n):
            if i != j:
                A[i, j] = quaternion.quaternion(0.01, 0.01, 0.01, 0.01)

    return A


def test_qgmres_well_conditioned_2x2():
    """Test QGMRES on a well-conditioned 2x2 system"""
    print("Testing QGMRES on well-conditioned 2x2 system...")

    # Create well-conditioned 2x2 system
    A = create_well_conditioned_matrix(2)
    b = np.array(
        [
            [quaternion.quaternion(1.0, 0.0, 0.0, 0.0)],
            [quaternion.quaternion(0.0, 1.0, 0.0, 0.0)],
        ],
        dtype=np.quaternion,
    )

    print(f"A shape: {A.shape}")
    print(f"b shape: {b.shape}")

    # Solve with QGMRES
    qgmres_solver = QGMRESSolver(tol=1e-12, max_iter=20, verbose=True)
    start_time = time.time()
    x_qgmres, info_qgmres = qgmres_solver.solve(A, b)
    qgmres_time = time.time() - start_time

    print(f"Q-GMRES solution shape: {x_qgmres.shape}")
    print(f"Q-GMRES iterations: {info_qgmres['iterations']}")
    print(f"Q-GMRES residual: {info_qgmres['residual']:.2e}")
    print(f"Q-GMRES time: {qgmres_time:.4f} seconds")

    # Solve with pseudoinverse for comparison
    pinv_solver = NewtonSchulzPseudoinverse(tol=1e-12, verbose=False)
    start_time = time.time()
    A_pinv, _, _ = pinv_solver.compute(A)
    x_pinv = quat_matmat(A_pinv, b)
    pinv_time = time.time() - start_time

    print(f"Pseudoinverse time: {pinv_time:.4f} seconds")

    # Compare solutions
    diff = quat_frobenius_norm(x_qgmres - x_pinv)
    print(f"||x_qgmres - x_pinv||_F = {diff:.2e}")

    # Check residuals
    residual_qgmres = quat_frobenius_norm(quat_matmat(A, x_qgmres) - b)
    residual_pinv = quat_frobenius_norm(quat_matmat(A, x_pinv) - b)
    print(f"||A*x_qgmres - b||_F = {residual_qgmres:.2e}")
    print(f"||A*x_pinv - b||_F = {residual_pinv:.2e}")

    # Verify accuracy
    assert diff < 1e-6, f"Q-GMRES and pseudoinverse solutions differ too much: {diff}"
    assert residual_qgmres < 1e-6, f"Q-GMRES residual too large: {residual_qgmres}"
    print("✓ 2x2 well-conditioned system test passed!")


def test_qgmres_well_conditioned_3x3():
    """Test QGMRES on a well-conditioned 3x3 system"""
    print("\nTesting QGMRES on well-conditioned 3x3 system...")

    # Create well-conditioned 3x3 system
    A = create_well_conditioned_matrix(3)
    b = np.array(
        [
            [quaternion.quaternion(1.0, 0.0, 0.0, 0.0)],
            [quaternion.quaternion(0.0, 1.0, 0.0, 0.0)],
            [quaternion.quaternion(0.0, 0.0, 1.0, 0.0)],
        ],
        dtype=np.quaternion,
    )

    print(f"A shape: {A.shape}")
    print(f"b shape: {b.shape}")

    # Solve with QGMRES
    qgmres_solver = QGMRESSolver(tol=1e-12, max_iter=20, verbose=True)
    start_time = time.time()
    x_qgmres, info_qgmres = qgmres_solver.solve(A, b)
    qgmres_time = time.time() - start_time

    print(f"Q-GMRES solution shape: {x_qgmres.shape}")
    print(f"Q-GMRES iterations: {info_qgmres['iterations']}")
    print(f"Q-GMRES residual: {info_qgmres['residual']:.2e}")
    print(f"Q-GMRES time: {qgmres_time:.4f} seconds")

    # Solve with pseudoinverse for comparison
    pinv_solver = NewtonSchulzPseudoinverse(tol=1e-12, verbose=False)
    start_time = time.time()
    A_pinv, _, _ = pinv_solver.compute(A)
    x_pinv = quat_matmat(A_pinv, b)
    pinv_time = time.time() - start_time

    print(f"Pseudoinverse time: {pinv_time:.4f} seconds")

    # Compare solutions
    diff = quat_frobenius_norm(x_qgmres - x_pinv)
    print(f"||x_qgmres - x_pinv||_F = {diff:.2e}")

    # Check residuals
    residual_qgmres = quat_frobenius_norm(quat_matmat(A, x_qgmres) - b)
    residual_pinv = quat_frobenius_norm(quat_matmat(A, x_pinv) - b)
    print(f"||A*x_qgmres - b||_F = {residual_qgmres:.2e}")
    print(f"||A*x_pinv - b||_F = {residual_pinv:.2e}")

    # Verify accuracy
    assert diff < 1e-6, f"Q-GMRES and pseudoinverse solutions differ too much: {diff}"
    assert residual_qgmres < 1e-6, f"Q-GMRES residual too large: {residual_qgmres}"
    print("✓ 3x3 well-conditioned system test passed!")


def test_qgmres_identity_system():
    """Test QGMRES on identity matrix system (should be exact)"""
    print("\nTesting QGMRES on identity matrix system...")

    # Create identity system
    n = 3
    A = np.zeros((n, n), dtype=np.quaternion)
    for i in range(n):
        A[i, i] = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)

    b = np.array(
        [
            [quaternion.quaternion(1.0, 0.0, 0.0, 0.0)],
            [quaternion.quaternion(0.0, 1.0, 0.0, 0.0)],
            [quaternion.quaternion(0.0, 0.0, 1.0, 0.0)],
        ],
        dtype=np.quaternion,
    )

    # Solve with QGMRES
    qgmres_solver = QGMRESSolver(tol=1e-12, max_iter=20, verbose=True)
    x_qgmres, info_qgmres = qgmres_solver.solve(A, b)

    print(f"Q-GMRES iterations: {info_qgmres['iterations']}")
    print(f"Q-GMRES residual: {info_qgmres['residual']:.2e}")

    # Check residuals
    residual_qgmres = quat_frobenius_norm(quat_matmat(A, x_qgmres) - b)
    print(f"||A*x_qgmres - b||_F = {residual_qgmres:.2e}")

    # For identity matrix, solution should be exact
    assert residual_qgmres < 1e-10, (
        f"Q-GMRES residual too large for identity system: {residual_qgmres}"
    )
    print("✓ Identity system test passed!")


def test_qgmres_convergence_history():
    """Test QGMRES convergence history"""
    print("\nTesting QGMRES convergence history...")

    # Create a simple 2x2 system
    A = create_well_conditioned_matrix(2)
    b = np.array(
        [
            [quaternion.quaternion(1.0, 0.0, 0.0, 0.0)],
            [quaternion.quaternion(0.0, 1.0, 0.0, 0.0)],
        ],
        dtype=np.quaternion,
    )

    # Solve with QGMRES
    qgmres_solver = QGMRESSolver(tol=1e-12, max_iter=20, verbose=False)
    x_qgmres, info_qgmres = qgmres_solver.solve(A, b)

    # Check that convergence history is available
    assert "residual_history" in info_qgmres, "Residual history not available"
    assert len(info_qgmres["residual_history"]) > 0, "Residual history is empty"

    print(f"Convergence history length: {len(info_qgmres['residual_history'])}")
    print(f"Final residual: {info_qgmres['residual']:.2e}")
    print("✓ Convergence history test passed!")


def main():
    """Run all accuracy tests"""
    print("=" * 60)
    print("TESTING QGMRES SOLVER ACCURACY")
    print("=" * 60)

    try:
        test_qgmres_well_conditioned_2x2()
        test_qgmres_well_conditioned_3x3()
        test_qgmres_identity_system()
        test_qgmres_convergence_history()

        print("\n" + "=" * 60)
        print("✓ ALL ACCURACY TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()
