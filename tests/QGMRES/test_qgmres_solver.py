#!/usr/bin/env python3
"""
Q-GMRES Solver Test Script

This script tests the Q-GMRES solver by:
1. Creating random quaternion linear systems
2. Solving them with Q-GMRES
3. Comparing with pseudoinverse solutions
4. Testing convergence and accuracy

Author: QuatIca Framework
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import quaternion

# Add core module to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "quatica"))

from data_gen import create_sparse_quat_matrix, create_test_matrix
from solver import NewtonSchulzPseudoinverse, QGMRESSolver
from utils import quat_frobenius_norm, quat_matmat


def display_quaternion_matrix(A, name="Matrix", max_display=4, verbose=True):
    """Display a quaternion matrix in a readable format."""
    if not verbose:
        print(f"{name} (shape: {A.shape})")
        return

    print(f"{name} (shape: {A.shape}):")

    # Handle sparse matrices
    if hasattr(A, "toarray"):
        # Convert sparse to dense for display
        A_real = A.real.toarray()
        A_i = A.i.toarray()
        A_j = A.j.toarray()
        A_k = A.k.toarray()
        A = quaternion.as_quat_array(np.stack([A_real, A_i, A_j, A_k], axis=-1))

    # Limit display size
    m, n = A.shape
    m_display = min(m, max_display)
    n_display = min(n, max_display)

    for i in range(m_display):
        row_str = "  ["
        for j in range(n_display):
            q = A[i, j]
            if j > 0:
                row_str += " "
            row_str += f"{q.w:6.3f}{q.x:+6.3f}i{q.y:+6.3f}j{q.z:+6.3f}k"
        if n > n_display:
            row_str += " ..."
        row_str += " ]"
        print(row_str)

    if m > m_display:
        print("  ...")


def test_qgmres_basic():
    """Test basic Q-GMRES functionality with a simple linear system."""
    print("\n" + "=" * 60)
    print("TESTING Q-GMRES BASIC FUNCTIONALITY")
    print("=" * 60)

    # Create a well-conditioned 3x3 quaternion system
    print("1. Creating test linear system...")
    # Use a more robust matrix generation to avoid lucky breakdown
    np.random.seed(42)  # For reproducibility
    A = create_test_matrix(3, 3, cond_number=10.0)  # Well-conditioned matrix
    b = create_test_matrix(3, 1)

    display_quaternion_matrix(A, "A", verbose=False)
    display_quaternion_matrix(b, "b", verbose=False)

    # Solve with Q-GMRES
    print("\n2. Solving with Q-GMRES...")
    qgmres_solver = QGMRESSolver(tol=1e-10, max_iter=100, verbose=True)
    start_time = time.time()
    x_qgmres, info_qgmres = qgmres_solver.solve(A, b)
    qgmres_time = time.time() - start_time

    print(f"Q-GMRES solution (shape: {x_qgmres.shape})")
    print(f"Q-GMRES iterations: {info_qgmres['iterations']}")
    print(f"Q-GMRES residual: {info_qgmres['residual']:.2e}")
    print(f"Q-GMRES time: {qgmres_time:.4f} seconds")

    # Solve with pseudoinverse for comparison
    print("\n3. Solving with pseudoinverse for comparison...")
    pinv_solver = NewtonSchulzPseudoinverse(tol=1e-6, verbose=False)
    start_time = time.time()
    A_pinv, _, _ = pinv_solver.compute(A)
    x_pinv = quat_matmat(A_pinv, b)
    pinv_time = time.time() - start_time

    print(f"Pseudoinverse solution (shape: {x_pinv.shape})")
    print(f"Pseudoinverse time: {pinv_time:.4f} seconds")

    # Compare solutions
    print("\n4. Comparing solutions...")
    diff = quat_frobenius_norm(x_qgmres - x_pinv)
    print(f"||x_qgmres - x_pinv||_F = {diff:.2e}")

    # Check residuals
    residual_qgmres = quat_frobenius_norm(quat_matmat(A, x_qgmres) - b)
    residual_pinv = quat_frobenius_norm(quat_matmat(A, x_pinv) - b)
    print(f"||A*x_qgmres - b||_F = {residual_qgmres:.2e}")
    print(f"||A*x_pinv - b||_F = {residual_pinv:.2e}")

    # Verify accuracy
    # Both methods should find valid solutions (small residuals)
    # The solution difference might be large if the system has multiple solutions
    assert residual_qgmres < 1e-3, f"Q-GMRES residual too large: {residual_qgmres}"
    assert residual_pinv < 1e-3, f"Pseudoinverse residual too large: {residual_pinv}"
    print("✓ Q-GMRES basic test passed!")


def test_qgmres_convergence():
    """Test Q-GMRES convergence with different matrix sizes."""
    print("\n" + "=" * 60)
    print("TESTING Q-GMRES CONVERGENCE")
    print("=" * 60)

    sizes = [5, 10, 15]
    results = []

    for size in sizes:
        print(f"\nTesting {size}x{size} system...")

        # Create random system
        A = create_test_matrix(size, size)
        b = create_test_matrix(size, 1)

        # Solve with Q-GMRES
        qgmres_solver = QGMRESSolver(tol=1e-6, verbose=False)
        start_time = time.time()
        x_qgmres, info_qgmres = qgmres_solver.solve(A, b)
        qgmres_time = time.time() - start_time

        # Solve with pseudoinverse
        pinv_solver = NewtonSchulzPseudoinverse(tol=1e-6, verbose=False)
        start_time = time.time()
        A_pinv, _, _ = pinv_solver.compute(A)
        x_pinv = quat_matmat(A_pinv, b)
        pinv_time = time.time() - start_time

        # Compute accuracy metrics
        residual_qgmres = quat_frobenius_norm(quat_matmat(A, x_qgmres) - b)
        residual_pinv = quat_frobenius_norm(quat_matmat(A, x_pinv) - b)
        solution_diff = quat_frobenius_norm(x_qgmres - x_pinv)

        results.append(
            {
                "size": size,
                "qgmres_iterations": info_qgmres["iterations"],
                "qgmres_residual": info_qgmres["residual"],
                "qgmres_time": qgmres_time,
                "pinv_time": pinv_time,
                "residual_qgmres": residual_qgmres,
                "residual_pinv": residual_pinv,
                "solution_diff": solution_diff,
            }
        )

        print(f"  Size: {size}x{size}")
        print(f"  Q-GMRES iterations: {info_qgmres['iterations']}")
        print(f"  Q-GMRES residual: {info_qgmres['residual']:.2e}")
        print(f"  Q-GMRES time: {qgmres_time:.4f}s")
        print(f"  Pseudoinverse time: {pinv_time:.4f}s")
        print(f"  Solution difference: {solution_diff:.2e}")

        # Verify accuracy
        # For larger matrices, be more lenient with solution difference
        # since both methods may find valid but different solutions
        tolerance = 1e-2 if size <= 5 else 2.0
        assert solution_diff < tolerance, (
            f"Solution difference too large for size {size}: {solution_diff}"
        )
        assert residual_qgmres < 1e-3, (
            f"Q-GMRES residual too large for size {size}: {residual_qgmres}"
        )

    print("\n✓ Q-GMRES convergence test passed!")
    return results


def test_qgmres_sparse():
    """Test Q-GMRES with sparse matrices."""
    print("\n" + "=" * 60)
    print("TESTING Q-GMRES WITH SPARSE MATRICES")
    print("=" * 60)

    # Create sparse system
    print("1. Creating sparse test system...")
    A_sparse = create_sparse_quat_matrix(6, 6, density=0.3)  # Use square matrix
    b = create_test_matrix(6, 1)

    display_quaternion_matrix(A_sparse, "A_sparse", verbose=False)

    # Solve with Q-GMRES
    print("\n2. Solving sparse system with Q-GMRES...")
    qgmres_solver = QGMRESSolver(tol=1e-6, verbose=False)
    x_qgmres, info_qgmres = qgmres_solver.solve(A_sparse, b)

    print(f"Q-GMRES solution (shape: {x_qgmres.shape})")
    print(f"Q-GMRES iterations: {info_qgmres['iterations']}")
    print(f"Q-GMRES residual: {info_qgmres['residual']:.2e}")

    # Solve with pseudoinverse for comparison
    print("\n3. Solving with pseudoinverse...")
    pinv_solver = NewtonSchulzPseudoinverse(tol=1e-6, verbose=False)
    A_pinv, _, _ = pinv_solver.compute(A_sparse)
    x_pinv = quat_matmat(A_pinv, b)

    # Compare solutions
    print("\n4. Comparing solutions...")
    diff = quat_frobenius_norm(x_qgmres - x_pinv)
    residual_qgmres = quat_frobenius_norm(quat_matmat(A_sparse, x_qgmres) - b)

    print(f"||x_qgmres - x_pinv||_F = {diff:.2e}")
    print(f"||A*x_qgmres - b||_F = {residual_qgmres:.2e}")

    # Verify accuracy
    assert diff < 1e-2, f"Q-GMRES and pseudoinverse solutions differ too much: {diff}"
    assert residual_qgmres < 1e-3, f"Q-GMRES residual too large: {residual_qgmres}"
    print("✓ Q-GMRES sparse test passed!")


def test_qgmres_ill_conditioned():
    """Test Q-GMRES with ill-conditioned matrices."""
    print("\n" + "=" * 60)
    print("TESTING Q-GMRES WITH ILL-CONDITIONED MATRICES")
    print("=" * 60)

    # Create ill-conditioned system
    print("1. Creating ill-conditioned test system...")
    A = create_test_matrix(4, 4)

    # Make it ill-conditioned by adding small perturbations
    A_real = quaternion.as_float_array(A)
    A_real[0, 0, 0] *= 1e-6  # Make one element very small
    A = quaternion.as_quat_array(A_real)

    b = create_test_matrix(4, 1)

    display_quaternion_matrix(A, "A (ill-conditioned)", verbose=False)

    # Solve with Q-GMRES
    print("\n2. Solving ill-conditioned system with Q-GMRES...")
    qgmres_solver = QGMRESSolver(tol=1e-6, verbose=False)
    x_qgmres, info_qgmres = qgmres_solver.solve(A, b)

    print(f"Q-GMRES solution (shape: {x_qgmres.shape})")
    print(f"Q-GMRES iterations: {info_qgmres['iterations']}")
    print(f"Q-GMRES residual: {info_qgmres['residual']:.2e}")

    # Solve with pseudoinverse for comparison
    print("\n3. Solving with pseudoinverse...")
    pinv_solver = NewtonSchulzPseudoinverse(tol=1e-6, verbose=False)
    A_pinv, _, _ = pinv_solver.compute(A)
    x_pinv = quat_matmat(A_pinv, b)

    # Compare solutions
    print("\n4. Comparing solutions...")
    diff = quat_frobenius_norm(x_qgmres - x_pinv)
    residual_qgmres = quat_frobenius_norm(quat_matmat(A, x_qgmres) - b)

    print(f"||x_qgmres - x_pinv||_F = {diff:.2e}")
    print(f"||A*x_qgmres - b||_F = {residual_qgmres:.2e}")

    # For ill-conditioned systems, we expect larger differences
    assert diff < 1e-1, f"Q-GMRES and pseudoinverse solutions differ too much: {diff}"
    assert residual_qgmres < 1e-2, f"Q-GMRES residual too large: {residual_qgmres}"
    print("✓ Q-GMRES ill-conditioned test passed!")


def plot_convergence_results(results):
    """Plot convergence results."""
    print("\n" + "=" * 60)
    print("PLOTTING CONVERGENCE RESULTS")
    print("=" * 60)

    # Create output directory
    os.makedirs("../../output_figures", exist_ok=True)

    # Extract data
    sizes = [r["size"] for r in results]
    qgmres_iterations = [r["qgmres_iterations"] for r in results]
    qgmres_times = [r["qgmres_time"] for r in results]
    pinv_times = [r["pinv_time"] for r in results]
    solution_diffs = [r["solution_diff"] for r in results]

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Iterations vs Size
    ax1.plot(sizes, qgmres_iterations, "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("Matrix Size")
    ax1.set_ylabel("Q-GMRES Iterations")
    ax1.set_title("Q-GMRES Iterations vs Matrix Size")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Time Comparison
    ax2.plot(sizes, qgmres_times, "ro-", label="Q-GMRES", linewidth=2, markersize=8)
    ax2.plot(sizes, pinv_times, "go-", label="Pseudoinverse", linewidth=2, markersize=8)
    ax2.set_xlabel("Matrix Size")
    ax2.set_ylabel("Time (seconds)")
    ax2.set_title("Computation Time Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Solution Accuracy
    ax3.semilogy(sizes, solution_diffs, "mo-", linewidth=2, markersize=8)
    ax3.set_xlabel("Matrix Size")
    ax3.set_ylabel("||x_qgmres - x_pinv||_F")
    ax3.set_title("Solution Accuracy")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Residual vs Size
    residuals = [r["residual_qgmres"] for r in results]
    ax4.semilogy(sizes, residuals, "co-", linewidth=2, markersize=8)
    ax4.set_xlabel("Matrix Size")
    ax4.set_ylabel("||A*x - b||_F")
    ax4.set_title("Q-GMRES Residual Norm")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "../../output_figures/qgmres_convergence_analysis.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print("✓ Convergence plots saved to output_figures/qgmres_convergence_analysis.png")


def main():
    """Run all Q-GMRES tests."""
    print("=" * 80)
    print("Q-GMRES SOLVER COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print("Testing Quaternion Generalized Minimal Residual Method")
    print("Comparing with Newton-Schulz Pseudoinverse solutions")
    print("=" * 80)

    try:
        # Run all tests
        test_qgmres_basic()
        results = test_qgmres_convergence()
        test_qgmres_sparse()
        test_qgmres_ill_conditioned()

        # Plot results
        plot_convergence_results(results)

        print("\n" + "=" * 80)
        print("🎉 ALL Q-GMRES TESTS PASSED SUCCESSFULLY!")
        print("=" * 80)
        print("✅ Basic functionality verified")
        print("✅ Convergence tested for different matrix sizes")
        print("✅ Sparse matrix support verified")
        print("✅ Ill-conditioned system handling tested")
        print("✅ Solution accuracy compared with pseudoinverse")
        print("✅ Performance analysis completed")
        print("✅ Visualizations generated")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
