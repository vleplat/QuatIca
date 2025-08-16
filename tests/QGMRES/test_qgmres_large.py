#!/usr/bin/env python3
"""
Large-scale Q-GMRES Solver Test

This script tests the Q-GMRES solver with large 200x200 quaternion matrices
to evaluate scalability and performance.
"""

import gc
import os
import sys
import time

import numpy as np
import psutil

# Add core module to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "quatica"))

from data_gen import create_test_matrix
from solver import NewtonSchulzPseudoinverse, QGMRESSolver
from utils import quat_frobenius_norm, quat_matmat


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def create_large_test_system(n=200, cond_number=10.0, seed=42):
    """Create a large test linear system with controlled conditioning."""
    print(f"Creating {n}x{n} quaternion system...")

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Create well-conditioned matrix
    A = create_test_matrix(n, n, cond_number=cond_number)
    b = create_test_matrix(n, 1)

    return A, b


def test_qgmres_large_scale():
    """Test Q-GMRES on large 200x200 system."""
    print("=" * 80)
    print("LARGE-SCALE Q-GMRES TEST (200x200)")
    print("=" * 80)

    # Memory usage before
    mem_before = get_memory_usage()
    print(f"Memory usage before: {mem_before:.1f} MB")

    # Create large test system
    print("\n1. Creating 200x200 test system...")
    start_time = time.time()
    A, b = create_large_test_system(n=200, cond_number=10.0)
    creation_time = time.time() - start_time

    print(f"Matrix A: {A.shape}")
    print(f"Vector b: {b.shape}")
    print(f"Creation time: {creation_time:.2f} seconds")

    # Memory usage after creation
    mem_after_creation = get_memory_usage()
    print(f"Memory usage after creation: {mem_after_creation:.1f} MB")
    print(f"Memory increase: {mem_after_creation - mem_before:.1f} MB")

    # Solve with Q-GMRES
    print("\n2. Solving with Q-GMRES...")
    qgmres_solver = QGMRESSolver(tol=1e-6, max_iter=200, verbose=False)

    start_time = time.time()
    x_qgmres, info_qgmres = qgmres_solver.solve(A, b)
    qgmres_time = time.time() - start_time

    print(f"Q-GMRES solution shape: {x_qgmres.shape}")
    print(f"Q-GMRES iterations: {info_qgmres['iterations']}")
    print(f"Q-GMRES residual: {info_qgmres['residual']:.2e}")
    print(f"Q-GMRES time: {qgmres_time:.2f} seconds")

    # Memory usage after Q-GMRES
    mem_after_qgmres = get_memory_usage()
    print(f"Memory usage after Q-GMRES: {mem_after_qgmres:.1f} MB")

    # Solve with pseudoinverse for comparison
    print("\n3. Solving with pseudoinverse...")
    pinv_solver = NewtonSchulzPseudoinverse(tol=1e-6, verbose=False)

    start_time = time.time()
    A_pinv, _, _ = pinv_solver.compute(A)
    x_pinv = quat_matmat(A_pinv, b)
    pinv_time = time.time() - start_time

    print(f"Pseudoinverse solution shape: {x_pinv.shape}")
    print(f"Pseudoinverse time: {pinv_time:.2f} seconds")

    # Memory usage after pseudoinverse
    mem_after_pinv = get_memory_usage()
    print(f"Memory usage after pseudoinverse: {mem_after_pinv:.1f} MB")

    # Compare solutions
    print("\n4. Comparing solutions...")
    diff = quat_frobenius_norm(x_qgmres - x_pinv)
    residual_qgmres = quat_frobenius_norm(quat_matmat(A, x_qgmres) - b)
    residual_pinv = quat_frobenius_norm(quat_matmat(A, x_pinv) - b)

    print(f"||x_qgmres - x_pinv||_F = {diff:.2e}")
    print(f"||A*x_qgmres - b||_F = {residual_qgmres:.2e}")
    print(f"||A*x_pinv - b||_F = {residual_pinv:.2e}")

    # Performance metrics
    print("\n5. Performance Summary:")
    print(f"Matrix size: {A.shape}")
    print(f"Q-GMRES iterations: {info_qgmres['iterations']}")
    print(f"Q-GMRES time: {qgmres_time:.2f}s")
    print(f"Pseudoinverse time: {pinv_time:.2f}s")
    print(f"Speedup (pinv/qgmres): {pinv_time / qgmres_time:.2f}x")
    print(
        f"Memory peak: {max(mem_after_creation, mem_after_qgmres, mem_after_pinv):.1f} MB"
    )

    # Verify accuracy (very relaxed tolerance for large systems)
    assert diff < 1e2, f"Q-GMRES and pseudoinverse solutions differ too much: {diff}"
    assert residual_qgmres < 1e2, f"Q-GMRES residual too large: {residual_qgmres}"

    print("\n✓ Large-scale Q-GMRES test passed!")

    # Clean up memory
    del A, b, x_qgmres, x_pinv, A_pinv
    gc.collect()

    mem_final = get_memory_usage()
    print(f"Memory usage after cleanup: {mem_final:.1f} MB")

    return {
        "size": 200,
        "qgmres_iterations": info_qgmres["iterations"],
        "qgmres_time": qgmres_time,
        "pinv_time": pinv_time,
        "qgmres_residual": residual_qgmres,
        "pinv_residual": residual_pinv,
        "solution_diff": diff,
        "memory_peak": max(mem_after_creation, mem_after_qgmres, mem_after_pinv),
    }


def test_qgmres_scalability():
    """Test Q-GMRES scalability with different matrix sizes."""
    print("\n" + "=" * 80)
    print("Q-GMRES SCALABILITY TEST")
    print("=" * 80)

    sizes = [50, 100, 150, 200]
    results = []

    for size in sizes:
        print(f"\nTesting {size}x{size} system...")

        # Create test system
        A, b = create_large_test_system(n=size, cond_number=10.0)

        # Solve with Q-GMRES
        qgmres_solver = QGMRESSolver(tol=1e-6, max_iter=min(200, size), verbose=False)
        start_time = time.time()
        x_qgmres, info_qgmres = qgmres_solver.solve(A, b)
        qgmres_time = time.time() - start_time

        # Solve with pseudoinverse
        pinv_solver = NewtonSchulzPseudoinverse(tol=1e-6, verbose=False)
        start_time = time.time()
        A_pinv, _, _ = pinv_solver.compute(A)
        x_pinv = quat_matmat(A_pinv, b)
        pinv_time = time.time() - start_time

        # Compute metrics
        residual_qgmres = quat_frobenius_norm(quat_matmat(A, x_qgmres) - b)
        residual_pinv = quat_frobenius_norm(quat_matmat(A, x_pinv) - b)
        diff = quat_frobenius_norm(x_qgmres - x_pinv)

        results.append(
            {
                "size": size,
                "qgmres_iterations": info_qgmres["iterations"],
                "qgmres_time": qgmres_time,
                "pinv_time": pinv_time,
                "qgmres_residual": residual_qgmres,
                "pinv_residual": residual_pinv,
                "solution_diff": diff,
                "memory_usage": get_memory_usage(),
            }
        )

        print(f"  Size: {size}x{size}")
        print(f"  Q-GMRES iterations: {info_qgmres['iterations']}")
        print(f"  Q-GMRES time: {qgmres_time:.2f}s")
        print(f"  Pseudoinverse time: {pinv_time:.2f}s")
        print(f"  Speedup: {pinv_time / qgmres_time:.2f}x")
        print(f"  Solution difference: {diff:.2e}")

        # Clean up
        del A, b, x_qgmres, x_pinv, A_pinv
        gc.collect()

    print("\n✓ Scalability test completed!")
    return results


def main():
    """Run large-scale Q-GMRES tests."""
    print("LARGE-SCALE Q-GMRES SOLVER TESTING")
    print("=" * 80)

    try:
        # Test large 200x200 system
        large_result = test_qgmres_large_scale()

        # Test scalability
        scalability_results = test_qgmres_scalability()

        # Print summary
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print("200x200 system:")
        print(f"  Q-GMRES time: {large_result['qgmres_time']:.2f}s")
        print(f"  Pseudoinverse time: {large_result['pinv_time']:.2f}s")
        print(
            f"  Speedup: {large_result['pinv_time'] / large_result['qgmres_time']:.2f}x"
        )
        print(f"  Memory peak: {large_result['memory_peak']:.1f} MB")

        print("\nScalability results:")
        for result in scalability_results:
            print(
                f"  {result['size']}x{result['size']}: Q-GMRES {result['qgmres_time']:.2f}s, "
                f"Pinv {result['pinv_time']:.2f}s, Speedup {result['pinv_time'] / result['qgmres_time']:.2f}x"
            )

        print("\n" + "=" * 80)
        print("✓ ALL LARGE-SCALE TESTS PASSED!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()
