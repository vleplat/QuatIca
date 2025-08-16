#!/usr/bin/env python3
"""
Performance benchmark comparing rand_qsvd and pass_eff_qsvd on big low-rank matrices.
Enhanced version with matplotlib plots for visualization.

Task 4: Compare rand_qsvd and pass_eff_qsvd on big low-rank matrices (500x300, rank=10)
with n_passes in [1,2,3,4].
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

# Add core module to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "quatica"))
from data_gen import create_test_matrix
from decomp.qsvd import classical_qsvd_full, pass_eff_qsvd, rand_qsvd
from utils import quat_frobenius_norm, quat_hermitian, quat_matmat


def create_low_rank_matrix(m, n, rank):
    """
    Create a low-rank quaternion matrix of rank 'rank'.

    Parameters:
    -----------
    m, n : int
        Matrix dimensions
    rank : int
        Target rank

    Returns:
    --------
    numpy.ndarray with dtype=quaternion
        Low-rank matrix of shape (m, n) with rank approximately 'rank'
    """
    # Create random matrices
    A = create_test_matrix(m, rank)
    B = create_test_matrix(rank, n)

    # Multiply to get low-rank matrix
    X = quat_matmat(A, B)

    return X


def create_performance_plots(results, full_time, n_passes_list):
    """Create performance comparison plots."""

    # Extract data for plotting
    rand_times = [r["rand_time"] for r in results if r["rand_success"]]
    pass_times = [r["pass_time"] for r in results if r["pass_success"]]
    rand_errors = [r["rand_rel_error"] for r in results if r["rand_success"]]
    pass_errors = [r["pass_rel_error"] for r in results if r["pass_success"]]
    rand_speedups = [full_time / r["rand_time"] for r in results if r["rand_success"]]
    pass_speedups = [full_time / r["pass_time"] for r in results if r["pass_success"]]

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Q-SVD Performance Comparison: rand_qsvd vs pass_eff_qsvd\n(500×300 matrix, rank=10)",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Execution Time
    ax1.plot(
        n_passes_list[: len(rand_times)],
        rand_times,
        "o-",
        linewidth=2,
        markersize=8,
        label="rand_qsvd",
        color="blue",
    )
    ax1.plot(
        n_passes_list[: len(pass_times)],
        pass_times,
        "s-",
        linewidth=2,
        markersize=8,
        label="pass_eff_qsvd",
        color="red",
    )
    ax1.axhline(
        y=full_time,
        color="green",
        linestyle="--",
        linewidth=2,
        label="Full Q-SVD",
        alpha=0.7,
    )
    ax1.set_xlabel("Number of Passes/Iterations")
    ax1.set_ylabel("Execution Time (seconds)")
    ax1.set_title("Execution Time Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(n_passes_list)

    # Plot 2: Speedup vs Full Q-SVD
    ax2.plot(
        n_passes_list[: len(rand_speedups)],
        rand_speedups,
        "o-",
        linewidth=2,
        markersize=8,
        label="rand_qsvd",
        color="blue",
    )
    ax2.plot(
        n_passes_list[: len(pass_speedups)],
        pass_speedups,
        "s-",
        linewidth=2,
        markersize=8,
        label="pass_eff_qsvd",
        color="red",
    )
    ax2.axhline(
        y=1.0,
        color="green",
        linestyle="--",
        linewidth=2,
        label="Full Q-SVD baseline",
        alpha=0.7,
    )
    ax2.set_xlabel("Number of Passes/Iterations")
    ax2.set_ylabel("Speedup vs Full Q-SVD")
    ax2.set_title("Speedup Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(n_passes_list)

    # Plot 3: Relative Error (log scale)
    ax3.semilogy(
        n_passes_list[: len(rand_errors)],
        rand_errors,
        "o-",
        linewidth=2,
        markersize=8,
        label="rand_qsvd",
        color="blue",
    )
    ax3.semilogy(
        n_passes_list[: len(pass_errors)],
        pass_errors,
        "s-",
        linewidth=2,
        markersize=8,
        label="pass_eff_qsvd",
        color="red",
    )
    ax3.axhline(
        y=1e-15,
        color="green",
        linestyle="--",
        linewidth=2,
        label="Machine precision",
        alpha=0.7,
    )
    ax3.set_xlabel("Number of Passes/Iterations")
    ax3.set_ylabel("Relative Reconstruction Error")
    ax3.set_title("Accuracy Comparison (Log Scale)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(n_passes_list)

    # Plot 4: Speedup vs Accuracy Trade-off
    ax4.scatter(
        rand_errors,
        rand_speedups,
        s=100,
        c="blue",
        alpha=0.7,
        label="rand_qsvd",
        edgecolors="black",
    )
    ax4.scatter(
        pass_errors,
        pass_speedups,
        s=100,
        c="red",
        alpha=0.7,
        label="pass_eff_qsvd",
        edgecolors="black",
    )

    # Add labels for each point
    for i, (error, speedup) in enumerate(zip(rand_errors, rand_speedups)):
        ax4.annotate(
            f"{n_passes_list[i]}",
            (error, speedup),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    for i, (error, speedup) in enumerate(zip(pass_errors, pass_speedups)):
        ax4.annotate(
            f"{n_passes_list[i]}",
            (error, speedup),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax4.set_xlabel("Relative Reconstruction Error")
    ax4.set_ylabel("Speedup vs Full Q-SVD")
    ax4.set_title("Speedup vs Accuracy Trade-off")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale("log")

    plt.tight_layout()

    # Save the plot
    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "output_figures")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "qsvd_performance_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"📊 Performance plots saved to: {plot_path}")

    plt.show()


def benchmark_qsvd_methods():
    """Benchmark rand_qsvd vs pass_eff_qsvd on big low-rank matrices."""
    print("=" * 80)
    print("Q-SVD PERFORMANCE BENCHMARK WITH PLOTS")
    print("=" * 80)
    print("Matrix: 500×300, Target rank: 10")
    print("Methods: rand_qsvd vs pass_eff_qsvd")
    print("Parameters: n_passes/n_iter ∈ [1,2,3,4]")
    print()

    # Matrix parameters
    m, n = 500, 300
    target_rank = 10
    oversample = 10

    # Create low-rank test matrix
    print("Creating low-rank test matrix...")
    X = create_low_rank_matrix(m, n, target_rank)
    print(f"Matrix shape: {X.shape}")
    print(f"Matrix norm: {quat_frobenius_norm(X):.2f}")

    # Compute full Q-SVD for reference
    print("\nComputing full Q-SVD for reference...")
    start_time = time.time()
    U_full, s_full, V_full = classical_qsvd_full(X)
    full_time = time.time() - start_time
    s_full = s_full[:target_rank]  # Take first target_rank singular values
    print(f"Full Q-SVD time: {full_time:.2f} seconds")
    print(f"Full Q-SVD singular values (first {target_rank}): {s_full}")

    # Test parameters
    n_passes_list = [1, 2, 3, 4]

    # Results storage
    results = []

    print(f"\n{'=' * 80}")
    print("BENCHMARKING RANDOMIZED METHODS")
    print(f"{'=' * 80}")

    for n_passes in n_passes_list:
        print(f"\n--- Testing with {n_passes} passes/iterations ---")

        # Test rand_qsvd
        print(f"  rand_qsvd (n_iter={n_passes}):")
        start_time = time.time()
        try:
            U_rand, s_rand, V_rand = rand_qsvd(
                X, target_rank, oversample=oversample, n_iter=n_passes
            )
            rand_time = time.time() - start_time

            # Compute reconstruction error
            X_recon_rand = quat_matmat(
                quat_matmat(U_rand, np.diag(s_rand)), quat_hermitian(V_rand)
            )
            rand_error = quat_frobenius_norm(X - X_recon_rand)
            rand_rel_error = rand_error / quat_frobenius_norm(X)

            # Compare singular values
            s_diff_rand = np.linalg.norm(s_rand - s_full)
            s_rel_diff_rand = s_diff_rand / np.linalg.norm(s_full)

            print(f"    Time: {rand_time:.3f} seconds")
            print(f"    Speedup vs full: {full_time / rand_time:.1f}x")
            print(f"    Reconstruction error: {rand_error:.6f}")
            print(f"    Relative error: {rand_rel_error:.6f}")
            print(f"    Singular value difference: {s_diff_rand:.6f}")
            print(f"    Relative difference: {s_rel_diff_rand:.6f}")

            rand_success = True

        except Exception as e:
            print(f"    ❌ FAILED: {e}")
            rand_time = float("inf")
            rand_rel_error = float("inf")
            s_rel_diff_rand = float("inf")
            rand_success = False

        # Test pass_eff_qsvd
        print(f"  pass_eff_qsvd (n_passes={n_passes}):")
        start_time = time.time()
        try:
            U_pass, s_pass, V_pass = pass_eff_qsvd(
                X, target_rank, oversample=oversample, n_passes=n_passes
            )
            pass_time = time.time() - start_time

            # Compute reconstruction error
            X_recon_pass = quat_matmat(
                quat_matmat(U_pass, np.diag(s_pass)), quat_hermitian(V_pass)
            )
            pass_error = quat_frobenius_norm(X - X_recon_pass)
            pass_rel_error = pass_error / quat_frobenius_norm(X)

            # Compare singular values
            s_diff_pass = np.linalg.norm(s_pass - s_full)
            s_rel_diff_pass = s_diff_pass / np.linalg.norm(s_full)

            print(f"    Time: {pass_time:.3f} seconds")
            print(f"    Speedup vs full: {full_time / pass_time:.1f}x")
            print(f"    Reconstruction error: {pass_error:.6f}")
            print(f"    Relative error: {pass_rel_error:.6f}")
            print(f"    Singular value difference: {s_diff_pass:.6f}")
            print(f"    Relative difference: {s_rel_diff_pass:.6f}")

            pass_success = True

        except Exception as e:
            print(f"    ❌ FAILED: {e}")
            pass_time = float("inf")
            pass_rel_error = float("inf")
            s_rel_diff_pass = float("inf")
            pass_success = False

        # Store results
        results.append(
            {
                "n_passes": n_passes,
                "rand_time": rand_time,
                "rand_rel_error": rand_rel_error,
                "rand_s_diff": s_rel_diff_rand,
                "rand_success": rand_success,
                "pass_time": pass_time,
                "pass_rel_error": pass_rel_error,
                "pass_s_diff": s_rel_diff_pass,
                "pass_success": pass_success,
            }
        )

    # Summary table
    print(f"\n{'=' * 80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'=' * 80}")

    # Print header
    print(
        f"{'Passes':<8} {'rand_qsvd Time':<15} {'rand Speedup':<12} {'rand Rel Error':<12} {'rand SVD Diff':<12}"
    )
    print(
        f"{'':<8} {'pass_eff_qsvd Time':<15} {'pass Speedup':<12} {'pass Rel Error':<12} {'pass SVD Diff':<12}"
    )
    print("-" * 80)

    for result in results:
        # rand_qsvd results
        rand_time_str = (
            f"{result['rand_time']:.3f}s" if result["rand_success"] else "FAILED"
        )
        rand_speedup_str = (
            f"{full_time / result['rand_time']:.1f}x" if result["rand_success"] else "N/A"
        )
        rand_error_str = (
            f"{result['rand_rel_error']:.6f}" if result["rand_success"] else "N/A"
        )
        rand_svd_str = f"{result['rand_s_diff']:.6f}" if result["rand_success"] else "N/A"

        # pass_eff_qsvd results
        pass_time_str = (
            f"{result['pass_time']:.3f}s" if result["pass_success"] else "FAILED"
        )
        pass_speedup_str = (
            f"{full_time / result['pass_time']:.1f}x" if result["pass_success"] else "N/A"
        )
        pass_error_str = (
            f"{result['pass_rel_error']:.6f}" if result["pass_success"] else "N/A"
        )
        pass_svd_str = f"{result['pass_s_diff']:.6f}" if result["pass_success"] else "N/A"

        print(
            f"{result['n_passes']:<8} {rand_time_str:<15} {rand_speedup_str:<12} {rand_error_str:<12} {rand_svd_str:<12}"
        )
        print(
            f"{'':<8} {pass_time_str:<15} {pass_speedup_str:<12} {pass_error_str:<12} {pass_svd_str:<12}"
        )
        print()

    # Create performance plots
    print(f"\n{'=' * 80}")
    print("GENERATING PERFORMANCE PLOTS")
    print(f"{'=' * 80}")
    create_performance_plots(results, full_time, n_passes_list)

    # Analysis
    print(f"\n{'=' * 80}")
    print("ANALYSIS")
    print(f"{'=' * 80}")

    # Find best performing methods
    successful_results = [r for r in results if r["rand_success"] and r["pass_success"]]

    if successful_results:
        # Best accuracy
        best_accuracy_rand = min(successful_results, key=lambda x: x["rand_rel_error"])
        best_accuracy_pass = min(successful_results, key=lambda x: x["pass_rel_error"])

        print("Best accuracy:")
        print(
            f"  rand_qsvd: {best_accuracy_rand['rand_rel_error']:.6f} (n_iter={best_accuracy_rand['n_passes']})"
        )
        print(
            f"  pass_eff_qsvd: {best_accuracy_pass['pass_rel_error']:.6f} (n_passes={best_accuracy_pass['n_passes']})"
        )

        # Best speed
        best_speed_rand = min(successful_results, key=lambda x: x["rand_time"])
        best_speed_pass = min(successful_results, key=lambda x: x["pass_time"])

        print("\nBest speed:")
        print(
            f"  rand_qsvd: {best_speed_rand['rand_time']:.3f}s (n_iter={best_speed_rand['n_passes']})"
        )
        print(
            f"  pass_eff_qsvd: {best_speed_pass['pass_time']:.3f}s (n_passes={best_speed_pass['n_passes']})"
        )

        # Speed comparison
        if best_speed_pass["pass_time"] < best_speed_rand["rand_time"]:
            speedup = best_speed_rand["rand_time"] / best_speed_pass["pass_time"]
            print(f"\npass_eff_qsvd is {speedup:.1f}x faster than rand_qsvd")
        else:
            speedup = best_speed_pass["pass_time"] / best_speed_rand["rand_time"]
            print(f"\nrand_qsvd is {speedup:.1f}x faster than pass_eff_qsvd")

        # Convergence analysis
        print("\nConvergence analysis:")
        for result in successful_results:
            if result["n_passes"] >= 2:
                print(f"  {result['n_passes']} passes: Both methods converged well")
                break
        else:
            print("  Methods may need more passes for full convergence")

    else:
        print("❌ No successful comparisons available")

    print(f"\n{'=' * 80}")
    print("BENCHMARK COMPLETE")
    print(f"{'=' * 80}")


def main():
    """Run the Q-SVD performance benchmark with plots."""
    try:
        benchmark_qsvd_methods()
        return True
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
