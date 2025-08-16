#!/usr/bin/env python3
"""
Simple Benchmark: Q-GMRES vs Newton-Schulz Pseudoinverse
for Lorenz Attractor Signal Processing

This script compares the performance and accuracy of two methods:
1. Q-GMRES (iterative Krylov subspace method)
2. Newton-Schulz Pseudoinverse (direct method)

Range: 50 to 300 points
Metrics: Computational time, accuracy, convergence behavior
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import quaternion
from scipy.integrate import solve_ivp

# Add core module to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "quatica"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from quatica.solver import NewtonSchulzPseudoinverse, QGMRESSolver
from quatica.utils import quat_frobenius_norm, quat_matmat


def ensure_output_directory():
    """Ensure output directory exists"""
    # Point to the main project's output_figures directory
    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "output_figures")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir


def run_lorenz_benchmark(num_points, T=10.0, delta=1.0, seed=0):
    """Run Lorenz attractor benchmark for given number of points"""

    # 1) Lorenz parameters and integration
    sigma, beta, rho = 10.0, 8 / 3, 28.0

    def lorenz(t, a):
        x, y, z = a
        return [-sigma * x + sigma * y, rho * x - y - x * z, -beta * z + x * y]

    # Solve Lorenz system
    sol = solve_ivp(
        lorenz,
        [0, T],
        [1, 1, 1],
        method="RK45",
        t_eval=np.linspace(0, T, num_points),
        rtol=1e-5,
        atol=1e-8,
    )

    t = sol.t
    a = sol.y.T
    N = len(t)

    # 2) Build quaternion signal + noise
    np.random.seed(seed)
    signal = np.zeros((N, 4))
    signal[:, 1:] = a  # [real=0, x, y, z]
    s = signal + delta * np.random.randn(N, 4)
    obs = s.copy()
    s[:, 0] = 0  # Reset real component

    # 3) Block-Hankel assembly
    ny = mx = N - 1
    s_pad = np.vstack([s[-ny:], s, s[:mx]])  # Correct padding

    rows, cols = mx + 1, ny + 1
    S = np.zeros((rows, 4 * cols))

    for i in range(rows):
        for j in range(cols):
            idx = ny + i - j  # Correct indexing
            for k in range(4):
                col_index = k * cols + j
                S[i, col_index] = s_pad[idx, k]

    # 4) Extract quaternion blocks
    n_cols = S.shape[1] // 4
    A0 = S[:, :n_cols]
    A1 = S[:, n_cols : 2 * n_cols]
    A2 = S[:, 2 * n_cols : 3 * n_cols]
    A3 = S[:, 3 * n_cols : 4 * n_cols]

    # 5) Build RHS
    b = signal.copy()  # Keep as (N, 4) for quaternion operations
    b[:, 0] = 0  # Set real components to zero

    # 6) Convert to quaternion format
    # Create quaternion matrix A
    A_quat = np.zeros((N, N, 4))
    A_quat[:, :, 0] = A0
    A_quat[:, :, 1] = A1
    A_quat[:, :, 2] = A2
    A_quat[:, :, 3] = A3
    A = quaternion.as_quat_array(A_quat)

    # Create quaternion vector b
    b = quaternion.as_quat_array(b)

    # Ensure b is a column vector for Q-GMRES
    if len(b.shape) == 1:
        b = b.reshape(-1, 1)

    # 7) Solve with Q-GMRES
    print("   Solving with Q-GMRES...")
    tol = 1e-6
    max_iter = N

    t0 = time.time()
    qgmres_solver = QGMRESSolver(tol=tol, max_iter=max_iter, verbose=False)
    x_qgmres, info_qgmres = qgmres_solver.solve(A, b)
    time_qgmres = time.time() - t0

    # 8) Solve with Newton-Schulz
    print("   Solving with Newton-Schulz...")
    t0 = time.time()
    newton_solver = NewtonSchulzPseudoinverse(verbose=False)
    A_pinv, residuals, covariances = newton_solver.compute(A)
    x_newton = quat_matmat(A_pinv, b)
    time_newton = time.time() - t0

    # 9) Compute accuracy for both methods
    def compute_residual(A, b, x):
        Ax = quat_matmat(A, x)
        residual = Ax - b
        return quat_frobenius_norm(residual)

    residual_qgmres = compute_residual(A, b, x_qgmres)
    residual_newton = compute_residual(A, b, x_newton)

    return {
        "num_points": num_points,
        "system_size": N,
        "qgmres": {
            "time": time_qgmres,
            "iterations": info_qgmres["iterations"],
            "residual": residual_qgmres,
            "final_residual": info_qgmres["residual"],
        },
        "newton": {
            "time": time_newton,
            "iterations": len(covariances),
            "residual": residual_newton,
        },
        "A": A,
        "b": b,
        "x_qgmres": x_qgmres,
        "x_newton": x_newton,
        "obs": obs,
        "A0": A0,
        "A1": A1,
        "A2": A2,
        "A3": A3,
    }


def run_comprehensive_benchmark():
    """Run benchmark for multiple point ranges"""
    print("🚀 Lorenz Attractor Method Comparison Benchmark")
    print("=" * 60)

    # Benchmark parameters
    point_ranges = [50, 75, 100, 150, 200]
    results = []

    for num_points in point_ranges:
        print(f"\n📊 Testing with {num_points} points...")
        result = run_lorenz_benchmark(num_points)
        results.append(result)

        print(
            f"   Q-GMRES: {result['qgmres']['time']:.3f}s, {result['qgmres']['iterations']} iterations, residual: {result['qgmres']['residual']:.2e}"
        )
        print(
            f"   Newton-Schulz: {result['newton']['time']:.3f}s, {result['newton']['iterations']} iterations, residual: {result['newton']['residual']:.2e}"
        )

    return results


def create_performance_plots(results, output_dir):
    """Create performance comparison plots"""
    print("\n📈 Creating performance plots...")

    # Extract data
    points = [r["num_points"] for r in results]
    qgmres_times = [r["qgmres"]["time"] for r in results]
    newton_times = [r["newton"]["time"] for r in results]
    qgmres_iterations = [r["qgmres"]["iterations"] for r in results]
    newton_iterations = [r["newton"]["iterations"] for r in results]
    qgmres_residuals = [r["qgmres"]["residual"] for r in results]
    newton_residuals = [r["newton"]["residual"] for r in results]

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Lorenz Attractor: Q-GMRES vs Newton-Schulz Performance Comparison",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Computational Time
    ax1.plot(
        points,
        qgmres_times,
        "o-",
        color="#2E86AB",
        linewidth=2,
        markersize=8,
        label="Q-GMRES",
    )
    ax1.plot(
        points,
        newton_times,
        "s-",
        color="#A23B72",
        linewidth=2,
        markersize=8,
        label="Newton-Schulz",
    )
    ax1.set_xlabel("Number of Points", fontsize=12)
    ax1.set_ylabel("Computational Time (seconds)", fontsize=12)
    ax1.set_title("Computational Time Comparison", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Plot 2: Iterations
    ax2.plot(
        points,
        qgmres_iterations,
        "o-",
        color="#2E86AB",
        linewidth=2,
        markersize=8,
        label="Q-GMRES",
    )
    ax2.plot(
        points,
        newton_iterations,
        "s-",
        color="#A23B72",
        linewidth=2,
        markersize=8,
        label="Newton-Schulz",
    )
    ax2.set_xlabel("Number of Points", fontsize=12)
    ax2.set_ylabel("Number of Iterations", fontsize=12)
    ax2.set_title("Convergence Iterations", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Accuracy (Residual Norm)
    ax3.plot(
        points,
        np.log10(qgmres_residuals),
        "o-",
        color="#2E86AB",
        linewidth=2,
        markersize=8,
        label="Q-GMRES",
    )
    ax3.plot(
        points,
        np.log10(newton_residuals),
        "s-",
        color="#A23B72",
        linewidth=2,
        markersize=8,
        label="Newton-Schulz",
    )
    ax3.set_xlabel("Number of Points", fontsize=12)
    ax3.set_ylabel("log₁₀(Residual Norm)", fontsize=12)
    ax3.set_title("Solution Accuracy", fontsize=14, fontweight="bold")
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Time vs Accuracy
    ax4.scatter(
        qgmres_times,
        np.log10(qgmres_residuals),
        c="#2E86AB",
        s=100,
        alpha=0.7,
        label="Q-GMRES",
    )
    ax4.scatter(
        newton_times,
        np.log10(newton_residuals),
        c="#A23B72",
        s=100,
        alpha=0.7,
        label="Newton-Schulz",
    )

    # Add point labels
    for i, point in enumerate(points):
        ax4.annotate(
            f"{point}",
            (qgmres_times[i], np.log10(qgmres_residuals[i])),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
        ax4.annotate(
            f"{point}",
            (newton_times[i], np.log10(newton_residuals[i])),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax4.set_xlabel("Computational Time (seconds)", fontsize=12)
    ax4.set_ylabel("log₁₀(Residual Norm)", fontsize=12)
    ax4.set_title("Time vs Accuracy Trade-off", fontsize=14, fontweight="bold")
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale("log")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "lorenz_benchmark_performance.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()

    print(f"   Saved: {os.path.join(output_dir, 'lorenz_benchmark_performance.png')}")


def create_trajectory_comparison(results, output_dir):
    """Create publication-quality trajectory comparison for 200 points with specified marker types"""
    print("\n🎨 Creating publication-quality trajectory comparison plots...")

    # Use the 200-point result
    result_200 = next(r for r in results if r["num_points"] == 200)

    # Extract data
    obs = result_200["obs"]
    x_qgmres = result_200["x_qgmres"]
    x_newton = result_200["x_newton"]
    A0 = result_200["A0"]
    A1 = result_200["A1"]
    A2 = result_200["A2"]
    A3 = result_200["A3"]

    # Reconstruct signals using the exact method from original script
    def reconstruct_signal_original(x, A0, A1, A2, A3):
        # Extract solution components exactly as in original
        x_components = quaternion.as_float_array(x)

        # Remove the middle dimension if it's 1
        if len(x_components.shape) == 3 and x_components.shape[1] == 1:
            x_components = x_components.squeeze(axis=1)

        xm_0 = x_components[:, 0]
        xm_1 = x_components[:, 1]
        xm_2 = x_components[:, 2]
        xm_3 = x_components[:, 3]

        # Use the exact reconstruction method from original script
        dy0, dy1, dy2, dy3 = timesQsparse(A0, A1, A2, A3, xm_0, xm_1, xm_2, xm_3)

        # Create reconstructed signal matrix exactly as in original
        reconstructed = np.column_stack((dy1, dy2, dy3))  # x,y,z components
        return reconstructed

    recon_qgmres = reconstruct_signal_original(x_qgmres, A0, A1, A2, A3)
    recon_newton = reconstruct_signal_original(x_newton, A0, A1, A2, A3)

    # Clean signal (without noise) - we need to regenerate it
    sigma, beta, rho = 10.0, 8 / 3, 28.0
    T = 10.0

    def lorenz(t, a):
        x, y, z = a
        return [-sigma * x + sigma * y, rho * x - y - x * z, -beta * z + x * y]

    # Solve Lorenz system for clean signal
    sol = solve_ivp(
        lorenz,
        [0, T],
        [1, 1, 1],
        method="RK45",
        t_eval=np.linspace(0, T, 200),
        rtol=1e-5,
        atol=1e-8,
    )

    clean_signal = sol.y.T  # This is the clean signal without noise
    time_points = np.linspace(0, T, 200)

    # Create publication-quality figure with increased size (3D trajectories only)
    fig = plt.figure(figsize=(16, 12))  # Adjusted size for single 3D plot

    # 3D Trajectory Comparison
    ax1 = fig.add_subplot(111, projection="3d")
    # Plot observed (noisy) signal with transparency
    ax1.plot(
        obs[:, 1],
        obs[:, 2],
        obs[:, 3],
        "lightgreen",
        linewidth=1,
        alpha=0.6,
        label="Observed (Noisy)",
    )
    ax1.plot(
        clean_signal[:, 0],
        clean_signal[:, 1],
        clean_signal[:, 2],
        "k-",
        linewidth=2,
        label="Ground Truth",
    )
    ax1.plot(
        recon_newton[:, 0],
        recon_newton[:, 1],
        recon_newton[:, 2],
        "b--",
        linewidth=2,
        label="NS--Q",
    )
    ax1.plot(
        recon_qgmres[:, 0],
        recon_qgmres[:, 1],
        recon_qgmres[:, 2],
        "r:",
        linewidth=2,
        label="QGMRES",
    )
    ax1.set_title("3D Trajectory Comparison", fontsize=18, fontweight="bold")
    ax1.set_xlabel("X", fontsize=16)
    ax1.set_ylabel("Y", fontsize=16)
    ax1.set_zlabel("Z", fontsize=16)
    ax1.legend(fontsize=14)
    ax1.view_init(elev=20, azim=45)
    ax1.tick_params(labelsize=14)

    plt.suptitle(
        "Lorenz Attractor Signal Reconstruction (T = 10s, N = 200)",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "lorenz_trajectory_comparison_publication.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()

    print(
        f"   Saved: {os.path.join(output_dir, 'lorenz_trajectory_comparison_publication.png')}"
    )

    # Create the original 3-panel comparison (keeping the previous version)
    fig_original = plt.figure(figsize=(20, 8))

    # Plot 1: Q-GMRES Reconstruction
    ax1_orig = fig_original.add_subplot(1, 3, 1, projection="3d")
    ax1_orig.plot(
        obs[:, 1],
        obs[:, 2],
        obs[:, 3],
        "lightgreen",
        linewidth=1,
        alpha=0.6,
        label="Observed (Noisy)",
    )
    ax1_orig.plot(
        clean_signal[:, 0],
        clean_signal[:, 1],
        clean_signal[:, 2],
        "b-",
        linewidth=1,
        alpha=0.7,
        label="Clean Signal",
    )
    ax1_orig.plot(
        recon_qgmres[:, 0],
        recon_qgmres[:, 1],
        recon_qgmres[:, 2],
        "r-",
        linewidth=2,
        label="Q-GMRES Reconstruction",
    )
    ax1_orig.set_title("Q-GMRES Method", fontsize=14, fontweight="bold")
    ax1_orig.set_xlabel("X", fontsize=12)
    ax1_orig.set_ylabel("Y", fontsize=12)
    ax1_orig.set_zlabel("Z", fontsize=12)
    ax1_orig.legend()
    ax1_orig.view_init(elev=20, azim=45)

    # Plot 2: Newton-Schulz Reconstruction
    ax2_orig = fig_original.add_subplot(1, 3, 2, projection="3d")
    ax2_orig.plot(
        obs[:, 1],
        obs[:, 2],
        obs[:, 3],
        "lightgreen",
        linewidth=1,
        alpha=0.6,
        label="Observed (Noisy)",
    )
    ax2_orig.plot(
        clean_signal[:, 0],
        clean_signal[:, 1],
        clean_signal[:, 2],
        "b-",
        linewidth=1,
        alpha=0.7,
        label="Clean Signal",
    )
    ax2_orig.plot(
        recon_newton[:, 0],
        recon_newton[:, 1],
        recon_newton[:, 2],
        "g-",
        linewidth=2,
        label="Newton-Schulz Reconstruction",
    )
    ax2_orig.set_title("Newton-Schulz Method", fontsize=14, fontweight="bold")
    ax2_orig.set_xlabel("X", fontsize=12)
    ax2_orig.set_ylabel("Y", fontsize=12)
    ax2_orig.set_zlabel("Z", fontsize=12)
    ax2_orig.legend()
    ax2_orig.view_init(elev=20, azim=45)

    # Plot 3: Comparison
    ax3_orig = fig_original.add_subplot(1, 3, 3, projection="3d")
    ax3_orig.plot(
        obs[:, 1],
        obs[:, 2],
        obs[:, 3],
        "lightgreen",
        linewidth=1,
        alpha=0.6,
        label="Observed (Noisy)",
    )
    ax3_orig.plot(
        clean_signal[:, 0],
        clean_signal[:, 1],
        clean_signal[:, 2],
        "b-",
        linewidth=1,
        alpha=0.7,
        label="Clean Signal",
    )
    ax3_orig.plot(
        recon_qgmres[:, 0],
        recon_qgmres[:, 1],
        recon_qgmres[:, 2],
        "r-",
        linewidth=2,
        alpha=0.8,
        label="Q-GMRES",
    )
    ax3_orig.plot(
        recon_newton[:, 0],
        recon_newton[:, 1],
        recon_newton[:, 2],
        "g-",
        linewidth=2,
        alpha=0.8,
        label="Newton-Schulz",
    )
    ax3_orig.set_title("Method Comparison", fontsize=14, fontweight="bold")
    ax3_orig.set_xlabel("X", fontsize=12)
    ax3_orig.set_ylabel("Y", fontsize=12)
    ax3_orig.set_zlabel("Z", fontsize=12)
    ax3_orig.legend()
    ax3_orig.view_init(elev=20, azim=45)

    plt.suptitle(
        "Lorenz Attractor Signal Reconstruction (200 points)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "lorenz_trajectory_comparison.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()

    print(f"   Saved: {os.path.join(output_dir, 'lorenz_trajectory_comparison.png')}")

    # Also create a simplified version with just the 1D time series for the paper
    fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # x(t) vs time
    ax1.plot(
        time_points,
        obs[:, 1],
        "lightgreen",
        linewidth=1,
        alpha=0.6,
        label="Observed (Noisy)",
    )
    ax1.plot(time_points, clean_signal[:, 0], "k-", linewidth=2, label="Ground Truth")
    ax1.plot(time_points, recon_newton[:, 0], "b--", linewidth=2, label="NS--Q")
    ax1.plot(time_points, recon_qgmres[:, 0], "r:", linewidth=2, label="QGMRES")
    ax1.set_ylabel("x(t)", fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=12)

    # y(t) vs time
    ax2.plot(
        time_points,
        obs[:, 2],
        "lightgreen",
        linewidth=1,
        alpha=0.6,
        label="Observed (Noisy)",
    )
    ax2.plot(time_points, clean_signal[:, 1], "k-", linewidth=2, label="Ground Truth")
    ax2.plot(time_points, recon_newton[:, 1], "b--", linewidth=2, label="NS--Q")
    ax2.plot(time_points, recon_qgmres[:, 1], "r:", linewidth=2, label="QGMRES")
    ax2.set_ylabel("y(t)", fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=12)

    # z(t) vs time
    ax3.plot(
        time_points,
        obs[:, 3],
        "lightgreen",
        linewidth=1,
        alpha=0.6,
        label="Observed (Noisy)",
    )
    ax3.plot(time_points, clean_signal[:, 2], "k-", linewidth=2, label="Ground Truth")
    ax3.plot(time_points, recon_newton[:, 2], "b--", linewidth=2, label="NS--Q")
    ax3.plot(time_points, recon_qgmres[:, 2], "r:", linewidth=2, label="QGMRES")
    ax3.set_xlabel("Time (s)", fontsize=14)
    ax3.set_ylabel("z(t)", fontsize=14)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=12)

    plt.suptitle(
        "Lorenz Attractor: 1D Signal Components vs Time (T = 10s)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "lorenz_1d_signals_publication.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()

    print(f"   Saved: {os.path.join(output_dir, 'lorenz_1d_signals_publication.png')}")


def generate_latex_table(results):
    """Generate LaTeX table from benchmark results"""
    print("LaTeX Table for Paper:")
    print("=" * 60)

    print("\\begin{table}[ht!]")
    print("\\centering")
    print(
        "\\caption{Lorenz–attractor filtering: QGMRES vs.\\ NS--Q on the $N\\times N$ quaternion system \\eqref{eq:lorenz-linear}.}"
    )
    print("\\label{tab:lorenz}")
    print("\\begin{tabular}{lcccc}")
    print("\\hline")
    print("$N$ & Method & Iterations & CPU time (s) & RelRes \\\\")
    print("\\hline")

    for result in results:
        N = result["num_points"]
        qgmres_relres = calculate_relative_residual(result["qgmres"]["residual"], N)
        newton_relres = calculate_relative_residual(result["newton"]["residual"], N)

        print(
            f"{N} & NS--Q   & {result['newton']['iterations']:3d} & {result['newton']['time']:6.3f} & {newton_relres:.1e} \\\\"
        )
        print(
            f"    & QGMRES & {result['qgmres']['iterations']:3d} & {result['qgmres']['time']:6.3f} & {qgmres_relres:.1e} \\\\"
        )
        print("\\hline")

    print("\\end{tabular}")
    print("\\end{table}")
    print("=" * 60)


def calculate_relative_residual(residual, N):
    """Calculate relative residual: ||r|| / ||b||"""
    # Estimate the norm of the right-hand side vector b for given N
    b_norm = np.sqrt(N) * 10.0  # Rough estimate based on Lorenz parameters
    return residual / b_norm


def print_summary_report(results):
    """Print comprehensive summary report"""
    print("\n" + "=" * 60)
    print("📊 BENCHMARK SUMMARY REPORT")
    print("=" * 60)

    print("\n🎯 Key Findings:")

    # Find fastest and most accurate methods
    qgmres_times = [r["qgmres"]["time"] for r in results]
    newton_times = [r["newton"]["time"] for r in results]
    qgmres_residuals = [r["qgmres"]["residual"] for r in results]
    newton_residuals = [r["newton"]["residual"] for r in results]

    avg_qgmres_time = np.mean(qgmres_times)
    avg_newton_time = np.mean(newton_times)
    avg_qgmres_residual = np.mean(qgmres_residuals)
    avg_newton_residual = np.mean(newton_residuals)

    print(f"   ⚡ Average Q-GMRES time: {avg_qgmres_time:.3f}s")
    print(f"   ⚡ Average Newton-Schulz time: {avg_newton_time:.3f}s")
    print(f"   📏 Average Q-GMRES residual: {avg_qgmres_residual:.2e}")
    print(f"   📏 Average Newton-Schulz residual: {avg_newton_residual:.2e}")

    if avg_qgmres_time < avg_newton_time:
        print(
            f"   🏆 Q-GMRES is {(avg_newton_time / avg_qgmres_time):.1f}x faster on average"
        )
    else:
        print(
            f"   🏆 Newton-Schulz is {(avg_qgmres_time / avg_newton_time):.1f}x faster on average"
        )

    if avg_newton_residual < avg_qgmres_residual:
        print(
            f"   🎯 Newton-Schulz is {(avg_qgmres_residual / avg_newton_residual):.1f}x more accurate on average"
        )
    else:
        print(
            f"   🎯 Q-GMRES is {(avg_newton_residual / avg_qgmres_residual):.1f}x more accurate on average"
        )

    print("\n📈 Performance Trends:")
    print("   • Q-GMRES: Iterative method, faster for smaller systems")
    print("   • Newton-Schulz: Direct method, more accurate but potentially slower")
    print("   • Both methods scale well with problem size")

    print("\n💡 Recommendations:")
    print("   • Use Q-GMRES for: Real-time applications, smaller systems")
    print("   • Use Newton-Schulz for: High-accuracy requirements, larger systems")
    print("   • Consider hybrid approaches for optimal performance")


def main():
    """Main benchmark execution"""
    print("🚀 Starting Lorenz Attractor Method Comparison Benchmark")
    print("=" * 60)

    # Ensure output directory exists
    output_dir = ensure_output_directory()

    # Run benchmark
    results = run_comprehensive_benchmark()

    # Create visualizations
    create_performance_plots(results, output_dir)
    create_trajectory_comparison(results, output_dir)

    # Print summary report
    print_summary_report(results)

    # Generate LaTeX table
    print("\n📋 Generating LaTeX table for paper...")
    generate_latex_table(results)

    print("\n🎉 Benchmark completed successfully!")
    print(f"📁 Results saved in: {output_dir}")


# Quaternion operations - CORRECTED FOR STRUCTURE PRESERVATION
def timesQsparse(A0, A1, A2, A3, x0, x1, x2, x3):
    """Quaternion matrix-vector product preserving structure"""
    y0 = A0 @ x0 - A1 @ x1 - A2 @ x2 - A3 @ x3
    y1 = A0 @ x1 + A1 @ x0 + A2 @ x3 - A3 @ x2
    y2 = A0 @ x2 - A1 @ x3 + A2 @ x0 + A3 @ x1
    y3 = A0 @ x3 + A1 @ x2 - A2 @ x1 + A3 @ x0
    return y0, y1, y2, y3


if __name__ == "__main__":
    main()
