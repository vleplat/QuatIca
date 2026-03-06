#!/usr/bin/env python3
"""
Simple Benchmark: Q-GMRES vs NS--Q vs LU (direct)
for Lorenz Attractor Signal Processing

This script compares the performance and accuracy of three methods:
1. Q-GMRES (iterative Krylov subspace method)
2. NS--Q (Newton-Schulz quaternion pseudoinverse, direct method)
3. LU (direct quaternion solve via LU factorization with partial pivoting)

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
import argparse

# Add core module to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "quatica"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from quatica.solver import NewtonSchulzPseudoinverse, QGMRESSolver
from quatica.utils import quat_frobenius_norm, quat_matmat
from quatica.decomp.LU import quaternion_lu


def _solve_lower_unit_diagonal(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve L y = b for y, where L is lower triangular with unit diagonal.

    Notes:
        This routine assumes the unknown appears on the right (matrix-vector product
        is Σ L_ij * y_j), matching QuatIca's conventions throughout the benchmark.
    """
    n = L.shape[0]
    y = np.zeros_like(b)
    for i in range(n):
        rhs = b[i, 0]
        for j in range(i):
            rhs = rhs - L[i, j] * y[j, 0]
        # L[i,i] = 1
        y[i, 0] = rhs
    return y


def _solve_upper(U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve U x = b for x, where U is upper triangular (quaternion diagonal).

    For the equation U_ii x_i = rhs (unknown on the right), we use left-multiplication
    by the inverse: x_i = U_ii^{-1} * rhs.
    """
    n = U.shape[0]
    x = np.zeros_like(b)
    for i in range(n - 1, -1, -1):
        rhs = b[i, 0]
        for j in range(i + 1, n):
            rhs = rhs - U[i, j] * x[j, 0]
        piv = U[i, i]
        x[i, 0] = (1.0 / piv) * rhs
    return x


def solve_via_quat_lu(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Direct solve of A x = b via quaternion LU with partial pivoting."""
    L, U, P = quaternion_lu(A, return_p=True)
    Pb = quat_matmat(P, b)
    y = _solve_lower_unit_diagonal(L, Pb)
    x = _solve_upper(U, y)
    return x


def ensure_output_directory():
    """Ensure output directory exists"""
    # Point to the main project's output_figures directory
    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "output_figures")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir


def _save_png_and_pdf(fig, path_png: str, *, dpi: int = 300) -> None:
    """Save a matplotlib figure as both PNG and PDF."""
    base, ext = os.path.splitext(path_png)
    if ext.lower() != ".png":
        path_png = base + ".png"
    path_pdf = base + ".pdf"
    fig.savefig(path_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(path_pdf, dpi=dpi, bbox_inches="tight", facecolor="white")


def run_lorenz_benchmark(num_points, T=10.0, delta=1.0, seed=0, methods=("qgmres", "newton", "lu")):
    """Run Lorenz attractor benchmark for given number of points."""
    methods = set(methods)

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

    x_qgmres = x_newton = x_lu = None
    time_qgmres = time_newton = time_lu = None
    info_qgmres = None
    covariances = None

    # 7) Solve with Q-GMRES (optional)
    if "qgmres" in methods:
        print("   Solving with Q-GMRES...")
        tol = 1e-6
        max_iter = N

        t0 = time.time()
        qgmres_solver = QGMRESSolver(tol=tol, max_iter=max_iter, verbose=False)
        x_qgmres, info_qgmres = qgmres_solver.solve(A, b)
        time_qgmres = time.time() - t0

    # 8) Solve with Newton-Schulz (optional)
    if "newton" in methods:
        print("   Solving with Newton-Schulz...")
        t0 = time.time()
        newton_solver = NewtonSchulzPseudoinverse(verbose=False)
        A_pinv, residuals, covariances = newton_solver.compute(A)
        x_newton = quat_matmat(A_pinv, b)
        time_newton = time.time() - t0

    # 9) Solve with direct LU baseline (optional)
    if "lu" in methods:
        print("   Solving with LU (direct)...")
        t0 = time.time()
        x_lu = solve_via_quat_lu(A, b)
        time_lu = time.time() - t0

    # 10) Compute accuracy for all methods
    def compute_residual(A, b, x):
        Ax = quat_matmat(A, x)
        residual = Ax - b
        return quat_frobenius_norm(residual)

    residual_qgmres = compute_residual(A, b, x_qgmres) if x_qgmres is not None else None
    residual_newton = compute_residual(A, b, x_newton) if x_newton is not None else None
    residual_lu = compute_residual(A, b, x_lu) if x_lu is not None else None

    out = {
        "num_points": num_points,
        "system_size": N,
        "A": A,
        "b": b,
        "obs": obs,
        "A0": A0,
        "A1": A1,
        "A2": A2,
        "A3": A3,
    }
    if x_qgmres is not None:
        out["qgmres"] = {
            "time": time_qgmres,
            "iterations": info_qgmres["iterations"],
            "residual": residual_qgmres,
            "final_residual": info_qgmres["residual"],
        }
        out["x_qgmres"] = x_qgmres
    if x_newton is not None:
        out["newton"] = {
            "time": time_newton,
            "iterations": len(covariances) if covariances is not None else None,
            "residual": residual_newton,
        }
        out["x_newton"] = x_newton
    if x_lu is not None:
        out["lu"] = {
            "time": time_lu,
            "iterations": 1,
            "residual": residual_lu,
        }
        out["x_lu"] = x_lu

    return out


def run_comprehensive_benchmark(*, point_ranges=None, methods=("qgmres", "newton", "lu")):
    """Run benchmark for multiple point ranges."""
    print("🚀 Lorenz Attractor Method Comparison Benchmark")
    print("=" * 60)

    # Benchmark parameters
    if point_ranges is None:
        point_ranges = [50, 75, 100, 150, 200]
    results = []

    for num_points in point_ranges:
        print(f"\n📊 Testing with {num_points} points...")
        result = run_lorenz_benchmark(num_points, methods=methods)
        results.append(result)

        if "qgmres" in result:
            print(
                f"   Q-GMRES: {result['qgmres']['time']:.3f}s, {result['qgmres']['iterations']} iterations, residual: {result['qgmres']['residual']:.2e}"
            )
        if "newton" in result:
            print(
                f"   NS--Q: {result['newton']['time']:.3f}s, {result['newton']['iterations']} iterations, residual: {result['newton']['residual']:.2e}"
            )
        if "lu" in result:
            print(
                f"   LU (direct): {result['lu']['time']:.3f}s, {result['lu']['iterations']} iterations, residual: {result['lu']['residual']:.2e}"
            )

    return results


def create_performance_plots(results, output_dir, *, show=True):
    """Create performance comparison plots"""
    print("\n📈 Creating performance plots...")

    # Extract data
    points = [r["num_points"] for r in results]
    have_qgmres = all("qgmres" in r for r in results)
    have_newton = all("newton" in r for r in results)
    have_lu = all("lu" in r for r in results)

    qgmres_times = [r["qgmres"]["time"] for r in results] if have_qgmres else None
    newton_times = [r["newton"]["time"] for r in results] if have_newton else None
    lu_times = [r["lu"]["time"] for r in results] if have_lu else None
    qgmres_iterations = [r["qgmres"]["iterations"] for r in results] if have_qgmres else None
    newton_iterations = [r["newton"]["iterations"] for r in results] if have_newton else None
    lu_iterations = [r["lu"]["iterations"] for r in results] if have_lu else None
    qgmres_residuals = [r["qgmres"]["residual"] for r in results] if have_qgmres else None
    newton_residuals = [r["newton"]["residual"] for r in results] if have_newton else None
    lu_residuals = [r["lu"]["residual"] for r in results] if have_lu else None

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Lorenz Attractor: Q-GMRES vs NS--Q vs LU (direct) Performance Comparison",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Computational Time
    if have_qgmres:
        ax1.plot(
            points,
            qgmres_times,
            "o-",
            color="#2E86AB",
            linewidth=2,
            markersize=8,
            label="Q-GMRES",
        )
    if have_newton:
        ax1.plot(
            points,
            newton_times,
            "s-",
            color="#A23B72",
            linewidth=2,
            markersize=8,
            label="NS--Q",
        )
    if have_lu:
        ax1.plot(
            points,
            lu_times,
            "^-",
            color="#2ca02c",
            linewidth=2,
            markersize=8,
            label="LU (direct)",
        )
    ax1.set_xlabel("Number of Points", fontsize=12)
    ax1.set_ylabel("Computational Time (seconds)", fontsize=12)
    ax1.set_title("Computational Time Comparison", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Plot 2: Iterations
    if have_qgmres:
        ax2.plot(
            points,
            qgmres_iterations,
            "o-",
            color="#2E86AB",
            linewidth=2,
            markersize=8,
            label="Q-GMRES",
        )
    if have_newton:
        ax2.plot(
            points,
            newton_iterations,
            "s-",
            color="#A23B72",
            linewidth=2,
            markersize=8,
            label="NS--Q",
        )
    if have_lu:
        ax2.plot(
            points,
            lu_iterations,
            "^-",
            color="#2ca02c",
            linewidth=2,
            markersize=8,
            label="LU (direct)",
        )
    ax2.set_xlabel("Number of Points", fontsize=12)
    ax2.set_ylabel("Number of Iterations", fontsize=12)
    ax2.set_title("Convergence Iterations", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Accuracy (Residual Norm)
    if have_qgmres:
        ax3.plot(
            points,
            np.log10(qgmres_residuals),
            "o-",
            color="#2E86AB",
            linewidth=2,
            markersize=8,
            label="Q-GMRES",
        )
    if have_newton:
        ax3.plot(
            points,
            np.log10(newton_residuals),
            "s-",
            color="#A23B72",
            linewidth=2,
            markersize=8,
            label="NS--Q",
        )
    if have_lu:
        ax3.plot(
            points,
            np.log10(lu_residuals),
            "^-",
            color="#2ca02c",
            linewidth=2,
            markersize=8,
            label="LU (direct)",
        )
    ax3.set_xlabel("Number of Points", fontsize=12)
    ax3.set_ylabel("log₁₀(Residual Norm)", fontsize=12)
    ax3.set_title("Solution Accuracy", fontsize=14, fontweight="bold")
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Time vs Accuracy
    if have_qgmres:
        ax4.scatter(
            qgmres_times,
            np.log10(qgmres_residuals),
            c="#2E86AB",
            s=100,
            alpha=0.7,
            label="Q-GMRES",
        )
    if have_newton:
        ax4.scatter(
            newton_times,
            np.log10(newton_residuals),
            c="#A23B72",
            s=100,
            alpha=0.7,
            label="NS--Q",
        )
    if have_lu:
        ax4.scatter(
            lu_times,
            np.log10(lu_residuals),
            c="#2ca02c",
            s=100,
            alpha=0.7,
            label="LU (direct)",
        )

    # Add point labels
    for i, point in enumerate(points):
        if have_qgmres:
            ax4.annotate(
                f"{point}",
                (qgmres_times[i], np.log10(qgmres_residuals[i])),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        if have_newton:
            ax4.annotate(
                f"{point}",
                (newton_times[i], np.log10(newton_residuals[i])),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        if have_lu:
            ax4.annotate(
                f"{point}",
                (lu_times[i], np.log10(lu_residuals[i])),
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
    _save_png_and_pdf(fig, os.path.join(output_dir, "lorenz_benchmark_performance.png"), dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"   Saved: {os.path.join(output_dir, 'lorenz_benchmark_performance.png')}")


def create_trajectory_comparison(results, output_dir, *, show=True, trajectory_n=200):
    """Create publication-quality trajectory comparison for 200 points with specified marker types"""
    print("\n🎨 Creating publication-quality trajectory comparison plots...")

    # Use requested trajectory result if available, otherwise fall back to the largest N.
    by_n = {r["num_points"]: r for r in results}
    if trajectory_n in by_n:
        result_200 = by_n[trajectory_n]
        n_used = trajectory_n
    else:
        n_used = max(by_n.keys())
        result_200 = by_n[n_used]

    # Extract data
    obs = result_200["obs"]
    x_qgmres = result_200.get("x_qgmres", None)
    x_newton = result_200.get("x_newton", None)
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

    recon_qgmres = (
        reconstruct_signal_original(x_qgmres, A0, A1, A2, A3) if x_qgmres is not None else None
    )
    recon_newton = (
        reconstruct_signal_original(x_newton, A0, A1, A2, A3) if x_newton is not None else None
    )

    # Plot styling for paper readability
    obs_color = "#ff00ff"  # neon magenta (high contrast in print)
    qgmres_color = "#d62728"  # red
    ns_color = "#1f77b4"  # blue

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
        t_eval=np.linspace(0, T, n_used),
        rtol=1e-5,
        atol=1e-8,
    )

    clean_signal = sol.y.T  # This is the clean signal without noise
    time_points = np.linspace(0, T, n_used)

    # Create publication-quality figure with increased size (3D trajectories only)
    fig = plt.figure(figsize=(16, 12))  # Adjusted size for single 3D plot

    # 3D Trajectory Comparison
    ax1 = fig.add_subplot(111, projection="3d")
    # Plot observed (noisy) signal with transparency
    ax1.plot(
        obs[:, 1],
        obs[:, 2],
        obs[:, 3],
        color=obs_color,
        linewidth=1.6,
        alpha=0.40,
        label="Observed (Noisy)",
    )
    ax1.plot(
        clean_signal[:, 0],
        clean_signal[:, 1],
        clean_signal[:, 2],
        "k-",
        linewidth=2.8,
        label="Ground Truth",
    )
    if recon_newton is not None:
        ax1.plot(
            recon_newton[:, 0],
            recon_newton[:, 1],
            recon_newton[:, 2],
            color=ns_color,
            linestyle="--",
            linewidth=2.6,
            label="NS--Q",
        )
    if recon_qgmres is not None:
        ax1.plot(
            recon_qgmres[:, 0],
            recon_qgmres[:, 1],
            recon_qgmres[:, 2],
            color=qgmres_color,
            linestyle=":",
            linewidth=2.6,
            label="QGMRES",
        )
    ax1.set_title("3D Trajectory Comparison", fontsize=18, fontweight="bold")
    ax1.set_xlabel("X", fontsize=16)
    ax1.set_ylabel("Y", fontsize=16)
    ax1.set_zlabel("Z", fontsize=16)
    ax1.legend(fontsize=16)
    ax1.view_init(elev=20, azim=45)
    ax1.tick_params(labelsize=14)

    plt.suptitle(
        f"Lorenz Attractor Signal Reconstruction (T = 10s, N = {n_used})",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    _save_png_and_pdf(
        fig, os.path.join(output_dir, "lorenz_trajectory_comparison_publication.png"), dpi=300
    )
    if show:
        plt.show()
    else:
        plt.close(fig)

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
        color=obs_color,
        linewidth=1,
        alpha=0.35,
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
    if recon_qgmres is not None:
        ax1_orig.plot(
            recon_qgmres[:, 0],
            recon_qgmres[:, 1],
            recon_qgmres[:, 2],
            color=qgmres_color,
            linewidth=2.4,
            label="Q-GMRES Reconstruction",
        )
    ax1_orig.set_title("Q-GMRES Method", fontsize=14, fontweight="bold")
    ax1_orig.set_xlabel("X", fontsize=12)
    ax1_orig.set_ylabel("Y", fontsize=12)
    ax1_orig.set_zlabel("Z", fontsize=12)
    ax1_orig.legend(fontsize=11)
    ax1_orig.view_init(elev=20, azim=45)

    # Plot 2: Newton-Schulz Reconstruction
    ax2_orig = fig_original.add_subplot(1, 3, 2, projection="3d")
    ax2_orig.plot(
        obs[:, 1],
        obs[:, 2],
        obs[:, 3],
        color=obs_color,
        linewidth=1,
        alpha=0.35,
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
    if recon_newton is not None:
        ax2_orig.plot(
            recon_newton[:, 0],
            recon_newton[:, 1],
            recon_newton[:, 2],
            color=ns_color,
            linewidth=2.4,
            label="NS--Q Reconstruction",
        )
    ax2_orig.set_title("Newton-Schulz Method", fontsize=14, fontweight="bold")
    ax2_orig.set_xlabel("X", fontsize=12)
    ax2_orig.set_ylabel("Y", fontsize=12)
    ax2_orig.set_zlabel("Z", fontsize=12)
    ax2_orig.legend(fontsize=11)
    ax2_orig.view_init(elev=20, azim=45)

    # Plot 3: Comparison
    ax3_orig = fig_original.add_subplot(1, 3, 3, projection="3d")
    ax3_orig.plot(
        obs[:, 1],
        obs[:, 2],
        obs[:, 3],
        color=obs_color,
        linewidth=1,
        alpha=0.35,
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
    if recon_qgmres is not None:
        ax3_orig.plot(
            recon_qgmres[:, 0],
            recon_qgmres[:, 1],
            recon_qgmres[:, 2],
            color=qgmres_color,
            linewidth=2.4,
            alpha=0.9,
            label="QGMRES",
        )
    if recon_newton is not None:
        ax3_orig.plot(
            recon_newton[:, 0],
            recon_newton[:, 1],
            recon_newton[:, 2],
            color=ns_color,
            linewidth=2.4,
            alpha=0.9,
            label="NS--Q",
        )
    ax3_orig.set_title("Method Comparison", fontsize=14, fontweight="bold")
    ax3_orig.set_xlabel("X", fontsize=12)
    ax3_orig.set_ylabel("Y", fontsize=12)
    ax3_orig.set_zlabel("Z", fontsize=12)
    ax3_orig.legend(fontsize=11)
    ax3_orig.view_init(elev=20, azim=45)

    plt.suptitle(
        "Lorenz Attractor Signal Reconstruction (200 points)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    _save_png_and_pdf(fig_original, os.path.join(output_dir, "lorenz_trajectory_comparison.png"), dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig_original)

    print(f"   Saved: {os.path.join(output_dir, 'lorenz_trajectory_comparison.png')}")

    # Also create a simplified version with just the 1D time series for the paper
    fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # x(t) vs time
    ax1.plot(
        time_points,
        obs[:, 1],
        color=obs_color,
        linewidth=1.8,
        alpha=0.55,
        marker=".",
        markersize=2,
        markevery=max(1, len(time_points) // 400),
        label="Observed (Noisy)",
    )
    ax1.plot(time_points, clean_signal[:, 0], "k-", linewidth=2.6, label="Ground Truth")
    if recon_newton is not None:
        ax1.plot(
            time_points,
            recon_newton[:, 0],
            linestyle="--",
            color=ns_color,
            linewidth=2.4,
            label="NS--Q",
        )
    if recon_qgmres is not None:
        ax1.plot(
            time_points,
            recon_qgmres[:, 0],
            linestyle=":",
            color=qgmres_color,
            linewidth=2.4,
            label="QGMRES",
        )
    ax1.set_ylabel("x(t)", fontsize=16)
    ax1.legend(fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=13)

    # y(t) vs time
    ax2.plot(
        time_points,
        obs[:, 2],
        color=obs_color,
        linewidth=1.8,
        alpha=0.55,
        marker=".",
        markersize=2,
        markevery=max(1, len(time_points) // 400),
        label="Observed (Noisy)",
    )
    ax2.plot(time_points, clean_signal[:, 1], "k-", linewidth=2.6, label="Ground Truth")
    if recon_newton is not None:
        ax2.plot(
            time_points,
            recon_newton[:, 1],
            linestyle="--",
            color=ns_color,
            linewidth=2.4,
            label="NS--Q",
        )
    if recon_qgmres is not None:
        ax2.plot(
            time_points,
            recon_qgmres[:, 1],
            linestyle=":",
            color=qgmres_color,
            linewidth=2.4,
            label="QGMRES",
        )
    ax2.set_ylabel("y(t)", fontsize=16)
    ax2.legend(fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=13)

    # z(t) vs time
    ax3.plot(
        time_points,
        obs[:, 3],
        color=obs_color,
        linewidth=1.8,
        alpha=0.55,
        marker=".",
        markersize=2,
        markevery=max(1, len(time_points) // 400),
        label="Observed (Noisy)",
    )
    ax3.plot(time_points, clean_signal[:, 2], "k-", linewidth=2.6, label="Ground Truth")
    if recon_newton is not None:
        ax3.plot(
            time_points,
            recon_newton[:, 2],
            linestyle="--",
            color=ns_color,
            linewidth=2.4,
            label="NS--Q",
        )
    if recon_qgmres is not None:
        ax3.plot(
            time_points,
            recon_qgmres[:, 2],
            linestyle=":",
            color=qgmres_color,
            linewidth=2.4,
            label="QGMRES",
        )
    ax3.set_xlabel("Time (s)", fontsize=16)
    ax3.set_ylabel("z(t)", fontsize=16)
    ax3.legend(fontsize=13)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=13)

    plt.suptitle(
        "Lorenz Attractor: 1D Signal Components vs Time (T = 10s)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    _save_png_and_pdf(fig2, os.path.join(output_dir, "lorenz_1d_signals_publication.png"), dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig2)

    print(f"   Saved: {os.path.join(output_dir, 'lorenz_1d_signals_publication.png')}")


def generate_latex_table(results):
    """Generate LaTeX table from benchmark results"""
    print("LaTeX Table for Paper:")
    print("=" * 60)

    print("\\begin{table}[ht!]")
    print("\\centering")
    # Build a method caption consistent with which solvers were run.
    methods_present = []
    if all("newton" in r for r in results):
        methods_present.append("NS--Q")
    if all("qgmres" in r for r in results):
        methods_present.append("QGMRES")
    if all("lu" in r for r in results):
        methods_present.append("LU (direct)")
    caption_methods = " vs.\\ ".join(methods_present) if methods_present else "methods"
    print(
        f"\\\\caption{{Lorenz–attractor filtering: {caption_methods} on the $N\\\\times N$ quaternion system \\\\eqref{{eq:lorenz-linear}}.}}"
    )
    print("\\label{tab:lorenz}")
    print("\\begin{tabular}{lcccc}")
    print("\\hline")
    print("$N$ & Method & Iterations & CPU time (s) & RelRes \\\\")
    print("\\hline")

    for result in results:
        N = result["num_points"]
        first_row = True
        if "newton" in result:
            newton_relres = calculate_relative_residual(result["newton"]["residual"], N)
            print(
                f"{N} & NS--Q   & {result['newton']['iterations']:3d} & {result['newton']['time']:6.3f} & {newton_relres:.1e} \\\\"
            )
            first_row = False
        if "qgmres" in result:
            qgmres_relres = calculate_relative_residual(result["qgmres"]["residual"], N)
            prefix = f"{N}" if first_row else "   "
            print(
                f"{prefix} & QGMRES & {result['qgmres']['iterations']:3d} & {result['qgmres']['time']:6.3f} & {qgmres_relres:.1e} \\\\"
            )
            first_row = False
        if "lu" in result:
            lu_relres = calculate_relative_residual(result["lu"]["residual"], N)
            prefix = f"{N}" if first_row else "   "
            print(
                f"{prefix} & LU     & {result['lu']['iterations']:3d} & {result['lu']['time']:6.3f} & {lu_relres:.1e} \\\\"
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

    stats = {}
    if all("qgmres" in r for r in results):
        stats["Q-GMRES"] = {
            "times": [r["qgmres"]["time"] for r in results],
            "residuals": [r["qgmres"]["residual"] for r in results],
        }
    if all("newton" in r for r in results):
        stats["NS--Q"] = {
            "times": [r["newton"]["time"] for r in results],
            "residuals": [r["newton"]["residual"] for r in results],
        }
    if all("lu" in r for r in results):
        stats["LU (direct)"] = {
            "times": [r["lu"]["time"] for r in results],
            "residuals": [r["lu"]["residual"] for r in results],
        }

    if not stats:
        print("   (No methods were run.)")
        return

    # Print averages
    avg_time = {}
    avg_res = {}
    for name, d in stats.items():
        avg_time[name] = float(np.mean(d["times"]))
        avg_res[name] = float(np.mean(d["residuals"]))
        print(f"   ⚡ Average {name} time: {avg_time[name]:.3f}s")
        print(f"   📏 Average {name} residual: {avg_res[name]:.2e}")

    # Fastest method
    fastest = min(avg_time.items(), key=lambda kv: kv[1])[0]
    most_accurate = min(avg_res.items(), key=lambda kv: kv[1])[0]
    print(f"\n   🏆 Fastest on average: {fastest}")
    print(f"   🎯 Most accurate on average: {most_accurate}")

    print("\n📈 Performance Trends:")
    if "Q-GMRES" in stats:
        print("   • Q-GMRES: Iterative method (often slower for large dense systems)")
    if "NS--Q" in stats:
        print("   • NS--Q: Newton–Schulz pseudoinverse iterations (dense, accuracy-controlled)")
    if "LU (direct)" in stats:
        print("   • LU (direct): Dense direct solve via LU factorization with partial pivoting")

    print("\n💡 Recommendations:")
    if "LU (direct)" in stats:
        print("   • Use LU (direct) for: Fast direct solves on moderate dense sizes")
    if "Q-GMRES" in stats:
        print("   • Use Q-GMRES for: Iterative solves / when avoiding factorization is important")
    if "NS--Q" in stats:
        print("   • Use NS--Q for: Pseudoinverse-style workflows and controlled iterative refinement")


def main():
    """Main benchmark execution"""
    print("🚀 Starting Lorenz Attractor Method Comparison Benchmark")
    print("=" * 60)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Run headless (no interactive figures). Figures are still saved to disk.",
    )
    parser.add_argument(
        "--points",
        type=str,
        default="50,75,100,150,200",
        help="Comma-separated list of N values to benchmark (e.g. 200,500,1000).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="qgmres,newton,lu",
        help="Comma-separated methods among {qgmres,newton,lu}. Example: newton,lu",
    )
    parser.add_argument(
        "--skip_trajectory",
        action="store_true",
        help="Skip trajectory reconstruction plots (recommended for large N).",
    )
    parser.add_argument(
        "--trajectory_n",
        type=int,
        default=200,
        help="Which N to use for trajectory plots (if available).",
    )
    args = parser.parse_args()
    show = not args.no_show
    if not show:
        plt.switch_backend("Agg")

    point_ranges = [int(x.strip()) for x in args.points.split(",") if x.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    # Ensure output directory exists
    output_dir = ensure_output_directory()

    # Run benchmark
    results = run_comprehensive_benchmark(point_ranges=point_ranges, methods=methods)

    # Create visualizations
    create_performance_plots(results, output_dir, show=show)
    if not args.skip_trajectory:
        create_trajectory_comparison(
            results,
            output_dir,
            show=show,
            trajectory_n=args.trajectory_n,
        )

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
