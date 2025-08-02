#!/usr/bin/env python3
"""
Lorenz Attractor Signal Processing with Q-GMRES
==============================================

This script implements Example 1 from the paper:
"Structure preserving quaternion generalized minimal residual method"
by Zhigang Jia and Michael K. Ng (2021).

This version follows the MATLAB code structure exactly.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys
import os

# Ensure we can import from core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from core.solver import QGMRESSolver
from core.utils import timesQsparse

# --- Lorenz system ODE ---
def lorenz_system(state, t, sigma, beta, rho):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# --- Plotting functions ---
def plot_3d_signal(signal, title="3D Signal", filename=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(signal[:, 1], signal[:, 2], signal[:, 3], linewidth=2, color='black')
    ax.set_xlabel('x(t)', fontsize=14)
    ax.set_ylabel('y(t)', fontsize=14)
    ax.set_zlabel('z(t)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True)
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_time_series(signal, t, title="Time Series", filename=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(t, signal[:, 1], 'r-', linewidth=2, label='x(t)')
    ax.plot(t, signal[:, 2], 'g-', linewidth=2, label='y(t)')
    ax.plot(t, signal[:, 3], 'b-', linewidth=2, label='z(t)')
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Amplitude', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True)
    ax.legend(fontsize=12)
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_residuals(residuals, title="Q-GMRES Residuals", filename=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(residuals, 'b-', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('Residual (log scale)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True)
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# --- Main script ---
def main():
    print("=" * 60)
    print("Lorenz Attractor Signal Processing with Q-GMRES (MATLAB-style)")
    print("=" * 60)

    # Parameters (as in MATLAB)
    sigma = 10
    beta = 8/3
    rho = 28
    T = 2  # Use small T for quick test
    t = np.linspace(0, T, 100)  # 100 points for T=2
    N = len(t)

    # 1. Generate Lorenz signal
    a = odeint(lorenz_system, [1, 1, 1], t, args=(sigma, beta, rho))
    signal = np.zeros((N, 4))
    signal[:, 1] = a[:, 0]  # x
    signal[:, 2] = a[:, 1]  # y
    signal[:, 3] = a[:, 2]  # z

    # 2. Add noise
    np.random.seed(0)
    delta = 1.0
    s = signal + np.random.randn(*signal.shape) * delta
    s[:, 0] = 0  # s(:,1)=0 in MATLAB
    obs = s.copy()

    # 3. Build extended signal for boundary conditions
    ny = N - 1
    mx = N - 1
    s_ext = np.vstack([s[-ny:], s, s[:mx]])

    # 4. Construct matrix S (N x 4N) as in MATLAB
    S = np.zeros((mx + 1, 4 * (ny + 1)))
    for i in range(mx + 1):
        for j in range(ny + 1):
            idx = ny + i - j
            S[i, j] = s_ext[idx, 0]  # real
            S[i, (ny + 1) + j] = s_ext[idx, 1]  # i
            S[i, 2 * (ny + 1) + j] = s_ext[idx, 2]  # j
            S[i, 3 * (ny + 1) + j] = s_ext[idx, 3]  # k

    # 5. Set up the square system
    A = S  # (N, 4N)
    # b = signal.flatten()  # (4N,)
    # b[:A.shape[0]] = 0  # b(1:N)=0 boundary condition

    # 6. Extract quaternion components (A0, A1, A2, A3), (b_0, b_1, b_2, b_3)
    Nq = N
    # Each block is (N, N)
    A0 = A[:, :Nq]
    A2 = A[:, Nq:2*Nq]
    A1 = A[:, 2*Nq:3*Nq]
    A3 = A[:, 3*Nq:4*Nq]
    # b components are columns of signal (not slices of flattened vector)
    b_0 = np.zeros(N)  # real part is always zero in MATLAB code
    b_1 = signal[:, 1]
    b_2 = signal[:, 2]
    b_3 = signal[:, 3]
    # No need to set boundary condition, already zero

    print(f"A0 shape: {A0.shape}")
    print(f"A1 shape: {A1.shape}")
    print(f"A2 shape: {A2.shape}")
    print(f"A3 shape: {A3.shape}")
    print(f"b_0 shape: {b_0.shape}")
    print(f"b_1 shape: {b_1.shape}")
    print(f"b_2 shape: {b_2.shape}")
    print(f"b_3 shape: {b_3.shape}")

    # 7. Call QGMRES (as in MATLAB)
    print("\nSolving with Q-GMRES...")
    tol = 1e-6
    qgmres = QGMRESSolver(tol=tol, max_iter=Nq, verbose=True)
    xm_0, xm_1, xm_2, xm_3, res, V0, V1, V2, V3, iter_count, resv = qgmres._GMRESQsparse(
        A0, A1, A2, A3, b_0, b_1, b_2, b_3, tol, Nq
    )
    print(f"Q-GMRES finished in {iter_count} iterations, final residual: {res:.2e}")

    # 8. Reconstruct signal (as in MATLAB)
    dy0, dy1, dy2, dy3 = timesQsparse(A0, A1, A2, A3, xm_0, xm_1, xm_2, xm_3)
    reconstructed = np.column_stack([dy0, dy1, dy2, dy3])

    # 9. Plot results (as in MATLAB)
    output_dir = "output_figures/lorenz_attractor"
    os.makedirs(output_dir, exist_ok=True)
    plot_3d_signal(obs, "Observed Signal (with noise)", f"{output_dir}/observed_3d.png")
    plot_time_series(obs, t, "Observed Signal Components", f"{output_dir}/observed_time_series.png")
    plot_3d_signal(reconstructed, "Q-GMRES Reconstructed Signal", f"{output_dir}/reconstructed_3d.png")
    plot_time_series(reconstructed, t, "Reconstructed Signal Components", f"{output_dir}/reconstructed_time_series.png")
    plot_3d_signal(signal, "Original Signal", f"{output_dir}/original_3d.png")
    plot_time_series(signal, t, "Original Signal Components", f"{output_dir}/original_time_series.png")
    if resv is not None:
        plot_residuals(np.array(resv)[:, 2], "Q-GMRES Convergence", f"{output_dir}/residuals.png")
    print(f"\nResults saved to {output_dir}/\n")
    print("=" * 60)
    print("Analysis complete!")
    print("=" * 60)

if __name__ == "__main__":
    main() 