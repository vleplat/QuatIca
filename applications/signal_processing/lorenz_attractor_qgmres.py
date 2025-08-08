#!/usr/bin/env python3
"""
Lorenz Attractor Signal Processing with Q-GMRES
==============================================

USAGE EXAMPLES:
==============

1. Default configuration (200 points, ~75s execution):
   python lorenz_attractor_qgmres.py

2. Fast execution (100 points, ~30s):
   python lorenz_attractor_qgmres.py --num_points 100

3. High resolution (500 points, ~5-10min):
   python lorenz_attractor_qgmres.py --num_points 500

4. Research quality (1000 points, ~20-30min):
   python lorenz_attractor_qgmres.py --num_points 1000

5. Save plots without displaying them:
   python lorenz_attractor_qgmres.py --no_show

6. Combine options:
   python lorenz_attractor_qgmres.py --num_points 300 --no_show

PERFORMANCE GUIDE:
=================
- 100 points: Fast testing, lower resolution
- 200 points: Balanced performance, good resolution (DEFAULT)
- 500 points: High resolution, publication quality
- 1000 points: Very high resolution, research quality

Fixed issues with:
1. Linear system construction (matrix S)
2. Quaternion handling and indexing
3. Plotting functions and component ordering
4. GMRES solver input/output handling
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
from matplotlib import gridspec
import sys
import os
import argparse

# Add core module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core'))

# Fixed plotting functions to match MATLAB's behavior
def createfigure3(YMatrix1):
    """Create 2D time series plot matching MATLAB's style"""
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    ax = fig.add_subplot(111)
    
    # Plot each component with specified colors
    plot1 = ax.plot(YMatrix1[:, 0], 'r-', linewidth=2, label='x(t)')
    plot2 = ax.plot(YMatrix1[:, 1], 'g-', linewidth=2, label='y(t)')
    plot3 = ax.plot(YMatrix1[:, 2], 'b-', linewidth=2, label='z(t)')
    
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Amplitude', fontsize=14)
    ax.set_title('3D Signal Components', fontsize=16, fontweight='bold')
    ax.grid(True)
    ax.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    return fig

def createfigure4(X1, Y1, Z1, title: str = '3D Trajectory'):
    """Create 3D trajectory plot matching MATLAB's style with configurable title"""
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot 3D trajectory
    ax.plot(X1, Y1, Z1, 'k-', linewidth=2)
    ax.set_xlabel('x(t)', fontsize=14)
    ax.set_ylabel('y(t)', fontsize=14)
    ax.set_zlabel('z(t)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True)
    ax.view_init(elev=30, azim=-37.5)  # Match MATLAB's default view
    return fig

def ensure_output_directory():
    """Create output_figures directory if it doesn't exist"""
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output_figures')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir

def save_high_res_plot(fig, filename, output_dir, dpi=300, show_plot=True):
    """Save plot in high resolution and optionally display it"""
    if show_plot:
        plt.show()  # Display the plot interactively
    
    full_path = os.path.join(output_dir, filename)
    fig.savefig(full_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved: {full_path}")
    plt.close(fig)  # Close figure to free memory

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Lorenz Attractor Signal Processing with Q-GMRES')
    parser.add_argument('--num_points', type=int, default=200, 
                       help='Number of points for Lorenz system integration (default: 200)')
    parser.add_argument('--no_show', action='store_true', 
                       help='Skip displaying plots (only save them)')
    args = parser.parse_args()
    
    # USER-CONFIGURABLE PARAMETER - NUM_POINTS
    # ==========================================
    # This parameter controls the resolution and computational cost:
    # - 100: Fast execution (~30s), lower resolution
    # - 200: Balanced performance (~75s), good resolution (DEFAULT)
    # - 500: High resolution (~5-10min), publication quality
    # - 1000: Very high resolution (~20-30min), research quality
    NUM_POINTS = args.num_points
    SHOW_PLOTS = not args.no_show
    # ==========================================
    
    print("=" * 60)
    print("Lorenz Attractor Signal Processing with Q-GMRES - CORRECTED")
    print("=" * 60)
    print(f"📊 CONFIGURATION: num_points = {NUM_POINTS}")
    print(f"⏱️  Expected execution time: {NUM_POINTS//100 * 35:.0f}-{NUM_POINTS//100 * 45:.0f} seconds")
    print("=" * 60)

    # Ensure output directory exists
    output_dir = ensure_output_directory()

    # 1) Lorenz parameters and integration
    sigma, beta, rho = 10.0, 8/3, 28.0
    T, delta, seed = 10.0, 1.0, 0
    num_points = NUM_POINTS  # Use user-specified value

    def lorenz(t, a):
        x, y, z = a
        return [
            -sigma*x + sigma*y,
            rho*x - y - x*z,
            -beta*z + x*y
        ]

    # Solve Lorenz system
    sol = solve_ivp(lorenz, [0, T], [1, 1, 1],
                    method='RK45', t_eval=np.linspace(0, T, num_points),
                    rtol=1e-5, atol=1e-8)
    
    t = sol.t
    a = sol.y.T
    N = len(t)

    # 2) Build quaternion signal + noise - CORRECTED INDEXING
    np.random.seed(seed)
    signal = np.zeros((N, 4))
    signal[:, 1:] = a  # [real=0, x, y, z]
    s = signal + delta * np.random.randn(N, 4)
    obs = s.copy()
    s[:, 0] = 0  # Reset real component

    # 3) Block-Hankel assembly - FIXED INDEXING
    ny = mx = N - 1
    s_pad = np.vstack([s[-ny:], s, s[:mx]])  # Correct padding
    
    rows, cols = mx + 1, ny + 1
    S = np.zeros((rows, 4 * cols))
    
    for i in range(rows):
        for j in range(cols):
            idx = ny + i - j  # Correct indexing (MATLAB: ny+1+i-j)
            for k in range(4):
                col_index = k * cols + j
                S[i, col_index] = s_pad[idx, k]

    # 4) Extract quaternion blocks - FIXED DIMENSIONS
    n_cols = S.shape[1] // 4
    A0 = S[:, :n_cols]
    A1 = S[:, n_cols:2*n_cols]
    A2 = S[:, 2*n_cols:3*n_cols]
    A3 = S[:, 3*n_cols:4*n_cols]

    # 5) Build RHS - PROPER COMPONENT SEPARATION
    b = signal.copy()  # Keep as (N, 4) for quaternion operations
    b[:, 0] = 0  # Set real components to zero
    
    # 6) Solve via Q-GMRES - USING OUR CURRENT SOLVER
    tol = 1e-6
    max_iter = N
    
    
    t0 = time.time()
    
    # Convert to quaternion format for our solver
    import quaternion
    
    # Create quaternion matrix A
    A_quat = np.zeros((N, N, 4))
    A_quat[:, :, 0] = A0
    A_quat[:, :, 1] = A1
    A_quat[:, :, 2] = A2
    A_quat[:, :, 3] = A3
    A_quat_array = quaternion.as_quat_array(A_quat)
    
    # Create quaternion vector b
    b_quat_array = quaternion.as_quat_array(b)
    
    # Ensure b is a column vector for Q-GMRES
    if len(b_quat_array.shape) == 1:
        b_quat_array = b_quat_array.reshape(-1, 1)
    
    # Use our Q-GMRES solver
    print("Starting Q-GMRES solve...")
    print(f"System dimensions: A: {A_quat_array.shape}, b: {b_quat_array.shape}")

    from solver import QGMRESSolver
    qgmres_solver = QGMRESSolver(tol=tol, max_iter=max_iter, verbose=True)
    x_solution, info = qgmres_solver.solve(A_quat_array, b_quat_array)
    
    # Extract solution components
    x_components = quaternion.as_float_array(x_solution)
    
    # Remove the middle dimension if it's 1
    if len(x_components.shape) == 3 and x_components.shape[1] == 1:
        x_components = x_components.squeeze(axis=1)
    
    xm_0 = x_components[:, 0]
    xm_1 = x_components[:, 1]
    xm_2 = x_components[:, 2]
    xm_3 = x_components[:, 3]
    
    iters = info['iterations']
    res = info['residual']
    resv = info.get('residuals', None)
    
    print(f"Q-GMRES done in {time.time()-t0:.3f}s, iters={iters}, res={res:.3e}")

    # 7) Plot observed signal - CORRECT COMPONENT ORDER
    # Observed signal components: x, y, z (skip real part)
    obs_xyz = obs[:, 1:4]  # Columns 1,2,3 = x,y,z
    fig1 = createfigure3(obs_xyz)
    save_high_res_plot(fig1, 'lorenz_observed_components.png', output_dir, show_plot=SHOW_PLOTS)
    
    fig2 = createfigure4(obs[:,1], obs[:,2], obs[:,3], title='Observed Trajectory (corrupted)')
    save_high_res_plot(fig2, 'lorenz_observed_trajectory.png', output_dir, show_plot=SHOW_PLOTS)

    # 8) Reconstruct signal - CORRECT COMPONENT ORDER
    dy0, dy1, dy2, dy3 = timesQsparse(A0, A1, A2, A3,
                                      xm_0, xm_1, xm_2, xm_3)
    
    # Create reconstructed signal matrix
    reconstructed = np.column_stack((dy1, dy2, dy3))  # x,y,z components
    
    fig3 = createfigure3(reconstructed)
    save_high_res_plot(fig3, 'lorenz_reconstructed_components.png', output_dir, show_plot=SHOW_PLOTS)
    
    fig4 = createfigure4(dy1, dy2, dy3, title='Recovered Trajectory')
    save_high_res_plot(fig4, 'lorenz_reconstructed_trajectory.png', output_dir, show_plot=SHOW_PLOTS)

    # 9) Residual history
    if resv is not None:
        fig7 = plt.figure(figsize=(10, 6))
        plt.semilogy(resv[:, 2], 'b-o', linewidth=2)
        plt.title("GMRESQ Residual History", fontsize=16)
        plt.xlabel("Iteration", fontsize=14)
        plt.ylabel("Residual Norm (log scale)", fontsize=14)
        plt.grid(True)
        save_high_res_plot(fig7, 'lorenz_residual_history.png', output_dir, show_plot=SHOW_PLOTS)

    print("=" * 60)
    print("Analysis complete! High-resolution plots saved to output_figures/")
    print("=" * 60)

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