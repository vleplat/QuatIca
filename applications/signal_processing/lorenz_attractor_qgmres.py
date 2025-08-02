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
import time

# Add core module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core'))

from solver import QGMRESSolver
from utils import timesQsparse

# --- Plotting functions ---
def createfigure3(YMatrix1):
    """Create 3D time series plot"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(YMatrix1[:, 0], 'r-', linewidth=2, label='x(t)')
    ax.plot(YMatrix1[:, 1], 'g-', linewidth=2, label='y(t)')
    ax.plot(YMatrix1[:, 2], 'b-', linewidth=2, label='z(t)')
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Amplitude', fontsize=14)
    ax.set_title('3D Signal Components', fontsize=16, fontweight='bold')
    ax.grid(True)
    ax.legend(fontsize=12)
    plt.show()

def createfigure4(X1, Y1, Z1):
    """Create 3D trajectory plot"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(X1, Y1, Z1, linewidth=2, color='black')
    ax.set_xlabel('x(t)', fontsize=14)
    ax.set_ylabel('y(t)', fontsize=14)
    ax.set_zlabel('z(t)', fontsize=14)
    ax.set_title('3D Trajectory', fontsize=16, fontweight='bold')
    ax.grid(True)
    plt.show()

# --- Main script ---
def main():
    print("=" * 60)
    print("Lorenz Attractor Signal Processing with Q-GMRES (MATLAB-style)")
    print("=" * 60)

    # 1) Lorenz parameters and integration
    sigma, beta, rho = 10.0, 8/3, 28.0
    T, delta, seed = 10.0, 1.0, 0

    def lorenz(t, a, sigma=10, beta=8/3, rho=28):
        x, y, z = a
        return [
            -sigma*x + sigma*y,
             rho*x - y - x*z,
            -beta*z + x*y
        ]

    t_span = (0.0, T)
    # Let solve_ivp pick its own adaptive time steps
    from scipy.integrate import solve_ivp
    sol = solve_ivp(lorenz, t_span, [1,1,1], method='RK45',
                    atol=1e-6, rtol=1e-3)
    t = sol.t           # time points (1D array)
    a = sol.y.T         # shape (len(t), 3)
    N = a.shape[0]

    

    # 2) Build quaternion signal + noise
    np.random.seed(seed)
    signal = np.zeros((N, 4))
    signal[:, 1:] = a
    s = signal + delta * np.random.randn(N, 4)
    obs = s.copy()
    s[:, 0] = 0

    # 3) Block-Hankel assembly
    ny, mx = N-1, N-1
    s_pad = np.vstack([s[-ny:], s, s[:mx]])
    rows, cols = mx+1, ny+1  # both = N
    S = np.zeros((rows, 4*cols))
    for i in range(rows):
        for j in range(cols):
            idx = ny + i - j
            S[i,     j       ] = s_pad[idx, 0]
            S[i, cols + j     ] = s_pad[idx, 1]
            S[i, 2*cols + j   ] = s_pad[idx, 2]
            S[i, 3*cols + j   ] = s_pad[idx, 3]

    # 4) Extract quaternion blocks A0…A3
    A0 = S[:,       :N]
    A1 = S[:,   N:  2*N]
    A2 = S[:, 2*N:  3*N]
    A3 = S[:, 3*N:  4*N]

    # 5) Build RHS b0…b3
    b = signal.reshape(-1)
    b[:N] = 0
    b0 = b[       :N]
    b1 = b[N:    2*N]
    b2 = b[2*N:  3*N]
    b3 = b[3*N:  4*N]

    # 6) Initial guess (w)
    w = np.ones((ny+1, 4))
    w /= np.sum(w)
    x0 = w.copy()

    # 7) Solve via Q-GMRES
    tol = 1e-6
    print("start solving with Q-GMRES")
    print(f"Matrix shapes: A0={A0.shape}, A1={A1.shape}, A2={A2.shape}, A3={A3.shape}")
    print(f"RHS shapes: b0={b0.shape}, b1={b1.shape}, b2={b2.shape}, b3={b3.shape}")
    print(f"N={N}")
    t0 = time.time()
    
    # Convert to quaternion format
    import quaternion
    
    # Create quaternion matrix A
    A_quat = np.zeros((N, N, 4))
    A_quat[:, :, 0] = A0
    A_quat[:, :, 1] = A1
    A_quat[:, :, 2] = A2
    A_quat[:, :, 3] = A3
    A_quat_array = quaternion.as_quat_array(A_quat)
    print(f"A_quat_array shape: {A_quat_array.shape}")
    print(f"A_quat_array dtype: {A_quat_array.dtype}")
    
    # Create quaternion vector b
    b_quat = np.zeros((N, 4))
    b_quat[:, 0] = b0
    b_quat[:, 1] = b1
    b_quat[:, 2] = b2
    b_quat[:, 3] = b3
    b_quat_array = quaternion.as_quat_array(b_quat)
    
    # Ensure b is a column vector for Q-GMRES
    if len(b_quat_array.shape) == 1:  # If it's 1D, reshape to column vector
        b_quat_array = b_quat_array.reshape(-1, 1)
    
    # Use Q-GMRES solver
    from solver import QGMRESSolver
    qgmres_solver = QGMRESSolver(tol=tol, max_iter=N, verbose=True)
    x_solution, info = qgmres_solver.solve(A_quat_array, b_quat_array)
    
    # Extract components from solution
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

    # 8) Plot observed signal
    # createfigure3(np.column_stack((obs[:,2], obs[:,3], obs[:,1])))
    # createfigure4(obs[:,2], obs[:,3], obs[:,1])

    XYZ = np.column_stack((obs[:,1], obs[:,2], obs[:,3]))  # take cols 1,2,3
    createfigure3(XYZ)
    createfigure4(obs[:,1], obs[:,2], obs[:,3])

    # 9) Reconstruct and plot
    dy0, dy1, dy2, dy3 = timesQsparse(A0, A1, A2, A3,
                                      xm_0, xm_1, xm_2, xm_3)
    createfigure3(np.column_stack((dy2, dy1, dy3)))
    createfigure4(dy2, dy1, dy3)
    # REC = np.column_stack((dy1, dy2, dy3))
    # createfigure3(REC)
    # createfigure4(dy1, dy2, dy3)

    # 10) Plot RHS comp.
    createfigure3(np.column_stack((b2, b1, b3)))
    createfigure4(b2, b1, b3)
    # createfigure3(np.column_stack((b1, b2, b3)))
    #createfigure4(b1, b2, b3)

    # 11) Residual history
    if resv is not None:
        plt.figure()
        plt.semilogy(resv[:, 2])
        plt.title("GMRESQ residual (component 3)")
        plt.xlabel("Iteration")
        plt.ylabel("Residual")
        plt.grid(True)
        plt.show()

    print("=" * 60)
    print("Analysis complete!")
    print("=" * 60)

if __name__ == "__main__":
    main() 