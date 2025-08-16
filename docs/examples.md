# Examples

Quick copy-paste examples to get you started with QuatIca.

## 🚀 Quick Start Commands

### Learn the Framework

```bash
# Start here - interactive tutorial with visualizations
python run_analysis.py tutorial

# Core functionality demo (comprehensive overview)
python run_analysis.py demo
```

### Linear System Solving

```bash
# Basic Q-GMRES solver test
python run_analysis.py qgmres

# Q-GMRES with LU preconditioning benchmark
python run_analysis.py qgmres_bench
```

### Signal Processing

```bash
# Lorenz attractor processing (default quality)
python run_analysis.py lorenz_signal

# Fast testing (100 points, ~30s)
python run_analysis.py lorenz_signal --num_points 100

# High quality (500 points, ~5-10min)
python run_analysis.py lorenz_signal --num_points 500

# Method comparison benchmark
python run_analysis.py lorenz_benchmark
```

### Image Processing

```bash
# Real image completion
python run_analysis.py image_completion

# Quaternion image deblurring (recommended)
python run_analysis.py image_deblurring --size 64 --lam 1e-3 --snr 40 --ns_mode fftT --fftT_order 3 --ns_iters 12

# Synthetic image completion
python run_analysis.py synthetic

# Test pseudoinverse on synthetic matrices
python run_analysis.py synthetic_matrices
```

### Matrix Decompositions

```bash
# Eigenvalue decomposition test
python run_analysis.py eigenvalue_test

# Quaternion Schur decomposition demo
python run_analysis.py schur_demo

# Schur with custom matrix size
python run_analysis.py schur_demo 25

# Compare Newton-Schulz variants
python run_analysis.py ns_compare
```

## 📝 Code Examples

### Basic Quaternion Matrix Operations

```python
import numpy as np
import quaternion
from quatica.utils import quat_matmat, quat_frobenius_norm, quat_eye

# Create quaternion matrices
A = quaternion.as_quat_array(np.random.randn(4, 4, 4))
B = quaternion.as_quat_array(np.random.randn(4, 4, 4))

# Matrix multiplication
C = quat_matmat(A, B)

# Compute Frobenius norm
norm_A = quat_frobenius_norm(A)
print(f"Frobenius norm: {norm_A:.6f}")

# Identity matrix
I = quat_eye(4)
```

### Pseudoinverse Computation

```python
from quatica.solver import NewtonSchulzPseudoinverse, HigherOrderNewtonSchulzPseudoinverse

# Standard Newton-Schulz pseudoinverse
ns_solver = NewtonSchulzPseudoinverse(gamma=0.5, max_iter=100, tol=1e-6)
A_pinv, residuals, covariances = ns_solver.compute(A)

# Higher-order Newton-Schulz (cubic convergence)
hon_solver = HigherOrderNewtonSchulzPseudoinverse(max_iter=50, tol=1e-8)
A_pinv_hon, residuals_hon, covariances_hon = hon_solver.compute(A)
```

### Q-GMRES Linear System Solving

```python
from quatica.solver import QGMRESSolver

# Create Q-GMRES solver
qgmres_solver = QGMRESSolver(tol=1e-6, max_iter=100, verbose=True)

# Solve linear system A*x = b
b = quaternion.as_quat_array(np.random.randn(4, 1, 4))
x, info = qgmres_solver.solve(A, b)

print(f"Converged in {info['iterations']} iterations")
print(f"Final residual: {info['residual']:.2e}")

# With LU preconditioning for better convergence
qgmres_lu = QGMRESSolver(tol=1e-6, max_iter=100, preconditioner='left_lu')
x_prec, info_prec = qgmres_lu.solve(A, b)
```

### Matrix Decompositions

```python
from quatica.decomp.qsvd import qr_qua, classical_qsvd, classical_qsvd_full
from quatica.decomp.eigen import quaternion_eigendecomposition
from quatica.decomp.LU import quaternion_lu

# QR decomposition
Q, R = qr_qua(A)

# Quaternion SVD (truncated)
U, s, V = classical_qsvd(A, rank=2)

# Full SVD
U_full, s_full, V_full = classical_qsvd_full(A)

# Eigendecomposition (Hermitian matrices only)
A_hermitian = A + quat_hermitian(A)  # Make it Hermitian
eigenvals, eigenvecs = quaternion_eigendecomposition(A_hermitian)

# LU decomposition
L, U, P = quaternion_lu(A, return_p=True)
```

### Custom Matrix Creation

```python
# Create Pauli matrices in quaternion format
def create_pauli_matrices():
    # σ₀ (identity)
    sigma_0_array = np.zeros((2, 2, 4))
    sigma_0_array[0, 0, 0] = 1.0  # (0,0) real
    sigma_0_array[1, 1, 0] = 1.0  # (1,1) real

    # σₓ (sigma_x)
    sigma_x_array = np.zeros((2, 2, 4))
    sigma_x_array[0, 1, 0] = 1.0  # (0,1) real
    sigma_x_array[1, 0, 0] = 1.0  # (1,0) real

    # Convert to quaternion arrays (simplified approach)
    sigma_0 = quaternion.as_quat_array(sigma_0_array)
    sigma_x = quaternion.as_quat_array(sigma_x_array)

    return sigma_0, sigma_x

sigma_0, sigma_x = create_pauli_matrices()
```

### Visualization

```python
from quatica.visualization import Visualizer

# Plot residual convergence
Visualizer.plot_residuals(residuals, title="Newton-Schulz Convergence")

# Visualize matrix components
Visualizer.visualize_matrix(A, component=0, title="Real Component")  # w component
Visualizer.visualize_matrix(A, component=1, title="i Component")     # x component

# Visualize absolute values
Visualizer.visualize_matrix_abs(A, title="Matrix Absolute Values")
```

### Matrix Generation

```python
from quatica.data_gen import create_test_matrix, create_sparse_quat_matrix, generate_random_unitary_matrix

# Random dense matrix
A_dense = create_test_matrix(m=50, n=30, rank=20)

# Sparse matrix
A_sparse = create_sparse_quat_matrix(m=100, n=100, density=0.1)

# Random unitary matrix
U_unitary = generate_random_unitary_matrix(n=10)
```

## 🎯 Performance Tips

### Optimal Setup

```bash
# Use numpy>=2.3.2 for 10-15x speedup
pip install --upgrade "numpy>=2.3.2"

# Check your numpy version
python -c "import numpy; print(f'numpy: {numpy.__version__}')"
```

### Choose the Right Algorithm

- **Small matrices (<200)**: Standard algorithms
- **Large matrices (≥200)**: Use LU-preconditioned Q-GMRES
- **Pseudoinverse**: Newton-Schulz for speed, Higher-Order NS for accuracy
- **SVD**: Classical for accuracy, randomized for speed on large matrices

### Memory Optimization

```python
# For large matrices, use sparse representations when possible
from quatica.utils import SparseQuaternionMatrix

# Create sparse matrix instead of dense
A_sparse = create_sparse_quat_matrix(1000, 1000, density=0.01)

# Convert dense to sparse if mostly zeros
# (implement conversion based on density threshold)
```

## 📊 Output Files

All examples save results to:

- **`output_figures/`**: Application plots and visualizations
- **`validation_output/`**: Unit test validation figures
- **Console output**: Detailed analysis and metrics

Results include PSNR/SSIM metrics for image processing and convergence analysis for iterative methods.
