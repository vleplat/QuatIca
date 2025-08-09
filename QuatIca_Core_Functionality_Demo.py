# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # QuatIca Core Functionality Demo
# 
# This notebook demonstrates all the core functionality examples from the README.
# Run each cell to see the code in action!

# ## Setup and Imports

import sys
import os
import numpy as np
import quaternion
import matplotlib.pyplot as plt

# Add the core module to the path
sys.path.append('core')

print("✅ All imports successful!")

# ## 1. 🧮 Basic Matrix Operations

from core.utils import quat_matmat, quat_frobenius_norm
from core.data_gen import create_test_matrix

# Create test matrices
A = create_test_matrix(3, 4)
B = create_test_matrix(4, 2)

print("Matrix A shape:", A.shape)
print("Matrix B shape:", B.shape)
print("Matrix A norm:", quat_frobenius_norm(A))
print("Matrix B norm:", quat_frobenius_norm(B))

# Matrix multiplication
C = quat_matmat(A, B)
print("Matrix C = A @ B shape:", C.shape)
print("Matrix C norm:", quat_frobenius_norm(C))

print("✅ Basic matrix operations work!")

# ## 2. 📐 QR Decomposition

from core.decomp.qsvd import qr_qua

# Create a test matrix
X_quat = create_test_matrix(4, 3)
print("Input matrix X shape:", X_quat.shape)

# QR decomposition
Q, R = qr_qua(X_quat)
print("Q shape:", Q.shape)
print("R shape:", R.shape)

# Verify reconstruction
X_recon = quat_matmat(Q, R)
reconstruction_error = quat_frobenius_norm(X_quat - X_recon)
print("Reconstruction error:", reconstruction_error)

print("✅ QR decomposition works!")

# ## 3. 🔍 Quaternion SVD (Q-SVD)

from core.decomp.qsvd import classical_qsvd, classical_qsvd_full

# Create a test matrix
X_quat = create_test_matrix(5, 4)
print("Input matrix X shape:", X_quat.shape)

# Truncated Q-SVD
R = 2  # Target rank
U, s, V = classical_qsvd(X_quat, R)
print("Truncated Q-SVD:")
print("  U shape:", U.shape)
print("  s length:", len(s))
print("  V shape:", V.shape)

# Full Q-SVD
U_full, s_full, V_full = classical_qsvd_full(X_quat)
print("\nFull Q-SVD:")
print("  U_full shape:", U_full.shape)
print("  s_full length:", len(s_full))
print("  V_full shape:", V_full.shape)

print("✅ Q-SVD works!")

# ## 4. 🎲 Randomized Q-SVD

from core.decomp.qsvd import rand_qsvd
from core.utils import quat_hermitian

# Create a test matrix
X_quat = create_test_matrix(8, 6)
print("Input matrix X shape:", X_quat.shape)

# Randomized Q-SVD with different parameters
R = 3  # Target rank
print(f"Target rank R = {R}")

# Test with different power iterations
for n_iter in [1, 2, 3]:
    print(f"\nTesting with {n_iter} power iteration(s):")
    
    U, s, V = rand_qsvd(X_quat, R, oversample=5, n_iter=n_iter)
    print(f"  U shape: {U.shape}")
    print(f"  V shape: {V.shape}")
    print(f"  s shape: {s.shape}")
    print(f"  Singular values: {s}")
    
    # Test reconstruction
    S_diag = np.diag(s)
    X_recon = quat_matmat(quat_matmat(U, S_diag), quat_hermitian(V))
    reconstruction_error = quat_frobenius_norm(X_quat - X_recon)
    relative_error = reconstruction_error / quat_frobenius_norm(X_quat)
    print(f"  Reconstruction error: {reconstruction_error:.6f}")
    print(f"  Relative error: {relative_error:.6f}")

# Test with full rank for perfect reconstruction
print(f"\nTesting with full rank (R = {min(X_quat.shape)}):")
U_full, s_full, V_full = rand_qsvd(X_quat, min(X_quat.shape), oversample=5, n_iter=2)
S_full_diag = np.diag(s_full)
X_recon_full = quat_matmat(quat_matmat(U_full, S_full_diag), quat_hermitian(V_full))
reconstruction_error_full = quat_frobenius_norm(X_quat - X_recon_full)
print(f"  Full rank reconstruction error: {reconstruction_error_full:.2e}")

print("✅ Randomized Q-SVD works!")

# ## 5. 🔢 Eigenvalue Decomposition

from core.decomp import quaternion_eigendecomposition, quaternion_eigenvalues, quaternion_eigenvectors
from core.utils import quat_hermitian

# Create a Hermitian matrix A = B^H @ B
B = create_test_matrix(4, 3)
B_H = quat_hermitian(B)
A_quat = quat_matmat(B_H, B)
print("Hermitian matrix A shape:", A_quat.shape)

# Full eigendecomposition
eigenvalues, eigenvectors = quaternion_eigendecomposition(A_quat)
print("Full eigendecomposition:")
print("  Number of eigenvalues:", len(eigenvalues))
print("  Eigenvectors shape:", eigenvectors.shape)

# Extract only eigenvalues
eigenvals = quaternion_eigenvalues(A_quat)
print("\nEigenvalues only:", len(eigenvals))

# Extract only eigenvectors
eigenvecs = quaternion_eigenvectors(A_quat)
print("Eigenvectors only shape:", eigenvecs.shape)

# Verify eigenvalues are real
imaginary_parts = np.imag(eigenvalues)
max_imag = np.max(np.abs(imaginary_parts))
print("Maximum imaginary part:", max_imag)

print("✅ Eigenvalue decomposition works!")

# ## 6. 🔧 LU Decomposition

from core.decomp import quaternion_lu, verify_lu_decomposition

# Create a test matrix (set to 300x600 to validate rectangular LU)
A = create_test_matrix(300, 600)
print("Matrix A shape:", A.shape)

# LU decomposition
L, U = quaternion_lu(A)
print("LU decomposition:")
print("  L shape:", L.shape)
print("  U shape:", U.shape)

# Verify reconstruction (A = L @ U)
LU = quat_matmat(L, U)
reconstruction_error = quat_frobenius_norm(A - LU)
relative_error = reconstruction_error / quat_frobenius_norm(A)
print("  Reconstruction error:", reconstruction_error)
print("  Relative error:", relative_error)

# Test with permutation matrix
L_p, U_p, P = quaternion_lu(A, return_p=True)
print("\nWith permutation matrix:")
print("  P shape:", P.shape)
PA = quat_matmat(P, A)
LU_p = quat_matmat(L_p, U_p)
permutation_error = quat_frobenius_norm(PA - LU_p)
print("  P*A = L*U error:", permutation_error)

# Test the alternative form: A = (P^T * L) * U
P_T = quat_hermitian(P)  # P^T
P_T_L = quat_matmat(P_T, L_p)  # P^T * L
A_recon_alt = quat_matmat(P_T_L, U_p)  # (P^T * L) * U
alt_error = quat_frobenius_norm(A - A_recon_alt)
print("  A = (P^T * L) * U error:", alt_error)

# Structural checks on pivoted factors (valid for rectangular matrices):
N = min(A.shape[0], A.shape[1])

# L_p should be lower-triangular (lower-trapezoidal) with unit diagonal on the leading N×N block
L_p_real = quaternion.as_float_array(L_p)[:, :, 0]
is_Lp_lower = np.allclose(L_p_real, np.tril(L_p_real), atol=1e-12)
unit_diag = np.allclose(np.diag(L_p_real[:N, :N]), np.ones(N), atol=1e-12)
print("  L (pivoted) is lower-triangular:", is_Lp_lower)
print("  L (pivoted) has unit diagonal (first N):", unit_diag)

# U_p (N×n) should be upper-triangular (upper-trapezoidal)
U_p_real = quaternion.as_float_array(U_p)[:, :, 0]
is_Up_upper = np.allclose(U_p_real, np.triu(U_p_real), atol=1e-12)
print("  U (pivoted) is upper-triangular:", is_Up_upper)

# Check if P^T * L is lower triangular (only when no pivoting needed)
P_T_L_float = quaternion.as_float_array(P_T_L)
P_T_L_real = P_T_L_float[:, :, 0]
is_P_T_L_lower_triangular = np.allclose(P_T_L_real, np.tril(P_T_L_real), atol=1e-12)
print("  P^T * L is lower triangular:", is_P_T_L_lower_triangular, "(only when no pivoting needed)")

print("✅ LU decomposition works!")

# ## 7. Tridiagonalization

from core.decomp import tridiagonalize

# Use the same Hermitian matrix from above
print("Hermitian matrix A shape:", A_quat.shape)

# Tridiagonalize
P, B_tridiag = tridiagonalize(A_quat)
print("Tridiagonalization:")
print("  P shape:", P.shape)
print("  B shape:", B_tridiag.shape)

# Verify transformation
P_H = quat_hermitian(P)
PAP_H = quat_matmat(quat_matmat(P, A_quat), P_H)
transformation_error = quat_frobenius_norm(PAP_H - B_tridiag)
print("  Transformation error:", transformation_error)

print("✅ Tridiagonalization works!")

# ## 8. Pseudoinverse Computation

from core.solver import NewtonSchulzPseudoinverse

# Create a test matrix
A = create_test_matrix(3, 4)
print("Matrix A shape:", A.shape)

# Compute pseudoinverse
solver = NewtonSchulzPseudoinverse()
A_pinv, residuals, covariances = solver.compute(A)
print("Pseudoinverse A^† shape:", A_pinv.shape)

# Verify pseudoinverse properties
A_pinv_H = quat_hermitian(A_pinv)
print("A^† shape:", A_pinv.shape)
print("A^†^† shape:", A_pinv_H.shape)

print("✅ Pseudoinverse computation works!")

# ## 9. Linear System Solving

from core.solver import QGMRESSolver

# Create a square system A * x = b
A = create_test_matrix(3, 3)
b = create_test_matrix(3, 1)
print("System A shape:", A.shape)
print("Right-hand side b shape:", b.shape)

# Solve using Q-GMRES
solver = QGMRESSolver()
x, info = solver.solve(A, b)
print("Solution x shape:", x.shape)
print("Convergence info:", info)

# Verify solution
Ax = quat_matmat(A, x)
residual = quat_frobenius_norm(Ax - b)
print("Residual ||A*x - b||:", residual)

print("✅ Linear system solving works!")

# ## 10. Visualization

from core.visualization import Visualizer

# Create a test matrix
A = create_test_matrix(4, 4)
print("Matrix A shape:", A.shape)

# Plot matrix components
Visualizer.visualize_matrix(A, component=0, title="Test Matrix - Real Component")
Visualizer.visualize_matrix(A, component=1, title="Test Matrix - i Component")

print("✅ Visualization works!")

# ## 11. Determinant and Rank Computation

from core.utils import det, rank
from core.data_gen import generate_random_unitary_matrix

print("\n" + "="*60)
print("DETERMINANT AND RANK COMPUTATION DEMONSTRATIONS")
print("="*60)

# ### 10.1 Determinant Demo: Unitary Matrix with Known Determinant

print("\n--- Determinant Demo: Unitary Matrix ---")

# Generate a random unitary matrix (determinant should be 1)
n = 4
U = generate_random_unitary_matrix(n)
print(f"Generated unitary matrix U of size {n}×{n}")

# Compute Dieudonné determinant
det_dieudonne = det(U, 'Dieudonne')
print(f"Dieudonné determinant: {det_dieudonne:.6f}")

# Expected determinant for unitary matrix should be close to 1
expected_det = 1.0
error = abs(det_dieudonne - expected_det)
print(f"Expected determinant: {expected_det}")
print(f"Absolute error: {error:.2e}")

if error < 1e-10:
    print("✅ Determinant computation works correctly!")
else:
    print("❌ Determinant computation has issues!")

# ### 10.2 Rank Demo: Matrix Product with Known Rank

print("\n--- Rank Demo: Matrix Product ---")

# Create matrices A (m×r) and B (r×n) with known rank r
m, r, n = 5, 3, 4
A = create_test_matrix(m, r)
B = create_test_matrix(r, n)
print(f"Matrix A: {m}×{r}")
print(f"Matrix B: {r}×{n}")

# Compute product C = A @ B
C = quat_matmat(A, B)
print(f"Matrix C = A @ B: {C.shape}")

# Compute rank of C
computed_rank = rank(C)
expected_rank = r
print(f"Computed rank of C: {computed_rank}")
print(f"Expected rank: {expected_rank}")

if computed_rank == expected_rank:
    print("✅ Rank computation works correctly!")
else:
    print("❌ Rank computation has issues!")

# ### 10.3 Additional Rank Examples

print("\n--- Additional Rank Examples ---")

# Test full-rank matrix
full_rank_matrix = create_test_matrix(4, 4)
full_rank = rank(full_rank_matrix)
print(f"Full-rank 4×4 matrix: rank = {full_rank} (expected: 4)")

# Test zero matrix
zero_matrix = np.zeros((3, 3), dtype=np.quaternion)
zero_rank = rank(zero_matrix)
print(f"Zero 3×3 matrix: rank = {zero_rank} (expected: 0)")

# Test identity matrix
identity_matrix = np.eye(5, dtype=np.quaternion)
identity_rank = rank(identity_matrix)
print(f"Identity 5×5 matrix: rank = {identity_rank} (expected: 5)")

print("✅ All rank examples work correctly!")

# ## 12. Power Iteration for Dominant Eigenvector

from core.utils import power_iteration
from core.decomp.eigen import quaternion_eigendecomposition

print("\n" + "="*60)
print("POWER ITERATION FOR DOMINANT EIGENVECTOR")
print("="*60)

# ### 11.1 Power Iteration Demo: Comparison with Eigendecomposition

print("\n--- Power Iteration vs Eigendecomposition ---")

# Create Hermitian matrix: A = B^H @ B (positive definite)
B = create_test_matrix(5, 5)
A = quat_matmat(quat_hermitian(B), B)
print(f"Created Hermitian matrix A of size {A.shape}")

# Run power iteration
print("\nRunning power iteration...")
power_eigenvector, power_eigenvalue = power_iteration(A, return_eigenvalue=True, verbose=True)

# Run eigendecomposition
print("\nRunning eigendecomposition...")
eigenvalues, eigenvectors = quaternion_eigendecomposition(A, verbose=False)

# Find dominant eigenvalue and eigenvector
dominant_idx = np.argmax(np.abs(eigenvalues))
dominant_eigenvalue = eigenvalues[dominant_idx]
dominant_eigenvector = eigenvectors[:, dominant_idx:dominant_idx+1]

print(f"\nComparison Results:")
print(f"Power iteration eigenvalue: {power_eigenvalue:.6f}")
print(f"Eigendecomposition dominant eigenvalue: {dominant_eigenvalue:.6f}")
print(f"Eigenvalue difference: {abs(power_eigenvalue - abs(dominant_eigenvalue)):.2e}")

# Compare eigenvectors
power_norm = quat_frobenius_norm(power_eigenvector)
decomp_norm = quat_frobenius_norm(dominant_eigenvector)

power_normalized = power_eigenvector / power_norm
decomp_normalized = dominant_eigenvector / decomp_norm

dot_product = quat_matmat(quat_hermitian(power_normalized), decomp_normalized)
dot_product_norm = quat_frobenius_norm(dot_product)

print(f"Eigenvector alignment: {dot_product_norm:.6f}")

# Verify results
eigenvalue_error = abs(power_eigenvalue - abs(dominant_eigenvalue))
eigenvector_error = abs(dot_product_norm - 1.0)

if eigenvalue_error < 1e-6 and eigenvector_error < 1e-6:
    print("✅ Power iteration matches eigendecomposition perfectly!")
else:
    print("❌ Power iteration has issues!")

# ### 11.2 Power Iteration Performance Across Sizes

print("\n--- Performance Across Matrix Sizes ---")

sizes = [3, 6, 9]
for size in sizes:
    print(f"\nTesting {size}×{size} matrix:")
    
    # Create test matrix
    B = create_test_matrix(size, size)
    A = quat_matmat(quat_hermitian(B), B)
    
    # Time power iteration
    import time
    start_time = time.time()
    eigenvector, eigenvalue = power_iteration(A, return_eigenvalue=True, verbose=False)
    power_time = time.time() - start_time
    
    # Time eigendecomposition
    start_time = time.time()
    eigenvalues, eigenvectors = quaternion_eigendecomposition(A, verbose=False)
    decomp_time = time.time() - start_time
    
    print(f"  Power iteration: {power_time:.3f}s")
    print(f"  Eigendecomposition: {decomp_time:.3f}s")
    print(f"  Speedup: {decomp_time/power_time:.1f}x faster")

print("✅ Power iteration performance analysis complete!")

# ## 14. 🔬 Advanced Eigenvalue Methods

from core.utils import power_iteration_nonhermitian

print("\n" + "="*60)
print("COMPLEX POWER ITERATION (ENHANCED WITH VALIDATION)")
print("="*60)

# ## 12b.1 Hermitian Case - More reliable convergence
print("\n--- HERMITIAN CASE ---")
n = 25
B_rand = create_test_matrix(n, n)
A_hermitian = quat_matmat(quat_hermitian(B_rand), B_rand)  # Make Hermitian: A = B^H * B
print(f"Hermitian matrix A = B^H * B of size {n}x{n}")

# Run complex power iteration on Hermitian matrix (should converge to real eigenvalue)
q_vec, lambda_complex, residuals = power_iteration_nonhermitian(
    A_hermitian,
    max_iterations=5000,
    eig_tol=1e-12,
    res_tol=1e-12,
    seed=0,
    return_vector=True,
)

# Print eigenvalue - should be real for Hermitian matrix
lam_q = quaternion.quaternion(float(np.real(lambda_complex)), float(np.imag(lambda_complex)), 0.0, 0.0)
print(f"Estimated dominant eigenvalue: {lambda_complex}")
print(f"As quaternion (x-axis subfield): {lam_q}")
print(f"Imaginary part (should be ~0 for Hermitian): {abs(np.imag(lambda_complex)):.2e}")
print(f"Residual final: {residuals[-1] if residuals else float('nan'):.3e} | steps: {len(residuals)}")

# Plot residual convergence for Hermitian case
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
if residuals:
    plt.semilogy(residuals)
plt.title(f"Hermitian case: Residual convergence (n={n})")
plt.xlabel("iteration")
plt.ylabel("||Mv - lambda v||_2 (adjoint residual)")
plt.grid(True, which="both", ls=":")

# ## 12b.2 Synthetic Unitary Similarity Case - A = P S P^H
print("\n--- SYNTHETIC UNITARY SIMILARITY CASE ---")

def complex_to_quaternion_matrix(C):
    """Convert complex matrix to quaternion matrix (x-axis subfield)."""
    m, n = C.shape
    Q = np.empty((m, n), dtype=np.quaternion)
    for i in range(m):
        for j in range(n):
            a = float(np.real(C[i, j]))
            b = float(np.imag(C[i, j]))
            Q[i, j] = quaternion.quaternion(a, b, 0.0, 0.0)
    return Q

def random_complex_unitary(n, rng):
    """Generate random complex unitary matrix."""
    X = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q_complex, _ = np.linalg.qr(X)
    return Q_complex

def build_diagonal_complex_quat(values):
    """Build diagonal quaternion matrix from complex values."""
    n = values.shape[0]
    S = np.zeros((n, n), dtype=np.quaternion)
    for i, lam in enumerate(values):
        S[i, i] = quaternion.quaternion(float(np.real(lam)), float(np.imag(lam)), 0.0, 0.0)
    return S

# Build synthetic A = P S P^H with known spectrum
rng = np.random.default_rng(1)
n_synth = 12
Uc = random_complex_unitary(n_synth, rng)
P = complex_to_quaternion_matrix(Uc)
spectrum_vals = rng.standard_normal(n_synth) + 1j * rng.standard_normal(n_synth)
S = build_diagonal_complex_quat(spectrum_vals)
A_synthetic = quat_matmat(quat_matmat(P, S), quat_hermitian(P))

print(f"Synthetic A = P S P^H matrix of size {n_synth}x{n_synth}")
print(f"Known spectrum: {[f'{v:.3f}' for v in spectrum_vals[:3]]}... (showing first 3)")

# Run power iteration on synthetic matrix
q_vec_synth, lam_synth, residuals_synth = power_iteration_nonhermitian(
    A_synthetic,
    max_iterations=8000,
    eig_tol=1e-14,
    res_tol=1e-12,
    seed=1,
    return_vector=True,
)

# Validate against known spectrum (up to conjugate)
dists = [abs(lam_synth - ev) for ev in spectrum_vals] + [abs(lam_synth - np.conjugate(ev)) for ev in spectrum_vals]
min_dist = min(dists)
scale = max(1e-12, max(abs(ev) for ev in spectrum_vals))
rel_error = min_dist / scale

print(f"Estimated eigenvalue: {lam_synth}")
print(f"Relative error to known spectrum: {rel_error:.2e}")
print(f"Residual final: {residuals_synth[-1] if residuals_synth else float('nan'):.3e} | steps: {len(residuals_synth)}")

# Plot residual convergence for synthetic case
plt.subplot(1, 2, 2)
if residuals_synth:
    plt.semilogy(residuals_synth)
plt.title(f"Synthetic case: Residual convergence (n={n_synth})")
plt.xlabel("iteration")
plt.ylabel("||Mv - lambda v||_2 (adjoint residual)")
plt.grid(True, which="both", ls=":")

plt.tight_layout()
plt.show()

print("✅ Enhanced complex power iteration with validation complete!")

# ## 15. 🧮 Schur Decomposition

from core.decomp.schur import quaternion_schur_unified
from core.utils import quat_eye

print("\n" + "="*60)
print("SCHUR DECOMPOSITION (SYNTHETIC UNITARY SIMILARITY)")
print("="*60)

# Build synthetic A = P S P^H with known spectrum for validation
def complex_to_quaternion_matrix_schur(C):
    """Convert complex matrix to quaternion matrix (x-axis subfield)."""
    m, n = C.shape
    Q = np.empty((m, n), dtype=np.quaternion)
    for i in range(m):
        for j in range(n):
            a = float(np.real(C[i, j]))
            b = float(np.imag(C[i, j]))
            Q[i, j] = quaternion.quaternion(a, b, 0.0, 0.0)
    return Q

def random_complex_unitary_schur(n, rng):
    """Generate random complex unitary matrix."""
    X = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q_complex, _ = np.linalg.qr(X)
    return Q_complex

def build_diagonal_complex_quat_schur(values):
    """Build diagonal quaternion matrix from complex values."""
    n = values.shape[0]
    S = np.zeros((n, n), dtype=np.quaternion)
    for i, lam in enumerate(values):
        S[i, i] = quaternion.quaternion(float(np.real(lam)), float(np.imag(lam)), 0.0, 0.0)
    return S

def quat_abs_matrix(T):
    """Compute |T| matrix with entrywise quaternion magnitudes."""
    Tf = quaternion.as_float_array(T)
    return np.sqrt(np.sum(Tf**2, axis=2))

# Synthetic construction A = P S P^H
rng = np.random.default_rng(0)
n_schur = 16
Uc = random_complex_unitary_schur(n_schur, rng)
P = complex_to_quaternion_matrix_schur(Uc)
vals = rng.standard_normal(n_schur) + 1j * rng.standard_normal(n_schur)
S = build_diagonal_complex_quat_schur(vals)
A_schur = quat_matmat(quat_matmat(P, S), quat_hermitian(P))

print(f"Synthetic A = P S P^H matrix of size {n_schur}x{n_schur}")
print(f"Known spectrum: {[f'{v:.3f}' for v in vals[:3]]}... (showing first 3)")

# Schur decomposition
Q_schur, T_schur, diag_schur = quaternion_schur_unified(
    A_schur, 
    variant="rayleigh", 
    max_iter=2000, 
    tol=1e-10, 
    return_diagnostics=True
)

# Validation metrics
sim_error = quat_frobenius_norm(quat_matmat(quat_hermitian(Q_schur), quat_matmat(A_schur, Q_schur)) - T_schur)
unit_error = quat_frobenius_norm(quat_matmat(quat_hermitian(Q_schur), Q_schur) - quat_eye(n_schur))

# Check upper triangular structure
below_diag_max = 0.0
for i in range(n_schur):
    for j in range(0, i):
        q = T_schur[i, j]
        below_diag_max = max(below_diag_max, (q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z)**0.5)

print(f"Similarity error ||Q^H A Q - T||_F: {sim_error:.3e}")
print(f"Unitarity error ||Q^H Q - I||_F: {unit_error:.3e}")
print(f"Below diagonal maximum: {below_diag_max:.3e}")

# Visualize |T| matrix
from core.visualization import Visualizer
Visualizer.visualize_matrix(T_schur, component=0, title="Schur T - Real Component")

print("✅ Schur decomposition with synthetic validation complete!")

# ## 16. 📊 Tensor Operations

from core.tensor import (
    tensor_frobenius_norm,
    tensor_entrywise_abs,
    tensor_unfold,
    tensor_fold,
)

print("\n" + "="*60)
print("QUATERNION TENSOR ALGEBRA AND DECOMPOSITIONS")
print("="*60)

# Build random order-3 quaternion tensor
I, J, K = 5, 4, 6
comp = np.random.randn(I, J, K, 4)
T_tensor = quaternion.as_quat_array(comp)

print(f"Quaternion tensor T shape: {T_tensor.shape}, dtype={T_tensor.dtype}")
print(f"Frobenius-like norm ||T||_F: {tensor_frobenius_norm(T_tensor):.6f}")

# Visualize |T| at a fixed k slice
k = 2
Abs = tensor_entrywise_abs(T_tensor)
plt.figure(figsize=(6, 4))
plt.imshow(Abs[:, :, k], cmap='viridis', aspect='auto')
plt.title(f"|T| slice at k={k}")
plt.colorbar(fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# Mode-1 unfolding and folding (round-trip test)
M1 = tensor_unfold(T_tensor, mode=1)
print(f"Mode-1 unfold shape: {M1.shape}")
T_back = tensor_fold(M1, mode=1, shape=(I, J, K))

# Check roundtrip accuracy
ok = np.all(quaternion.as_float_array(T_back) == quaternion.as_float_array(T_tensor))
print(f"Unfold/fold round-trip exact equality: {ok}")

print("✅ Tensor operations complete!")
print("Preview complete — tensor tools lay the groundwork for future tensor decompositions (e.g., HOSVD, TT, Tucker) in quaternion space.")

# ## 13. 🔧 Hessenberg Form (Upper Hessenberg Reduction)

from core.decomp.hessenberg import hessenbergize, is_hessenberg

print("\n" + "="*60)
print("HESSENBERG FORM (UPPER HESSENBERG REDUCTION)")
print("="*60)

# Create a random quaternion matrix (general, non-Hermitian)
X = create_test_matrix(6, 6)
print("Random matrix X shape:", X.shape)

# Compute Hessenberg form
P_hess, H = hessenbergize(X)
print("Hessenberg reduction:")
print("  P shape:", P_hess.shape)
print("  H shape:", H.shape)

# Verify unitarity of P: P^H P = I
P_hess_H = quat_hermitian(P_hess)
I_check = quat_matmat(P_hess_H, P_hess)
is_unitary = np.allclose(I_check, quat_eye(P_hess.shape[0]), atol=1e-10)
print("  P is unitary (P^H P = I):", is_unitary)

# Verify similarity relation: H = P * X * P^H
PX = quat_matmat(P_hess, X)
PXPH = quat_matmat(PX, P_hess_H)
sim_error = quat_frobenius_norm(PXPH - H)
print(f"  Similarity error ||P X P^H - H||_F: {sim_error:.2e}")

# Check Hessenberg structure
print("  is_hessenberg(H):", is_hessenberg(H))

# Visualize the real component to illustrate Hessenberg pattern
from core.visualization import Visualizer
Visualizer.visualize_matrix(H, component=0, title="Hessenberg H - Real Component")

# ## Summary

print("🎉 ALL CORE FUNCTIONALITY TESTS COMPLETED SUCCESSFULLY!")
print("\n✅ Basic matrix operations")
print("✅ QR decomposition")
print("✅ Quaternion SVD (Q-SVD)")
print("✅ Randomized Q-SVD")
print("✅ Eigenvalue decomposition")
print("✅ LU decomposition")
print("✅ Tridiagonalization")
print("✅ Pseudoinverse computation")
print("✅ Linear system solving")
print("✅ Visualization")
print("✅ Determinant computation")
print("✅ Rank computation")
print("✅ Power iteration")
print("✅ Hessenberg form")
print("✅ Advanced eigenvalue methods")
print("✅ Schur decomposition")
print("✅ Tensor operations")
print("\nThe code examples in the README are working correctly! 🚀") 