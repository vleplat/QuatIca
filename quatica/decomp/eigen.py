"""
Quaternion Matrix Eigendecomposition Module

This module provides eigenvalue decomposition algorithms for HERMITIAN quaternion matrices.

⚠️  IMPORTANT: This implementation is ONLY for Hermitian quaternion matrices.
   For non-Hermitian matrices, consider using the adjoint matrix approach.

Main functions:
- quaternion_eigendecomposition: Eigendecomposition for Hermitian quaternion matrices
- quaternion_eigenvalues: Extract eigenvalues only (Hermitian matrices only)
- quaternion_eigenvectors: Extract eigenvectors only (Hermitian matrices only)

Algorithm: Tridiagonalization + Standard Eigendecomposition
- Tridiagonalize Hermitian matrix A: P * A * P^H = B
- Compute eigendecomposition of tridiagonal matrix B using numpy.linalg.eig
- Transform eigenvectors back: V = P^H * V_B

Limitations:
- Only works for Hermitian quaternion matrices
- For non-Hermitian matrices, use adjoint matrix approach
- For single quaternions, use adjoint matrix approach

References:
- MATLAB QTFM (Quaternion Toolbox for MATLAB) implementation: http://qtfm.sourceforge.net/ by Stephen J. Sangwine & Nicolas Le Bihan
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import quaternion
from utils import quat_frobenius_norm, quat_hermitian, quat_matmat

# Import tridiagonalize using absolute path
sys.path.append(os.path.dirname(__file__))
import tridiagonalize


def quaternion_eigendecomposition(A_quat, verbose=False):
    """
    Compute eigendecomposition of Hermitian quaternion matrix.

    Parameters:
    -----------
    A_quat : numpy.ndarray with dtype=quaternion
        Input quaternion matrix (must be square and Hermitian)
    verbose : bool
        Whether to print convergence information

    Returns:
    --------
    tuple : (eigenvalues, eigenvectors)
        eigenvalues : complex array of eigenvalues
        eigenvectors : quaternion matrix of eigenvectors
    """
    m, n = A_quat.shape
    if m != n:
        raise ValueError("Matrix must be square for eigendecomposition")

    if verbose:
        print(f"Starting eigendecomposition for {m}x{m} quaternion matrix")

    # Check if matrix is Hermitian
    A_hermitian = quat_hermitian(A_quat)
    is_hermitian = np.allclose(A_quat, A_hermitian, atol=1e-10)

    if verbose:
        print(f"Matrix is Hermitian: {is_hermitian}")

    if not is_hermitian:
        print("Warning: Matrix is not Hermitian.")
        print("This implementation only works for Hermitian quaternion matrices.")
        print("For non-Hermitian matrices, consider using the adjoint matrix approach.")
        raise ValueError("Matrix must be Hermitian for this eigendecomposition method")

    if m == 1:
        # 1x1 Hermitian matrix: eigenvalue is the real part of the single element
        element = A_quat[0, 0]
        eigenvalue = complex(element.w, 0.0)  # Hermitian matrix has real eigenvalues
        eigenvector = np.array([[1.0]], dtype=np.quaternion)
        return np.array([eigenvalue]), eigenvector

    # For Hermitian matrices, use tridiagonalization approach
    if verbose:
        print(f"Using tridiagonalization approach for {m}x{m} Hermitian matrix")

    # Step 1: Tridiagonalize A to get P and B where P * A * P^H = B
    P, B = tridiagonalize.tridiagonalize(A_quat)

    if verbose:
        print("Tridiagonalization completed")
        print(f"P shape: {P.shape}")
        print(f"B shape: {B.shape}")

    # Step 2: Convert tridiagonal matrix B to complex form for numpy.linalg.eig
    B_complex = np.zeros((m, m), dtype=complex)
    for i in range(m):
        for j in range(m):
            element = B[i, j]
            B_complex[i, j] = complex(element.w, element.x)

    # Step 3: Compute eigendecomposition of the tridiagonal matrix using numpy
    eigenvalues, eigenvectors_complex = np.linalg.eig(B_complex)

    if verbose:
        print("Eigendecomposition of tridiagonal matrix completed")
        print(f"Eigenvalues: {eigenvalues}")

    # Step 4: Convert eigenvectors back to quaternion format
    eigenvectors_B = np.zeros((m, m), dtype=np.quaternion)
    for i in range(m):
        for j in range(m):
            real_part = eigenvectors_complex[i, j].real
            imag_part = eigenvectors_complex[i, j].imag
            eigenvectors_B[i, j] = quaternion.quaternion(real_part, imag_part, 0, 0)

    # Step 5: Transform eigenvectors back: eigenvectors_A = P^H * eigenvectors_B
    P_H = quat_hermitian(P)
    eigenvectors = quat_matmat(P_H, eigenvectors_B)

    if verbose:
        print("Eigenvectors transformed back to original space")
        print(f"Final eigenvectors shape: {eigenvectors.shape}")

    return eigenvalues, eigenvectors


def quaternion_eigenvalues(A_quat, verbose=False):
    """
    Compute only eigenvalues of Hermitian quaternion matrix.

    Parameters:
    -----------
    A_quat : numpy.ndarray with dtype=quaternion
        Input quaternion matrix (must be square and Hermitian)
    verbose : bool
        Whether to print convergence information

    Returns:
    --------
    eigenvalues : complex array
        Eigenvalues of the matrix
    """
    eigenvalues, _ = quaternion_eigendecomposition(A_quat, verbose)
    return eigenvalues


def quaternion_eigenvectors(A_quat, verbose=False):
    """
    Compute only eigenvectors of Hermitian quaternion matrix.

    Parameters:
    -----------
    A_quat : numpy.ndarray with dtype=quaternion
        Input quaternion matrix (must be square and Hermitian)
    verbose : bool
        Whether to print convergence information

    Returns:
    --------
    eigenvectors : quaternion matrix
        Matrix of eigenvectors (columns are eigenvectors)
    """
    _, eigenvectors = quaternion_eigendecomposition(A_quat, verbose)
    return eigenvectors


def verify_eigendecomposition(A_quat, eigenvalues, eigenvectors, verbose=False):
    """
    Verify eigendecomposition by checking A * v = λ * v for each eigenpair.

    Parameters:
    -----------
    A_quat : quaternion matrix
    eigenvalues : complex array
    eigenvectors : quaternion matrix (columns are eigenvectors)
    verbose : bool

    Returns:
    --------
    dict : verification results
    """
    m, n = A_quat.shape
    max_error = 0.0
    errors = []

    for i in range(min(len(eigenvalues), eigenvectors.shape[1])):
        # Extract eigenvector
        v = eigenvectors[:, i : i + 1]  # Column vector

        # Compute A * v
        Av = quat_matmat(A_quat, v)

        # Compute λ * v (convert complex eigenvalue to quaternion)
        lambda_val = eigenvalues[i]
        lambda_quat = quaternion.quaternion(lambda_val.real, lambda_val.imag, 0, 0)
        lambda_v = lambda_quat * v

        # Compute error
        error = quat_frobenius_norm(Av - lambda_v)
        errors.append(error)
        max_error = max(max_error, error)

        if verbose:
            print(f"  Eigenvalue {i}: λ = {lambda_val:.6f} + {lambda_val.imag:.6f}i")
            print(f"    Error: {error:.2e}")

    result = {
        "max_error": max_error,
        "mean_error": np.mean(errors),
        "errors": errors,
        "success": max_error < 1e-6,
    }

    if verbose:
        print("Eigendecomposition verification:")
        print(f"  Max error: {max_error:.2e}")
        print(f"  Mean error: {result['mean_error']:.2e}")
        print(f"  Success: {result['success']}")

    return result
