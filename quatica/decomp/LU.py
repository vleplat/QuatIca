"""
Quaternion LU Decomposition Module

This module provides LU decomposition for quaternion matrices using Gaussian elimination
with partial pivoting, following the MATLAB QTFM implementation.

Algorithm: Gaussian elimination with partial pivoting
- Partial pivoting based on modulus of quaternion elements
- In-place computation (L and U stored in the same matrix)
- Handles all cases: m > n, m == n, m < n
- Supports both 2-output (L, U) and 3-output (L, U, P) modes

References:
- MATLAB QTFM (Quaternion Toolbox for MATLAB) implementation: http://qtfm.sourceforge.net/ by Stephen J. Sangwine & Nicolas Le Bihan
- Golub, G. H., & Van Loan, C. F. (1996). Matrix Computations (3rd ed.)
- Algorithm 3.2.1, section 3.2.6, modified along the lines of section 3.2.11
"""

import os
import sys

import numpy as np
import quaternion

# Add core module to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import quat_frobenius_norm, quat_matmat


def quaternion_modulus(A):
    """
    Compute the modulus (magnitude) of quaternion matrix elements.

    Parameters:
    -----------
    A : numpy.ndarray with dtype=quaternion
        Input quaternion matrix

    Returns:
    --------
    numpy.ndarray : Real array of moduli
    """
    if isinstance(A, np.ndarray) and A.dtype == np.quaternion:
        comp = quaternion.as_float_array(A)
        return np.sqrt(np.sum(comp**2, axis=-1))
    else:
        raise ValueError("Input must be a quaternion array")


def quaternion_triu(A, k=0):
    """
    Extract upper triangular part of quaternion matrix.

    Parameters:
    -----------
    A : numpy.ndarray with dtype=quaternion
        Input quaternion matrix
    k : int, optional
        Diagonal offset (default: 0)

    Returns:
    --------
    numpy.ndarray : Upper triangular quaternion matrix
    """
    if not isinstance(A, np.ndarray) or A.dtype != np.quaternion:
        raise ValueError("Input must be a quaternion array")

    m, n = A.shape
    result = np.zeros_like(A)

    for i in range(m):
        for j in range(n):
            if j >= i + k:
                result[i, j] = A[i, j]

    return result


def quaternion_tril(A, k=0):
    """
    Extract lower triangular part of quaternion matrix.

    Parameters:
    -----------
    A : numpy.ndarray with dtype=quaternion
        Input quaternion matrix
    k : int, optional
        Diagonal offset (default: 0)

    Returns:
    --------
    numpy.ndarray : Lower triangular quaternion matrix
    """
    if not isinstance(A, np.ndarray) or A.dtype != np.quaternion:
        raise ValueError("Input must be a quaternion array")

    m, n = A.shape
    result = np.zeros_like(A)

    for i in range(m):
        for j in range(n):
            if j <= i + k:
                result[i, j] = A[i, j]

    return result


def quaternion_lu(A, return_p=False):
    """
    LU decomposition of quaternion matrix using Gaussian elimination with partial pivoting.

    Parameters:
    -----------
    A : numpy.ndarray with dtype=quaternion
        Input quaternion matrix of shape (m, n)
    return_p : bool, optional
        Whether to return permutation matrix P (default: False)

    Returns:
    --------
    tuple : (L, U) or (L, U, P)
        L : m×N quaternion matrix (lower triangular with unit diagonal)
        U : N×n quaternion matrix (upper triangular)
        P : m×m permutation matrix (only if return_p=True)
        where N = min(m, n)

    Notes:
    ------
    - Uses partial pivoting based on modulus of quaternion elements
    - Handles all matrix shapes: m > n, m == n, m < n
    - If return_p=False, L is permuted so that A = L * U
    - If return_p=True, P * A = L * U
    """
    if not isinstance(A, np.ndarray) or A.dtype != np.quaternion:
        raise ValueError("Input must be a quaternion array")

    m, n = A.shape
    N = min(m, n)  # Number of elements on the diagonal

    # Create a copy of A for in-place modification
    A_work = A.copy()

    # Permutation vector for row swaps
    IP = list(range(m))

    # Gaussian elimination with partial pivoting
    for j in range(N):
        # Partial pivoting: find largest element in column j from position j downwards
        col_moduli = quaternion_modulus(A_work[j:m, j])
        k = np.argmax(col_moduli) + 1  # +1 because we start from j

        if k != 1:  # If k == 1, largest element is already at A(j, j)
            # Swap rows j and j + k - 1
            l = j + k - 1
            IP[j], IP[l] = IP[l], IP[j]

            # Swap rows in A_work
            A_work[j, :], A_work[l, :] = A_work[l, :].copy(), A_work[j, :].copy()

        if j == m - 1:
            break  # If true, j+1:m would be an empty range

        # Scale column j below diagonal: A(j+1:m, j) = A(j+1:m, j) ./ A(j, j)
        pivot = A_work[j, j]
        pivot_modulus = quaternion_modulus(np.array([[pivot]]))[0, 0]
        if pivot_modulus < 1e-15:  # Use small threshold for numerical stability
            raise ValueError(f"Zero pivot encountered at position ({j}, {j})")

        for i in range(j + 1, m):
            A_work[i, j] = A_work[i, j] / pivot

        if j == n - 1:
            break  # If true, j+1:n would be an empty range

        # Update submatrix: A(j+1:m, j+1:n) = A(j+1:m, j+1:n) - A(j+1:m, j) * A(j, j+1:n)
        # Use matrix multiplication for better numerical stability
        if j + 1 < m and j + 1 < n:
            # Extract the submatrices
            submatrix = A_work[j + 1 : m, j + 1 : n]
            col_vector = A_work[j + 1 : m, j : j + 1]  # Column j from row j+1 to m
            row_vector = A_work[j : j + 1, j + 1 : n]  # Row j from column j+1 to n

            # Update: submatrix = submatrix - col_vector * row_vector
            update = quat_matmat(col_vector, row_vector)
            A_work[j + 1 : m, j + 1 : n] = submatrix - update

    # Extract L and U from the modified matrix
    # U is the upper triangular part of A_work(1:N, :)
    U = quaternion_triu(A_work[:N, :])

    # L is the lower triangular part of A_work(:, 1:N) with unit diagonal
    # The multipliers are stored in the lower triangular part of A_work
    L = np.zeros((m, N), dtype=np.quaternion)
    for i in range(m):
        for j in range(N):
            if i > j:
                L[i, j] = A_work[i, j]  # These are the multipliers
            elif i == j:
                L[i, j] = quaternion.quaternion(1, 0, 0, 0)  # Unit diagonal
            else:
                L[i, j] = quaternion.quaternion(0, 0, 0, 0)  # Zero above diagonal

    # Ensure U is properly upper triangular
    U = quaternion_triu(U)

    if return_p:
        # Create permutation matrix P
        P = np.zeros((m, m), dtype=np.quaternion)
        for i in range(m):
            P[i, IP[i]] = quaternion.quaternion(1, 0, 0, 0)
        return L, U, P
    else:
        # Permute L so that A = L * U (equivalent to multiplication by P^T)
        # Create inverse permutation: IX[i] = j means original row j is now row i
        IX = [0] * m
        for i in range(m):
            IX[IP[i]] = i

        # Apply inverse permutation to L
        L_permuted = np.zeros_like(L)
        for i in range(m):
            L_permuted[IX[i], :] = L[i, :]

        return L_permuted, U


def verify_lu_decomposition(A, L, U, P=None, verbose=False):
    """
    Verify LU decomposition by checking reconstruction accuracy.

    Parameters:
    -----------
    A : numpy.ndarray with dtype=quaternion
        Original quaternion matrix
    L : numpy.ndarray with dtype=quaternion
        Lower triangular matrix
    U : numpy.ndarray with dtype=quaternion
        Upper triangular matrix
    P : numpy.ndarray with dtype=quaternion, optional
        Permutation matrix
    verbose : bool, optional
        Whether to print verification results

    Returns:
    --------
    dict : Verification results
    """
    if P is not None:
        # Check P * A = L * U
        PA = quat_matmat(P, A)
        LU = quat_matmat(L, U)
        reconstruction_error = quat_frobenius_norm(PA - LU)
        original_norm = quat_frobenius_norm(PA)
    else:
        # Check A = L * U
        LU = quat_matmat(L, U)
        reconstruction_error = quat_frobenius_norm(A - LU)
        original_norm = quat_frobenius_norm(A)

    relative_error = reconstruction_error / original_norm if original_norm > 0 else 0

    if verbose:
        print("LU Decomposition Verification:")
        print(f"  Reconstruction error: {reconstruction_error:.2e}")
        print(f"  Relative error: {relative_error:.2e}")
        print(f"  Verification: {'✅ PASSED' if relative_error < 1e-12 else '❌ FAILED'}")

    return {
        "reconstruction_error": reconstruction_error,
        "relative_error": relative_error,
        "passed": relative_error < 1e-12,
    }
