"""
Quaternion Matrix Tridiagonalization Module

This module implements tridiagonalization of HERMITIAN quaternion matrices
using Householder transformations.

⚠️  IMPORTANT: This implementation is ONLY for Hermitian quaternion matrices.
   For non-Hermitian matrices, consider using the adjoint matrix approach.

Main functions:
- tridiagonalize: Convert Hermitian quaternion matrix to tridiagonal form
- householder_matrix: Compute Householder transformation matrix
- householder_vector: Compute Householder vector and zeta value
- internal_tridiagonalizer: Recursive tridiagonalization algorithm

Algorithm: Householder Transformations
- Uses Householder transformations to eliminate subdiagonal elements
- Converts Hermitian matrix A to tridiagonal form B: P * A * P^H = B
- B is real tridiagonal with same eigenvalues as A
- P is unitary transformation matrix

Limitations:
- Only works for Hermitian quaternion matrices
- For non-Hermitian matrices, use adjoint matrix approach

References:
- MATLAB QTFM (Quaternion Toolbox for MATLAB) implementation: http://qtfm.sourceforge.net/ by Stephen J. Sangwine & Nicolas Le Bihan
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import quaternion
from utils import quat_frobenius_norm, quat_hermitian, quat_matmat


def householder_vector(a, v):
    """
    Calculate a Householder vector u with norm sqrt(2) and value zeta.

    Parameters:
    -----------
    a : quaternion array
        Vector to reflect
    v : numpy array (real)
        Target vector (must be real)

    Returns:
    --------
    tuple : (u, zeta)
        u : quaternion array (Householder vector)
        zeta : quaternion (scalar value)
    """
    if a.shape != v.shape:
        raise ValueError("Input parameters must be vectors of the same size")

    if np.any(np.imag(v) != 0):
        raise ValueError(
            "Parameter v may not be a quaternion vector (mathematical limitation)"
        )

    # Check if column or row vector
    col_vector = len(a.shape) == 1 or a.shape[1] == 1

    alpha = quat_frobenius_norm(a)

    if alpha == 0:
        u = a - a  # Zero vector of same type as a
        zeta = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
        return u, zeta

    # Compute romega
    if col_vector:
        romega = np.sum(a * v)
    else:
        romega = np.sum(v * a)

    r = abs(romega)

    if r != 0:
        zeta = -romega / r
    else:
        zeta = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)

    mu = np.sqrt(alpha * (alpha + r))

    if col_vector:
        u = (a - (zeta * v) * alpha) / mu
    else:
        u = (alpha * (v * zeta) - a).conj() / mu

    return u, zeta


def householder_matrix(a, v):
    """
    Compute Householder matrix that will zero all elements of a except those
    corresponding to non-zero elements of v.

    Parameters:
    -----------
    a : quaternion array
        Vector to reflect
    v : numpy array (real)
        Target vector (must be real)

    Returns:
    --------
    h : quaternion matrix
        Householder transformation matrix
    """
    if a.shape != v.shape:
        raise ValueError("Input parameters must be vectors of the same size")

    # Check if column or row vector
    col_vector = len(a.shape) == 1 or a.shape[1] == 1

    h = np.eye(len(a), dtype=np.quaternion)

    n = np.linalg.norm(v)  # If v has zero norm, return identity matrix

    if n != 0:
        u, zeta = householder_vector(a, v / n)

        if col_vector:
            # Compute u * u^H (outer product)
            uuH = np.zeros((len(u), len(u)), dtype=np.quaternion)
            for i in range(len(u)):
                for j in range(len(u)):
                    uuH[i, j] = u[i] * u[j].conj()

            h = (1.0 / zeta) * (h - uuH)
        else:
            # For row vectors, compute u^T * conj(u)
            uTu = np.zeros((len(u), len(u)), dtype=np.quaternion)
            for i in range(len(u)):
                for j in range(len(u)):
                    uTu[i, j] = u[i].conj() * u[j]

            h = (h - uTu) * (1.0 / zeta)

    return h


def internal_tridiagonalizer(A):
    """
    Internal recursive tridiagonalization algorithm.

    Parameters:
    -----------
    A : quaternion matrix
        Hermitian matrix to tridiagonalize

    Returns:
    --------
    tuple : (P, B)
        P : quaternion matrix (unitary transformation)
        B : quaternion matrix (tridiagonal result)
    """
    r, c = A.shape

    if r != c:
        raise ValueError("Internal tridiagonalize error - non-square matrix")

    # Initialize P as identity matrix
    P = np.eye(r, dtype=np.quaternion)

    # Compute and apply Householder transformation to first row and column
    # We omit the first element of the first column since it's already real (A is Hermitian)
    # We apply P on both sides of A so that the first row and column are nulled out
    # (apart from the first two elements in each case)

    # Extract the first column (excluding first element)
    first_col = A[1:, 0]

    # Create unit vector e1 for Householder transformation
    e1 = np.zeros(r - 1)
    e1[0] = 1.0

    # Compute Householder matrix for the submatrix
    H_sub = householder_matrix(first_col, e1)

    # Embed H_sub into full-size matrix
    P[1:, 1:] = H_sub

    # Apply transformation: B = P * A * P^H
    P_H = quat_hermitian(P)
    B = quat_matmat(quat_matmat(P, A), P_H)

    # Apply algorithm recursively to sub-matrices of B, except when 2x2
    if r > 2:
        Q = np.eye(r, dtype=np.quaternion)
        Q_sub, B_sub = internal_tridiagonalizer(B[1:, 1:])
        Q[1:, 1:] = Q_sub
        B[1:, 1:] = B_sub
        P = quat_matmat(Q, P)

    return P, B


def check_tridiagonal(B):
    """
    Verify tridiagonalization results and convert to exactly tridiagonal form.

    Parameters:
    -----------
    B : quaternion matrix
        Result from tridiagonalization

    Returns:
    --------
    B_clean : quaternion matrix
        Cleaned tridiagonal matrix
    """
    r, c = B.shape

    # Extract the three diagonals
    diag_main = np.array([B[i, i] for i in range(min(r, c))])
    diag_upper = np.array([B[i, i + 1] for i in range(min(r, c - 1))])
    diag_lower = np.array([B[i + 1, i] for i in range(min(r - 1, c))])

    # Extract off-diagonal part
    off_diag = np.zeros_like(B)
    for i in range(r):
        for j in range(c):
            if abs(i - j) > 1:  # Off-tridiagonal elements
                off_diag[i, j] = B[i, j]

    # Find largest on- and off-tridiagonal elements
    D = np.concatenate([diag_main, diag_upper, diag_lower])
    T1 = np.max(np.abs(off_diag))  # Largest off-tridiagonal element
    T2 = np.max(np.abs(D))  # Largest tridiagonal element

    tolerance = 1e-12 * 1e3  # Empirically determined tolerance

    if T1 > T2 * tolerance:
        print("Warning: Result of tridiagonalization was not accurately tridiagonal.")
        print(f"Largest on- and off-tridiagonal moduli were: {T2}, {T1}")

    # Verify diagonal elements have negligible vector parts
    diag_scalar = np.array([float(diag_main[i].w) for i in range(len(diag_main))])
    diag_vector = np.array(
        [
            np.sqrt(diag_main[i].x ** 2 + diag_main[i].y ** 2 + diag_main[i].z ** 2)
            for i in range(len(diag_main))
        ]
    )

    T1_scalar = np.max(np.abs(diag_scalar))
    T2_vector = np.max(diag_vector)

    if T2_vector > T1_scalar * tolerance:
        print("Warning: Result of tridiagonalization was not accurately scalar.")
        print(
            f"Largest on-tridiagonal vector and scalar moduli were: {T2_vector}, {T1_scalar}"
        )

    # Clean up the result: subtract off-diagonal part and take scalar part
    B_clean = B - off_diag

    # Convert to scalar (real) form
    for i in range(r):
        for j in range(c):
            element = B_clean[i, j]
            B_clean[i, j] = quaternion.quaternion(float(element.w), 0.0, 0.0, 0.0)

    return B_clean


def tridiagonalize(A):
    """
    Tridiagonalize Hermitian matrix A, such that P * A * P^H = B and
    P^H * B * P = A. B is real, P is unitary, and B has the same eigenvalues as A.

    Parameters:
    -----------
    A : quaternion matrix
        Hermitian matrix to tridiagonalize

    Returns:
    --------
    tuple : (P, B)
        P : quaternion matrix (unitary transformation matrix)
        B : quaternion matrix (tridiagonal result)
    """
    r, c = A.shape

    if r != c:
        raise ValueError("Cannot tridiagonalize a non-square matrix")

    if r < 2:
        raise ValueError("Cannot tridiagonalize a matrix smaller than 2 by 2")

    # Check if matrix is Hermitian
    A_H = quat_hermitian(A)
    if not np.allclose(A, A_H, atol=1e-10):
        raise ValueError("Matrix to be tridiagonalized is not (accurately) Hermitian")

    # Perform tridiagonalization
    P, B = internal_tridiagonalizer(A)

    # Verify and clean up the result
    B = check_tridiagonal(B)

    return P, B
