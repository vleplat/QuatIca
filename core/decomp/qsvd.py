"""
Quaternion SVD Implementations for QuatIca

This module provides Q-SVD routines leveraging existing primitives in utils.py:

1. Classical Q-SVD via real-block embedding and LAPACK ✅ WORKING
2. Full Classical Q-SVD (no truncation) ✅ WORKING
3. QR decomposition for quaternion matrices ✅ WORKING
4. Randomized Q-SVD (rand_qsvd) using Gaussian sketching + power iterations ⚠️ PLACEHOLDER
5. Pass-efficient Q-SVD (pass_eff_qsvd) alternating a single QR per pass ⚠️ PLACEHOLDER

All routines operate on quaternion arrays (numpy.quaternion) and reuse utilities:
- real_expand(Q), real_contract(R, m, n)
- quat_matmat(A, B), quat_hermitian(A)
- quat_frobenius_norm(A)

References:
- Ahmadi-Asl, S., Nobakht Kooshkghazi, M., & Leplat, V. (2025). 
  Pass-efficient Randomized Algorithms for Low-rank Approximation of Quaternion Matrices.
  arXiv:2507.13731

Future Work:
- Ma, R.-R., & Bai, Z.-J. (2018). A Structure-Preserving One-Sided Jacobi Method 
  for Computing the SVD of a Quaternion Matrix. arXiv:1811.08671
  This paper presents a more advanced, structure-preserving Q-SVD algorithm that will
  be implemented in future releases for improved accuracy and efficiency.
"""

import numpy as np
import quaternion
from scipy.linalg import qr
import sys
import os

# Add core module to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import real_expand, real_contract, quat_matmat, quat_hermitian, quat_frobenius_norm


def qr_qua(X_quat):
    """
    QR decomposition of quaternion matrix using real-block embedding.
    
    Parameters:
    -----------
    X_quat : numpy.ndarray with dtype=quaternion
        Input quaternion matrix of shape (m, n)
    
    Returns:
    --------
    tuple : (Q_quat, R_quat)
        Q_quat : m×n quaternion matrix with orthonormal columns
        R_quat : n×n upper triangular quaternion matrix
    
    Notes:
    ------
    Uses real-block embedding + SciPy's QR + contraction back to quaternion.
    Produces perfect reconstruction and orthonormal Q matrix.
    """
    m, n = X_quat.shape
    
    # 1) Real-block embedding
    A = real_expand(X_quat)            # 4m×4n
    
    # 2) Full QR (not economic)
    Qr, Rr = qr(A)                     # Qr: 4m×4m, Rr: 4m×4n
    
    # 3) Extract the relevant parts
    Qr_thin = Qr[:, :4*n]              # 4m×4n (first 4n columns)
    Rr_thin = Rr[:4*n, :]              # 4n×4n (first 4n rows)
    
    # 4) Convert back to quaternion using the original real_contract
    Q_quat = real_contract(Qr_thin, m, n)
    R_quat = real_contract(Rr_thin, n, n)
    
    return Q_quat, R_quat


def classical_qsvd(X_quat, R):
    """
    Classical Q-SVD via real-block embedding and LAPACK.
    
    Parameters:
    -----------
    X_quat : numpy.ndarray with dtype=quaternion
        Input quaternion matrix of shape (m, n)
    R : int
        Target rank for truncation
    
    Returns:
    --------
    tuple : (U_quat, s, V_quat)
        U_quat : m×R quaternion matrix with orthonormal columns
        s : R-length array of singular values
        V_quat : n×R quaternion matrix with orthonormal columns
    
    Notes:
    ------
    Complexity: O((4m)(4n)min(4m,4n)), leverages optimized LAPACK.
    Handles degenerate singular values from real-block embedding by extracting
    every 4th singular value to obtain true quaternion singular values.
    
    For reconstruction: X ≈ U @ diag(s) @ V^H (when R < min(m,n))
    """
    m, n = X_quat.shape
    
    # Embed to real
    A_real = real_expand(X_quat)              # 4m×4n
    
    # Full SVD on real (get ALL singular values)
    U_real, s, Vt_real = np.linalg.svd(A_real, full_matrices=True)
    
    # The full SVD gives us ALL singular values
    # U_real: 4m × 4m
    # Vt_real: 4n × 4n  
    # s: min(4m, 4n) singular values
    
    # Convert back to quaternion using the original real_contract
    Uq = real_contract(U_real, m, m)  # m×m quaternion
    Vq = real_contract(Vt_real.T, n, n)  # n×n quaternion
    
    # Take every 4th singular value for quaternion SVD
    # For m×n input: take s[0], s[4], s[8], ..., s[4*(min(m,n)-1)]
    min_dim = min(m, n)
    s_quat = []
    for i in range(min_dim):
        s_quat.append(s[4*i])
    
    # Convert to numpy array
    s_quat = np.array(s_quat)
    
    # Truncate to rank R
    Uq = Uq[:, :R]                            # m×R
    Vq = Vq[:, :R]                            # n×R
    s_quat = s_quat[:R]                       # R largest singular values
    
    return Uq, s_quat, Vq


def classical_qsvd_full(X_quat):
    """
    Full Classical Q-SVD via real-block embedding and LAPACK.
    
    Parameters:
    -----------
    X_quat : numpy.ndarray with dtype=quaternion
        Input quaternion matrix of shape (m, n)
    
    Returns:
    --------
    tuple : (U_quat, s, V_quat)
        U_quat : m×m quaternion matrix with orthonormal columns
        s : min(m,n)-length array of singular values
        V_quat : n×n quaternion matrix with orthonormal columns
    
    Notes:
    ------
    Returns the FULL Q-SVD decomposition without truncation.
    For reconstruction: X = U @ Σ @ V^H where Σ is the diagonal matrix with s.
    """
    m, n = X_quat.shape
    
    # Embed to real
    A_real = real_expand(X_quat)              # 4m×4n
    
    # Full SVD on real (get ALL singular values)
    U_real, s, Vt_real = np.linalg.svd(A_real, full_matrices=True)
    
    # The full SVD gives us ALL singular values
    # U_real: 4m × 4m
    # Vt_real: 4n × 4n  
    # s: min(4m, 4n) singular values
    
    # Convert back to quaternion using the original real_contract
    Uq = real_contract(U_real, m, m)  # m×m quaternion
    Vq = real_contract(Vt_real.T, n, n)  # n×n quaternion
    
    # Take every 4th singular value for quaternion SVD
    # For m×n input: take s[0], s[4], s[8], ..., s[4*(min(m,n)-1)]
    min_dim = min(m, n)
    s_quat = []
    for i in range(min_dim):
        s_quat.append(s[4*i])
    
    # Convert to numpy array
    s_quat = np.array(s_quat)
    
    return Uq, s_quat, Vq


def rand_qsvd(X_quat, R, oversample=10, n_iter=2):
    """
    Randomized Q-SVD using Gaussian sketching + power iterations.
    
    Parameters:
    -----------
    X_quat : numpy.ndarray with dtype=quaternion
        Input quaternion matrix of shape (m, n)
    R : int
        Target rank
    oversample : int, optional
        Oversampling parameter (default: 10)
    n_iter : int, optional
        Number of power iterations (default: 2)
    
    Returns:
    --------
    tuple : (U_quat, s, V_quat)
        U_quat : m×R quaternion matrix with orthonormal columns
        s : R-length array of singular values
        V_quat : n×R quaternion matrix with orthonormal columns
    
    Notes:
    ------
    Cost: O(mn(R+P)) + O((R+P)²n) where P = oversample.
    ⚠️ PLACEHOLDER IMPLEMENTATION - needs proper testing and validation.
    """
    m, n = X_quat.shape
    P = oversample
    
    # 1) Gaussian sketch
    Omega = np.random.randn(n, R + P)
    # Convert Omega to quaternion matrix
    Omega_quat = quaternion.as_quat_array(Omega.reshape(n, R + P, 1))
    Y = quat_matmat(X_quat, Omega_quat)
    
    # 2) Orthonormalize
    Q, _ = qr_qua(Y)
    
    # 3) Power iterations
    for _ in range(n_iter):
        Q2, _ = qr_qua(quat_matmat(quat_hermitian(X_quat), Q))
        Q, _ = qr_qua(quat_matmat(X_quat, Q2))
    
    # 4) Small projection
    B = quat_matmat(quat_hermitian(Q), X_quat)  # (R+P)×n
    
    # 5) Real SVD
    B_real = real_expand(B)                    # 4(R+P)×4n
    U_r, s, Vt_r = np.linalg.svd(B_real, full_matrices=False)
    
    # 6) Lift and truncate
    U_small_reshaped = U_r[:, :R].reshape(Q.shape[1], 4, R).transpose(0, 2, 1)
    U_small = quaternion.as_quat_array(U_small_reshaped)
    V_small_reshaped = Vt_r[:R, :].T.reshape(n, 4, R).transpose(0, 2, 1)
    V_small = quaternion.as_quat_array(V_small_reshaped)
    
    U_quat = quat_matmat(Q, U_small)
    V_quat = V_small
    
    return U_quat, s[:R], V_quat


def pass_eff_qsvd(X_quat, R, oversample=10, n_passes=2):
    """
    Pass-efficient Q-SVD alternating single multiplies + QR.
    
    Parameters:
    -----------
    X_quat : numpy.ndarray with dtype=quaternion
        Input quaternion matrix of shape (m, n)
    R : int
        Target rank
    oversample : int, optional
        Oversampling parameter (default: 10)
    n_passes : int, optional
        Number of passes over the matrix (default: 2)
    
    Returns:
    --------
    tuple : (U_quat, s, V_quat)
        U_quat : m×R quaternion matrix with orthonormal columns
        s : R-length array of singular values
        V_quat : n×R quaternion matrix with orthonormal columns
    
    Notes:
    ------
    Advantage: fewer large passes over X, good cache behavior.
    ⚠️ PLACEHOLDER IMPLEMENTATION - needs proper testing and validation.
    """
    m, n = X_quat.shape
    P, v = oversample, n_passes
    
    # 1) Initial random matrix (dense real)
    Q1 = np.random.randn(n, R + P)
    
    for i in range(1, v + 1):
        if i % 2 == 1:
            # Convert Q1 to quaternion matrix
            Q1_quat = quaternion.as_quat_array(Q1.reshape(n, R + P, 1))
            Q2, R2 = qr_qua(quat_matmat(X_quat, Q1_quat))
        else:
            Q1, R1 = qr_qua(quat_matmat(quat_hermitian(X_quat), Q2))
    
    # 2) Small Rmat
    Rmat = R1 if v % 2 == 0 else R2
    
    # 3) SVD on Rmat
    R_real = real_expand(Rmat)
    U_r, s, Vt_r = np.linalg.svd(R_real, full_matrices=False)
    
    # 4) Lift & truncate
    U_small_reshaped = U_r[:, :R].reshape(Q2.shape[1], 4, R).transpose(0, 2, 1)
    U_small = quaternion.as_quat_array(U_small_reshaped)
    V_small_reshaped = Vt_r[:R, :].T.reshape(Q1.shape[1], 4, R).transpose(0, 2, 1)
    V_small = quaternion.as_quat_array(V_small_reshaped)
    
    U_quat = quat_matmat(Q2 if v % 2 else Q1, U_small)
    V_quat = quat_matmat(Q1 if v % 2 else Q2, V_small)
    
    return U_quat, s[:R], V_quat 