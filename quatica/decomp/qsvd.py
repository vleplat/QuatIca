"""
Quaternion SVD Implementations for QuatIca

This module provides Q-SVD routines leveraging existing primitives in utils.py:

1. Classical Q-SVD via real-block embedding and LAPACK
2. Full Classical Q-SVD (no truncation)
3. QR decomposition for quaternion matrices
4. Randomized Q-SVD (rand_qsvd) using Gaussian sketching + power iterations
5. Pass-efficient Q-SVD (pass_eff_qsvd) alternating a single QR per pass

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

import os
import sys

import numpy as np
import quaternion
from scipy.linalg import qr

# Add core module to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import quat_hermitian, quat_matmat, real_contract, real_expand


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
    A = real_expand(X_quat)  # 4m×4n

    # 2) Full QR (not economic)
    Qr, Rr = qr(A)  # Qr: 4m×4m, Rr: 4m×4n

    # 3) Extract the relevant parts
    # Handle the case where the matrix might be wide (m < n)
    if 4 * m >= 4 * n:
        # Tall matrix: extract thin QR
        Qr_thin = Qr[:, : 4 * n]  # 4m×4n (first 4n columns)
        Rr_thin = Rr[: 4 * n, :]  # 4n×4n (first 4n rows)
    else:
        # Wide matrix: use full QR
        Qr_thin = Qr  # 4m×4m (full Q)
        Rr_thin = Rr  # 4m×4n (full R)

    # 4) Convert back to quaternion using the original real_contract
    if 4 * m >= 4 * n:
        # Tall matrix: standard conversion
        Q_quat = real_contract(Qr_thin, m, n)
        R_quat = real_contract(Rr_thin, n, n)
    else:
        # Wide matrix: adjust dimensions
        Q_quat = real_contract(Qr_thin, m, m)  # Q is m×m
        R_quat = real_contract(Rr_thin, m, n)  # R is m×n

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
    A_real = real_expand(X_quat)  # 4m×4n

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
        s_quat.append(s[4 * i])

    # Convert to numpy array
    s_quat = np.array(s_quat)

    # Truncate to rank R
    Uq = Uq[:, :R]  # m×R
    Vq = Vq[:, :R]  # n×R
    s_quat = s_quat[:R]  # R largest singular values

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
    A_real = real_expand(X_quat)  # 4m×4n

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
        s_quat.append(s[4 * i])

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
    """
    m, n = X_quat.shape
    P = oversample

    # 1) O=randn(n,R+P) - Create random matrix
    O = np.random.randn(n, R + P)
    # Convert to quaternion matrix (real part only)
    O_components = np.zeros((n, R + P, 4))
    O_components[..., 0] = O  # Real part
    O_quat = quaternion.as_quat_array(O_components)

    # 2) [Q_1,~]=qr_qua(X*O) - QR of X*O
    Q1, _ = qr_qua(quat_matmat(X_quat, O_quat))

    # 3) Power iterations: for i=1:q
    for i in range(n_iter):
        # [Q_2,~]=qr_qua(X'*Q_1)
        Q2_temp = quat_matmat(quat_hermitian(X_quat), Q1)
        # Handle the case where Q2_temp might not be tall enough for thin QR
        if Q2_temp.shape[0] >= Q2_temp.shape[1]:
            Q2, _ = qr_qua(Q2_temp)
        else:
            # Use full QR and take the first columns
            Q2_full, _ = qr_qua(Q2_temp)
            Q2 = Q2_full[:, : Q2_temp.shape[1]]

        # [Q_1,~]=qr_qua(X*Q_2)
        Q1_temp = quat_matmat(X_quat, Q2)
        if Q1_temp.shape[0] >= Q1_temp.shape[1]:
            Q1, _ = qr_qua(Q1_temp)
        else:
            Q1_full, _ = qr_qua(Q1_temp)
            Q1 = Q1_full[:, : Q1_temp.shape[1]]

    # 4) [Q_2,RR]=qr_qua(X'*Q_1) - Final QR
    Q2_temp = quat_matmat(quat_hermitian(X_quat), Q1)
    if Q2_temp.shape[0] >= Q2_temp.shape[1]:
        Q2, RR = qr_qua(Q2_temp)
    else:
        # Use full QR and take the first columns
        Q2_full, RR_full = qr_qua(Q2_temp)
        Q2 = Q2_full[:, : Q2_temp.shape[1]]
        RR = RR_full[: Q2_temp.shape[1], :]

    # 5) [V_,S,U_]=svd(RR) - SVD of RR
    RR_real = real_expand(RR)
    V_, S, U_ = np.linalg.svd(RR_real, full_matrices=False)

    # Take every 4th singular value for quaternion SVD
    s = S[::4][:R]

    # 6) V=Q_2*V_; U=Q_1*U_ - Lift back
    V_small = real_contract(V_[:, : 4 * R], Q2.shape[1], R)
    U_small = real_contract(U_[: 4 * R, :].T, Q1.shape[1], R)

    V_quat = quat_matmat(Q2, V_small)
    U_quat = quat_matmat(Q1, U_small)

    return U_quat, s, V_quat


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
    """
    m, n = X_quat.shape
    P, v = oversample, n_passes

    # 1) Initial random matrix (dense real) - same as MATLAB
    Q1 = np.random.randn(n, R + P)

    # 2) Alternating passes - matching MATLAB logic
    for i in range(1, v + 1):  # MATLAB: for i=1:v
        if i % 2 == 1:  # MATLAB: if rem(i,2)~=0
            # [Q_2,R_2]=qr_qua(X*Q_1)
            # Convert real matrix to quaternion matrix properly
            if isinstance(Q1, np.ndarray) and Q1.dtype != np.quaternion:
                Q1_components = np.zeros((n, R + P, 4))
                Q1_components[..., 0] = Q1  # Real part
                Q1_quat = quaternion.as_quat_array(Q1_components)
            else:
                Q1_quat = Q1  # Already quaternion
            Q2, R2 = qr_qua(quat_matmat(X_quat, Q1_quat))
        else:
            # [Q_1,R_1]=qr_qua(X'*Q_2)
            Q1, R1 = qr_qua(quat_matmat(quat_hermitian(X_quat), Q2))

    # 3) Final SVD - matching MATLAB logic
    if v % 2 == 0:  # MATLAB: if rem(v,2)==0
        # [V_,S,U_]=svd(R_1)
        R_real = real_expand(R1)
        V_, S, U_ = np.linalg.svd(R_real, full_matrices=False)
    else:
        # [U_,S,V_]=svd(R_2)
        R_real = real_expand(R2)
        U_, S, V_ = np.linalg.svd(R_real, full_matrices=False)

    # 4) Lift back - matching MATLAB: V=Q_1*V_; U=Q_2*U_
    # Take every 4th singular value for quaternion SVD (like rand_qsvd)
    s = S[::4][:R]

    if v % 2 == 0:
        # V=Q_1*V_; U=Q_2*U_
        V_small = real_contract(V_[:, : 4 * R], Q1.shape[1], R)
        U_small = real_contract(U_[: 4 * R, :].T, Q2.shape[1], R)
        V_quat = quat_matmat(Q1, V_small)
        U_quat = quat_matmat(Q2, U_small)
    else:
        # U=Q_2*U_; V=Q_1*V_
        U_small = real_contract(U_[:, : 4 * R], Q2.shape[1], R)
        V_small = real_contract(V_[: 4 * R, :].T, Q1.shape[1], R)
        U_quat = quat_matmat(Q2, U_small)
        V_quat = quat_matmat(Q1, V_small)

    return U_quat, s, V_quat
