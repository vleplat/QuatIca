"""
Quaternion Matrix Hessenberg Reduction Module

This module implements reduction of a general quaternion matrix to upper
Hessenberg form using Householder similarity transformations.

Key properties:
- For a square matrix A, there exists a unitary P such that H = P * A * P^H
  is upper Hessenberg (all entries strictly below the first subdiagonal are 0).

Notes:
- We reuse the quaternion Householder machinery already implemented for
  Hermitian tridiagonalization. Here, we adapt it to the general (non-Hermitian)
  Hessenberg case by zeroing elements below the first subdiagonal column-by-column.

Constraints:
- Per project rules, this module is added without modifying existing core
  functions. It only depends on utilities and existing Householder helpers.
"""

import os
import sys
from typing import Tuple

import numpy as np
import quaternion  # type: ignore

# Add parent directory to path for imports within core package structure
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import quat_hermitian, quat_matmat  # noqa: E402

# Import Householder helpers from tridiagonalization module
from .tridiagonalize import householder_matrix  # noqa: E402


def is_hessenberg(H: np.ndarray, atol: float = 1e-12) -> bool:
    """Return True if quaternion matrix H is upper Hessenberg within tolerance.

    Upper Hessenberg means H[i, j] == 0 for all i > j + 1.
    """
    rows, cols = H.shape
    for i in range(rows):
        for j in range(cols):
            if i > j + 1:
                hij = H[i, j]
                if (
                    (abs(hij.w) > atol)
                    or (abs(hij.x) > atol)
                    or (abs(hij.y) > atol)
                    or (abs(hij.z) > atol)
                ):
                    return False
    return True


def check_hessenberg(H: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    """Clean tiny entries below first subdiagonal to exact zeros (for readability).

    Parameters:
    - H: quaternion matrix suspected to be upper Hessenberg
    - atol: tolerance below which entries are zeroed
    """
    rows, cols = H.shape
    H_clean = H.copy()
    for i in range(rows):
        for j in range(cols):
            if i > j + 1:
                hij = H_clean[i, j]
                if (
                    (abs(hij.w) <= atol)
                    and (abs(hij.x) <= atol)
                    and (abs(hij.y) <= atol)
                    and (abs(hij.z) <= atol)
                ):
                    H_clean[i, j] = quaternion.quaternion(0.0, 0.0, 0.0, 0.0)
    return H_clean


def _embed_householder_submatrix(H_sub: np.ndarray, offset: int, n: int) -> np.ndarray:
    """Embed a Householder matrix H_sub (size k x k) into an n x n identity.

    The submatrix is placed at rows/cols [offset:n, offset:n].
    """
    H_full = np.eye(n, dtype=np.quaternion)
    H_full[offset:n, offset:n] = H_sub
    return H_full


def hessenbergize(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce a general square quaternion matrix A to upper Hessenberg form.

    Returns (P, H) such that H = P * A * P^H and H is upper Hessenberg.

    Parameters:
    - A: (n x n) quaternion numpy array

    Returns:
    - P: unitary quaternion matrix accumulating the similarity transforms
    - H: upper Hessenberg quaternion matrix
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Hessenberg reduction requires a square matrix")

    n = A.shape[0]

    # Trivial sizes: no reduction needed
    if n <= 2:
        P_id = np.eye(n, dtype=np.quaternion)
        return P_id, A.copy()

    # Copy to avoid modifying input
    H = A.copy()
    P = np.eye(n, dtype=np.quaternion)

    # For each column k, zero elements below the first subdiagonal in column k
    for k in range(0, n - 2):
        # Target vector is the part of column k below the first subdiagonal: indices k+2..n-1
        col_segment = H[k + 1 :, k]  # length n - (k+1)

        if col_segment.shape[0] <= 1:
            continue  # nothing to zero

        # Build target unit vector e1 of length m = n - (k+1)
        m = col_segment.shape[0]
        e1 = np.zeros(m)
        e1[0] = 1.0

        # Compute Householder matrix to map col_segment to [*, 0, 0, ..., 0]^T
        H_sub = householder_matrix(col_segment, e1)

        # Embed into full-size matrix operating on rows/cols k+1..n-1
        Hk = _embed_householder_submatrix(H_sub, k + 1, n)

        # Apply similarity transform: H ← Hk * H * Hk^H
        Hk_H = quat_hermitian(Hk)
        H = quat_matmat(quat_matmat(Hk, H), Hk_H)

        # Accumulate transformation: P ← Hk * P
        P = quat_matmat(Hk, P)

    # Clean tiny entries and return
    H = check_hessenberg(H)
    return P, H


__all__ = [
    "hessenbergize",
    "is_hessenberg",
    "check_hessenberg",
]
