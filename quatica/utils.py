from typing import List, Tuple

import numpy as np
import quaternion
from scipy import sparse

# Optimized quaternion matrix operations


def quat_matmat(A, B):
    """
    Multiply two quaternion matrices (supports dense × dense, sparse × dense, dense × sparse, sparse × sparse).

    Performs quaternion matrix multiplication using optimized algorithms based on the
    input types. Handles mixed sparse/dense operations efficiently.

    Parameters:
    -----------
    A : np.ndarray or SparseQuaternionMatrix
        First quaternion matrix
    B : np.ndarray or SparseQuaternionMatrix
        Second quaternion matrix

    Returns:
    --------
    np.ndarray or SparseQuaternionMatrix
        Result of quaternion matrix multiplication A @ B

    Notes:
    ------
    The function automatically selects the appropriate multiplication algorithm:
    - Dense × Dense: Component-wise quaternion multiplication
    - Sparse × Dense/Sparse: Uses sparse matrix multiplication routines
    - Dense × Sparse: Uses left multiplication method
    """
    if isinstance(A, SparseQuaternionMatrix):
        return A @ B
    elif isinstance(B, SparseQuaternionMatrix):
        # For dense @ sparse, use left_multiply
        return B.left_multiply(A)
    else:
        # Both dense
        A_comp = quaternion.as_float_array(A)
        B_comp = quaternion.as_float_array(B)
        Aw, Ax, Ay, Az = A_comp[..., 0], A_comp[..., 1], A_comp[..., 2], A_comp[..., 3]
        Bw, Bx, By, Bz = B_comp[..., 0], B_comp[..., 1], B_comp[..., 2], B_comp[..., 3]
        Cw = Aw @ Bw - Ax @ Bx - Ay @ By - Az @ Bz
        Cx = Aw @ Bx + Ax @ Bw + Ay @ Bz - Az @ By
        Cy = Aw @ By - Ax @ Bz + Ay @ Bw + Az @ Bx
        Cz = Aw @ Bz + Ax @ By - Ay @ Bx + Az @ Bw
        C = np.stack([Cw, Cx, Cy, Cz], axis=-1)
        return quaternion.as_quat_array(C)


def quat_frobenius_norm(A: np.ndarray) -> float:
    """
    Compute the Frobenius norm of a quaternion matrix (dense or sparse).

    Calculates ||A||_F = sqrt(sum(|A_ij|^2)) where |A_ij| is the modulus
    of the quaternion at position (i,j).

    Parameters:
    -----------
    A : np.ndarray or SparseQuaternionMatrix
        Input quaternion matrix

    Returns:
    --------
    float
        Frobenius norm of the matrix

    Notes:
    ------
    For sparse matrices, the norm is computed efficiently by summing
    the squared norms of each component separately.
    """
    if isinstance(A, SparseQuaternionMatrix):
        real_norm = A.real.power(2).sum()
        i_norm = A.i.power(2).sum()
        j_norm = A.j.power(2).sum()
        k_norm = A.k.power(2).sum()
        return np.sqrt(real_norm + i_norm + j_norm + k_norm)
    else:
        comp = quaternion.as_float_array(A)
        return np.sqrt(np.sum(comp**2))


def quat_hermitian(A: np.ndarray) -> np.ndarray:
    """
    Return the conjugate transpose (Hermitian) of quaternion matrix A (dense or sparse).

    Computes A^H = (A*)^T where A* is the complex conjugate and T is transpose.
    For quaternions q = w + xi + yj + zk, the conjugate is q* = w - xi - yj - zk.

    Parameters:
    -----------
    A : np.ndarray or SparseQuaternionMatrix
        Input quaternion matrix

    Returns:
    --------
    np.ndarray or SparseQuaternionMatrix
        Conjugate transpose A^H of the input matrix

    Notes:
    ------
    The Hermitian (conjugate transpose) is fundamental in quaternion linear algebra
    and appears in definitions of unitary matrices, eigenvalue problems, and norms.
    """
    if isinstance(A, SparseQuaternionMatrix):
        return A.conjugate().transpose()
    else:
        return np.transpose(np.conjugate(A))


def quat_eye(n: int) -> np.ndarray:
    """
    Create an n×n identity quaternion matrix.

    Generates the quaternion identity matrix I where I_ij = δ_ij * (1 + 0i + 0j + 0k),
    i.e., ones on the diagonal and zeros elsewhere.

    Parameters:
    -----------
    n : int
        Size of the square identity matrix

    Returns:
    --------
    np.ndarray
        An n×n quaternion identity matrix

    Notes:
    ------
    The quaternion identity matrix satisfies A @ I = I @ A = A for any
    n×n quaternion matrix A.
    """
    I = np.zeros((n, n), dtype=np.quaternion)
    np.fill_diagonal(I, quaternion.quaternion(1, 0, 0, 0))
    return I


def quat_abs_scalar(q: quaternion.quaternion) -> float:
    """
    Return the modulus |q| of a quaternion scalar q.

    Computes the absolute value (modulus) of a quaternion q = w + xi + yj + zk
    as |q| = sqrt(w² + x² + y² + z²).

    Parameters:
    -----------
    q : quaternion.quaternion
        Input quaternion scalar

    Returns:
    --------
    float
        Modulus (absolute value) of the quaternion

    Notes:
    ------
    The quaternion modulus satisfies |q₁ * q₂| = |q₁| * |q₂| and is used
    in defining norms and distances in quaternion space.
    """
    return float(np.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z))


def induced_matrix_norm_1(A: np.ndarray) -> float:
    """Matrix 1-norm induced by vector 1-norm: max column sum of |A_ij|.

    Notes:
      - Supports dense quaternion ndarrays. For sparse, convert to dense first or
        use component-space helpers.
    """
    if not isinstance(A, np.ndarray) or A.dtype != np.quaternion:
        raise ValueError("A must be a dense quaternion ndarray")
    m, n = A.shape
    max_col_sum = 0.0
    for j in range(n):
        col_sum = 0.0
        for i in range(m):
            col_sum += quat_abs_scalar(A[i, j])
        if col_sum > max_col_sum:
            max_col_sum = col_sum
    return max_col_sum


def induced_matrix_norm_inf(A: np.ndarray) -> float:
    """Matrix infinity-norm induced by vector infinity-norm: max row sum of |A_ij|.

    Notes:
      - Supports dense quaternion ndarrays. For sparse, convert to dense first or
        use component-space helpers.
    """
    if not isinstance(A, np.ndarray) or A.dtype != np.quaternion:
        raise ValueError("A must be a dense quaternion ndarray")
    m, n = A.shape
    max_row_sum = 0.0
    for i in range(m):
        row_sum = 0.0
        for j in range(n):
            row_sum += quat_abs_scalar(A[i, j])
        if row_sum > max_row_sum:
            max_row_sum = row_sum
    return max_row_sum


def spectral_norm_2(A: np.ndarray) -> float:
    """Matrix 2-norm (spectral norm): largest singular value of A.

    Delegates to quaternion SVD implementation.
    """
    if not isinstance(A, np.ndarray) or A.dtype != np.quaternion:
        raise ValueError("A must be a dense quaternion ndarray")
    # Import here to avoid circular imports at module load
    from decomp.qsvd import classical_qsvd_full

    _U, s, _V = classical_qsvd_full(A)
    return float(np.max(s)) if len(s) else 0.0


def matrix_norm(A: np.ndarray, ord: str | int | float | None = None) -> float:
    """Compute common matrix norms for quaternion matrices.

    ord supported:
      - None or 'fro' or 'F': Frobenius norm
      - 1: Induced 1-norm (max column sum)
      - np.inf or 'inf': Induced infinity-norm (max row sum)
      - 2: Spectral norm (largest singular value)
    """
    if ord in (None, "fro", "F"):
        return quat_frobenius_norm(A)
    if ord == 1:
        return induced_matrix_norm_1(A)
    if ord == 2:
        return spectral_norm_2(A)
    if ord == np.inf or ord == "inf":
        return induced_matrix_norm_inf(A)
    raise ValueError(f"Unsupported ord for matrix_norm: {ord}")


def _is_hermitian_quat(A: np.ndarray, atol: float = 1e-12) -> bool:
    """
    Lightweight Hermitian check for quaternion matrices.

    Tests whether a quaternion matrix A satisfies A = A^H within tolerance,
    where A^H is the conjugate transpose.

    Parameters:
    -----------
    A : np.ndarray
        Quaternion matrix to test
    atol : float, optional
        Absolute tolerance for comparison (default: 1e-12)

    Returns:
    --------
    bool
        True if matrix is Hermitian within tolerance, False otherwise

    Notes:
    ------
    This is an internal function used for optimization in other algorithms.
    For user-facing Hermitian checks, use ishermitian() which includes
    additional validation and error handling.
    """
    try:
        return np.allclose(A, quat_hermitian(A), atol=atol)
    except Exception:
        return False


def real_expand(Q):
    """
    Convert quaternion matrix Q to real block matrix representation.
    Given an m×n quaternion array Q, return a (4m)×(4n) real block matrix
    [[Qw, -Qx, -Qy, -Qz],
     [Qx,  Qw, -Qz,  Qy],
     [Qy,  Qz,  Qw, -Qx],
     [Qz, -Qy,  Qx,  Qw]]
    """
    if isinstance(Q, np.ndarray) and Q.dtype == np.quaternion:
        m, n = Q.shape
        Q_array = quaternion.as_float_array(Q)  # Shape: (m, n, 4)

        # Create the real block matrix
        R = np.zeros((4 * m, 4 * n))

        for i in range(m):
            for j in range(n):
                w, x, y, z = Q_array[i, j]
                # Block position
                bi, bj = 4 * i, 4 * j
                # Fill the 4x4 block
                R[bi : bi + 4, bj : bj + 4] = np.array(
                    [[w, -x, -y, -z], [x, w, -z, y], [y, z, w, -x], [z, -y, x, w]]
                )
        return R
    else:
        raise ValueError("Input must be a quaternion array")


def real_contract(R, m, n):
    """
    Convert real block matrix R back to quaternion matrix.
    Invert real_expand back into an m×n quaternion array
    """
    if R.shape != (4 * m, 4 * n):
        raise ValueError(f"Expected shape (4*{m}, 4*{n}), got {R.shape}")

    Q_array = np.zeros((m, n, 4))

    for i in range(m):
        for j in range(n):
            bi, bj = 4 * i, 4 * j
            block = R[bi : bi + 4, bj : bj + 4]
            # Extract quaternion components from the block
            w = block[0, 0]
            x = block[1, 0]
            y = block[2, 0]
            z = block[3, 0]
            Q_array[i, j] = [w, x, y, z]

    return quaternion.as_quat_array(Q_array)


def compute_real_svd_pinv(X_real):
    """
    Compute pseudoinverse using SVD in the real domain
    """
    U, s, Vt = np.linalg.svd(X_real, full_matrices=False)
    # Handle small singular values
    threshold = 1e-12
    s_inv = np.where(s > threshold, 1.0 / s, 0.0)
    return Vt.T @ np.diag(s_inv) @ U.T


class SparseQuaternionMatrix:
    def __init__(self, real, i, j, k, shape):
        self.real = real.tocsr()
        self.i = i.tocsr()
        self.j = j.tocsr()
        self.k = k.tocsr()
        self.shape = shape

    def __matmul__(self, other):
        if isinstance(other, SparseQuaternionMatrix):
            return self.sparse_multiply(other)
        elif isinstance(other, (np.ndarray, quaternion.quaternion)):
            return self.dense_multiply(other)
        else:
            raise TypeError("Unsupported type for multiplication")

    def dense_multiply(self, B):
        B_comp = quaternion.as_float_array(B)
        B_real = B_comp[..., 0]
        B_i = B_comp[..., 1]
        B_j = B_comp[..., 2]
        B_k = B_comp[..., 3]
        real_out = self.real @ B_real - self.i @ B_i - self.j @ B_j - self.k @ B_k
        i_out = self.real @ B_i + self.i @ B_real + self.j @ B_k - self.k @ B_j
        j_out = self.real @ B_j - self.i @ B_k + self.j @ B_real + self.k @ B_i
        k_out = self.real @ B_k + self.i @ B_j - self.j @ B_i + self.k @ B_real
        result = np.stack([real_out, i_out, j_out, k_out], axis=-1)
        return quaternion.as_quat_array(result)

    def sparse_multiply(self, other):
        real_part = (
            self.real @ other.real
            - self.i @ other.i
            - self.j @ other.j
            - self.k @ other.k
        )
        i_part = (
            self.real @ other.i
            + self.i @ other.real
            + self.j @ other.k
            - self.k @ other.j
        )
        j_part = (
            self.real @ other.j
            - self.i @ other.k
            + self.j @ other.real
            + self.k @ other.i
        )
        k_part = (
            self.real @ other.k
            + self.i @ other.j
            - self.j @ other.i
            + self.k @ other.real
        )
        return SparseQuaternionMatrix(
            real_part, i_part, j_part, k_part, (self.shape[0], other.shape[1])
        )

    def left_multiply(self, A_dense):
        A_comp = quaternion.as_float_array(A_dense)
        A_real = sparse.csr_matrix(A_comp[..., 0])
        A_i = sparse.csr_matrix(A_comp[..., 1])
        A_j = sparse.csr_matrix(A_comp[..., 2])
        A_k = sparse.csr_matrix(A_comp[..., 3])
        temp = SparseQuaternionMatrix(A_real, A_i, A_j, A_k, A_dense.shape)
        return temp @ self

    def conjugate(self):
        return SparseQuaternionMatrix(
            self.real.conjugate(),
            -self.i.conjugate(),
            -self.j.conjugate(),
            -self.k.conjugate(),
            self.shape,
        )

    def transpose(self):
        return SparseQuaternionMatrix(
            self.real.transpose(),
            self.i.transpose(),
            self.j.transpose(),
            self.k.transpose(),
            (self.shape[1], self.shape[0]),
        )

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float, np.floating)):
            return SparseQuaternionMatrix(
                self.real * scalar,
                self.i * scalar,
                self.j * scalar,
                self.k * scalar,
                self.shape,
            )
        else:
            raise TypeError("Can only multiply SparseQuaternionMatrix by a scalar.")

    def __rmul__(self, scalar):
        return self.__mul__(scalar)


# =============================================================================
# Q-GMRES FUNCTIONS
# =============================================================================
# These functions are based on MATLAB implementations by Zhigang Jia and colleagues
# for the Quaternion Generalized Minimal Residual Method (Q-GMRES)
#
# References:
# - Zhigang Jia and Michael K. Ng, "Structure Preserving Quaternion Generalized
#   Minimal Residual Method", SIMAX, 2021
# - Various MATLAB functions by Zhigang Jia (2014-2020)
# =============================================================================


def normQsparse(A0, A1, A2, A3, opt=None):
    """
    Compute norm of quaternion matrix in component format (A0, A1, A2, A3).

    Parameters:
    -----------
    A0, A1, A2, A3 : numpy.ndarray or scipy.sparse matrix
        Quaternion matrix components
    opt : str, optional
        Norm type: 'd' (dual), '2' (2-norm), '1' (1-norm), or None (Frobenius)

    Returns:
    --------
    float : The computed norm

    Notes:
    ------
    Based on MATLAB implementation by Zhigang Jia (Apr 27, 2020)
    """
    if opt is None:
        # Default: Frobenius norm of [A0, A2, A1, A3]
        if hasattr(A0, "toarray"):
            # Handle sparse matrices
            A0_arr, A2_arr, A1_arr, A3_arr = (
                A0.toarray(),
                A2.toarray(),
                A1.toarray(),
                A3.toarray(),
            )
        else:
            A0_arr, A2_arr, A1_arr, A3_arr = A0, A2, A1, A3

        # Stack matrices horizontally (concatenate columns) as in MATLAB
        stacked = np.hstack([A0_arr, A2_arr, A1_arr, A3_arr])

        # Use appropriate norm based on dimensionality
        if stacked.ndim == 1:
            return np.linalg.norm(stacked, 2)  # 2-norm for vectors
        else:
            return np.linalg.norm(stacked, "fro")  # Frobenius norm for matrices

    elif opt == "d":
        # Dual norm: norm([2*A1-A2-A3, 2*A2-A3-A1, 2*A3-A1-A2], 'fro') / 3
        if hasattr(A0, "toarray"):
            A0_arr, A1_arr, A2_arr, A3_arr = (
                A0.toarray(),
                A1.toarray(),
                A2.toarray(),
                A3.toarray(),
            )
        else:
            A0_arr, A1_arr, A2_arr, A3_arr = A0, A1, A2, A3

        term1 = 2 * A1_arr - A2_arr - A3_arr
        term2 = 2 * A2_arr - A3_arr - A1_arr
        term3 = 2 * A3_arr - A1_arr - A2_arr

        stacked = np.hstack([term1, term2, term3])
        if stacked.ndim == 1:
            return np.linalg.norm(stacked, 2) / 3
        else:
            return np.linalg.norm(stacked, "fro") / 3

    elif opt == "2":
        # 2-norm: largest singular value
        if hasattr(A0, "toarray"):
            A0_arr, A2_arr, A1_arr, A3_arr = (
                A0.toarray(),
                A2.toarray(),
                A1.toarray(),
                A3.toarray(),
            )
        else:
            A0_arr, A2_arr, A1_arr, A3_arr = A0, A2, A1, A3

        stacked = np.hstack([A0_arr, A2_arr, A1_arr, A3_arr])
        _, s, _ = np.linalg.svd(stacked, full_matrices=False)
        return np.max(s)

    elif opt == "1":
        # 1-norm of element-wise square root
        if hasattr(A0, "toarray"):
            A0_arr, A1_arr, A2_arr, A3_arr = (
                A0.toarray(),
                A1.toarray(),
                A2.toarray(),
                A3.toarray(),
            )
        else:
            A0_arr, A1_arr, A2_arr, A3_arr = A0, A1, A2, A3

        sqrt_sum = np.sqrt(
            np.abs(A0_arr) + np.abs(A1_arr) + np.abs(A2_arr) + np.abs(A3_arr)
        )
        return np.linalg.norm(sqrt_sum, 1)

    else:
        # Other norms: apply to element-wise square root
        if hasattr(A0, "toarray"):
            A0_arr, A1_arr, A2_arr, A3_arr = (
                A0.toarray(),
                A1.toarray(),
                A2.toarray(),
                A3.toarray(),
            )
        else:
            A0_arr, A1_arr, A2_arr, A3_arr = A0, A1, A2, A3

        sqrt_sum = np.sqrt(
            np.abs(A0_arr) + np.abs(A1_arr) + np.abs(A2_arr) + np.abs(A3_arr)
        )
        return np.linalg.norm(sqrt_sum, opt)


def timesQsparse(B0, B1, B2, B3, C0, C1, C2, C3):
    """
    Quaternion matrix multiplication in component format.

    Parameters:
    -----------
    B0, B1, B2, B3 : numpy.ndarray or scipy.sparse matrix
        First quaternion matrix components
    C0, C1, C2, C3 : numpy.ndarray or scipy.sparse matrix
        Second quaternion matrix components

    Returns:
    --------
    tuple : (A0, A1, A2, A3) - Result quaternion matrix components

    Notes:
    ------
    Implements quaternion multiplication: A = B * C
    Based on MATLAB implementation by Zhigang Jia
    """
    # Handle sparse matrices by converting to dense for computation
    if hasattr(B0, "toarray"):
        B0_arr, B1_arr, B2_arr, B3_arr = (
            B0.toarray(),
            B1.toarray(),
            B2.toarray(),
            B3.toarray(),
        )
    else:
        B0_arr, B1_arr, B2_arr, B3_arr = B0, B1, B2, B3

    if hasattr(C0, "toarray"):
        C0_arr, C1_arr, C2_arr, C3_arr = (
            C0.toarray(),
            C1.toarray(),
            C2.toarray(),
            C3.toarray(),
        )
    else:
        C0_arr, C1_arr, C2_arr, C3_arr = C0, C1, C2, C3

    # Helper function to handle scalar-matrix multiplication
    def safe_multiply(a, b):
        if np.isscalar(a) and not np.isscalar(b):
            return a * b
        elif np.isscalar(b) and not np.isscalar(a):
            return a * b
        else:
            return a @ b

    # Quaternion multiplication: A = B * C
    # A0 = B0*C0 - B2*C2 - B1*C1 - B3*C3
    A0 = (
        safe_multiply(B0_arr, C0_arr)
        - safe_multiply(B2_arr, C2_arr)
        - safe_multiply(B1_arr, C1_arr)
        - safe_multiply(B3_arr, C3_arr)
    )

    # A2 = B0*C2 + B2*C0 - B1*C3 + B3*C1
    A2 = (
        safe_multiply(B0_arr, C2_arr)
        + safe_multiply(B2_arr, C0_arr)
        - safe_multiply(B1_arr, C3_arr)
        + safe_multiply(B3_arr, C1_arr)
    )

    # A1 = B0*C1 + B2*C3 + B1*C0 - B3*C2
    A1 = (
        safe_multiply(B0_arr, C1_arr)
        + safe_multiply(B2_arr, C3_arr)
        + safe_multiply(B1_arr, C0_arr)
        - safe_multiply(B3_arr, C2_arr)
    )

    # A3 = B0*C3 - B2*C1 + B1*C2 + B3*C0
    A3 = (
        safe_multiply(B0_arr, C3_arr)
        - safe_multiply(B2_arr, C1_arr)
        + safe_multiply(B1_arr, C2_arr)
        + safe_multiply(B3_arr, C0_arr)
    )

    return A0, A1, A2, A3


def A2A0123(A):
    """
    Extract component matrices from real matrix A = [A0 A2 A1 A3].

    Parameters:
    -----------
    A : numpy.ndarray
        Real matrix with columns arranged as [A0 A2 A1 A3]
        where each component matrix has n columns

    Returns:
    --------
    tuple : (A0, A1, A2, A3) component matrices

    Notes:
    ------
    Based on MATLAB implementation by Zhigang Jia (Aug 14, 2014)
    """
    n = A.shape[1] // 4  # Each component matrix has n columns

    A0 = A[:, 0:n]
    A2 = A[:, n : 2 * n]
    A1 = A[:, 2 * n : 3 * n]
    A3 = A[:, 3 * n : 4 * n]

    return A0, A1, A2, A3


def normQ(A, opt=None):
    """
    Compute norm of quaternion matrix A.

    Parameters:
    -----------
    A : numpy.ndarray with dtype=quaternion
        Quaternion matrix
    opt : str, optional
        Norm type: 'd' (dual) or None (Frobenius)

    Returns:
    --------
    float : The computed norm

    Notes:
    ------
    Based on MATLAB implementation by Zhigang Jia (Jan 24, 2018)
    """
    if opt is None:
        # Default: Frobenius norm
        return quat_frobenius_norm(A)

    elif opt == "d":
        # Dual norm: convert to component format and use normQsparse
        A_comp = quaternion.as_float_array(A)
        A0, A1, A2, A3 = A_comp[..., 0], A_comp[..., 1], A_comp[..., 2], A_comp[..., 3]
        return normQsparse(A0, A1, A2, A3, "d")

    else:
        # Other norms: apply directly to quaternion matrix
        # Convert to real representation for standard norms
        A_real = quaternion.as_float_array(A)
        return np.linalg.norm(A_real, opt)


def Realp(A1, A2, A3, A4):
    """
    Convert quaternion matrix components to real block matrix representation.

    Parameters:
    -----------
    A1, A2, A3, A4 : numpy.ndarray or scalar
        Quaternion matrix components (can be scalars or matrices)

    Returns:
    --------
    numpy.ndarray : Real block matrix AR = [A1 -A2 -A3 -A4; A2 A1 -A4 A3; A3 A4 A1 -A2; A4 -A3 A2 A1]

    Notes:
    ------
    Based on MATLAB implementation by Zhigang Jia
    """
    # Handle scalar inputs
    if np.isscalar(A1):
        # Scalar case: create 4x4 matrix
        AR = np.array(
            [[A1, -A2, -A3, -A4], [A2, A1, -A4, A3], [A3, A4, A1, -A2], [A4, -A3, A2, A1]]
        )
        return AR
    else:
        # Matrix case: create block matrix
        m, n = A1.shape

        # Create the real block matrix
        AR = np.zeros((4 * m, 4 * n))

        # Fill the 4x4 blocks
        AR[0:m, 0:n] = A1
        AR[0:m, n : 2 * n] = -A2
        AR[0:m, 2 * n : 3 * n] = -A3
        AR[0:m, 3 * n : 4 * n] = -A4

        AR[m : 2 * m, 0:n] = A2
        AR[m : 2 * m, n : 2 * n] = A1
        AR[m : 2 * m, 2 * n : 3 * n] = -A4
        AR[m : 2 * m, 3 * n : 4 * n] = A3

        AR[2 * m : 3 * m, 0:n] = A3
        AR[2 * m : 3 * m, n : 2 * n] = A4
        AR[2 * m : 3 * m, 2 * n : 3 * n] = A1
        AR[2 * m : 3 * m, 3 * n : 4 * n] = -A2

        AR[3 * m : 4 * m, 0:n] = A4
        AR[3 * m : 4 * m, n : 2 * n] = -A3
        AR[3 * m : 4 * m, 2 * n : 3 * n] = A2
        AR[3 * m : 4 * m, 3 * n : 4 * n] = A1

        return AR


def ggivens(x1, x2):
    """
    Generate quaternion Givens rotation matrix.

    Parameters:
    -----------
    x1, x2 : numpy.ndarray
        Quaternion vectors (4-component arrays)

    Returns:
    --------
    numpy.ndarray : 8x8 real Givens rotation matrix G such that G'*[x1;x2] = [||x||_2; 0]

    Notes:
    ------
    Based on MATLAB implementation by Zhigang Jia, Musheng Wei, Meixiang Zhao and Yong Chen (Mar 24, 2017)
    """
    # Compute norm of the combined vector
    t = np.linalg.norm(np.concatenate([x1, x2]))

    if t <= np.finfo(float).eps:
        # If norm is too small, return identity matrix
        # q1=eye(4,1); q2=zeros(4,1); q3=zeros(4,1); q4=eye(4,1)
        q1 = np.array([1, 0, 0, 0])
        q2 = np.array([0, 0, 0, 0])
        q3 = np.array([0, 0, 0, 0])
        q4 = np.array([1, 0, 0, 0])
    else:
        # Normalize vectors
        q1 = x1 / t
        q2 = x2 / t

        # Compute norms of q1 and q2
        t_norms = np.array([np.linalg.norm(q1), np.linalg.norm(q2)])

        if t_norms[0] < t_norms[1]:
            # q3 = [t(2); 0; 0; 0]
            q3 = np.array([t_norms[1], 0, 0, 0])
            # q4 = Realp(q2(1),q2(2),q2(3),q2(4))*[q1(1);-q1(2);-q1(3);-q1(4)]/(-t(2))
            q1_conj = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
            q4 = Realp(q2[0], q2[1], q2[2], q2[3]) @ q1_conj / (-t_norms[1])
        else:
            # q4 = [t(1); 0; 0; 0]
            q4 = np.array([t_norms[0], 0, 0, 0])
            # q3 = Realp(q1(1),q1(2),q1(3),q1(4))*[q2(1);-q2(2);-q2(3);-q2(4)]/(-t(1))
            q2_conj = np.array([q2[0], -q2[1], -q2[2], -q2[3]])
            q3 = Realp(q1[0], q1[1], q1[2], q1[3]) @ q2_conj / (-t_norms[0])

    # G = Realp([q1(1),q3(1);q2(1),q4(1)],[q1(2),q3(2);q2(2),q4(2)],[q1(3),q3(3);q2(3),q4(3)],[q1(4),q3(4);q2(4),q4(4)])
    # Create 2x2 quaternion matrices for each component
    q1_mat = np.array([[q1[0], q3[0]], [q2[0], q4[0]]])
    q2_mat = np.array([[q1[1], q3[1]], [q2[1], q4[1]]])
    q3_mat = np.array([[q1[2], q3[2]], [q2[2], q4[2]]])
    q4_mat = np.array([[q1[3], q3[3]], [q2[3], q4[3]]])

    # Apply Realp to get the 8x8 real matrix
    G = Realp(q1_mat, q2_mat, q3_mat, q4_mat)

    return G


def GRSGivens(g1, g2=None, g3=None, g4=None):
    """
    Generate quaternion Givens rotation matrix for the last column.

    Parameters:
    -----------
    g1 : float or numpy.ndarray
        First quaternion component or 4-vector
    g2, g3, g4 : float, optional
        Additional quaternion components (only used if g1 is scalar)

    Returns:
    --------
    numpy.ndarray : 4x4 real Givens rotation matrix

    Notes:
    ------
    Based on MATLAB implementation by Zhigang Jia
    """
    if g2 is None and g3 is None and g4 is None:
        # Case: GRSGivens(g1) where g1 is a 4-vector
        g1 = np.array(g1)
        if np.allclose(g1[1:3], 0):
            return np.eye(4)
        else:
            g1 = g1 / np.linalg.norm(g1)
            return Realp(g1[0], g1[1], g1[2], g1[3])
    else:
        # Case: GRSGivens(g1, g2, g3, g4)
        if np.allclose([g2, g3, g4], 0):
            return np.eye(4)
        else:
            r = np.linalg.norm([g1, g2, g3, g4])
            return Realp(g1 / r, g2 / r, g3 / r, g4 / r)


def Hess_QR_ggivens(Hess):
    """
    QR decomposition of quaternion Hessenberg matrix using Givens rotations.

    Parameters:
    -----------
    Hess : numpy.ndarray
        Real block matrix representation of quaternion Hessenberg matrix
        Hess = [A0; A1; A2; A3] where A0, A1, A2, A3 are the quaternion components

    Returns:
    --------
    tuple : (W, Hess) where W*Hess gives the QR factorization

    Notes:
    ------
    Based on MATLAB implementation by Zhigang Jia (Apr 21, 2020)
    """
    m, n = Hess.shape
    m = m // 4  # Each quaternion component has m rows

    # Initialize W as [eye(m), zeros(m), zeros(m), zeros(m)]
    W = np.zeros((m, 4 * m))
    W[:, 0:m] = np.eye(m)

    # Apply Givens rotations for subdiagonal elements
    for s in range(m - 1):
        # Extract quaternion vectors for current position
        x1 = np.array(
            [Hess[s, s], Hess[m + s, s], Hess[2 * m + s, s], Hess[3 * m + s, s]]
        )
        x2 = np.array(
            [
                Hess[s + 1, s],
                Hess[m + s + 1, s],
                Hess[2 * m + s + 1, s],
                Hess[3 * m + s + 1, s],
            ]
        )

        # Generate Givens rotation
        G = ggivens(x1, x2)

        # Apply rotation to Hess matrix
        # Select 8 rows: [s:s+1, s+m:s+m+1, s+2*m:s+2*m+1, s+3*m:s+3*m+1]
        rows_to_update = [
            s,
            s + 1,
            s + m,
            s + m + 1,
            s + 2 * m,
            s + 2 * m + 1,
            s + 3 * m,
            s + 3 * m + 1,
        ]
        Hess[rows_to_update, s:n] = G.T @ Hess[rows_to_update, s:n]

        # Update W matrix
        # Select 8 columns: [s:s+1, s+m:s+m+1, s+2*m:s+2*m+1, s+3*m:s+3*m+1]
        cols_to_update = [
            s,
            s + 1,
            s + m,
            s + m + 1,
            s + 2 * m,
            s + 2 * m + 1,
            s + 3 * m,
            s + 3 * m + 1,
        ]
        W[:, cols_to_update] = W[:, cols_to_update] @ G

    # Handle the last column (special case)
    # s = m
    G = GRSGivens(
        Hess[m - 1, n - 1],
        Hess[2 * m - 1, n - 1],
        Hess[3 * m - 1, n - 1],
        Hess[4 * m - 1, n - 1],
    )
    W[:, [m - 1, 2 * m - 1, 3 * m - 1, 4 * m - 1]] = (
        W[:, [m - 1, 2 * m - 1, 3 * m - 1, 4 * m - 1]] @ G
    )
    Hess[[m - 1, 2 * m - 1, 3 * m - 1, 4 * m - 1], n - 1] = (
        G.T @ Hess[[m - 1, 2 * m - 1, 3 * m - 1, 4 * m - 1], n - 1]
    )

    # Final reshaping to match MATLAB output format
    # W = [W(1:m,1:m), -W(1:m,2*m+1:3*m), -W(1:m,m+1:2*m), -W(1:m,3*m+1:4*m)]
    W_final = np.zeros((m, 4 * m))
    W_final[:, 0:m] = W[:, 0:m]
    W_final[:, m : 2 * m] = -W[:, 2 * m : 3 * m]
    W_final[:, 2 * m : 3 * m] = -W[:, m : 2 * m]
    W_final[:, 3 * m : 4 * m] = -W[:, 3 * m : 4 * m]

    # Hess = [Hess(1:m,1:n), Hess(2*m+1:3*m,1:n), Hess(m+1:2*m,1:n), Hess(3*m+1:4*m,1:n)]
    Hess_final = np.zeros((m, 4 * n))
    Hess_final[:, 0:n] = Hess[0:m, 0:n]
    Hess_final[:, n : 2 * n] = Hess[2 * m : 3 * m, 0:n]
    Hess_final[:, 2 * n : 3 * n] = Hess[m : 2 * m, 0:n]
    Hess_final[:, 3 * n : 4 * n] = Hess[3 * m : 4 * m, 0:n]

    return W_final, Hess_final


def absQsparse(A0, A1, A2, A3):
    """
    Compute absolute value/norm of quaternion in component format.

    Parameters:
    -----------
    A0, A1, A2, A3 : float or numpy.ndarray
        Quaternion components

    Returns:
    --------
    tuple : (r, s0, s1, s2, s3) where r is the norm and s0,s1,s2,s3 are normalized components

    Notes:
    ------
    Based on MATLAB implementation by Zhigang Jia
    """
    r = np.sqrt(A0**2 + A1**2 + A2**2 + A3**2)
    s0 = A0 / (r + np.finfo(float).eps)
    s1 = A1 / (r + np.finfo(float).eps)
    s2 = A2 / (r + np.finfo(float).eps)
    s3 = A3 / (r + np.finfo(float).eps)

    return r, s0, s1, s2, s3


def dotinvQsparse(A0, A1, A2, A3):
    """
    Compute quaternion inverse in component format.

    Parameters:
    -----------
    A0, A1, A2, A3 : float or numpy.ndarray
        Quaternion components

    Returns:
    --------
    tuple : (inv0, inv1, inv2, inv3) - Inverse quaternion components

    Notes:
    ------
    Based on MATLAB implementation by Zhigang Jia
    """
    inv = A0**2 + A2**2 + A1**2 + A3**2
    inv0 = A0 / (inv + np.finfo(float).eps)
    inv1 = -A1 / (inv + np.finfo(float).eps)
    inv2 = -A2 / (inv + np.finfo(float).eps)
    inv3 = -A3 / (inv + np.finfo(float).eps)

    return inv0, inv1, inv2, inv3


def UtriangleQsparse(R0, R1, R2, R3, b0, b1, b2, b3, tol=1e-14):
    """
    Solve quaternion upper triangular system R * x = b using backward substitution.

    Parameters:
    -----------
    R0, R1, R2, R3 : numpy.ndarray
        Upper triangular matrix components
    b0, b1, b2, b3 : numpy.ndarray
        Right-hand side vector components
    tol : float, optional
        Tolerance for checking zero elements (default: 1e-14)

    Returns:
    --------
    tuple : (b0, b1, b2, b3) - Solution vector components (overwrites input b)

    Notes:
    ------
    Based on MATLAB implementation by Zhigang Jia & Xuan Liu (Apr 22, 2020)
    Implements backward substitution: Algorithm 3.1.2, page 89, Matrix Computations, Golub and Van Loan, 3rd ed.
    """
    m, n = R0.shape
    rb = b0.shape[0]

    if m == n and m == rb:
        # Get the last diagonal element
        Unn_0, Unn_1, Unn_2, Unn_3 = (
            R0[n - 1, n - 1],
            R1[n - 1, n - 1],
            R2[n - 1, n - 1],
            R3[n - 1, n - 1],
        )

        # Check if the diagonal element is non-zero
        r, _, _, _, _ = absQsparse(Unn_0, Unn_1, Unn_2, Unn_3)
        if r > tol:
            # Solve for the last element
            delta0, delta1, delta2, delta3 = dotinvQsparse(Unn_0, Unn_1, Unn_2, Unn_3)

            # Convert scalars to arrays if needed
            if np.isscalar(delta0):
                delta0, delta1, delta2, delta3 = (
                    np.array([delta0]),
                    np.array([delta1]),
                    np.array([delta2]),
                    np.array([delta3]),
                )

            b0[n - 1, :], b1[n - 1, :], b2[n - 1, :], b3[n - 1, :] = timesQsparse(
                delta0,
                delta1,
                delta2,
                delta3,
                b0[n - 1, :],
                b1[n - 1, :],
                b2[n - 1, :],
                b3[n - 1, :],
            )

            # Backward substitution for remaining elements
            for i in range(n - 2, -1, -1):
                # Get diagonal element
                Uii_0, Uii_1, Uii_2, Uii_3 = R0[i, i], R1[i, i], R2[i, i], R3[i, i]

                # Check if the diagonal element is non-zero
                r, _, _, _, _ = absQsparse(Uii_0, Uii_1, Uii_2, Uii_3)
                if r > tol:
                    # Compute R(i, i+1:n) * b(i+1:n, :)
                    delta0, delta1, delta2, delta3 = timesQsparse(
                        R0[i, i + 1 : n],
                        R1[i, i + 1 : n],
                        R2[i, i + 1 : n],
                        R3[i, i + 1 : n],
                        b0[i + 1 : n, :],
                        b1[i + 1 : n, :],
                        b2[i + 1 : n, :],
                        b3[i + 1 : n, :],
                    )

                    # Subtract from b(i, :)
                    delta0 = b0[i, :] - delta0
                    delta1 = b1[i, :] - delta1
                    delta2 = b2[i, :] - delta2
                    delta3 = b3[i, :] - delta3

                    # Solve for b(i, :)
                    beta0, beta1, beta2, beta3 = dotinvQsparse(Uii_0, Uii_1, Uii_2, Uii_3)

                    # Convert scalars to arrays if needed
                    if np.isscalar(beta0):
                        beta0, beta1, beta2, beta3 = (
                            np.array([beta0]),
                            np.array([beta1]),
                            np.array([beta2]),
                            np.array([beta3]),
                        )

                    b0[i, :], b1[i, :], b2[i, :], b3[i, :] = timesQsparse(
                        beta0, beta1, beta2, beta3, delta0, delta1, delta2, delta3
                    )
                else:
                    print(f"U0({i},{i})=0 ! No solution but least square solution!")
                    b0[i, :] = 0
                    b1[i, :] = 0
                    b2[i, :] = 0
                    b3[i, :] = 0
        else:
            print("U0(n,n)=0 ! No solution but least square solution!")
            b0[n - 1, :] = 0
            b1[n - 1, :] = 0
            b2[n - 1, :] = 0
            b3[n - 1, :] = 0
    else:
        raise ValueError("The sizes of R and b are not consistent!")

    return b0, b1, b2, b3


def ishermitian(A, tol=None):
    """
    Check if a quaternion matrix is Hermitian to within the given tolerance.

    Parameters:
    -----------
    A : numpy.ndarray with dtype=quaternion
        Quaternion matrix to test
    tol : float, optional
        Tolerance for comparison (default: machine epsilon)

    Returns:
    --------
    bool : True if matrix is Hermitian within tolerance

    Notes:
    ------
    A matrix A is Hermitian if A = A^H where A^H is the conjugate transpose.
    """
    if tol is None:
        tol = np.finfo(float).eps

    r, c = A.shape

    if r != c:
        raise ValueError("Cannot test whether a non-square matrix is Hermitian.")

    # Compute A - A^H and check if it's within tolerance
    A_H = quat_hermitian(A)
    diff = A - A_H

    # Normalize by maximum absolute value
    max_abs = np.max(np.abs(A))
    if max_abs == 0:
        return True  # Zero matrix is Hermitian

    # Check if normalized difference is within tolerance
    normalized_diff = np.abs(diff) / max_abs
    return not np.any(normalized_diff > tol)


def det(X, d):
    """
    Compute determinant of a quaternion matrix.

    Parameters:
    -----------
    X : numpy.ndarray with dtype=quaternion
        Square quaternion matrix
    d : str
        Determinant type:
        - 'Moore': Product of eigenvalues (requires Hermitian matrix)
        - 'Dieudonné' or 'Dieudonne': Product of singular values
        - 'Study': Determinant of the adjoint matrix

    Returns:
    --------
    complex or float : The computed determinant

    Notes:
    ------
    - Moore determinant can be negative or complex, but requires Hermitian matrix
    - Dieudonné determinant is always real
    - Study determinant is the square of Dieudonné determinant
    """
    r, c = X.shape

    if r != c:
        raise ValueError("Matrix must be square.")

    if d in ["Dieudonné", "Dieudonne"]:
        # Dieudonné determinant: product of singular values
        from decomp.qsvd import classical_qsvd_full

        _, s, _ = classical_qsvd_full(X)
        return np.prod(s)

    elif d == "Study":
        # Study determinant: determinant of the adjoint
        # For now, we'll use a simplified approach
        # TODO: Implement proper adjoint computation
        raise NotImplementedError("Study determinant not yet implemented")

    elif d == "Moore":
        # Moore determinant: product of eigenvalues (requires Hermitian)
        if not ishermitian(X):
            raise ValueError(
                "Cannot compute Moore determinant of a non-Hermitian matrix."
            )

        from decomp import quaternion_eigenvalues

        eigenvalues = quaternion_eigenvalues(X)
        return np.prod(eigenvalues)

    else:
        raise ValueError(f"Unrecognized determinant type: {d}")


def rank(X, tol=None):
    """
    Compute the rank of a quaternion matrix by counting non-zero singular values.

    Parameters:
    -----------
    X : numpy.ndarray with dtype=quaternion
        Quaternion matrix of any shape (m, n)
    tol : float, optional
        Tolerance for considering singular values as non-zero
        (default: machine epsilon * max(m,n) * max singular value)

    Returns:
    --------
    int : The rank of the matrix (number of non-zero singular values)

    Notes:
    ------
    The rank is computed by:
    1. Computing the SVD of the matrix: X = U @ S @ V^H
    2. Counting singular values above the tolerance threshold
    3. The rank equals the number of non-zero singular values

    The tolerance is automatically adjusted based on matrix size and magnitude
    to handle numerical precision issues.
    """
    m, n = X.shape

    # Compute SVD
    from decomp.qsvd import classical_qsvd_full

    _, s, _ = classical_qsvd_full(X)

    # Set tolerance if not provided
    if tol is None:
        # Use machine epsilon scaled by matrix size and largest singular value
        max_sv = np.max(s) if len(s) > 0 else 0
        tol = np.finfo(float).eps * max(m, n) * max_sv

    # Count singular values above tolerance
    rank_count = np.sum(s > tol)

    return int(rank_count)


def power_iteration(
    A, max_iterations=100, tol=1e-10, return_eigenvalue=False, verbose=False
):
    """
    Compute the dominant eigenvector of a quaternion matrix using power iteration.

    Parameters:
    -----------
    A : numpy.ndarray with dtype=quaternion
        Square quaternion matrix of size n×n
    max_iterations : int, optional
        Maximum number of iterations (default: 100)
    tol : float, optional
        Convergence tolerance for vector norm difference (default: 1e-10)
    return_eigenvalue : bool, optional
        If True, also return an eigenvalue estimate (default: False)
    verbose : bool, optional
        If True, print convergence information and non-Hermitian warning (default: False)

    Returns:
    --------
    numpy.ndarray or tuple:
        - If return_eigenvalue=False: dominant eigenvector (n×1 quaternion vector)
        - If return_eigenvalue=True: (eigenvector, eigenvalue_estimate) tuple

    Notes:
    ------
    - Intended primarily for HERMITIAN quaternion matrices, where eigenvalues are real.
      In that case this routine returns a meaningful dominant eigenpair.
    - For general (non-Hermitian) quaternion matrices, the returned scalar when
      return_eigenvalue=True is a magnitude-based Rayleigh-quotient estimate (real),
      which can be interpreted as a real-part/magnitude heuristic, not a true complex
      eigenvalue. For non-Hermitian matrices, use `power_iteration_nonhermitian` which
      returns complex eigenvalues (in a fixed complex subfield) and quaternion eigenvectors.

    Algorithm:
    1) Start with a random vector
    2) Iterate v_{k+1} = A v_k / ||A v_k||
    3) Stop when ||v_{k+1} - v_k|| < tol or max_iterations reached
    """
    from data_gen import create_test_matrix

    # Input validation
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square")

    n = A.shape[0]
    if n == 0:
        raise ValueError("Matrix A cannot be empty")

    # Optional notice for non-Hermitian inputs
    try:
        if verbose and not ishermitian(A):
            print(
                "[power_iteration] Notice: A appears non-Hermitian; eigenvalue estimate will be real (magnitude-based). For complex eigenvalues, use power_iteration_nonhermitian()."
            )
    except Exception:
        # If ishermitian fails (e.g., dtype mismatch), ignore and proceed
        pass

    # Initialize with random vector
    v_k = create_test_matrix(n, 1)  # n×1 random quaternion vector

    # Normalize initial vector
    v_k_norm = quat_frobenius_norm(v_k)
    if v_k_norm == 0:
        raise ValueError("Initial vector has zero norm")
    v_k = v_k / v_k_norm

    prev_norm_diff = float("inf")

    for iteration in range(max_iterations):
        # Compute A * v_k
        Av_k = quat_matmat(A, v_k)

        # Compute norm of result
        Av_k_norm = quat_frobenius_norm(Av_k)

        # Check for breakdown (zero norm)
        if Av_k_norm == 0:
            if verbose:
                print(f"Breakdown at iteration {iteration}: Av_k has zero norm")
            break

        # Store previous vector for convergence check
        v_k_prev = v_k.copy()

        # Normalize to get new vector
        v_k = Av_k / Av_k_norm

        # Check convergence: ||v_k - v_k_prev||
        norm_diff = quat_frobenius_norm(v_k - v_k_prev)

        if verbose:
            print(f"Iteration {iteration}: norm_diff = {norm_diff:.2e}")

        # Check if convergence criterion is met
        if norm_diff < tol:
            if verbose:
                print(
                    f"Converged at iteration {iteration} with norm_diff = {norm_diff:.2e}"
                )
            break

        # Check for stagnation (no improvement)
        if abs(norm_diff - prev_norm_diff) < tol * 1e-3:
            if verbose:
                print(f"Stagnation detected at iteration {iteration}")
            break

        prev_norm_diff = norm_diff

    # Compute eigenvalue if requested
    if return_eigenvalue:
        # Rayleigh quotient: λ = (v^H * A * v) / (v^H * v)
        v_H = quat_hermitian(v_k)
        numerator = quat_matmat(quat_matmat(v_H, A), v_k)
        denominator = quat_matmat(v_H, v_k)

        # For quaternion matrices, eigenvalue might be complex
        # We return the magnitude as the eigenvalue
        eigenvalue = quat_frobenius_norm(numerator) / quat_frobenius_norm(denominator)

        return v_k, eigenvalue

    return v_k


# =============================================================================
# Experimental: Complex power iteration for NON-Hermitian quaternion matrices
# =============================================================================
def quaternion_to_complex_adjoint(A: np.ndarray, axis: str = "x") -> np.ndarray:
    """Map quaternion matrix A to 2n×2n complex adjoint matrix for a chosen complex subfield.

    By default the complex subfield is tied to the x-axis (unit i):
      A = W + X i + Y j + Z k → C = W + i X, D = Y + i Z,  Adj(A) = [[C, D], [-D*, C*]]

    Parameters
    ----------
    A : np.ndarray (n×n, dtype=np.quaternion)
    axis : {"x"}
        Imaginary axis defining the complex subfield. Currently only "x" is supported.

    Notes
    -----
    Extending to other axes requires a consistent symplectic embedding relative to that axis.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.dtype != np.quaternion:
        raise ValueError("A must be a square quaternion matrix")
    if axis != "x":
        raise NotImplementedError(
            "quaternion_to_complex_adjoint currently supports axis='x' only"
        )
    n = A.shape[0]
    Af = quaternion.as_float_array(A)  # (n,n,4) -> (w,x,y,z)
    W, X, Y, Z = Af[..., 0], Af[..., 1], Af[..., 2], Af[..., 3]
    C = W + 1j * X
    D = Y + 1j * Z
    M = np.zeros((2 * n, 2 * n), dtype=complex)
    M[0:n, 0:n] = C
    M[0:n, n : 2 * n] = D
    M[n : 2 * n, 0:n] = -np.conjugate(D)
    M[n : 2 * n, n : 2 * n] = np.conjugate(C)
    return M


def _power_iteration_complex(
    M: np.ndarray,
    max_iter: int = 1000,
    eig_tol: float = 1e-12,
    res_tol: float | None = 1e-10,
    seed: int = 0,
) -> Tuple[complex, np.ndarray, List[float]]:
    """Standard complex power iteration on matrix M returning (lambda, v, residuals)."""
    rng = np.random.default_rng(seed)
    n = M.shape[0]
    v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    v = v / (np.linalg.norm(v) + np.finfo(float).eps)
    lam = 0.0 + 0.0j
    residuals: List[float] = []
    for _ in range(max_iter):
        w = M @ v
        nw = np.linalg.norm(w)
        if not np.isfinite(nw) or nw == 0.0:
            break
        v_new = w / nw
        Mv = M @ v_new
        lam_new = np.vdot(v_new, Mv) / np.vdot(v_new, v_new)
        res = float(np.linalg.norm(Mv - lam_new * v_new))
        residuals.append(res)
        if res_tol is not None and res <= res_tol:
            v = v_new
            lam = lam_new
            break
        if np.abs(lam_new - lam) <= eig_tol * max(1.0, np.abs(lam_new)):
            v = v_new
            lam = lam_new
            break
        v = v_new
        lam = lam_new
    return lam, v, residuals


def power_iteration_nonhermitian(
    A: np.ndarray,
    max_iterations: int = 5000,
    eig_tol: float = 1e-12,
    res_tol: float | None = 1e-10,
    seed: int = 0,
    return_vector: bool = True,
    eigenvalue_format: str = "complex",
    subfield_axis: str = "x",
    block_purify: bool = True,
):
    """Power iteration for NON-Hermitian quaternion matrices (experimental).

    Returns a complex eigenvalue estimate (in a chosen complex subfield) and a
    corresponding quaternion eigenvector approximation.

    Parameters
    ----------
    A : np.ndarray
        Square quaternion matrix (non-Hermitian allowed)
    max_iterations : int
        Maximum iterations for the complex power iteration
    eig_tol : float
        Tolerance for eigenvalue stabilization
    res_tol : float | None
        Residual tolerance on ||Mv - lambda v|| to declare convergence
    seed : int
        RNG seed for initialization
    return_vector : bool
        If True, also return the quaternion eigenvector approximation
    eigenvalue_format : {"complex", "quaternion"}
        Output format for the eigenvalue. Default is "complex".
    subfield_axis : {"x"}
        Imaginary axis defining complex subfield for the adjoint (currently only "x").

    Notes
    -----
    - For Hermitian matrices, prefer `power_iteration`, which returns a quaternion
      eigenvector and a real-valued eigenvalue magnitude.
    - Eigenvalues appear in conjugate pairs in the adjoint; this returns one complex root.
    """
    # Fast path for Hermitian: use classical quaternion power iteration for stability
    if _is_hermitian_quat(A):
        v_h, lam_mag = power_iteration(
            A, max_iterations=max_iterations, tol=eig_tol, return_eigenvalue=True
        )
        # Build a small residual curve in quaternion space for reporting
        lam_q = quaternion.quaternion(float(lam_mag), 0.0, 0.0, 0.0)
        v_h_col = v_h.reshape(A.shape[0], 1)
        res_val = quat_frobenius_norm(quat_matmat(A, v_h_col) - v_h_col * lam_q)
        residuals = [float(res_val)]
        if eigenvalue_format == "quaternion":
            lam_out = quaternion.quaternion(float(lam_mag), 0.0, 0.0, 0.0)
        else:
            lam_out = complex(float(lam_mag), 0.0)
        if return_vector:
            return (
                v_h.reshape(
                    A.shape[0],
                ),
                lam_out,
                residuals,
            )
        return lam_out, residuals

    M = quaternion_to_complex_adjoint(A, axis=subfield_axis)
    # For Hermitian quaternion matrices, the adjoint is block-structured; ensure the
    # returned eigenvalue is real (within tolerance) by projecting to the dominant block.
    lam, v_c, residuals = _power_iteration_complex(
        M, max_iter=max_iterations, eig_tol=eig_tol, res_tol=res_tol, seed=seed
    )
    # Map complex vector back to quaternion vector using symplectic form: q = u + j v
    n2 = v_c.shape[0]
    if n2 % 2 != 0:
        raise RuntimeError("Adjoint vector length is not even; mapping failed")
    n = n2 // 2
    u = v_c[:n]
    w = v_c[n:]
    # Optional block purification for cases with near block-diagonal adjoint (e.g., complex embedding)
    if block_purify:
        norm_u = np.linalg.norm(u)
        norm_w = np.linalg.norm(w)
        if norm_u >= norm_w:
            w = np.zeros_like(w)
            # Recompute eigenvalue on purified vector
            v_proj = np.concatenate([u, w])
        else:
            u = np.zeros_like(u)
            v_proj = np.concatenate([u, w])
        # Normalize projected vector and recompute Rayleigh quotient and residuals tail for reporting
        denom = np.linalg.norm(v_proj) + np.finfo(float).eps
        v_proj = v_proj / denom
        Mv = M @ v_proj
        lam = np.vdot(v_proj, Mv) / np.vdot(v_proj, v_proj)
        res_tail = float(np.linalg.norm(Mv - lam * v_proj))
        if residuals:
            residuals[-1] = res_tail
    # u = a + i b, w = c + i d → quaternion(a,b,c,d)
    a, b = np.real(u), np.imag(u)
    c, d = np.real(w), np.imag(w)
    q_arr = np.stack([a, b, c, d], axis=-1)
    q_vec = quaternion.as_quat_array(q_arr).reshape(
        n,
    )
    # Normalize quaternion eigenvector to unit Frobenius norm for stability
    q_norm = quat_frobenius_norm(q_vec.reshape(n, 1))
    if q_norm > 0:
        q_vec = (q_vec / q_norm).reshape(
            n,
        )
    # If A is Hermitian, force eigenvalue to be real (numerical cleanup)
    if _is_hermitian_quat(A):
        lam = complex(float(np.real(lam)), 0.0)
    # Format eigenvalue
    if eigenvalue_format == "quaternion":
        lam_out = quaternion.quaternion(
            float(np.real(lam)), float(np.imag(lam)), 0.0, 0.0
        )
    else:
        lam_out = lam
    if return_vector:
        return q_vec, lam_out, residuals
    return lam_out, residuals


# =====================================================================================
# KERNEL/NULL SPACE FUNCTIONS
# =====================================================================================


def quat_null_space(
    A: np.ndarray, side: str = "right", rtol: float = 1e-10
) -> np.ndarray:
    """
    Compute the null space (kernel) of a quaternion matrix using Q-SVD.

    Parameters:
    -----------
    A : numpy.ndarray with dtype=quaternion
        Input quaternion matrix of shape (m, n)
    side : str, optional
        'right' for right null space (null(A)), 'left' for left null space (null(A^H))
        Default: 'right'
    rtol : float, optional
        Relative tolerance for determining rank (singular values <= rtol * max(s) are zero)
        Default: 1e-10

    Returns:
    --------
    N : numpy.ndarray with dtype=quaternion
        For side='right': null space matrix of shape (n, n-rank) such that A @ N ≈ 0
        For side='left': null space matrix of shape (m, m-rank) such that N^H @ A ≈ 0

    Notes:
    ------
    Uses Q-SVD: A = U @ Σ @ V^H
    - Right null space: columns of V corresponding to zero singular values
    - Left null space: columns of U corresponding to zero singular values

    Examples:
    ---------
    >>> A = create_test_matrix(5, 3)  # 5x3 matrix, rank ≤ 3
    >>> N_right = quat_null_space(A, side='right')  # null(A)
    >>> N_left = quat_null_space(A, side='left')    # null(A^H)
    >>> print(f"Right null space: {N_right.shape}")  # (3, 3-rank)
    >>> print(f"Left null space: {N_left.shape}")   # (5, 5-rank)
    """
    if side not in ["right", "left"]:
        raise ValueError(f"side must be 'right' or 'left', got '{side}'")

    # Import Q-SVD function
    from decomp.qsvd import classical_qsvd_full

    # Compute full Q-SVD
    U, s, V = classical_qsvd_full(A)

    # Determine numerical rank
    if len(s) == 0:
        rank = 0
    else:
        rank = np.sum(s > rtol * s[0])  # s[0] is largest singular value

    if side == "right":
        # Right null space: columns of V corresponding to zero singular values
        n = A.shape[1]
        if rank == n:
            # Full rank: no null space
            return np.empty((n, 0), dtype=np.quaternion)
        else:
            # Return last (n-rank) columns of V
            return V[:, rank:]

    else:  # side == 'left'
        # Left null space: columns of U corresponding to zero singular values
        m = A.shape[0]
        if rank == m:
            # Full rank: no null space
            return np.empty((m, 0), dtype=np.quaternion)
        else:
            # Return last (m-rank) columns of U
            return U[:, rank:]


def quat_null_right(A: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    """
    Compute the right null space of a quaternion matrix: null(A).

    Parameters:
    -----------
    A : numpy.ndarray with dtype=quaternion
        Input quaternion matrix of shape (m, n)
    rtol : float, optional
        Relative tolerance for determining rank
        Default: 1e-10

    Returns:
    --------
    N : numpy.ndarray with dtype=quaternion
        Right null space matrix of shape (n, n-rank) such that A @ N ≈ 0

    Notes:
    ------
    Convenience function equivalent to quat_null_space(A, side='right', rtol=rtol)
    """
    return quat_null_space(A, side="right", rtol=rtol)


def quat_null_left(A: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    """
    Compute the left null space of a quaternion matrix: null(A^H).

    Parameters:
    -----------
    A : numpy.ndarray with dtype=quaternion
        Input quaternion matrix of shape (m, n)
    rtol : float, optional
        Relative tolerance for determining rank
        Default: 1e-10

    Returns:
    --------
    N : numpy.ndarray with dtype=quaternion
        Left null space matrix of shape (m, m-rank) such that N^H @ A ≈ 0

    Notes:
    ------
    Convenience function equivalent to quat_null_space(A, side='left', rtol=rtol)
    """
    return quat_null_space(A, side="left", rtol=rtol)


def quat_kernel(A: np.ndarray, side: str = "right", rtol: float = 1e-10) -> np.ndarray:
    """
    Compute the kernel (null space) of a quaternion matrix.

    Alias for quat_null_space() with identical functionality.

    Parameters:
    -----------
    A : numpy.ndarray with dtype=quaternion
        Input quaternion matrix of shape (m, n)
    side : str, optional
        'right' for ker(A), 'left' for ker(A^H)
        Default: 'right'
    rtol : float, optional
        Relative tolerance for determining rank
        Default: 1e-10

    Returns:
    --------
    N : numpy.ndarray with dtype=quaternion
        Kernel matrix such that A @ N ≈ 0 (right) or N^H @ A ≈ 0 (left)
    """
    return quat_null_space(A, side=side, rtol=rtol)
