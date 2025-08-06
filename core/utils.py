import numpy as np
import quaternion
from scipy import sparse

# Optimized quaternion matrix operations

def quat_matmat(A, B):
    """Multiply two quaternion matrices (supports dense × dense, sparse × dense, dense × sparse, sparse × sparse)."""
    if isinstance(A, SparseQuaternionMatrix):
        return A @ B
    elif isinstance(B, SparseQuaternionMatrix):
        # For dense @ sparse, use left_multiply
        return B.left_multiply(A)
    else:
        # Both dense
        A_comp = quaternion.as_float_array(A)
        B_comp = quaternion.as_float_array(B)
        Aw, Ax, Ay, Az = A_comp[...,0], A_comp[...,1], A_comp[...,2], A_comp[...,3]
        Bw, Bx, By, Bz = B_comp[...,0], B_comp[...,1], B_comp[...,2], B_comp[...,3]
        Cw = Aw @ Bw - Ax @ Bx - Ay @ By - Az @ Bz
        Cx = Aw @ Bx + Ax @ Bw + Ay @ Bz - Az @ By
        Cy = Aw @ By - Ax @ Bz + Ay @ Bw + Az @ Bx
        Cz = Aw @ Bz + Ax @ By - Ay @ Bx + Az @ Bw
        C = np.stack([Cw, Cx, Cy, Cz], axis=-1)
        return quaternion.as_quat_array(C)


def quat_frobenius_norm(A: np.ndarray) -> float:
    """Compute the Frobenius norm of a quaternion matrix (dense or sparse)."""
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
    """Return the conjugate transpose (Hermitian) of quaternion matrix A (dense or sparse)."""
    if isinstance(A, SparseQuaternionMatrix):
        return A.conjugate().transpose()
    else:
        return np.transpose(np.conjugate(A))


def quat_eye(n: int) -> np.ndarray:
    """Create an n×n identity quaternion matrix."""
    I = np.zeros((n, n), dtype=np.quaternion)
    np.fill_diagonal(I, quaternion.quaternion(1, 0, 0, 0))
    return I

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
        R = np.zeros((4*m, 4*n))
        
        for i in range(m):
            for j in range(n):
                w, x, y, z = Q_array[i, j]
                # Block position
                bi, bj = 4*i, 4*j
                # Fill the 4x4 block
                R[bi:bi+4, bj:bj+4] = np.array([
                    [ w, -x, -y, -z],
                    [ x,  w, -z,  y],
                    [ y,  z,  w, -x],
                    [ z, -y,  x,  w]
                ])
        return R
    else:
        raise ValueError("Input must be a quaternion array")

def real_contract(R, m, n):
    """
    Convert real block matrix R back to quaternion matrix.
    Invert real_expand back into an m×n quaternion array
    """
    if R.shape != (4*m, 4*n):
        raise ValueError(f"Expected shape (4*{m}, 4*{n}), got {R.shape}")
    
    Q_array = np.zeros((m, n, 4))
    
    for i in range(m):
        for j in range(n):
            bi, bj = 4*i, 4*j
            block = R[bi:bi+4, bj:bj+4]
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
        real_out = (self.real @ B_real - self.i @ B_i - self.j @ B_j - self.k @ B_k)
        i_out = (self.real @ B_i + self.i @ B_real + self.j @ B_k - self.k @ B_j)
        j_out = (self.real @ B_j - self.i @ B_k + self.j @ B_real + self.k @ B_i)
        k_out = (self.real @ B_k + self.i @ B_j - self.j @ B_i + self.k @ B_real)
        result = np.stack([real_out, i_out, j_out, k_out], axis=-1)
        return quaternion.as_quat_array(result)

    def sparse_multiply(self, other):
        real_part = (self.real @ other.real - self.i @ other.i - self.j @ other.j - self.k @ other.k)
        i_part = (self.real @ other.i + self.i @ other.real + self.j @ other.k - self.k @ other.j)
        j_part = (self.real @ other.j - self.i @ other.k + self.j @ other.real + self.k @ other.i)
        k_part = (self.real @ other.k + self.i @ other.j - self.j @ other.i + self.k @ other.real)
        return SparseQuaternionMatrix(real_part, i_part, j_part, k_part, (self.shape[0], other.shape[1]))

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
            self.shape
        )
    def transpose(self):
        return SparseQuaternionMatrix(
            self.real.transpose(),
            self.i.transpose(),
            self.j.transpose(),
            self.k.transpose(),
            (self.shape[1], self.shape[0])
        )

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float, np.floating)):
            return SparseQuaternionMatrix(
                self.real * scalar,
                self.i * scalar,
                self.j * scalar,
                self.k * scalar,
                self.shape
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
        if hasattr(A0, 'toarray'):
            # Handle sparse matrices
            A0_arr, A2_arr, A1_arr, A3_arr = A0.toarray(), A2.toarray(), A1.toarray(), A3.toarray()
        else:
            A0_arr, A2_arr, A1_arr, A3_arr = A0, A2, A1, A3
        
        # Stack matrices horizontally (concatenate columns) as in MATLAB
        stacked = np.hstack([A0_arr, A2_arr, A1_arr, A3_arr])
        
        # Use appropriate norm based on dimensionality
        if stacked.ndim == 1:
            return np.linalg.norm(stacked, 2)  # 2-norm for vectors
        else:
            return np.linalg.norm(stacked, 'fro')  # Frobenius norm for matrices
    
    elif opt == 'd':
        # Dual norm: norm([2*A1-A2-A3, 2*A2-A3-A1, 2*A3-A1-A2], 'fro') / 3
        if hasattr(A0, 'toarray'):
            A0_arr, A1_arr, A2_arr, A3_arr = A0.toarray(), A1.toarray(), A2.toarray(), A3.toarray()
        else:
            A0_arr, A1_arr, A2_arr, A3_arr = A0, A1, A2, A3
        
        term1 = 2*A1_arr - A2_arr - A3_arr
        term2 = 2*A2_arr - A3_arr - A1_arr
        term3 = 2*A3_arr - A1_arr - A2_arr
        
        stacked = np.hstack([term1, term2, term3])
        if stacked.ndim == 1:
            return np.linalg.norm(stacked, 2) / 3
        else:
            return np.linalg.norm(stacked, 'fro') / 3
    
    elif opt == '2':
        # 2-norm: largest singular value
        if hasattr(A0, 'toarray'):
            A0_arr, A2_arr, A1_arr, A3_arr = A0.toarray(), A2.toarray(), A1.toarray(), A3.toarray()
        else:
            A0_arr, A2_arr, A1_arr, A3_arr = A0, A2, A1, A3
        
        stacked = np.hstack([A0_arr, A2_arr, A1_arr, A3_arr])
        _, s, _ = np.linalg.svd(stacked, full_matrices=False)
        return np.max(s)
    
    elif opt == '1':
        # 1-norm of element-wise square root
        if hasattr(A0, 'toarray'):
            A0_arr, A1_arr, A2_arr, A3_arr = A0.toarray(), A1.toarray(), A2.toarray(), A3.toarray()
        else:
            A0_arr, A1_arr, A2_arr, A3_arr = A0, A1, A2, A3
        
        sqrt_sum = np.sqrt(np.abs(A0_arr) + np.abs(A1_arr) + np.abs(A2_arr) + np.abs(A3_arr))
        return np.linalg.norm(sqrt_sum, 1)
    
    else:
        # Other norms: apply to element-wise square root
        if hasattr(A0, 'toarray'):
            A0_arr, A1_arr, A2_arr, A3_arr = A0.toarray(), A1.toarray(), A2.toarray(), A3.toarray()
        else:
            A0_arr, A1_arr, A2_arr, A3_arr = A0, A1, A2, A3
        
        sqrt_sum = np.sqrt(np.abs(A0_arr) + np.abs(A1_arr) + np.abs(A2_arr) + np.abs(A3_arr))
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
    if hasattr(B0, 'toarray'):
        B0_arr, B1_arr, B2_arr, B3_arr = B0.toarray(), B1.toarray(), B2.toarray(), B3.toarray()
    else:
        B0_arr, B1_arr, B2_arr, B3_arr = B0, B1, B2, B3
    
    if hasattr(C0, 'toarray'):
        C0_arr, C1_arr, C2_arr, C3_arr = C0.toarray(), C1.toarray(), C2.toarray(), C3.toarray()
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
    A0 = safe_multiply(B0_arr, C0_arr) - safe_multiply(B2_arr, C2_arr) - safe_multiply(B1_arr, C1_arr) - safe_multiply(B3_arr, C3_arr)
    
    # A2 = B0*C2 + B2*C0 - B1*C3 + B3*C1
    A2 = safe_multiply(B0_arr, C2_arr) + safe_multiply(B2_arr, C0_arr) - safe_multiply(B1_arr, C3_arr) + safe_multiply(B3_arr, C1_arr)
    
    # A1 = B0*C1 + B2*C3 + B1*C0 - B3*C2
    A1 = safe_multiply(B0_arr, C1_arr) + safe_multiply(B2_arr, C3_arr) + safe_multiply(B1_arr, C0_arr) - safe_multiply(B3_arr, C2_arr)
    
    # A3 = B0*C3 - B2*C1 + B1*C2 + B3*C0
    A3 = safe_multiply(B0_arr, C3_arr) - safe_multiply(B2_arr, C1_arr) + safe_multiply(B1_arr, C2_arr) + safe_multiply(B3_arr, C0_arr)
    
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
    A2 = A[:, n:2*n]
    A1 = A[:, 2*n:3*n]
    A3 = A[:, 3*n:4*n]
    
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
    
    elif opt == 'd':
        # Dual norm: convert to component format and use normQsparse
        A_comp = quaternion.as_float_array(A)
        A0, A1, A2, A3 = A_comp[..., 0], A_comp[..., 1], A_comp[..., 2], A_comp[..., 3]
        return normQsparse(A0, A1, A2, A3, 'd')
    
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
        AR = np.array([
            [A1, -A2, -A3, -A4],
            [A2,  A1, -A4,  A3],
            [A3,  A4,  A1, -A2],
            [A4, -A3,  A2,  A1]
        ])
        return AR
    else:
        # Matrix case: create block matrix
        m, n = A1.shape
        
        # Create the real block matrix
        AR = np.zeros((4*m, 4*n))
        
        # Fill the 4x4 blocks
        AR[0:m, 0:n] = A1
        AR[0:m, n:2*n] = -A2
        AR[0:m, 2*n:3*n] = -A3
        AR[0:m, 3*n:4*n] = -A4
        
        AR[m:2*m, 0:n] = A2
        AR[m:2*m, n:2*n] = A1
        AR[m:2*m, 2*n:3*n] = -A4
        AR[m:2*m, 3*n:4*n] = A3
        
        AR[2*m:3*m, 0:n] = A3
        AR[2*m:3*m, n:2*n] = A4
        AR[2*m:3*m, 2*n:3*n] = A1
        AR[2*m:3*m, 3*n:4*n] = -A2
        
        AR[3*m:4*m, 0:n] = A4
        AR[3*m:4*m, n:2*n] = -A3
        AR[3*m:4*m, 2*n:3*n] = A2
        AR[3*m:4*m, 3*n:4*n] = A1
        
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
            return Realp(g1/r, g2/r, g3/r, g4/r)


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
    W = np.zeros((m, 4*m))
    W[:, 0:m] = np.eye(m)
    
    # Apply Givens rotations for subdiagonal elements
    for s in range(m-1):
        # Extract quaternion vectors for current position
        x1 = np.array([Hess[s, s], Hess[m+s, s], Hess[2*m+s, s], Hess[3*m+s, s]])
        x2 = np.array([Hess[s+1, s], Hess[m+s+1, s], Hess[2*m+s+1, s], Hess[3*m+s+1, s]])
        
        # Generate Givens rotation
        G = ggivens(x1, x2)
        
        # Apply rotation to Hess matrix
        # Select 8 rows: [s:s+1, s+m:s+m+1, s+2*m:s+2*m+1, s+3*m:s+3*m+1]
        rows_to_update = [s, s+1, s+m, s+m+1, s+2*m, s+2*m+1, s+3*m, s+3*m+1]
        Hess[rows_to_update, s:n] = G.T @ Hess[rows_to_update, s:n]
        
        # Update W matrix
        # Select 8 columns: [s:s+1, s+m:s+m+1, s+2*m:s+2*m+1, s+3*m:s+3*m+1]
        cols_to_update = [s, s+1, s+m, s+m+1, s+2*m, s+2*m+1, s+3*m, s+3*m+1]
        W[:, cols_to_update] = W[:, cols_to_update] @ G
    
    # Handle the last column (special case)
    # s = m
    G = GRSGivens(Hess[m-1, n-1], Hess[2*m-1, n-1], Hess[3*m-1, n-1], Hess[4*m-1, n-1])
    W[:, [m-1, 2*m-1, 3*m-1, 4*m-1]] = W[:, [m-1, 2*m-1, 3*m-1, 4*m-1]] @ G
    Hess[[m-1, 2*m-1, 3*m-1, 4*m-1], n-1] = G.T @ Hess[[m-1, 2*m-1, 3*m-1, 4*m-1], n-1]
    
    # Final reshaping to match MATLAB output format
    # W = [W(1:m,1:m), -W(1:m,2*m+1:3*m), -W(1:m,m+1:2*m), -W(1:m,3*m+1:4*m)]
    W_final = np.zeros((m, 4*m))
    W_final[:, 0:m] = W[:, 0:m]
    W_final[:, m:2*m] = -W[:, 2*m:3*m]
    W_final[:, 2*m:3*m] = -W[:, m:2*m]
    W_final[:, 3*m:4*m] = -W[:, 3*m:4*m]
    
    # Hess = [Hess(1:m,1:n), Hess(2*m+1:3*m,1:n), Hess(m+1:2*m,1:n), Hess(3*m+1:4*m,1:n)]
    Hess_final = np.zeros((m, 4*n))
    Hess_final[:, 0:n] = Hess[0:m, 0:n]
    Hess_final[:, n:2*n] = Hess[2*m:3*m, 0:n]
    Hess_final[:, 2*n:3*n] = Hess[m:2*m, 0:n]
    Hess_final[:, 3*n:4*n] = Hess[3*m:4*m, 0:n]
    
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
        Unn_0, Unn_1, Unn_2, Unn_3 = R0[n-1, n-1], R1[n-1, n-1], R2[n-1, n-1], R3[n-1, n-1]
        
        # Check if the diagonal element is non-zero
        r, _, _, _, _ = absQsparse(Unn_0, Unn_1, Unn_2, Unn_3)
        if r > tol:
            # Solve for the last element
            delta0, delta1, delta2, delta3 = dotinvQsparse(Unn_0, Unn_1, Unn_2, Unn_3)
            
            # Convert scalars to arrays if needed
            if np.isscalar(delta0):
                delta0, delta1, delta2, delta3 = np.array([delta0]), np.array([delta1]), np.array([delta2]), np.array([delta3])
            
            b0[n-1, :], b1[n-1, :], b2[n-1, :], b3[n-1, :] = timesQsparse(
                delta0, delta1, delta2, delta3,
                b0[n-1, :], b1[n-1, :], b2[n-1, :], b3[n-1, :]
            )
            
            # Backward substitution for remaining elements
            for i in range(n-2, -1, -1):
                # Get diagonal element
                Uii_0, Uii_1, Uii_2, Uii_3 = R0[i, i], R1[i, i], R2[i, i], R3[i, i]
                
                # Check if the diagonal element is non-zero
                r, _, _, _, _ = absQsparse(Uii_0, Uii_1, Uii_2, Uii_3)
                if r > tol:
                    # Compute R(i, i+1:n) * b(i+1:n, :)
                    delta0, delta1, delta2, delta3 = timesQsparse(
                        R0[i, i+1:n], R1[i, i+1:n], R2[i, i+1:n], R3[i, i+1:n],
                        b0[i+1:n, :], b1[i+1:n, :], b2[i+1:n, :], b3[i+1:n, :]
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
                        beta0, beta1, beta2, beta3 = np.array([beta0]), np.array([beta1]), np.array([beta2]), np.array([beta3])
                    
                    b0[i, :], b1[i, :], b2[i, :], b3[i, :] = timesQsparse(
                        beta0, beta1, beta2, beta3,
                        delta0, delta1, delta2, delta3
                    )
                else:
                    print(f'U0({i},{i})=0 ! No solution but least square solution!')
                    b0[i, :] = 0
                    b1[i, :] = 0
                    b2[i, :] = 0
                    b3[i, :] = 0
        else:
            print('U0(n,n)=0 ! No solution but least square solution!')
            b0[n-1, :] = 0
            b1[n-1, :] = 0
            b2[n-1, :] = 0
            b3[n-1, :] = 0
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
        raise ValueError('Cannot test whether a non-square matrix is Hermitian.')
    
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
        raise ValueError('Matrix must be square.')
    
    if d in ['Dieudonné', 'Dieudonne']:
        # Dieudonné determinant: product of singular values
        from decomp.qsvd import classical_qsvd_full
        _, s, _ = classical_qsvd_full(X)
        return np.prod(s)
    
    elif d == 'Study':
        # Study determinant: determinant of the adjoint
        # For now, we'll use a simplified approach
        # TODO: Implement proper adjoint computation
        raise NotImplementedError('Study determinant not yet implemented')
    
    elif d == 'Moore':
        # Moore determinant: product of eigenvalues (requires Hermitian)
        if not ishermitian(X):
            raise ValueError('Cannot compute Moore determinant of a non-Hermitian matrix.')
        
        from decomp import quaternion_eigenvalues
        eigenvalues = quaternion_eigenvalues(X)
        return np.prod(eigenvalues)
    
    else:
        raise ValueError(f'Unrecognized determinant type: {d}')


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