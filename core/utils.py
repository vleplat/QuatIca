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