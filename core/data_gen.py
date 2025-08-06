import sys
import os

# Add parent directory to path to import core modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core'))
import numpy as np
import quaternion
from scipy import sparse
from utils import quat_matmat, SparseQuaternionMatrix

# Import QR decomposition from decomp module
from decomp.qsvd import qr_qua


def create_sparse_quat_matrix(m: int, n: int, density: float = 0.1) -> SparseQuaternionMatrix:
    """Create a random sparse quaternion matrix of size m×n using CSR format."""
    real_part = sparse.random(m, n, density=density, format='csr')
    imag_i = sparse.random(m, n, density=density, format='csr')
    imag_j = sparse.random(m, n, density=density, format='csr')
    imag_k = sparse.random(m, n, density=density, format='csr')
    return SparseQuaternionMatrix(real_part, imag_i, imag_j, imag_k, (m, n))


def create_test_matrix(m: int, n: int, rank: int = None, cond_number: float = None) -> np.ndarray:
    """Generate a random dense quaternion matrix with optional rank and conditioning."""
    if rank is None:
        rank = min(m, n)
    # Random real components for A and B factors
    A_real = np.random.randn(m, rank, 4)
    B_real = np.random.randn(rank, n, 4)
    # Apply conditioning if requested
    if cond_number:
        U, _ = np.linalg.qr(np.random.randn(rank, rank))
        V, _ = np.linalg.qr(np.random.randn(rank, rank))
        S = np.logspace(0, np.log10(cond_number), rank)
        B_real = (U @ np.diag(S) @ V.T)[:, :, np.newaxis] * B_real
    # Convert to quaternion arrays
    A_quat = quaternion.as_quat_array(A_real.reshape(-1, 4)).reshape(m, rank)
    B_quat = quaternion.as_quat_array(B_real.reshape(-1, 4)).reshape(rank, n)
    # Return product A * B
    return quat_matmat(A_quat, B_quat)


def generate_random_unitary_matrix(n: int) -> np.ndarray:
    """
    Generate a random unitary quaternion matrix of size n×n.
    
    This function creates a random unitary matrix by:
    1. Generating a random n×n quaternion matrix
    2. Computing its QR decomposition
    3. Returning the Q matrix, which is guaranteed to be unitary
    
    Parameters:
    -----------
    n : int
        Size of the square unitary matrix to generate
        
    Returns:
    --------
    np.ndarray
        An n×n unitary quaternion matrix Q satisfying Q^H * Q = I
        
    Notes:
    ------
    The generated matrix is truly unitary (orthogonal in quaternion space)
    and can be used for various applications including:
    - Random rotations in 4D space
    - Unitary transformations
    - Orthogonal basis generation
    - Testing unitary matrix algorithms
    """
    # Generate a random n×n quaternion matrix
    random_matrix = create_test_matrix(n, n)
    
    # Compute QR decomposition
    Q, R = qr_qua(random_matrix)
    
    # Return the Q matrix (unitary part)
    return Q


def small_test_Mat() -> np.ndarray:
    """
    Create a validation quaternion matrix with known theoretical properties.
    
    This function returns a specific 2×3 quaternion matrix that serves as a
    validation test for our pseudoinverse implementation. The matrix has
    known theoretical properties that allow us to verify the correctness
    of our numerical algorithms.
    
    The matrix corresponds to Example 5.2 from the reference paper:
    
    A = [1  i+2k  3]
        [i  6+j   7]
    
    The theoretical pseudoinverse A^† (computed using Maple package) is:
    ⎛ -47/347 + 21/694 i + 11/694 j - 21/694 k    63/347 - 28/347 i + 21/694 j - 101/694 k ⎞
    ⎜ -11/694 - 347 i - 11/694 k                   61/694 + 21/694 i - 6/347 j + 21/347 k  ⎟
    ⎝  57/347 + 49/694 i + 77/694 k                21/347 - 21/694 i - 33/694 k            ⎠
    
    Returns:
        np.ndarray: A 2×3 quaternion matrix for validation testing
        
    Reference:
        Huang, L., Wang, Q.-W., & Zhang, Y. (2015). The Moore–Penrose inverses 
        of matrices over quaternion polynomial rings. Linear Algebra and its 
        Applications, 475, 45-61. https://doi.org/10.1016/j.laa.2015.02.004
        
        Example 5.2 provides the exact theoretical pseudoinverse for comparison.
        
    Note:
        This matrix is designed to test the accuracy and convergence properties
        of quaternion pseudoinverse algorithms. It should converge quickly and
        achieve high numerical accuracy, with results closely matching the
        theoretical pseudoinverse from the paper.
    """
    A_real = np.zeros((2,3,4))
    A_real[0,0,0]=1
    A_real[0,1,1]=1
    A_real[0,1,3]=2
    A_real[0,2,0]=3
    A_real[1,0,1]=1
    A_real[1,1,0]=6
    A_real[1,1,2]=1
    A_real[1,2,0]=7
    return quaternion.as_quat_array(A_real.reshape(-1, 4)).reshape(2,3)

def theoretical_pseudoinverse_example_5_2() -> np.ndarray:
    """
    Compute the theoretical pseudoinverse from Example 5.2 of the reference paper.
    
    This function returns the exact theoretical pseudoinverse A^† for the matrix
    A = [1 i+2k 3; i 6+j 7] as computed using the Maple package in the paper.
    
    Returns:
        np.ndarray: The 3×2 theoretical pseudoinverse matrix
        
    Note:
        This provides the exact analytical solution for comparison with our
        numerical implementation. The values are computed using the exact
        fractions from the paper.
    """
    # Theoretical values from Example 5.2 (exact fractions)
    # A^† matrix from the paper (3×2 matrix)
    
    # Note: The paper shows the matrix in a specific format
    # We need to be very careful about the exact values and signs
    
    # Based on the paper's Example 5.2, the pseudoinverse should be:
    # Row 1: [47/347 + 21/694 i + 11/694 j + 0  k, -21/694  - 11/347 i +  0 j -  11/694 k]
    # Row 2: [-63/347 - 28/347 i + 21/694 j -101/694k, 61/694 + 21/694 i - 6/347 j + 21/347 k]  (corrected)
    # Row 3: [57/347 + 49/694 i + 77/694 k, 21/347 - 21/694 i - 33/694 k]
    
    A_theoretical = np.array([
        # Row 1
        [47/347, 21/694, 11/694, 0],  # 47/347 + 21/694 i + 11/694 j + 0  k
        [-21/694, -11/347, 0, -11/694], # -21/694  - 11/347 i +  0 j -  11/694 k
        
        # Row 2  
        [-63/347, -28/347, 21/694, -101/694],            # -63/347 - 28/347 i + 21/694 j -101/694k(corrected from paper typo)
        [61/694, 21/694, -6/347, 21/347],    # 61/694 + 21/694 i - 6/347 j + 21/347 k
        
        # Row 3
        [57/347, 49/694, 0, 77/694],         # 57/347 + 49/694 i + 77/694 k
        [21/347, -21/694, 0, -33/694]        # 21/347 - 21/694 i - 33/694 k
    ]).reshape(3, 2, 4)
    
    return quaternion.as_quat_array(A_theoretical.reshape(-1, 4)).reshape(3, 2)

