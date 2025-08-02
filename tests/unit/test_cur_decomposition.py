import sys
import os

# Add parent directory to path to import core modules
sys.path.append(os.path.join(os.path.dirname(__file__), \'..\', \'..\', \'core\'))
import numpy as np
import quaternion
from utils import quat_matmat, quat_frobenius_norm
from solver import NewtonSchulzPseudoinverse

def test_cur_decomposition():
    """Test CUR decomposition with a simple quaternion matrix"""
    print("Testing CUR Decomposition with Quaternion Matrix")
    print("=" * 50)
    
    # Create a simple 4x4 quaternion matrix
    size = 4
    # Create a rank-deficient matrix
    A = np.random.randn(size, size)
    B = np.random.randn(size, size)
    # Make it rank-deficient
    rank = size // 2
    U, S, Vt = np.linalg.svd(A @ B)
    S[rank:] = 0
    matrix = U @ np.diag(S) @ Vt
    
    # Convert to quaternion matrix (pure quaternions)
    quat_data = np.stack([np.zeros((size, size)), matrix, matrix*0.5, matrix*0.3], axis=-1)
    X = quaternion.as_quat_array(quat_data)
    
    print(f"Original matrix shape: {X.shape}")
    print(f"Original matrix norm: {quat_frobenius_norm(X):.6f}")
    
    # Test CUR decomposition
    rank_col = 2
    rank_row = 2
    
    # Randomly sample columns and rows
    col_indices = np.random.choice(size, rank_col, replace=False)
    row_indices = np.random.choice(size, rank_row, replace=False)
    
    print(f"Column indices: {col_indices}")
    print(f"Row indices: {row_indices}")
    
    # Extract column and row samples
    C = X[:, col_indices]  # size x rank_col
    R = X[row_indices, :]  # rank_row x size
    
    print(f"C shape: {C.shape}")
    print(f"R shape: {R.shape}")
    
    # Use our existing solver for pseudoinverses
    solver = NewtonSchulzPseudoinverse(gamma=1.0, max_iter=10, tol=1e-6, verbose=False)
    
    print("Computing C pseudoinverse...")
    C_pinv, _, _ = solver.compute(C)  # rank_col x size
    print(f"C_pinv shape: {C_pinv.shape}")
    
    print("Computing R pseudoinverse...")
    R_pinv, _, _ = solver.compute(R)  # size x rank_row
    print(f"R_pinv shape: {R_pinv.shape}")
    
    # Compute middle matrix U = C^† X R^†
    print("Computing U matrix...")
    U = quat_matmat(quat_matmat(C_pinv, X), R_pinv)  # rank_col x rank_row
    print(f"U shape: {U.shape}")
    
    # Compute CUR approximation
    print("Computing CUR approximation...")
    X_approx = quat_matmat(quat_matmat(C, U), R)
    print(f"X_approx shape: {X_approx.shape}")
    
    # Compute error
    error = quat_frobenius_norm(X - X_approx)
    print(f"Reconstruction error: {error:.6f}")
    
    # Test with different ranks
    print("\nTesting different ranks:")
    for r in [1, 2, 3]:
        if r <= size:
            col_indices = np.random.choice(size, r, replace=False)
            row_indices = np.random.choice(size, r, replace=False)
            
            C = X[:, col_indices]
            R = X[row_indices, :]
            
            C_pinv, _, _ = solver.compute(C)
            R_pinv, _, _ = solver.compute(R)
            
            U = quat_matmat(quat_matmat(C_pinv, X), R_pinv)
            X_approx = quat_matmat(quat_matmat(C, U), R)
            
            error = quat_frobenius_norm(X - X_approx)
            print(f"Rank {r}x{r}: error = {error:.6f}")
    
    print("\nCUR decomposition test completed successfully!")

if __name__ == "__main__":
    test_cur_decomposition() 