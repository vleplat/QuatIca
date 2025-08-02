import sys
import os

# Add parent directory to path to import core modules
sys.path.append(os.path.join(os.path.dirname(__file__), \'..\', \'..\', \'core\'))
import numpy as np
import quaternion
from utils import quat_matmat, quat_frobenius_norm, quat_eye

def debug_residual():
    """Debug the residual computation"""
    
    # Simple test case
    m = 3
    n = 4
    
    # Create random X
    X_real = np.random.randn(m, n)
    X_i = np.random.randn(m, n)
    X_j = np.random.randn(m, n)
    X_k = np.random.randn(m, n)
    X_quat = np.stack([X_real, X_i, X_j, X_k], axis=-1)
    X = quaternion.as_quat_array(X_quat)
    
    print(f"X shape: {X.shape}")
    
    # Create random weights
    W1_real = np.random.randn(n, 3) * 0.1
    W1_i = np.random.randn(n, 3) * 0.1
    W1_j = np.random.randn(n, 3) * 0.1
    W1_k = np.random.randn(n, 3) * 0.1
    W1_quat = np.stack([W1_real, W1_i, W1_j, W1_k], axis=-1)
    W1 = quaternion.as_quat_array(W1_quat)
    
    W2_real = np.random.randn(3, m) * 0.1
    W2_i = np.random.randn(3, m) * 0.1
    W2_j = np.random.randn(3, m) * 0.1
    W2_k = np.random.randn(3, m) * 0.1
    W2_quat = np.stack([W2_real, W2_i, W2_j, W2_k], axis=-1)
    W2 = quaternion.as_quat_array(W2_quat)
    
    print(f"W1 shape: {W1.shape}")
    print(f"W2 shape: {W2.shape}")
    
    # Compute X @ W1 @ W2
    temp = quat_matmat(X, W1)
    print(f"X @ W1 shape: {temp.shape}")
    XW = quat_matmat(temp, W2)
    print(f"X @ W1 @ W2 shape: {XW.shape}")
    
    # Create identity matrix
    I = quat_eye(m)
    print(f"I shape: {I.shape}")
    
    # Compute residual
    residual = XW - I
    residual_norm = quat_frobenius_norm(residual)
    
    print(f"Residual norm: {residual_norm:.6f}")
    print(f"XW norm: {quat_frobenius_norm(XW):.6f}")
    print(f"I norm: {quat_frobenius_norm(I):.6f}")

if __name__ == "__main__":
    debug_residual() 