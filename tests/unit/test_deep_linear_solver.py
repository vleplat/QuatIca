import sys
import os

# Add parent directory to path to import core modules
sys.path.append(os.path.join(os.path.dirname(__file__), \'..\', \'..\', \'core\'))
import numpy as np
import quaternion
from solver import DeepLinearNewtonSchulz
from utils import quat_matmat, quat_frobenius_norm

def test_deep_linear_solver():
    """Test the deep linear Newton-Schulz solver"""
    print("Testing Deep Linear Newton-Schulz Solver...")
    
    # Create a simple test case
    n_samples = 10
    input_dim = 8
    
    # Create random quaternion input matrix
    X_real = np.random.randn(n_samples, input_dim)
    X_i = np.random.randn(n_samples, input_dim)
    X_j = np.random.randn(n_samples, input_dim)
    X_k = np.random.randn(n_samples, input_dim)
    
    X_quat = np.stack([X_real, X_i, X_j, X_k], axis=-1)
    X = quaternion.as_quat_array(X_quat)
    
    print(f"Input matrix shape: {X.shape}")
    
    # Define layer dimensions (autoencoder-like)
    layers = [input_dim, 6, 4, 6, input_dim]  # 8 -> 6 -> 4 -> 6 -> 8
    print(f"Layer dimensions: {layers}")
    
    # Initialize and run the solver
    solver = DeepLinearNewtonSchulz(gamma=0.9, max_iter=30, verbose=True)
    weights, residuals, deviations = solver.compute(X, layers)
    
    print(f"\nOptimized {len(weights)} layers:")
    for i, W in enumerate(weights):
        print(f"Layer {i+1}: {W.shape}")
    
    # Test reconstruction
    print("\nTesting reconstruction...")
    Y = X.copy()
    for W in weights:
        Y = quat_matmat(Y, W)
    
    # Compute reconstruction error
    error = quat_frobenius_norm(Y - X)
    print(f"Final reconstruction error: {error:.6f}")
    
    # Plot convergence if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot total deviations
        ax1.plot(deviations, 'b-', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Total Deviation')
        ax1.set_title('Convergence: Total Deviation')
        ax1.grid(True, alpha=0.3)
        
        # Plot reconstruction errors
        if 'total_reconstruction' in residuals:
            ax2.plot(residuals['total_reconstruction'], 'r-', linewidth=2)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Reconstruction Error')
            ax2.set_title('Convergence: Reconstruction Error')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../../output_figures/deep_linear_convergence.png', dpi=300, bbox_inches='tight')
        print("Convergence plot saved to: deep_linear_convergence.png")
        plt.show()
        
    except ImportError:
        print("Matplotlib not available, skipping plots")
    
    return weights, residuals, deviations

if __name__ == "__main__":
    test_deep_linear_solver() 