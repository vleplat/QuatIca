import sys
import os

# Add parent directory to path to import core modules
sys.path.append(os.path.join(os.path.dirname(__file__), \'..\', \'..\', \'core\'))
import numpy as np
import quaternion
import matplotlib.pyplot as plt
from solver import DeepLinearNewtonSchulz
from utils import quat_matmat, quat_frobenius_norm, quat_eye

def test_initialization_sensitivity():
    """Test the sensitivity of Newton-Schulz to initialization by running multiple trials."""
    
    # Test parameters
    m = 3  # number of samples
    n = 4  # number of features
    layer_dims = [n, 3, m]  # 4 -> 3 -> 3 (last dimension = m)
    n_trials = 10
    
    print(f"Testing initialization sensitivity:")
    print(f"  m (samples): {m}")
    print(f"  n (features): {n}")
    print(f"  layer_dims: {layer_dims}")
    print(f"  n_trials: {n_trials}")
    
    # Store results
    final_errors = []
    convergence_histories = []
    successful_trials = 0
    
    for trial in range(n_trials):
        print(f"\n--- Trial {trial + 1}/{n_trials} ---")
        
        # Set different random seed for each trial
        np.random.seed(trial)
        
        # Create random X quaternion matrix
        X_real = np.random.randn(m, n)
        X_i = np.random.randn(m, n)
        X_j = np.random.randn(m, n)
        X_k = np.random.randn(m, n)
        X_quat = np.stack([X_real, X_i, X_j, X_k], axis=-1)
        X = quaternion.as_quat_array(X_quat)
        
        try:
            # Run Newton-Schulz solver with random initialization
            solver = DeepLinearNewtonSchulz(gamma=0.5, max_iter=20, verbose=True, inner_iterations=10, random_init=True)
            weights, residuals, deviations = solver.compute(X, layer_dims)
            
            # Compute final error
            W = quat_matmat(weights[0], weights[1])
            XW = quat_matmat(X, W)
            I_m = quat_eye(m)
            error = quat_frobenius_norm(XW - I_m)
            
            final_errors.append(error)
            convergence_histories.append(residuals['total_reconstruction'])
            
            print(f"  Final error: {error:.6f}")
            
            if error < 0.01:  # Consider successful if error < 0.01
                successful_trials += 1
                print(f"  ✅ SUCCESS")
            else:
                print(f"  ❌ FAILED")
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            final_errors.append(float('inf'))
            convergence_histories.append([])
    
    # Summary statistics
    print(f"\n=== SUMMARY ===")
    print(f"Successful trials: {successful_trials}/{n_trials} ({100*successful_trials/n_trials:.1f}%)")
    print(f"Final errors: {[f'{e:.6f}' for e in final_errors]}")
    
    # Plot convergence histories
    plt.figure(figsize=(12, 8))
    
    # Plot all convergence curves
    for i, history in enumerate(convergence_histories):
        if len(history) > 0:
            iterations = range(1, len(history) + 1)
            if final_errors[i] < 0.01:
                plt.semilogy(iterations, history, 'g-', alpha=0.3, linewidth=1)
            else:
                plt.semilogy(iterations, history, 'r-', alpha=0.3, linewidth=1)
    
    # Plot mean convergence
    max_len = max(len(h) for h in convergence_histories if len(h) > 0)
    if max_len > 0:
        mean_history = []
        for iter_idx in range(max_len):
            values = [h[iter_idx] for h in convergence_histories if len(h) > iter_idx]
            if values:
                mean_history.append(np.mean(values))
        
        iterations = range(1, len(mean_history) + 1)
        plt.semilogy(iterations, mean_history, 'b-o', linewidth=3, markersize=8, label='Mean convergence')
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Iteration')
    plt.ylabel('Residual Error (log scale)')
    plt.title(f'Newton-Schulz Initialization Sensitivity\n{successful_trials}/{n_trials} successful trials')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../../output_figures/initialization_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Sensitivity plot saved as 'initialization_sensitivity.png'")

if __name__ == "__main__":
    test_initialization_sensitivity() 