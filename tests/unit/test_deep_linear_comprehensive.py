import sys
import os

# Add parent directory to path to import core modules
sys.path.append(os.path.join(os.path.dirname(__file__), \'..\', \'..\', \'core\'))
import numpy as np
import quaternion
from solver import DeepLinearNewtonSchulz
from utils import quat_matmat, quat_frobenius_norm

def create_synthetic_deep_linear_data(n_samples=20, input_dim=8, layer_dims=[8, 6, 4, 6, 8]):
    """
    Create synthetic data by generating random weights and applying them to input
    
    Args:
        n_samples: Number of samples
        input_dim: Input dimension
        layer_dims: Layer dimensions [d0, d1, ..., dk]
    
    Returns:
        X: Synthetic data matrix
        true_weights: Original weights used to generate X
    """
    print("Creating synthetic deep linear data...")
    
    # Generate random input data
    X_input_real = np.random.randn(n_samples, input_dim)
    X_input_i = np.random.randn(n_samples, input_dim)
    X_input_j = np.random.randn(n_samples, input_dim)
    X_input_k = np.random.randn(n_samples, input_dim)
    
    X_input_quat = np.stack([X_input_real, X_input_i, X_input_j, X_input_k], axis=-1)
    X_input = quaternion.as_quat_array(X_input_quat)
    
    print(f"Input data shape: {X_input.shape}")
    
    # Generate random weights
    true_weights = []
    for i in range(len(layer_dims)-1):
        # Generate weights with some structure (not completely random)
        W_real = np.random.randn(layer_dims[i], layer_dims[i+1]) * 0.1
        W_i = np.random.randn(layer_dims[i], layer_dims[i+1]) * 0.1
        W_j = np.random.randn(layer_dims[i], layer_dims[i+1]) * 0.1
        W_k = np.random.randn(layer_dims[i], layer_dims[i+1]) * 0.1
        
        # Add some identity-like structure for better conditioning
        if layer_dims[i] == layer_dims[i+1]:
            W_real += np.eye(layer_dims[i]) * 0.5
        
        W_quat = np.stack([W_real, W_i, W_j, W_k], axis=-1)
        true_weights.append(quaternion.as_quat_array(W_quat))
        print(f"True weight {i+1} shape: {true_weights[i].shape}")
    
    # Return the original input data (not the transformed data)
    # The goal is to find weights such that X @ W_1 @ W_2 @ ... @ W_n ≈ I
    print(f"Synthetic data shape: {X_input.shape}")
    print(f"Data norm: {quat_frobenius_norm(X_input):.6f}")
    
    return X_input, true_weights

def test_decomposition_accuracy(X, true_weights, recovered_weights):
    """Test how well we recovered the original weights"""
    print("\n" + "="*50)
    print("DECOMPOSITION ACCURACY TEST")
    print("="*50)
    
    # Test 1: Reconstruction accuracy
    print("\n1. Reconstruction Accuracy:")
    
    # Original reconstruction
    Y_original = X.copy()
    for W in true_weights:
        Y_original = quat_matmat(Y_original, W)
    original_error = quat_frobenius_norm(Y_original - X)
    print(f"   Original reconstruction error: {original_error:.6f}")
    
    # Recovered reconstruction
    Y_recovered = X.copy()
    for W in recovered_weights:
        Y_recovered = quat_matmat(Y_recovered, W)
    recovered_error = quat_frobenius_norm(Y_recovered - X)
    print(f"   Recovered reconstruction error: {recovered_error:.6f}")
    
    # Test 2: Weight similarity
    print("\n2. Weight Similarity:")
    for i, (true_W, recovered_W) in enumerate(zip(true_weights, recovered_weights)):
        weight_diff = quat_frobenius_norm(true_W - recovered_W)
        weight_similarity = 1.0 - (weight_diff / quat_frobenius_norm(true_W))
        print(f"   Layer {i+1}: similarity = {weight_similarity:.4f} (diff = {weight_diff:.6f})")
    
    # Test 3: Overall performance
    print("\n3. Overall Performance:")
    if recovered_error < original_error * 1.1:  # Within 10% of original
        print("   ✅ SUCCESS: Recovered weights provide good reconstruction")
    else:
        print("   ❌ FAILURE: Recovered weights don't provide good reconstruction")
    
    return recovered_error, original_error

def test_deep_linear_decomposition():
    """Comprehensive test of deep linear decomposition"""
    print("="*60)
    print("COMPREHENSIVE DEEP LINEAR DECOMPOSITION TEST")
    print("="*60)
    
    # Test parameters
    n_samples = 15
    input_dim = 8
    layer_dims = [input_dim, 6, 4, 6, n_samples]  # Last dimension = m = n_samples
    
    print(f"Test parameters:")
    print(f"  n_samples: {n_samples}")
    print(f"  input_dim: {input_dim}")
    print(f"  layer_dims: {layer_dims}")
    
    # Step 1: Create synthetic data
    print("\n" + "-"*30)
    print("STEP 1: Creating synthetic data")
    print("-"*30)
    X, true_weights = create_synthetic_deep_linear_data(n_samples, input_dim, layer_dims)
    
    # Step 2: Test decomposition with different parameters
    print("\n" + "-"*30)
    print("STEP 2: Testing decomposition")
    print("-"*30)
    
    # Test with different learning rates
    gammas = [0.5, 0.7, 0.9]
    best_error = float('inf')
    best_weights = None
    best_gamma = None
    
    for gamma in gammas:
        print(f"\nTesting with gamma = {gamma}")
        print("-" * 20)
        
        try:
            solver = DeepLinearNewtonSchulz(gamma=gamma, max_iter=25, verbose=False)
            recovered_weights, residuals, deviations = solver.compute(X, layer_dims)
            
            # Test reconstruction
            Y_test = X.copy()
            for W in recovered_weights:
                Y_test = quat_matmat(Y_test, W)
            test_error = quat_frobenius_norm(Y_test - X)
            
            print(f"Final reconstruction error: {test_error:.6f}")
            
            if test_error < best_error:
                best_error = test_error
                best_weights = recovered_weights
                best_gamma = gamma
                
        except Exception as e:
            print(f"Error with gamma {gamma}: {e}")
            continue
    
    if best_weights is None:
        print("❌ All decomposition attempts failed!")
        return
    
    print(f"\nBest result: gamma = {best_gamma}, error = {best_error:.6f}")
    
    # Step 3: Test decomposition accuracy
    print("\n" + "-"*30)
    print("STEP 3: Testing decomposition accuracy")
    print("-"*30)
    
    test_decomposition_accuracy(X, true_weights, best_weights)
    
    # Step 4: Plot convergence for best result
    print("\n" + "-"*30)
    print("STEP 4: Plotting convergence")
    print("-"*30)
    
    try:
        import matplotlib.pyplot as plt
        
        # Re-run with best gamma and verbose=True to get residuals
        solver = DeepLinearNewtonSchulz(gamma=best_gamma, max_iter=25, verbose=True)
        final_weights, final_residuals, final_deviations = solver.compute(X, layer_dims)
        
        plt.figure(figsize=(12, 8))
        
        # Plot reconstruction error
        plt.subplot(2, 2, 1)
        if 'total_reconstruction' in final_residuals:
            plt.plot(final_residuals['total_reconstruction'], 'b-', linewidth=2)
            plt.xlabel('Iteration')
            plt.ylabel('Reconstruction Error')
            plt.title(f'Convergence (γ={best_gamma})')
            plt.grid(True, alpha=0.3)
        
        # Plot deviations
        plt.subplot(2, 2, 2)
        plt.plot(final_deviations, 'r-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Total Deviation')
        plt.title('Total Deviation')
        plt.grid(True, alpha=0.3)
        
        # Plot weight norms
        plt.subplot(2, 2, 3)
        weight_norms = [quat_frobenius_norm(W) for W in final_weights]
        plt.bar(range(1, len(weight_norms)+1), weight_norms)
        plt.xlabel('Layer')
        plt.ylabel('Weight Norm')
        plt.title('Weight Matrix Norms')
        plt.grid(True, alpha=0.3)
        
        # Plot final reconstruction vs original
        plt.subplot(2, 2, 4)
        Y_final = X.copy()
        for W in final_weights:
            Y_final = quat_matmat(Y_final, W)
        
        # Convert to real for plotting
        X_real = quaternion.as_float_array(X)[:, 0]  # Take real part
        Y_real = quaternion.as_float_array(Y_final)[:, 0]
        
        plt.scatter(X_real.flatten(), Y_real.flatten(), alpha=0.6)
        plt.plot([X_real.min(), X_real.max()], [X_real.min(), X_real.max()], 'r--', linewidth=2)
        plt.xlabel('Original Data')
        plt.ylabel('Reconstructed Data')
        plt.title('Reconstruction Quality')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../../output_figures/deep_linear_comprehensive_test.png', dpi=300, bbox_inches='tight')
        print("Comprehensive test plot saved to: deep_linear_comprehensive_test.png")
        plt.show()
        
    except ImportError:
        print("Matplotlib not available, skipping plots")
    
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)

if __name__ == "__main__":
    test_deep_linear_decomposition() 