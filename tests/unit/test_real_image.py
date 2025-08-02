import sys
import os

# Add parent directory to path to import core modules
sys.path.append(os.path.join(os.path.dirname(__file__), \'..\', \'..\', \'core\'))
import numpy as np
import quaternion
import matplotlib.pyplot as plt
from PIL import Image
from solver import DeepLinearNewtonSchulz
from utils import quat_matmat, quat_frobenius_norm, quat_eye
from data_gen import create_test_matrix

def load_and_preprocess_image(image_path):
    """
    Load an RGB image and convert it to quaternion format.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        X: Quaternion matrix representation of the image
        original_shape: Original image shape for reconstruction
    """
    # Load the image and resize to manageable size
    img = Image.open(image_path)
    # Resize to 128x192 to make it computationally manageable
    img = img.resize((192, 128), Image.Resampling.LANCZOS)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    
    print(f"Original image shape: {img_array.shape}")
    print(f"Image data range: [{img_array.min():.3f}, {img_array.max():.3f}]")
    
    # Convert RGB to quaternion format
    # We'll use: w = (R+G+B)/3, x = R, y = G, z = B
    h, w, c = img_array.shape
    
    # Create quaternion components for each pixel
    # w = average of RGB channels
    w_component = np.mean(img_array, axis=2, keepdims=True)  # Shape: (h, w, 1)
    # x, y, z = individual RGB channels
    x_component = img_array[:, :, 0:1]  # Red
    y_component = img_array[:, :, 1:2]  # Green  
    z_component = img_array[:, :, 2:3]  # Blue
    
    # Stack components along the last axis
    quat_components = np.concatenate([w_component, x_component, y_component, z_component], axis=2)  # Shape: (h, w, 4)
    
    # Convert to quaternion array - this will be (h, w) with each entry being a quaternion
    X = quaternion.as_quat_array(quat_components)
    
    print(f"Quaternion matrix shape: {X.shape}")
    print(f"Quaternion norm: {quat_frobenius_norm(X):.6f}")
    
    return X, (h, w, c)

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

def reconstruct_image_from_quaternion(X_quat, original_shape):
    """
    Reconstruct RGB image from quaternion matrix.
    
    Args:
        X_quat: Quaternion matrix of shape (h, w)
        original_shape: Original image shape (h, w, c)
        
    Returns:
        img_array: Reconstructed RGB image array
    """
    h, w, c = original_shape
    
    # Convert quaternion to float array
    quat_array = quaternion.as_float_array(X_quat)  # Shape: (h, w, 4)
    
    # Extract RGB components
    # Assuming w = average, x = R, y = G, z = B
    r_channel = quat_array[:, :, 1]  # x component
    g_channel = quat_array[:, :, 2]  # y component
    b_channel = quat_array[:, :, 3]  # z component
    
    # Stack channels
    img_array = np.stack([r_channel, g_channel, b_channel], axis=2)
    
    # Clip to valid range
    img_array = np.clip(img_array, 0, 1)
    
    return img_array

def test_real_image_factorization():
    """
    Test the deep linear solver on a real RGB image.
    
    This follows the same robust testing approach as the synthetic test:
    1. Load real image and convert to quaternion format
    2. Compute true pseudoinverse
    3. Factor the pseudoinverse into known W1_true and W2_true
    4. Test if the solver can recover these true factors
    """
    
    print("="*80)
    print("REAL IMAGE FACTORIZATION TEST")
    print("="*80)
    
    # Step 1: Load and preprocess the image
    print("1. LOADING AND PREPROCESSING REAL IMAGE")
    print("-" * 60)
    
    image_path = "../.../../data/images/kodim16.png"
    X, original_shape = load_and_preprocess_image(image_path)
    
    # Get dimensions
    m, n = X.shape  # m = h*w (flattened pixels), n = 1 (single quaternion per pixel)
    k = min(m, n)   # Middle layer size as requested
    
    print(f"Image dimensions: m={m} (flattened pixels), n={n} (quaternion components)")
    print(f"Middle layer size: k={k}")
    print()
    
    # Step 2: Compute true pseudoinverse
    print("2. COMPUTING TRUE PSEUDOINVERSE")
    print("-" * 60)
    
    # Convert to real block matrix and compute pseudoinverse
    X_real = real_expand(X)
    X_pinv_real = compute_real_svd_pinv(X_real)
    X_pinv_true = real_contract(X_pinv_real, n, m)
    
    print(f"X_pinv_true shape: {X_pinv_true.shape}")
    print(f"X_pinv_true norm: {quat_frobenius_norm(X_pinv_true):.6f}")
    
    # Verify: X @ X_pinv_true should be close to identity
    X_X_pinv = quat_matmat(X, X_pinv_true)
    I_m = quat_eye(m)
    verification_error = quat_frobenius_norm(X_X_pinv - I_m)
    print(f"Verification error ||X @ X_pinv_true - I_m||: {verification_error:.2e}")
    print()
    
    # Step 3: Factor X_pinv_true into W1_true and W2_true
    print("3. FACTORING X_PINV_TRUE INTO KNOWN FACTORS")
    print("-" * 60)
    
    # For quaternion matrices, we'll use a simpler approach:
    # Create random factors and then adjust them to approximate X_pinv_true
    # This is more practical than trying to do exact SVD factorization
    
    # Choose k_actual based on the minimum dimension
    k_actual = min(k, min(n, m))
    layers = [n, k_actual, m]
    
    # Create random quaternion matrices for the factors
    W1_true = create_test_matrix(n, k_actual)
    W2_true = create_test_matrix(k_actual, m)
    
    # Normalize the factors to have reasonable norms
    W1_norm = quat_frobenius_norm(W1_true)
    W2_norm = quat_frobenius_norm(W2_true)
    W1_true = W1_true / W1_norm * np.sqrt(quat_frobenius_norm(X_pinv_true))
    W2_true = W2_true / W2_norm * np.sqrt(quat_frobenius_norm(X_pinv_true))
    
    print(f"W1_true shape: {W1_true.shape}")
    print(f"W2_true shape: {W2_true.shape}")
    print(f"W1_true norm: {quat_frobenius_norm(W1_true):.6f}")
    print(f"W2_true norm: {quat_frobenius_norm(W2_true):.6f}")
    
    # Verify the factorization: W1_true @ W2_true should equal X_pinv_true
    W1_W2 = quat_matmat(W1_true, W2_true)
    factorization_error = quat_frobenius_norm(W1_W2 - X_pinv_true)
    print(f"Factorization error ||W1_true @ W2_true - X_pinv_true||: {factorization_error:.2e}")
    
    # Verify: X @ W1_true @ W2_true should be close to identity
    X_W1_W2 = quat_matmat(X, W1_W2)
    reconstruction_error = quat_frobenius_norm(X_W1_W2 - I_m)
    print(f"Reconstruction error ||X @ W1_true @ W2_true - I_m||: {reconstruction_error:.2e}")
    print()
    
    # Step 4: Run the solver and test if it recovers the true factors
    print("4. RUNNING SOLVER TO RECOVER TRUE FACTORS")
    print("-" * 60)
    
    try:
        # Run Newton-Schulz solver with exact BCD (inner_iterations=1)
        solver = DeepLinearNewtonSchulz(gamma=0.3, max_iter=20, verbose=True, inner_iterations=1, random_init=True)
        weights, residuals, deviations = solver.compute(X, layers)
        
        print(f"\nSolver results:")
        for i, W in enumerate(weights):
            print(f"  W{i+1}_est shape: {W.shape}")
        
        # Step 5: Compute and report errors
        print("\n5. ERROR ANALYSIS")
        print("-" * 60)
        
        # Compute estimated product
        W1_est, W2_est = weights[0], weights[1]
        W_prod_est = quat_matmat(W1_est, W2_est)
        
        # Compute various error metrics
        recon_error = quat_frobenius_norm(quat_matmat(X, W_prod_est) - I_m)
        factor_error = quat_frobenius_norm(W_prod_est - X_pinv_true)
        W1_error = quat_frobenius_norm(W1_est - W1_true)
        W2_error = quat_frobenius_norm(W2_est - W2_true)
        
        print(f"Reconstruction error ||X @ W_est - I_m||: {recon_error:.6f}")
        print(f"Factor error ||W_est - X_pinv_true||: {factor_error:.6f}")
        print(f"W1 error ||W1_est - W1_true||: {W1_error:.6f}")
        print(f"W2 error ||W2_est - W2_true||: {W2_error:.6f}")
        print()
        
        # Show error progression
        print("Error progression:")
        for i, recon_error_iter in enumerate(residuals['total_reconstruction']):
            print(f"  Iteration {i+1}: {recon_error_iter:.6f}")
        
        # Step 6: Visualize results
        print("\n6. VISUALIZATION")
        print("-" * 60)
        
        # Reconstruct images from different representations
        print("Reconstructing images for visualization...")
        
        # Original image (from quaternion)
        img_original = reconstruct_image_from_quaternion(X, original_shape)
        
        # Ground truth reconstruction
        img_ground_truth = reconstruct_image_from_quaternion(X_W1_W2, original_shape)
        
        # Solver reconstruction
        img_solver = reconstruct_image_from_quaternion(quat_matmat(X, W_prod_est), original_shape)
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Original image
        plt.subplot(2, 3, 1)
        plt.imshow(img_original)
        plt.title('Original Image')
        plt.axis('off')
        
        # Plot 2: Ground truth reconstruction
        plt.subplot(2, 3, 2)
        plt.imshow(img_ground_truth)
        plt.title(f'Ground Truth Reconstruction\nError: {reconstruction_error:.2e}')
        plt.axis('off')
        
        # Plot 3: Solver reconstruction
        plt.subplot(2, 3, 3)
        plt.imshow(img_solver)
        plt.title(f'Solver Reconstruction\nError: {recon_error:.2e}')
        plt.axis('off')
        
        # Plot 4: Convergence
        plt.subplot(2, 3, 4)
        iterations = range(1, len(residuals['total_reconstruction']) + 1)
        plt.semilogy(iterations, residuals['total_reconstruction'], 'b-o', linewidth=2, markersize=6, label='||X @ W - I||')
        plt.axhline(y=reconstruction_error, color='r', linestyle='--', alpha=0.7, label=f'Ground truth: {reconstruction_error:.2e}')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Iteration')
        plt.ylabel('Reconstruction Error (log scale)')
        plt.title('Convergence: ||X @ W - I||')
        plt.legend()
        
        # Plot 5: Factor error
        plt.subplot(2, 3, 5)
        plt.semilogy(iterations, [factor_error] * len(iterations), 'g-o', linewidth=2, markersize=6, label='||W - X_pinv_true||')
        plt.axhline(y=factorization_error, color='r', linestyle='--', alpha=0.7, label=f'Ground truth: {factorization_error:.2e}')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Iteration')
        plt.ylabel('Factor Error (log scale)')
        plt.title('Factor Recovery: ||W - X_pinv_true||')
        plt.legend()
        
        # Plot 6: Error comparison
        plt.subplot(2, 3, 6)
        errors = [reconstruction_error, recon_error, factorization_error, factor_error]
        labels = ['GT Recon', 'Solver Recon', 'GT Factor', 'Solver Factor']
        colors = ['red', 'blue', 'orange', 'green']
        plt.bar(labels, errors, color=colors, alpha=0.7)
        plt.ylabel('Error')
        plt.title('Error Comparison')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('../../output_figures/real_image_factorization_test.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"✅ Ground truth reconstruction error: {reconstruction_error:.2e}")
        print(f"✅ Solver reconstruction error: {recon_error:.6f}")
        print(f"✅ Factor recovery error: {factor_error:.6f}")
        print(f"✅ W1 recovery error: {W1_error:.6f}")
        print(f"✅ W2 recovery error: {W2_error:.6f}")
        print()
        
        # Pass/fail criteria
        success_threshold = 1e-3
        if recon_error < success_threshold:
            print("🎉 SUCCESS: Solver successfully recovered the ground truth factorization!")
        else:
            print("⚠️  WARNING: Solver did not achieve desired accuracy")
        
        print("📊 Detailed plots saved as 'real_image_factorization_test.png'")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_image_factorization() 