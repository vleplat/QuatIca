from tabnanny import verbose
import numpy as np
import quaternion
from utils import quat_matmat, quat_frobenius_norm, quat_hermitian, quat_eye

class NewtonSchulzPseudoinverse:
    """Compute the Moore–Penrose pseudoinverse of quaternion matrices via damped Newton–Schulz."""
    def __init__(self, gamma: float = 0.5, max_iter: int = 100, tol: float = 1e-6, verbose: bool = False, compute_residuals: bool = True) -> None:
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.compute_residuals = compute_residuals

    def compute(self, A: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]], list[float]]:
        m, n = A.shape
        # If A is sparse, convert to dense quaternion array for computation
        from utils import SparseQuaternionMatrix
        if isinstance(A, SparseQuaternionMatrix):
            A = quaternion.as_quat_array(
                np.stack([
                    A.real.toarray(),
                    A.i.toarray(),
                    A.j.toarray(),
                    A.k.toarray()
                ], axis=-1)
            )
        # Determine left vs. right pseudoinverse
        if m >= n:
            I_target = quat_eye(n); use_left = True
        else:
            I_target = quat_eye(m); use_left = False
        # Initialization
        alpha = 1.0 / (quat_frobenius_norm(A)**2)
        X = alpha * quat_hermitian(A)
        residuals = {'AXA-A': [], 'XAX-X': [], 'AX-herm': [], 'XA-herm': []}
        covariances = []
        # Iterations
        for k in range(self.max_iter):
            # Covariance deviation
            cov_prod = quat_matmat(X, A) if use_left else quat_matmat(A, X)
            cov_dev = cov_prod - I_target
            cov_norm = quat_frobenius_norm(cov_dev)
            covariances.append(cov_norm)
            # Update
            update = quat_matmat(cov_dev, X) if use_left else quat_matmat(X, cov_dev)
            X = X - self.gamma * update
            # MP residuals (only if requested)
            if self.compute_residuals:
                AX = quat_matmat(A, X); XA = quat_matmat(X, A)
                residuals['AXA-A'].append(quat_frobenius_norm(quat_matmat(AX, A) - A))
                residuals['XAX-X'].append(quat_frobenius_norm(quat_matmat(XA, X) - X))
                residuals['AX-herm'].append(quat_frobenius_norm(AX - quat_hermitian(AX)))
                residuals['XA-herm'].append(quat_frobenius_norm(XA - quat_hermitian(XA)))
                # Verbose logging
                max_res = max(residuals[key][-1] for key in residuals)
                if self.verbose and (k < 5 or k % 10 == 0 or max_res < self.tol):
                    print(f"Iter {k+1}: max_res={max_res:.2e}, cov_dev={cov_norm:.2e}")
                if max_res < self.tol:
                    if self.verbose: print(f"Converged at iter {k+1}")
                    break
            else:
                # Simple convergence check based on covariance deviation
                if self.verbose and (k < 5 or k % 10 == 0):
                    print(f"Iter {k+1}: cov_dev={cov_norm:.2e}")
                if cov_norm < self.tol:
                    if self.verbose: print(f"Converged at iter {k+1}")
                    break
        return X, residuals, covariances


class DeepLinearNewtonSchulz:
    """
    Deep Linear Newton-Schulz solver for quaternion matrices.
    
    Implements the alternating gradient method for deep linear networks using
    proper Newton-Schulz updates for each layer individually.
    
    The method optimizes ||X * W_1 * ... * W_d - X|| -> min
    where each W_i is updated using the damped Newton-Schulz formula:
    W_i^(k+1) = W_i^(k) - γ * W_i^(k) * (hat_W * hat_X * W_i^(k) - I)
    """
    
    def __init__(self, gamma: float = 0.9, max_iter: int = 20, tol: float = 1e-6, 
                 verbose: bool = False, compute_residuals: bool = True, inner_iterations: int = 1,
                 random_init: bool = False) -> None:
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.compute_residuals = compute_residuals
        self.inner_iterations = inner_iterations
        self.random_init = random_init
        self.NSPSolver = NewtonSchulzPseudoinverse(gamma=1, max_iter=100, tol=tol, verbose=False, compute_residuals=compute_residuals)
    
    def compute(self, X: np.ndarray, layers: list[int]) -> tuple[list[np.ndarray], dict[str, list[float]], list[float]]:
        """
        Compute deep linear decomposition using alternating Newton-Schulz updates.
        
        Args:
            X: Input quaternion matrix (n_samples x input_dim)
            layers: List of layer dimensions [d0, d1, ..., dk] 
                    where d0 = input_dim, dk = output_dim
        
        Returns:
            weights: List of optimized quaternion weight matrices
            residuals: Dictionary of residual norms over iterations
            deviations: List of total deviation norms over iterations
        """
        n_samples, input_dim = X.shape
        
        # Validate layer dimensions
        if layers[0] != input_dim:
            raise ValueError(f"First layer dimension {layers[0]} must match input dimension {input_dim}")
        
        # Initialize weights with better conditioning
        # We want W = W_1 * W_2 * ... * W_n ≈ X^†
        # So we start with a good approximation of X^† and factor it
        
        # First, compute a rough pseudoinverse of X using SVD-like approach
        # For quaternion matrices, we'll use a simple approach
        X_norm = quat_frobenius_norm(X)
        if X_norm > 0:
            # Initialize with scaled identity-like structure
            # The idea is to start close to a pseudoinverse approximation
            # scale_factor = 1.0 / (X_norm * np.sqrt(n_samples)) 
            scale_factor = 1.0 / (X_norm ) ** (1/len(layers))
            print(f"Scale factor: {scale_factor}")
        else:
            scale_factor = 0.1
        
        weights = []
        for i in range(len(layers)-1):
            # Initialize with scaled identity structure
            # This helps ensure the product is well-conditioned
            if self.random_init:
                # Random initialization with controlled scale
                # Use Xavier/Glorot initialization for better conditioning
                fan_in = layers[i]
                fan_out = layers[i+1]
                scale = np.sqrt(2.0 / (fan_in + fan_out)) * scale_factor 
                print(f"Scale factor: {scale}")
                
                W_real = np.random.randn(layers[i], layers[i+1]) * scale
                W_i = np.random.randn(layers[i], layers[i+1]) * scale
                W_j = np.random.randn(layers[i], layers[i+1]) * scale
                W_k = np.random.randn(layers[i], layers[i+1]) * scale
            else:
                # Deterministic initialization with scaled identity structure
                print(f"Scale factor: {scale_factor}")
                W_real = np.eye(layers[i], layers[i+1]) * scale_factor 
                W_i = np.zeros((layers[i], layers[i+1])) * scale_factor 
                W_j = np.zeros((layers[i], layers[i+1])) * scale_factor 
                W_k = np.eye(layers[i], layers[i+1]) * scale_factor 
            
            # Add small random perturbation for exploration
            # W_real += np.random.randn(layers[i], layers[i+1]) * scale_factor * 0.1
            # W_i += np.random.randn(layers[i], layers[i+1]) * scale_factor * 0.1
            # W_j += np.random.randn(layers[i], layers[i+1]) * scale_factor * 0.1
            # W_k += np.random.randn(layers[i], layers[i+1]) * scale_factor * 0.1
            
            W_quat = np.stack([W_real, W_i, W_j, W_k], axis=-1)
            weights.append(quaternion.as_quat_array(W_quat))
        
        if self.verbose:
            print(f"Initialized {len(weights)} layers with dimensions: {layers}")
        
        # Initialize tracking variables
        residuals = {'total_reconstruction': [], 'layer_deviations': []}
        deviations = []
        
        # Alternating Newton-Schulz updates with inner iterations for each layer
        for iter in range(self.max_iter):
            # Update each layer in sequence with multiple inner iterations
            for i in range(len(weights)):
                # Inner iterations: update the same layer multiple times before moving to next
                for inner_iter in range(self.inner_iterations):
                    # Current weight
                    W_i = weights[i]
                    
                    # Implement the EXACT Newton-Schulz alternating update from the paper:
                    # W_i^(k+1) = W_i^(k) - γ W_i^(k) (Ŵ Ĥ W_i^(k) - I)
                    # Where Ĥ = X W_1 ... W_{i-1} and Ŵ = W_{i+1} ... W_d
                    
                    # Compute Ĥ = X W_1 ... W_{i-1}
                    hat_X = X.copy()
                    for j in range(i):
                        hat_X = quat_matmat(hat_X, weights[j])
                    
                    # Compute Ŵ = W_{i+1} ... W_d
                    hat_W = None
                    if i < len(weights) - 1:
                        hat_W = weights[i+1].copy()
                        for j in range(i+2, len(weights)):
                            hat_W = quat_matmat(hat_W, weights[j])
                    
                    # Compute Ŵ Ĥ W_i^(k) (note the order: Ŵ first, then Ĥ)
                    if hat_W is None:
                        # Last layer: Ŵ is identity
                        product = quat_matmat(hat_X, W_i)
                    else:
                        # Intermediate layer: compute Ŵ Ĥ W_i^(k)
                        temp = quat_matmat(hat_W, hat_X)
                        product = quat_matmat(temp, W_i)
                    
                    # Create identity matrix of appropriate size
                    I_product = np.zeros(product.shape, dtype=np.quaternion)
                    min_dim = min(product.shape[0], product.shape[1])
                    for k in range(min_dim):
                        I_product[k, k] = quaternion.quaternion(1, 0, 0, 0)
                    
                    # Compute (Ŵ Ĥ W_i^(k) - I)
                    deviation = product - I_product
                    
                    # Compute W_i^(k) (Ŵ Ĥ W_i^(k) - I)
                    # update_term = quat_matmat(W_i, deviation)
                    
                    # Apply the exact Newton-Schulz update: W_i^(k+1) = W_i^(k) - γ update_term
                    # weights[i] = W_i - self.gamma * update_term
                    # Exact BCD
                    hat_X_PI = self.NSPSolver.compute(hat_X)[0]
                    if hat_W is not None:
                        hat_W_PI = self.NSPSolver.compute(hat_W)[0]
                        weights[i] = quat_matmat(hat_X_PI, hat_W_PI)
                    else:
                        # For the last layer, hat_W is None (identity), so just use hat_X_PI
                        weights[i] = hat_X_PI
                
                # Clip weights to prevent numerical instability
                weight_norm = quat_frobenius_norm(weights[i])
                if weight_norm > 3.0:
                    weights[i] = weights[i] * (3.0 / weight_norm)
            
            # Compute overall reconstruction error (X * W - I)
            Y = X.copy()
            for W in weights:
                Y = quat_matmat(Y, W)
            I = np.zeros(Y.shape, dtype=np.quaternion)
            min_dim = min(Y.shape[0], Y.shape[1])
            for k in range(min_dim):
                I[k, k] = quaternion.quaternion(1, 0, 0, 0)
            recon_error = quat_frobenius_norm(Y - I)
            residuals['total_reconstruction'].append(recon_error)
            deviations.append(recon_error)
            
            if self.verbose and (iter < 5 or iter % 5 == 0):
                print(f"Iteration {iter+1}: Reconstruction error = {recon_error:.6f}")
            
            # Check convergence
            if recon_error < self.tol:
                if self.verbose:
                    print(f"Converged after {iter+1} iterations")
                break
        
        return weights, residuals, deviations