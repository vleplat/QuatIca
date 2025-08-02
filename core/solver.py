from tabnanny import verbose
import numpy as np
import quaternion
from utils import (quat_matmat, quat_frobenius_norm, quat_hermitian, quat_eye,
                   normQsparse, timesQsparse, A2A0123, Realp, ggivens, GRSGivens, 
                   Hess_QR_ggivens, absQsparse, dotinvQsparse, UtriangleQsparse)

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


class QGMRESSolver:
    """
    Quaternion Generalized Minimal Residual (Q-GMRES) solver.
    
    Implements the Q-GMRES algorithm for solving quaternion linear systems
    A * x = b, where A is a quaternion matrix and b is a quaternion vector.
    
    Based on the implementation by Zhigang Jia and Michael K. Ng:
    "Structure Preserving Quaternion Generalized Minimal Residual Method", SIMAX, 2021
    """
    
    def __init__(self, tol: float = 1e-6, max_iter: int = None, verbose: bool = False) -> None:
        """
        Initialize Q-GMRES solver.
        
        Parameters:
        -----------
        tol : float, optional
            Tolerance for convergence (default: 1e-6)
        max_iter : int, optional
            Maximum number of iterations (default: None, uses matrix dimension)
        verbose : bool, optional
            Whether to print convergence information (default: False)
        """
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
    
    def solve(self, A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Solve the quaternion linear system A * x = b using Q-GMRES.
        
        Parameters:
        -----------
        A : np.ndarray
            Quaternion matrix (m x n)
        b : np.ndarray
            Quaternion right-hand side vector (m x 1)
        
        Returns:
        --------
        x : np.ndarray
            Solution vector (n x 1)
        info : dict
            Information about the solution process including:
            - 'iterations': Number of iterations performed
            - 'residual': Final residual norm
            - 'residual_history': List of residual norms
            - 'converged': Whether the method converged
        """
        # Convert quaternion matrices to component format
        A0, A1, A2, A3 = self._quat_to_components(A)
        b0, b1, b2, b3 = self._quat_to_components(b)
        
        # Check that A is square
        if A0.shape[0] != A0.shape[1]:
            raise ValueError(f"Q-GMRES requires square matrices. Got matrix of shape {A0.shape}")
        
        # Get dimensions
        N = A0.shape[1]  # Number of columns in A
        if self.max_iter is None:
            self.max_iter = N
        
        # Call the core GMRES implementation
        xm_0, xm_1, xm_2, xm_3, res, V0, V1, V2, V3, iter_count, resv = self._GMRESQsparse(
            A0, A1, A2, A3, b0, b1, b2, b3, self.tol, self.max_iter
        )
        
        # Convert solution back to quaternion format
        x = self._components_to_quat(xm_0, xm_1, xm_2, xm_3)
        
        # Prepare info dictionary
        info = {
            'iterations': iter_count,
            'residual': res,
            'residual_history': resv if resv is not None else [],
            'converged': res < self.tol,
            'V0': V0, 'V1': V1, 'V2': V2, 'V3': V3  # Krylov basis vectors
        }
        
        if self.verbose:
            print(f"Q-GMRES converged in {iter_count} iterations with residual {res:.2e}")
        
        return x, info
    
    def _GMRESQsparse(self, A0, A1, A2, A3, b_0, b_1, b_2, b_3, tol, maxit):
        """
        Core Q-GMRES implementation in component format.
        
        This is a direct translation of the MATLAB GMRESQsparse function.
        """
        N = A0.shape[1]
        ninf = normQsparse(A0, A1, A2, A3)
        
        if maxit is None:
            maxit = N
        
        # Initialize solution
        x0_0 = np.zeros((N, 1))
        x0_1 = x0_0.copy()
        x0_2 = x0_0.copy()
        x0_3 = x0_0.copy()
        
        resv = []
        
        # Main GMRES loop: for m=1:N (building Krylov subspace incrementally)
        for m in range(1, N + 1):
            # Compute residual: r0 = b - A * x0
            delta0, delta1, delta2, delta3 = timesQsparse(A0, A1, A2, A3, x0_0, x0_1, x0_2, x0_3)
            r0_0 = b_0 - delta0
            r0_2 = b_2 - delta2
            r0_1 = b_1 - delta1
            r0_3 = b_3 - delta3
            
            # Normalize residual to get first basis vector
            beta = normQsparse(r0_0, r0_1, r0_2, r0_3)
            v1_0 = r0_0 / beta
            v1_1 = r0_1 / beta
            v1_2 = r0_2 / beta
            v1_3 = r0_3 / beta
            
            # Initialize Krylov basis
            V0 = np.zeros((N, m))
            V1 = np.zeros((N, m))
            V2 = np.zeros((N, m))
            V3 = np.zeros((N, m))
            V0[:, 0] = v1_0.flatten()
            V1[:, 0] = v1_1.flatten()
            V2[:, 0] = v1_2.flatten()
            V3[:, 0] = v1_3.flatten()
            
            # Initialize Hessenberg matrix (m+1 x m to accommodate subdiagonal)
            H0 = np.zeros((m+1, m))
            H1 = np.zeros((m+1, m))
            H2 = np.zeros((m+1, m))
            H3 = np.zeros((m+1, m))
            
            # Arnoldi iteration: for j=1:m
            for j in range(m):
                # Compute A * v_j
                v_0, v_1, v_2, v_3 = timesQsparse(A0, A1, A2, A3, 
                                                  V0[:, j:j+1], V1[:, j:j+1], 
                                                  V2[:, j:j+1], V3[:, j:j+1])
                
                # Modified Gram-Schmidt orthogonalization: for i=1:j
                for i in range(j + 1):
                    # Compute inner product: <v_i, A*v_j>
                    v_i_conj_0 = V0[:, i:i+1].T  # Shape: (1, N)
                    v_i_conj_1 = -V1[:, i:i+1].T
                    v_i_conj_2 = -V2[:, i:i+1].T
                    v_i_conj_3 = -V3[:, i:i+1].T
                    
                    # Compute inner product using timesQsparse: (1×N) * (N×1) = (1×1)
                    H0[i, j], H1[i, j], H2[i, j], H3[i, j] = timesQsparse(
                        v_i_conj_0, v_i_conj_1, v_i_conj_2, v_i_conj_3,
                        v_0, v_1, v_2, v_3
                    )
                    
                    # The result should be a scalar (1x1 matrix), extract the scalar value
                    if hasattr(H0[i, j], 'shape') and H0[i, j].shape == (1, 1):
                        H0[i, j] = H0[i, j][0, 0]
                        H1[i, j] = H1[i, j][0, 0]
                        H2[i, j] = H2[i, j][0, 0]
                        H3[i, j] = H3[i, j][0, 0]
                    
                    # Subtract projection: v = v - <v_i, A*v_j> * v_i
                    delta0, delta1, delta2, delta3 = timesQsparse(
                        V0[:, i:i+1], V1[:, i:i+1], V2[:, i:i+1], V3[:, i:i+1],
                        H0[i, j], H1[i, j], H2[i, j], H3[i, j]
                    )
                    v_0 = v_0 - delta0
                    v_1 = v_1 - delta1
                    v_2 = v_2 - delta2
                    v_3 = v_3 - delta3
                
                # Compute norm of remaining vector
                if j < N:
                    H0[j+1, j] = normQsparse(v_0, v_1, v_2, v_3)
                    H1[j+1, j] = 0
                    H2[j+1, j] = 0
                    H3[j+1, j] = 0
                    
                    # Check for lucky breakdown
                    if abs(H0[j+1, j]) + ninf == ninf:
                        if self.verbose:
                            print('Lucky breakdown occurred!')
                        return x0_0, x0_1, x0_2, x0_3, 0, V0, V1, V2, V3, m, resv
                    
                    # Normalize next basis vector
                    if j < m - 1:
                        V0[:, j+1] = (v_0 / H0[j+1, j]).flatten()
                        V1[:, j+1] = (v_1 / H0[j+1, j]).flatten()
                        V2[:, j+1] = (v_2 / H0[j+1, j]).flatten()
                        V3[:, j+1] = (v_3 / H0[j+1, j]).flatten()
                    elif j == m - 1:
                        v_0 = v_0 / H0[j+1, j]
                        v_1 = v_1 / H0[j+1, j]
                        v_2 = v_2 / H0[j+1, j]
                        v_3 = v_3 / H0[j+1, j]
            
            # Construct full Krylov basis
            if m < N:
                Vm_0 = np.column_stack([V0, v_0])
                Vm_1 = np.column_stack([V1, v_1])
                Vm_2 = np.column_stack([V2, v_2])
                Vm_3 = np.column_stack([V3, v_3])
            else:
                Vm_0, Vm_1, Vm_2, Vm_3 = V0, V1, V2, V3
            
            # Compute projection of right-hand side onto Krylov subspace
            # Vm has shape (N, m+1) if m < N, or (N, m) if m == N
            # We need to project r0 onto the Krylov subspace
            bm_0, bm_1, bm_2, bm_3 = timesQsparse(
                Vm_0.T, -Vm_1.T, -Vm_2.T, -Vm_3.T,
                r0_0, r0_1, r0_2, r0_3
            )
            
            # QR decomposition of Hessenberg matrix using Givens rotations
            Hess = np.vstack([H0, H1, H2, H3])
            U, R = Hess_QR_ggivens(Hess)
            U0, U1, U2, U3 = A2A0123(U)
            R0, R1, R2, R3 = A2A0123(R)
            
            # Apply Q^T to right-hand side
            # Ensure bm has the correct shape to match U matrix
            # U0 has shape (m, m), so bm should have shape (m, 1)
            if bm_0.shape[0] != U0.shape[0]:
                if bm_0.shape[0] > U0.shape[0]:
                    # Truncate if too large
                    bm_0 = bm_0[:U0.shape[0], :]
                    bm_1 = bm_1[:U0.shape[0], :]
                    bm_2 = bm_2[:U0.shape[0], :]
                    bm_3 = bm_3[:U0.shape[0], :]
                else:
                    # Pad with zeros if too small
                    bm_0 = np.vstack([bm_0, np.zeros((U0.shape[0] - bm_0.shape[0], 1))])
                    bm_1 = np.vstack([bm_1, np.zeros((U0.shape[0] - bm_1.shape[0], 1))])
                    bm_2 = np.vstack([bm_2, np.zeros((U0.shape[0] - bm_2.shape[0], 1))])
                    bm_3 = np.vstack([bm_3, np.zeros((U0.shape[0] - bm_3.shape[0], 1))])
            
            bm2_0, bm2_1, bm2_2, bm2_3 = timesQsparse(
                U0.T, -U1.T, -U2.T, -U3.T,
                bm_0, bm_1, bm_2, bm_3
            )
            
            # Solve upper triangular system R * y = Q^T * b
            ym_0, ym_1, ym_2, ym_3 = UtriangleQsparse(
                R0[:m, :m], R1[:m, :m], R2[:m, :m], R3[:m, :m],
                bm2_0[:m], bm2_1[:m], bm2_2[:m], bm2_3[:m]
            )
            
            # Compute solution: x = x0 + V * y
            delta0, delta1, delta2, delta3 = timesQsparse(
                V0, V1, V2, V3, ym_0, ym_1, ym_2, ym_3
            )
            xm_0 = delta0 + x0_0
            xm_1 = delta1 + x0_1
            xm_2 = delta2 + x0_2
            xm_3 = delta3 + x0_3
            
            # Compute residual norm
            delta0, delta1, delta2, delta3 = timesQsparse(A0, A1, A2, A3, xm_0, xm_1, xm_2, xm_3)
            res_xm = normQsparse(b_0 - delta0, b_1 - delta1, b_2 - delta2, b_3 - delta3) / normQsparse(b_0, b_1, b_2, b_3)
            res = res_xm
            
            # Store residual history
            delta0, delta1, delta2, delta3 = timesQsparse(H0, H1, H2, H3, ym_0, ym_1, ym_2, ym_3)
            res_ym = normQsparse(bm_0 - delta0, bm_1 - delta1, bm_2 - delta2, bm_3 - delta3) / normQsparse(bm_0, bm_1, bm_2, bm_3)
            resv.append([m, res_ym, res_xm])
            
            # Check convergence (MATLAB: if res<tol || m>maxit)
            if res < tol or m > maxit:
                iter = m
                break
            else:
                # Update initial guess for next iteration (restart)
                x0_0, x0_1, x0_2, x0_3 = xm_0, xm_1, xm_2, xm_3
                iter = m
        
        return xm_0, xm_1, xm_2, xm_3, res, V0, V1, V2, V3, iter, resv
    
    def _quat_to_components(self, A):
        """Convert quaternion matrix to component format."""
        # Check if it's a sparse quaternion matrix
        if hasattr(A, 'real') and hasattr(A, 'i') and hasattr(A, 'j') and hasattr(A, 'k'):
            # Sparse quaternion matrix
            A0 = A.real.toarray()
            A1 = A.i.toarray()
            A2 = A.j.toarray()
            A3 = A.k.toarray()
        elif hasattr(A, 'dtype') and A.dtype == np.quaternion:
            # Dense quaternion array
            A_real = quaternion.as_float_array(A)
            A0 = A_real[..., 0]
            A1 = A_real[..., 1]
            A2 = A_real[..., 2]
            A3 = A_real[..., 3]
        else:
            # Assume it's already in component format
            A0, A1, A2, A3 = A
        return A0, A1, A2, A3
    
    def _components_to_quat(self, A0, A1, A2, A3):
        """Convert component format back to quaternion matrix."""
        # Stack components and convert to quaternion array
        A_real = np.stack([A0, A1, A2, A3], axis=-1)
        return quaternion.as_quat_array(A_real)