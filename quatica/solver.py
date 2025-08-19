import time

import numpy as np
import quaternion

# Support both package and script import contexts for quatica.utils
try:
    from .utils import (
        A2A0123,
        Hess_QR_ggivens,
        UtriangleQsparse,
        normQsparse,
        quat_eye,
        quat_frobenius_norm,
        quat_hermitian,
        quat_matmat,
        timesQsparse,
    )
except Exception:  # fallback for direct script runs
    from utils import (
        A2A0123,
        Hess_QR_ggivens,
        UtriangleQsparse,
        normQsparse,
        quat_eye,
        quat_frobenius_norm,
        quat_hermitian,
        quat_matmat,
        timesQsparse,
    )


class NewtonSchulzPseudoinverse:
    """Compute the Moore–Penrose pseudoinverse of quaternion matrices via damped Newton–Schulz."""

    def __init__(
        self,
        gamma: float = 0.5,
        max_iter: int = 100,
        tol: float = 1e-6,
        verbose: bool = False,
        compute_residuals: bool = True,
    ) -> None:
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.compute_residuals = compute_residuals

    def compute(
        self, A: np.ndarray
    ) -> tuple[np.ndarray, dict[str, list[float]], list[float]]:
        m, n = A.shape
        # If A is sparse, convert to dense quaternion array for computation
        # Import SparseQuaternionMatrix with package-safe fallback
        try:
            from .utils import SparseQuaternionMatrix
        except Exception:
            from utils import SparseQuaternionMatrix
        if isinstance(A, SparseQuaternionMatrix):
            A = quaternion.as_quat_array(
                np.stack(
                    [A.real.toarray(), A.i.toarray(), A.j.toarray(), A.k.toarray()],
                    axis=-1,
                )
            )
        # Determine left vs. right pseudoinverse
        if m >= n:
            I_target = quat_eye(n)
            use_left = True
        else:
            I_target = quat_eye(m)
            use_left = False
        # Initialization
        alpha = 1.0 / (quat_frobenius_norm(A) ** 2)
        X = alpha * quat_hermitian(A)
        residuals = {"AXA-A": [], "XAX-X": [], "AX-herm": [], "XA-herm": []}
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
                AX = quat_matmat(A, X)
                XA = quat_matmat(X, A)
                residuals["AXA-A"].append(quat_frobenius_norm(quat_matmat(AX, A) - A))
                residuals["XAX-X"].append(quat_frobenius_norm(quat_matmat(XA, X) - X))
                residuals["AX-herm"].append(quat_frobenius_norm(AX - quat_hermitian(AX)))
                residuals["XA-herm"].append(quat_frobenius_norm(XA - quat_hermitian(XA)))
                # Verbose logging
                max_res = max(residuals[key][-1] for key in residuals)
                if self.verbose and (k < 5 or k % 10 == 0 or max_res < self.tol):
                    print(f"Iter {k + 1}: max_res={max_res:.2e}, cov_dev={cov_norm:.2e}")
                if max_res < self.tol:
                    if self.verbose:
                        print(f"Converged at iter {k + 1}")
                    break
            else:
                # Simple convergence check based on covariance deviation
                if self.verbose and (k < 5 or k % 10 == 0):
                    print(f"Iter {k + 1}: cov_dev={cov_norm:.2e}")
                if cov_norm < self.tol:
                    if self.verbose:
                        print(f"Converged at iter {k + 1}")
                    break
        return X, residuals, covariances


class HigherOrderNewtonSchulzPseudoinverse:
    """
    Third-order Newton–Schulz pseudoinverse solver (no damping).

    Iteration (T := X_k):
      X_{k+1} = 3 T - 3 T A T + T (A T)^2

    Initialization:
      T_0 = A^H / ||A||_F^2

    Residuals tracked per iteration:
      E1 = ||A X A - A||_F
      E2 = ||X A X - X||_F
      E3 = ||(A X)^H - A X||_F
      E4 = ||(X A)^H - X A||_F
    """

    def __init__(
        self, max_iter: int = 100, tol: float = 0.0, verbose: bool = False
    ) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def compute(
        self, A: np.ndarray
    ) -> tuple[np.ndarray, dict[str, list[float]], list[float]]:
        m, n = A.shape
        # Initialization
        alpha = 1.0 / (quat_frobenius_norm(A) ** 2 + 1e-30)
        T = alpha * quat_hermitian(A)
        residuals = {"AXA-A": [], "XAX-X": [], "AX-herm": [], "XA-herm": []}
        times_per_iter: list[float] = []

        for k in range(self.max_iter):
            t0 = time.time()
            # Core third-order NS update
            AT = quat_matmat(A, T)
            AT_sq = quat_matmat(AT, AT)
            TAT = quat_matmat(quat_matmat(T, A), T)
            T = 3 * T - 3 * TAT + quat_matmat(T, AT_sq)

            # Residuals
            AX = quat_matmat(A, T)
            XA = quat_matmat(T, A)
            residuals["AXA-A"].append(quat_frobenius_norm(quat_matmat(AX, A) - A))
            residuals["XAX-X"].append(quat_frobenius_norm(quat_matmat(XA, T) - T))
            residuals["AX-herm"].append(quat_frobenius_norm(AX - quat_hermitian(AX)))
            residuals["XA-herm"].append(quat_frobenius_norm(XA - quat_hermitian(XA)))
            times_per_iter.append(time.time() - t0)

            if self.verbose and (k < 5 or k % 10 == 0):
                print(
                    f"Iter {k + 1}: E1={residuals['AXA-A'][-1]:.2e} E2={residuals['XAX-X'][-1]:.2e}"
                )

            if self.tol > 0.0 and residuals["AXA-A"][-1] < self.tol:
                break

        return T, residuals, times_per_iter


class DeepLinearNewtonSchulz:
    """
    Deep Linear Newton-Schulz solver for quaternion matrices.

    Implements the alternating gradient method for deep linear networks using
    proper Newton-Schulz updates for each layer individually.

    The method optimizes ||X * W_1 * ... * W_d - I||_F -> min
    where each W_i is updated using the damped Newton-Schulz formula:
    W_i^(k+1) = W_i^(k) - γ * W_i^(k) * (hat_W * hat_X * W_i^(k) - I)
    """

    def __init__(
        self,
        gamma: float = 0.9,
        max_iter: int = 20,
        tol: float = 1e-6,
        verbose: bool = False,
        compute_residuals: bool = True,
        inner_iterations: int = 1,
        random_init: bool = False,
    ) -> None:
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.compute_residuals = compute_residuals
        self.inner_iterations = inner_iterations
        self.random_init = random_init
        self.NSPSolver = NewtonSchulzPseudoinverse(
            gamma=1,
            max_iter=100,
            tol=tol,
            verbose=False,
            compute_residuals=compute_residuals,
        )

    def compute(
        self, X: np.ndarray, layers: list[int]
    ) -> tuple[list[np.ndarray], dict[str, list[float]], list[float]]:
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
            raise ValueError(
                f"First layer dimension {layers[0]} must match input dimension {input_dim}"
            )

        # Initialize weights with better conditioning
        # We want W = W_1 * W_2 * ... * W_n ≈ X^†
        # So we start with a good approximation of X^† and factor it

        # First, compute a rough pseudoinverse of X using SVD-like approach
        # For quaternion matrices, we'll use a simple approach
        X_norm = quat_frobenius_norm(X)
        if X_norm > 0:
            # Initialize with scaled identity-like structure
            # The idea is to start close to a pseudoinverse approximation
            # scale_factor = 1.0 / (X_norm * np.sqrt(n_samples))
            scale_factor = 1.0 / (X_norm) ** (1 / len(layers))
            print(f"Scale factor: {scale_factor}")
        else:
            scale_factor = 0.1

        weights = []
        for i in range(len(layers) - 1):
            # Initialize with scaled identity structure
            # This helps ensure the product is well-conditioned
            if self.random_init:
                # Random initialization with controlled scale
                # Use Xavier/Glorot initialization for better conditioning
                fan_in = layers[i]
                fan_out = layers[i + 1]
                scale = np.sqrt(2.0 / (fan_in + fan_out)) * scale_factor
                print(f"Scale factor: {scale}")

                W_real = np.random.randn(layers[i], layers[i + 1]) * scale
                W_i = np.random.randn(layers[i], layers[i + 1]) * scale
                W_j = np.random.randn(layers[i], layers[i + 1]) * scale
                W_k = np.random.randn(layers[i], layers[i + 1]) * scale
            else:
                # Deterministic initialization with scaled identity structure
                print(f"Scale factor: {scale_factor}")
                W_real = np.eye(layers[i], layers[i + 1]) * scale_factor
                W_i = np.zeros((layers[i], layers[i + 1])) * scale_factor
                W_j = np.zeros((layers[i], layers[i + 1])) * scale_factor
                W_k = np.eye(layers[i], layers[i + 1]) * scale_factor

            # Add small random perturbation for exploration
            # W_real += np.random.randn(layers[i], layers[i+1]) * scale_factor * 0.1
            # W_i += np.random.randn(layers[i], layers[i+1]) * scale_factor * 0.1
            # W_j += np.random.randn(layers[i], layers[i+1]) * scale_factor * 0.1
            # W_k += np.random.randn(layers[i], layers[i+1]) * scale_factor * 0.1

            W_quat = np.stack([W_real, W_i, W_j, W_k], axis=-1)
            weights.append(quaternion.as_quat_array(W_quat))

        if self.verbose:
            print(f"Initialized {len(weights)} layers with dimensions: {layers}")

        # Initialize tracking variables
        residuals = {"total_reconstruction": [], "layer_deviations": []}
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
                        hat_W = weights[i + 1].copy()
                        for j in range(i + 2, len(weights)):
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
                    product - I_product

                    # Compute W_i^(k) (Ŵ Ĥ W_i^(k) - I)
                    # update_term = quat_matmat(W_i, deviation)

                    # Apply the exact Newton-Schulz update: W_i^(k+1) = W_i^(k) - γ update_term
                    # weights[i] = W_i - self.gamma * update_term
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
            residuals["total_reconstruction"].append(recon_error)
            deviations.append(recon_error)

            if self.verbose and (iter < 5 or iter % 5 == 0):
                print(f"Iteration {iter + 1}: Reconstruction error = {recon_error:.6f}")

            # Check convergence
            if recon_error < self.tol:
                if self.verbose:
                    print(f"Converged after {iter + 1} iterations")
                break

        return weights, residuals, deviations


def _solve_lower_triangular_quat(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Solve L X = B for X where L is lower-triangular (dense quaternion), B (n×k).

    Uses forward substitution in quaternion arithmetic:
    X[i] = L[ii]^{-1} (B[i] - sum_{j<i} L[i,j] X[j]).

    Parameters:
    -----------
    L : np.ndarray
        Lower triangular quaternion matrix (n×n)
    B : np.ndarray
        Right-hand side quaternion matrix (n×k)

    Returns:
    --------
    np.ndarray
        Solution matrix X (n×k) such that L @ X = B

    Notes:
    ------
    This function performs forward substitution for dense quaternion matrices.
    Diagonal elements are inverted using quaternion conjugate division.
    """
    n = L.shape[0]
    k = B.shape[1]
    X = np.zeros_like(B)
    for i in range(n):
        rhs = B[i : i + 1, :].copy()
        if i > 0:
            acc = np.zeros_like(B[i : i + 1, :])
            for j in range(i):
                acc = acc + L[i : i + 1, j : j + 1] * X[j : j + 1, :]
            rhs = rhs - acc
        # Invert diagonal quaternion scalar L[i,i]
        diag = L[i, i]
        denom = (
            diag.w * diag.w + diag.x * diag.x + diag.y * diag.y + diag.z * diag.z + 1e-30
        )
        diag_inv = quaternion.quaternion(diag.w, -diag.x, -diag.y, -diag.z) * (
            1.0 / denom
        )
        # Multiply diag_inv on the left: X[i] = diag_inv * rhs
        for col in range(k):
            X[i, col] = diag_inv * rhs[0, col]
    return X


def _solve_upper_triangular_quat(U: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Solve U X = B for X where U is upper-triangular (dense quaternion), B (n×k).

    Uses backward substitution in quaternion arithmetic:
    X[i] = U[ii]^{-1} (B[i] - sum_{j>i} U[i,j] X[j]).

    Parameters:
    -----------
    U : np.ndarray
        Upper triangular quaternion matrix (n×n)
    B : np.ndarray
        Right-hand side quaternion matrix (n×k)

    Returns:
    --------
    np.ndarray
        Solution matrix X (n×k) such that U @ X = B

    Notes:
    ------
    This function performs backward substitution for dense quaternion matrices.
    Diagonal elements are inverted using quaternion conjugate division.
    """
    n = U.shape[0]
    k = B.shape[1]
    X = np.zeros_like(B)
    for i in range(n - 1, -1, -1):
        rhs = B[i : i + 1, :].copy()
        if i < n - 1:
            acc = np.zeros_like(B[i : i + 1, :])
            for j in range(i + 1, n):
                acc = acc + U[i : i + 1, j : j + 1] * X[j : j + 1, :]
            rhs = rhs - acc
        diag = U[i, i]
        denom = (
            diag.w * diag.w + diag.x * diag.x + diag.y * diag.y + diag.z * diag.z + 1e-30
        )
        diag_inv = quaternion.quaternion(diag.w, -diag.x, -diag.y, -diag.z) * (
            1.0 / denom
        )
        for col in range(k):
            X[i, col] = diag_inv * rhs[0, col]
    return X


class QGMRESSolver:
    """
    Quaternion Generalized Minimal Residual (Q-GMRES) solver.

    Implements the Q-GMRES algorithm for solving quaternion linear systems
    A * x = b, where A is a quaternion matrix and b is a quaternion vector.

    Based on the implementation by Zhigang Jia and Michael K. Ng:
    "Structure Preserving Quaternion Generalized Minimal Residual Method", SIMAX, 2021
    """

    def __init__(
        self,
        tol: float = 1e-6,
        max_iter: int = None,
        verbose: bool = False,
        preconditioner: str | None = None,
    ) -> None:
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
        self.preconditioner = preconditioner or "none"

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
        # Keep original for true residual reporting
        A_orig, b_orig = A, b

        # Optional left preconditioning using full LU: M = P^T L U ≈ A
        prec = self.preconditioner.lower()
        if prec == "left_lu":
            try:
                from decomp import quaternion_lu

                # Compute LU with permutation: P * A = L * U ⇒ A = P^T L U
                Lq, Uq, Pq = quaternion_lu(A, return_p=True)
                # Full left preconditioner: M = P^T L U ≈ A (exact if LU exact)
                # Apply M^{-1} = U^{-1} L^{-1} P via: P, then solve L, then solve U
                n = A.shape[0]
                # Apply to A by columns
                A_cols = [A[:, j : j + 1] for j in range(n)]
                A_tilde_cols = []
                for Aj in A_cols:
                    # Step 1: Apply permutation P * Aj
                    PAj = quat_matmat(Pq, Aj)
                    # Step 2: Solve L * Y = PAj (forward substitution)
                    Y = _solve_lower_triangular_quat(Lq, PAj)
                    # Step 3: Solve U * Z = Y (backward substitution)
                    Z = _solve_upper_triangular_quat(Uq, Y)
                    A_tilde_cols.append(Z)
                A = np.concatenate(A_tilde_cols, axis=1)
                # Apply to b
                # Step 1: Apply permutation P * b
                Pb = quat_matmat(Pq, b)
                # Step 2: Solve L * Y = Pb
                Yb = _solve_lower_triangular_quat(Lq, Pb)
                # Step 3: Solve U * x = Y
                b = _solve_upper_triangular_quat(Uq, Yb)
                if self.verbose:
                    print("[QGMRES] Applied left LU preconditioner (full M = P^T L U)")
            except Exception as e:
                if self.verbose:
                    print(
                        f"[QGMRES] LU preconditioning failed ({e}); continuing without preconditioner"
                    )

        # Convert quaternion matrices to component format
        A0, A1, A2, A3 = self._quat_to_components(A)
        b0, b1, b2, b3 = self._quat_to_components(b)

        # Check that A is square
        if A0.shape[0] != A0.shape[1]:
            raise ValueError(
                f"Q-GMRES requires square matrices. Got matrix of shape {A0.shape}"
            )

        # Get dimensions
        N = A0.shape[1]  # Number of columns in A
        if self.max_iter is None:
            self.max_iter = N

        # Call the core GMRES implementation
        xm_0, xm_1, xm_2, xm_3, res, V0, V1, V2, V3, iter_count, resv = (
            self._GMRESQsparse(A0, A1, A2, A3, b0, b1, b2, b3, self.tol, self.max_iter)
        )

        # Convert solution back to quaternion format
        x = self._components_to_quat(xm_0, xm_1, xm_2, xm_3)

        # Prepare info dictionary
        info = {
            "iterations": iter_count,
            "residual": res,
            "residual_history": resv if resv is not None else [],
            "converged": res < self.tol,
            "V0": V0,
            "V1": V1,
            "V2": V2,
            "V3": V3,  # Krylov basis vectors
        }

        # Compute true residual with respect to original (un-preconditioned) system
        try:
            r_true = quat_frobenius_norm(quat_matmat(A_orig, x) - b_orig)
            r_true /= quat_frobenius_norm(b_orig) + 1e-30
            info["residual_true"] = r_true
            # Replace primary residual with true residual for fair comparison across preconditioners
            info["residual"] = r_true
        except Exception:
            # Fallback: keep preconditioned residual only
            info["residual_true"] = info["residual"]

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
            delta0, delta1, delta2, delta3 = timesQsparse(
                A0, A1, A2, A3, x0_0, x0_1, x0_2, x0_3
            )
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
            H0 = np.zeros((m + 1, m))
            H1 = np.zeros((m + 1, m))
            H2 = np.zeros((m + 1, m))
            H3 = np.zeros((m + 1, m))

            # Arnoldi iteration: for j=1:m
            for j in range(m):
                # Compute A * v_j
                v_0, v_1, v_2, v_3 = timesQsparse(
                    A0,
                    A1,
                    A2,
                    A3,
                    V0[:, j : j + 1],
                    V1[:, j : j + 1],
                    V2[:, j : j + 1],
                    V3[:, j : j + 1],
                )

                # Modified Gram-Schmidt orthogonalization: for i=1:j
                for i in range(j + 1):
                    # Compute inner product: <v_i, A*v_j>
                    v_i_conj_0 = V0[:, i : i + 1].T  # Shape: (1, N)
                    v_i_conj_1 = -V1[:, i : i + 1].T
                    v_i_conj_2 = -V2[:, i : i + 1].T
                    v_i_conj_3 = -V3[:, i : i + 1].T

                    # Compute inner product using timesQsparse: (1×N) * (N×1) = (1×1)
                    H0[i, j], H1[i, j], H2[i, j], H3[i, j] = timesQsparse(
                        v_i_conj_0, v_i_conj_1, v_i_conj_2, v_i_conj_3, v_0, v_1, v_2, v_3
                    )

                    # The result should be a scalar (1x1 matrix), extract the scalar value
                    if hasattr(H0[i, j], "shape") and H0[i, j].shape == (1, 1):
                        H0[i, j] = H0[i, j][0, 0]
                        H1[i, j] = H1[i, j][0, 0]
                        H2[i, j] = H2[i, j][0, 0]
                        H3[i, j] = H3[i, j][0, 0]

                    # Subtract projection: v = v - <v_i, A*v_j> * v_i
                    delta0, delta1, delta2, delta3 = timesQsparse(
                        V0[:, i : i + 1],
                        V1[:, i : i + 1],
                        V2[:, i : i + 1],
                        V3[:, i : i + 1],
                        H0[i, j],
                        H1[i, j],
                        H2[i, j],
                        H3[i, j],
                    )
                    v_0 = v_0 - delta0
                    v_1 = v_1 - delta1
                    v_2 = v_2 - delta2
                    v_3 = v_3 - delta3

                # Compute norm of remaining vector
                if j < N:
                    H0[j + 1, j] = normQsparse(v_0, v_1, v_2, v_3)
                    H1[j + 1, j] = 0
                    H2[j + 1, j] = 0
                    H3[j + 1, j] = 0

                    # Check for lucky breakdown
                    if abs(H0[j + 1, j]) + ninf == ninf:
                        if self.verbose:
                            print("Lucky breakdown occurred!")
                        return x0_0, x0_1, x0_2, x0_3, 0, V0, V1, V2, V3, m, resv

                    # Normalize next basis vector
                    if j < m - 1:
                        V0[:, j + 1] = (v_0 / H0[j + 1, j]).flatten()
                        V1[:, j + 1] = (v_1 / H0[j + 1, j]).flatten()
                        V2[:, j + 1] = (v_2 / H0[j + 1, j]).flatten()
                        V3[:, j + 1] = (v_3 / H0[j + 1, j]).flatten()
                    elif j == m - 1:
                        v_0 = v_0 / H0[j + 1, j]
                        v_1 = v_1 / H0[j + 1, j]
                        v_2 = v_2 / H0[j + 1, j]
                        v_3 = v_3 / H0[j + 1, j]

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
                Vm_0.T, -Vm_1.T, -Vm_2.T, -Vm_3.T, r0_0, r0_1, r0_2, r0_3
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
                    bm_0 = bm_0[: U0.shape[0], :]
                    bm_1 = bm_1[: U0.shape[0], :]
                    bm_2 = bm_2[: U0.shape[0], :]
                    bm_3 = bm_3[: U0.shape[0], :]
                else:
                    # Pad with zeros if too small
                    bm_0 = np.vstack([bm_0, np.zeros((U0.shape[0] - bm_0.shape[0], 1))])
                    bm_1 = np.vstack([bm_1, np.zeros((U0.shape[0] - bm_1.shape[0], 1))])
                    bm_2 = np.vstack([bm_2, np.zeros((U0.shape[0] - bm_2.shape[0], 1))])
                    bm_3 = np.vstack([bm_3, np.zeros((U0.shape[0] - bm_3.shape[0], 1))])

            bm2_0, bm2_1, bm2_2, bm2_3 = timesQsparse(
                U0.T, -U1.T, -U2.T, -U3.T, bm_0, bm_1, bm_2, bm_3
            )

            # Solve upper triangular system R * y = Q^T * b
            ym_0, ym_1, ym_2, ym_3 = UtriangleQsparse(
                R0[:m, :m],
                R1[:m, :m],
                R2[:m, :m],
                R3[:m, :m],
                bm2_0[:m],
                bm2_1[:m],
                bm2_2[:m],
                bm2_3[:m],
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
            delta0, delta1, delta2, delta3 = timesQsparse(
                A0, A1, A2, A3, xm_0, xm_1, xm_2, xm_3
            )
            res_xm = normQsparse(
                b_0 - delta0, b_1 - delta1, b_2 - delta2, b_3 - delta3
            ) / normQsparse(b_0, b_1, b_2, b_3)
            res = res_xm

            # Store residual history
            delta0, delta1, delta2, delta3 = timesQsparse(
                H0, H1, H2, H3, ym_0, ym_1, ym_2, ym_3
            )
            res_ym = normQsparse(
                bm_0 - delta0, bm_1 - delta1, bm_2 - delta2, bm_3 - delta3
            ) / normQsparse(bm_0, bm_1, bm_2, bm_3)
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
        if hasattr(A, "real") and hasattr(A, "i") and hasattr(A, "j") and hasattr(A, "k"):
            # Sparse quaternion matrix
            A0 = A.real.toarray()
            A1 = A.i.toarray()
            A2 = A.j.toarray()
            A3 = A.k.toarray()
        elif hasattr(A, "dtype") and A.dtype == np.quaternion:
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


class RandomizedSketchProjectPseudoinverse:
    """
    Randomized Sketch-and-Project (RSP-Q) for quaternion matrix pseudoinverse.

    Implements the randomized sketch-and-project method for computing the
    Moore-Penrose pseudoinverse of quaternion matrices. This method provides
    global linear convergence in expectation through cheap randomized projections
    onto sketched identity constraints.

    The method works by drawing random sketches and projecting onto the constraint
    set {X: X*Y_k = Omega_k} where Y_k = A*Omega_k and Omega_k is a random sketch.
    This provides a block Kaczmarz-style approach that converges globally.
    """

    def __init__(
        self,
        block_size: int = 16,
        max_iter: int = 1000,
        tol: float = 1e-6,
        test_sketch_size: int = 8,
        verbose: bool = False,
        seed: int = None,
        column_solver: str = "qr",  # "qr" or "spd"
    ):
        """
        Initialize the RSP-Q solver.

        Parameters:
        -----------
        block_size : int, optional
            Size of the random sketch block r (default: 16)
        max_iter : int, optional
            Maximum number of iterations (default: 1000)
        tol : float, optional
            Convergence tolerance (default: 1e-6)
        test_sketch_size : int, optional
            Size of the test sketch for convergence check (default: 8)
        verbose : bool, optional
            Whether to print convergence information (default: False)
        seed : int, optional
            Random seed for reproducibility (default: None)
        """
        self.block_size = block_size
        self.max_iter = max_iter
        self.tol = tol
        self.test_sketch_size = test_sketch_size
        self.verbose = verbose
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.column_solver = column_solver.lower() if isinstance(column_solver, str) else "qr"

    def _generate_random_sketch(self, rows: int, cols: int) -> np.ndarray:
        """
        Generate a random quaternion sketch matrix with i.i.d. standard normal entries.

        Parameters:
        -----------
        rows : int
            Number of rows in the sketch
        cols : int
            Number of columns in the sketch

        Returns:
        --------
        np.ndarray
            Random quaternion matrix of shape (rows, cols)
        """
        # Generate 4 independent standard normal components
        real_part = np.random.randn(rows, cols)
        i_part = np.random.randn(rows, cols)
        j_part = np.random.randn(rows, cols)
        k_part = np.random.randn(rows, cols)

        # Stack and convert to quaternion array
        sketch_real = np.stack([real_part, i_part, j_part, k_part], axis=-1)
        return quaternion.as_quat_array(sketch_real)

    def _solve_spd_quat(
        self, G: np.ndarray, B: np.ndarray, tol: float = 1e-8, max_iter: int = 200
    ) -> tuple[np.ndarray, bool]:
        """
        Solve G X = B where G is quaternion Hermitian SPD (r x r) and B is (r x m).
        
        Uses a simple (unpreconditioned) Conjugate Gradient in quaternion arithmetic.
        This is a micro-solver used internally by the RSP-Q row variant.
        
        Parameters:
        -----------
        G : np.ndarray
            Quaternion Hermitian positive definite matrix of shape (r, r)
        B : np.ndarray
            Right-hand side quaternion matrix of shape (r, m)
        tol : float, optional
            Convergence tolerance for CG iterations (default: 1e-8)
        max_iter : int, optional
            Maximum number of CG iterations (default: 200)
            
        Returns:
        --------
        tuple[np.ndarray, bool]
            X : np.ndarray
                Solution matrix of shape (r, m) such that G @ X = B
            success : bool
                True if CG converged successfully, False otherwise
                
        Notes:
        ------
        - Applies mild symmetrization to suppress roundoff: G = 0.5 * (G + G^H)
        - Uses Fletcher-Reeves CG with real inner products
        - Returns success=False if CG fails to converge or encounters numerical issues
        - This is an internal helper function for RSP-Q row variant implementation
        """
        # Mild symmetrization to suppress roundoff
        G = 0.5 * (G + quat_hermitian(G))

        r, m = B.shape
        X = np.zeros((r, m), dtype=np.quaternion)

        def real_inner(u: np.ndarray, v: np.ndarray) -> float:
            s = 0.0
            for i in range(u.shape[0]):
                prod = u[i].conjugate() * v[i]  # u^H v (quaternion)
                s += quaternion.as_float_array(prod)[0]  # Re(u^H v)
            return float(s)

        ok = True
        for j in range(m):
            b = B[:, j].copy()
            x = np.zeros(r, dtype=np.quaternion)
            rvec = b.copy()
            p = rvec.copy()
            rsold = real_inner(rvec, rvec)
            bnorm = max(1e-16, np.sqrt(rsold))
            for _ in range(max_iter):
                Ap = quat_matmat(G, p.reshape(r, 1)).reshape(r)
                pAp = real_inner(p, Ap)
                if pAp <= 0 or abs(pAp) < 1e-18:
                    ok = False
                    break
                alpha = rsold / pAp
                x = x + alpha * p
                rvec = rvec - alpha * Ap
                rsnew = real_inner(rvec, rvec)
                if np.sqrt(rsnew) / bnorm <= tol:
                    break
                beta = rsnew / rsold
                p = rvec + beta * p
                rsold = rsnew
            X[:, j] = x
            if np.sqrt(rsold) / bnorm > max(1e-6, tol * 10):
                ok = False
        return X, ok

    def _invert_quat_small(self, A: np.ndarray, ns_iters: int = 12) -> np.ndarray:
        """
        Compute a small quaternion matrix inverse using Newton–Schulz iteration.
        
        This is a fallback method for small matrix inversion when CG-based methods fail.
        Assumes A is well-conditioned Hermitian positive definite (or close).
        
        Parameters:
        -----------
        A : np.ndarray
            Small quaternion matrix to invert, assumed to be well-conditioned
        ns_iters : int, optional
            Number of Newton-Schulz iterations (default: 12)
            
        Returns:
        --------
        np.ndarray
            Approximate inverse of A with same shape as A
            
        Notes:
        ------
        - Uses Newton-Schulz iteration: X_{k+1} = X_k (2I - A X_k)
        - Initializes with X_0 = (2/tr(A)) * I for better convergence
        - This is an internal helper function used as fallback in RSP-Q methods
        - Assumes A is small and well-conditioned for reliable convergence
        """
        r = A.shape[0]
        I = quat_eye(r)
        # Scale by approximate spectral norm via trace real part
        tr = 0.0
        for i in range(r):
            tr += quaternion.as_float_array(A[i, i])[0]
        alpha = 2.0 / max(abs(tr), 1e-12)
        X = alpha * I
        for _ in range(ns_iters):
            AX = quat_matmat(A, X)
            X = quat_matmat(X, (2.0 * I - AX))
        return X

    def _compute_pseudoinverse_thin(self, Y: np.ndarray) -> np.ndarray:
        """
        Compute pseudoinverse of thin matrix Y using (Y^H Y)^{-1} Y^H.
        
        Uses quaternion-native operations throughout. This method computes the
        pseudoinverse of a thin matrix Y by solving the normal equations.
        
        Parameters:
        -----------
        Y : np.ndarray
            Thin quaternion matrix of shape (m, r) with m >= r
            
        Returns:
        --------
        np.ndarray
            Pseudoinverse of Y, shape (r, m)
            
        Notes:
        ------
        - Computes Y^H Y and inverts it using Newton-Schulz iteration
        - Returns (Y^H Y)^{-1} Y^H as the pseudoinverse
        - Uses scaled identity initialization for better convergence
        - This is an internal helper function for RSP-Q column variant
        - Assumes Y is well-conditioned for reliable inversion
        """
        # Compute Y^H Y (this is r x r)
        Y_H = quat_hermitian(Y)
        YHY = quat_matmat(Y_H, Y)

        # For small matrices, we can use the existing quaternion solver infrastructure
        # We need to solve (Y^H Y) X = Y^H for X = (Y^H Y)^{-1} Y^H

        try:
            # Use the existing Newton-Schulz infrastructure for small matrix inversion
            # Initialize with scaled identity
            r = YHY.shape[0]
            I_r = quat_eye(r)

            # Simple quaternion matrix inversion using the identity A^{-1} ≈ (2/tr(A)) * I initially
            trace_YHY = sum(YHY[i, i] for i in range(r))
            trace_norm = abs(trace_YHY)

            if trace_norm > 1e-12:
                alpha = 2.0 / trace_norm
                YHY_inv = alpha * I_r

                # Few Newton-Schulz steps for small matrix: X_{k+1} = X_k (2I - A X_k)
                for _ in range(5):  # Just a few steps for small matrices
                    AX = quat_matmat(YHY, YHY_inv)
                    residual = 2.0 * I_r - AX
                    YHY_inv = quat_matmat(YHY_inv, residual)

                    # Check if we have reasonable convergence
                    test_product = quat_matmat(YHY, YHY_inv)
                    error = quat_frobenius_norm(test_product - I_r)
                    if error < 1e-8:
                        break
            else:
                # Matrix is essentially zero, return zero pseudoinverse
                return np.zeros_like(Y_H)

        except:
            # Fallback: return a crude approximation
            r = YHY.shape[0]
            YHY_inv = quat_eye(r) * 0.1

        # Return (Y^H Y)^{-1} Y^H
        return quat_matmat(YHY_inv, Y_H)

    def compute_column_variant(self, A: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Compute pseudoinverse using column variant RSP-Q for full column rank A.
        
        Solves XA = I_n by iterative sketch-and-project updates. This variant is
        designed for tall matrices (m >= n) and uses randomized sketching to
        reduce computational complexity.
        
        Parameters:
        -----------
        A : np.ndarray
            Full column rank quaternion matrix of shape (m, n) with m >= n
            
        Returns:
        --------
        tuple[np.ndarray, dict]
            X : np.ndarray
                Pseudoinverse of A, shape (n, m)
            convergence_info : dict
                Dictionary containing convergence information:
                - 'iterations': Number of iterations performed
                - 'residual_norms': List of residual norms per iteration
                - 'iteration_times': List of CPU times per iteration
                - 'total_time': Total computation time
                - 'converged': Boolean indicating if convergence was achieved
                
        Notes:
        ------
        - Initializes with X_0 = (1/||A||_F^2) * A^H
        - Uses random sketches of size block_size for each iteration
        - Supports both QR and SPD update strategies via column_solver parameter
        - Monitors convergence using a test sketch to avoid computing full residuals
        - Global linear convergence in expectation for well-conditioned matrices
        """
        m, n = A.shape

        if m < n:
            raise ValueError("Column variant requires m >= n (full column rank)")

        # Initialize X_0 = alpha * A^H with alpha = 1 / ||A||_F^2 (matches debug/reference)
        A_H = quat_hermitian(A)
        alpha = 1.0 / max(quat_frobenius_norm(A) ** 2, 1e-16)
        X = alpha * A_H

        # Generate test sketch for convergence monitoring
        Pi = self._generate_random_sketch(n, self.test_sketch_size)
        A_Pi = quat_matmat(A, Pi)  # Precompute A*Pi

        # Storage for convergence history
        residual_norms = []
        iteration_times = []

        if self.verbose:
            print("RSP-Q Column Variant: Starting (alpha=1/||A||_F^2)")
            print(f"Matrix size: {m}x{n}, block size: {self.block_size}")

        for k in range(self.max_iter):
            start_time = time.time()

            # Draw random sketch Omega_k
            Omega_k = self._generate_random_sketch(n, self.block_size)

            # Compute Y_k = A * Omega_k
            Y_k = quat_matmat(A, Omega_k)

            # Compute residual R_k = Omega_k - X_k * Y_k
            X_Y_k = quat_matmat(X, Y_k)
            R_k = Omega_k - X_Y_k

            # Compute update via selected strategy
            try:
                if self.column_solver == "spd":
                    # SPD-based thin pseudoinverse: Z = (Y^H Y)^{-1} Y^H
                    Y_H = quat_hermitian(Y_k)           # (r x m)
                    G = quat_matmat(Y_H, Y_k)           # (r x r)
                    # Stabilize
                    rdim = G.shape[0]
                    G = 0.5 * (G + quat_hermitian(G)) + 1e-10 * quat_eye(rdim)
                    Z_k, ok = self._solve_spd_quat(G, Y_H, tol=1e-8, max_iter=200)  # (r x m)
                    if not ok:
                        # NS fallback on G
                        G_inv = self._invert_quat_small(G, ns_iters=16)
                        Z_k = quat_matmat(G_inv, Y_H)
                    X = X + quat_matmat(R_k, Z_k)
                else:
                    # Default QR-based update: Y = U R, solve R Z = U^H
                    try:
                        from .decomp.qsvd import qr_qua  # package context
                    except Exception:
                        from quatica.decomp.qsvd import qr_qua  # script context
                    U_k, RY_k = qr_qua(Y_k)  # U_k: (m x r), RY_k: (r x r) upper
                    U_k_H = quat_hermitian(U_k)  # (r x m)
                    Z_k = _solve_upper_triangular_quat(RY_k, U_k_H)  # (r x m)
                    X = X + quat_matmat(R_k, Z_k)
            except Exception:
                if self.verbose:
                    print(f"Iteration {k}: {self.column_solver} update failed, redrawing sketch")
                continue

            # Check convergence using test sketch
            X_A_Pi = quat_matmat(X, A_Pi)
            residual = Pi - X_A_Pi
            residual_norm = quat_frobenius_norm(residual) / quat_frobenius_norm(Pi)

            iteration_time = time.time() - start_time
            residual_norms.append(residual_norm)
            iteration_times.append(iteration_time)

            if self.verbose and k % 10 == 0:
                print(f"Iteration {k}: residual = {residual_norm:.6e}")

            # Check for convergence
            if residual_norm <= self.tol:
                if self.verbose:
                    print(
                        f"Converged in {k + 1} iterations with residual {residual_norm:.6e}"
                    )
                break

        else:
            if self.verbose:
                print(
                    f"Maximum iterations ({self.max_iter}) reached. Final residual: {residual_norms[-1]:.6e}"
                )

        convergence_info = {
            "iterations": len(residual_norms),
            "residual_norms": residual_norms,
            "iteration_times": iteration_times,
            "total_time": sum(iteration_times),
            "converged": residual_norms[-1] <= self.tol if residual_norms else False,
        }

        return X, convergence_info

    def compute_row_variant(self, A: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        RSP-Q row variant: solve A X = I_m for full row rank A (m <= n).
        
        This variant is designed for wide matrices (m <= n) and uses left sketching
        to project onto sketched identity constraints. The update formula is:
        X <- X + Z^H (Z Z^H)^{-1} (S^H - Z X), where Z = S^H A.
        
        Parameters:
        -----------
        A : np.ndarray
            Full row rank quaternion matrix of shape (m, n) with m <= n
            
        Returns:
        --------
        tuple[np.ndarray, dict]
            X : np.ndarray
                Pseudoinverse of A, shape (n, m)
            info : dict
                Dictionary containing convergence information:
                - 'iterations': Number of iterations performed
                - 'residual_norms': List of residual norms per iteration
                - 'iteration_times': List of CPU times per iteration
                - 'total_time': Total computation time
                - 'converged': Boolean indicating if convergence was achieved
                
        Notes:
        ------
        - Initializes with X_0 = 0 (zero initialization is sufficient for RSP)
        - Uses left sketches S of size block_size for each iteration
        - Solves (Z Z^H) W = R using CG micro-solver with fallback to Newton-Schulz
        - Monitors convergence using proxy residual with test sketch
        - Applies tiny ridge regularization (1e-10) to Gram matrix for stability
        - Global linear convergence in expectation for well-conditioned matrices
        """
        m, n = A.shape
        if m > n:
            raise ValueError("Row variant requires m <= n (full row rank)")

        # Clamp block size to safe range
        r = max(1, min(self.block_size, m, n))
        # Init: zero is fine for RSP
        X = np.zeros((n, m), dtype=np.quaternion)

        # Cheap test-sketch to monitor AX ≈ I_m without forming AX every time
        Theta = self._generate_random_sketch(m, self.test_sketch_size)  # (m x s)
        Theta_norm = max(quat_frobenius_norm(Theta), 1e-30)

        residual_norms, iteration_times = [], []
        if self.verbose:
            print(f"RSP-Q Row Variant: m={m}, n={n}, block={r}")

        for k in range(self.max_iter):
            t0 = time.time()

            # Left sketch and block
            S = self._generate_random_sketch(m, r)  # (m x r)
            S_H = quat_hermitian(S)  # (r x m)
            Z = quat_matmat(S_H, A)  # (r x n)

            # Residual on sketched constraints
            ZX = quat_matmat(Z, X)  # (r x m)
            Rres = S_H - ZX  # (r x m)

            # Solve (Z Z^H) W = R, then X += Z^H W
            Z_H = quat_hermitian(Z)  # (n x r)
            ZZ_H = quat_matmat(Z, Z_H)  # (r x r)
            # Stabilize Gram matrix with tiny ridge
            ZZ_H = 0.5 * (ZZ_H + quat_hermitian(ZZ_H)) + 1e-10 * quat_eye(r)
            W, ok = self._solve_spd_quat(ZZ_H, Rres, tol=1e-8, max_iter=200)
            if not ok:
                # Fallback: explicit small inverse via Newton–Schulz
                ZZ_H_inv = self._invert_quat_small(ZZ_H, ns_iters=16)
                W = quat_matmat(ZZ_H_inv, Rres)
                ok = True
            if not ok:
                if self.verbose:
                    print(f"Iter {k}: ill-conditioned sketch; redrawing")
                continue
            X = X + quat_matmat(Z_H, W)  # (n x r) @ (r x m) -> (n x m)

            # Proxy stopping: test AX Theta ≈ Theta
            AX_Theta = quat_matmat(A, quat_matmat(X, Theta))  # (m x s)
            proxy = quat_frobenius_norm(Theta - AX_Theta) / Theta_norm

            elapsed = time.time() - t0
            residual_norms.append(proxy)
            iteration_times.append(elapsed)

            if self.verbose and k % 10 == 0:
                print(f"Iter {k}: proxy={proxy:.3e}")

            if proxy <= self.tol:
                if self.verbose:
                    print(f"Converged in {k+1} iters: proxy={proxy:.3e}")
                break
        else:
            if self.verbose:
                print(f"Reached max_iter; final proxy={residual_norms[-1]:.3e}")

        info = {
            "iterations": len(residual_norms),
            "residual_norms": residual_norms,
            "iteration_times": iteration_times,
            "total_time": sum(iteration_times),
            "converged": bool(residual_norms and residual_norms[-1] <= self.tol),
        }
        return X, info

    def compute(self, A: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Compute the Moore-Penrose pseudoinverse using RSP-Q.
        
        Automatically selects column or row variant based on matrix dimensions.
        This is the main entry point for RSP-Q pseudoinverse computation.
        
        Parameters:
        -----------
        A : np.ndarray
            Quaternion matrix of shape (m, n)
            
        Returns:
        --------
        tuple[np.ndarray, dict]
            X : np.ndarray
                Moore-Penrose pseudoinverse of A, shape (n, m)
            convergence_info : dict
                Dictionary containing convergence information from the selected variant
                
        Notes:
        ------
        - For m >= n (tall/square matrices): uses column variant (compute_column_variant)
        - For m < n (wide matrices): uses row variant (compute_row_variant)
        - Automatically clamps block_size to safe range [1, min(m, n)]
        - Both variants provide global linear convergence in expectation
        - The choice between variants is optimal for computational efficiency
        """
        m, n = A.shape
        # Clamp block size once
        self.block_size = max(1, min(self.block_size, m, n))

        if m >= n:
            # Use column variant for tall/square matrices
            return self.compute_column_variant(A)
        else:
            # Use row variant for wide matrices
            return self.compute_row_variant(A)


class HybridRSPNewtonSchulz:
    """
    Hybrid RSP-Q + NS (column variant): alternate T randomized sketch-and-project
    steps with one exact hyperpower (order p) step on the right residual.

    Parameters
    ----------
    r : int
        Sketch block size for RSP-Q phase
    p : int
        Hyperpower order for NS step (e.g., 2, 4, 8)
    T : int
        Number of RSP steps per cycle before one NS step
    tol : float
        Stopping tolerance for proxy residual using test sketch Pi
    max_iter : int
        Maximum total RSP steps (cycles*T bounded by this)
    verbose : bool
        Verbose logging
    seed : int | None
        RNG seed
    """

    def __init__(
        self,
        r: int = 12,
        p: int = 4,
        T: int = 5,
        tol: float = 1e-6,
        max_iter: int = 1000,
        verbose: bool = False,
        seed: int | None = None,
        column_solver: str = "qr",  # "qr" or "spd" for RSP step
    ) -> None:
        self.r = r
        self.p = p
        self.T = T
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.column_solver = column_solver.lower() if isinstance(column_solver, str) else "qr"

    def _rsp_step_column(self, A: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Perform one RSP-Q column step using either thin QR or SPD micro-solver.
        
        This is an internal helper method for the hybrid RSP+NS algorithm.
        
        Parameters:
        -----------
        A : np.ndarray
            Quaternion matrix of shape (m, n) with m >= n
        X : np.ndarray
            Current pseudoinverse approximation of shape (n, m)
            
        Returns:
        --------
        np.ndarray
            Updated pseudoinverse approximation of shape (n, m)
            
        Notes:
        ------
        - Draws random sketch Omega of size (n x r) where r is the block size
        - Computes Y = A @ Omega and residual R = Omega - X @ Y
        - Updates X using either QR decomposition or SPD solve based on column_solver
        - Returns X unchanged if the update step fails
        """
        m, n = A.shape
        # Draw right sketch Omega (n x r)
        real = np.random.randn(n, self.r)
        ii = np.random.randn(n, self.r)
        jj = np.random.randn(n, self.r)
        kk = np.random.randn(n, self.r)
        Omega = quaternion.as_quat_array(np.stack([real, ii, jj, kk], axis=-1))
        Y = quat_matmat(A, Omega)
        R_id = Omega - quat_matmat(X, Y)
        try:
            if self.column_solver == "spd":
                # SPD-based thin pseudoinverse on Y
                Y_H = quat_hermitian(Y)
                G = quat_matmat(Y_H, Y)
                rdim = G.shape[0]
                G = 0.5 * (G + quat_hermitian(G)) + 1e-10 * quat_eye(rdim)
                # Reuse RSP-Q micro-solver and fallback
                Z, ok = RandomizedSketchProjectPseudoinverse._solve_spd_quat(self, G, Y_H, tol=1e-8, max_iter=200)
                if not ok:
                    G_inv = RandomizedSketchProjectPseudoinverse._invert_quat_small(self, G, ns_iters=16)
                    Z = quat_matmat(G_inv, Y_H)
                return X + quat_matmat(R_id, Z)
            else:
                # QR path
                try:
                    from .decomp.qsvd import qr_qua
                except Exception:
                    from quatica.decomp.qsvd import qr_qua
                U, R = qr_qua(Y)
                U_H = quat_hermitian(U)
                Z = _solve_upper_triangular_quat(R, U_H)
                return X + quat_matmat(R_id, Z)
        except Exception:
            return X  # skip on failure

    def _ns_hyperpower_right(self, A: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Perform one exact NS/hyperpower step on right residual.
        
        This is an internal helper method for the hybrid RSP+NS algorithm.
        Implements the hyperpower update: X <- (sum_{i=0}^{p-1} F^i) X, where F = I - X A.
        
        Parameters:
        -----------
        A : np.ndarray
            Quaternion matrix of shape (m, n) with m >= n
        X : np.ndarray
            Current pseudoinverse approximation of shape (n, m)
            
        Returns:
        --------
        np.ndarray
            Updated pseudoinverse approximation of shape (n, m)
            
        Notes:
        ------
        - Computes F = I - X @ A as the right residual matrix
        - Accumulates S = I + F + F^2 + ... + F^{p-1} using p-1 matrix multiplications
        - Returns S @ X as the updated approximation
        - This provides exact hyperpower acceleration for the right residual
        """
        n = A.shape[1]
        I = quat_eye(n)
        F = I - quat_matmat(X, A)
        # Accumulate S = I + F + F^2 + ... + F^{p-1}
        S = I.copy()
        F_power = F.copy()
        for _ in range(1, self.p):
            S = S + F_power
            F_power = quat_matmat(F_power, F)
        return quat_matmat(S, X)

    def compute(self, A: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Compute pseudoinverse using hybrid RSP-Q + NS algorithm.
        
        Alternates T randomized sketch-and-project steps with one exact hyperpower
        (order p) step on the right residual. This combines the efficiency of
        randomized methods with the accuracy of exact Newton-Schulz iterations.
        
        Parameters:
        -----------
        A : np.ndarray
            Quaternion matrix of shape (m, n) with m >= n (full column rank)
            
        Returns:
        --------
        tuple[np.ndarray, dict]
            X : np.ndarray
                Pseudoinverse of A, shape (n, m)
            info : dict
                Dictionary containing convergence information:
                - 'iterations_rsp': Number of RSP iterations performed
                - 'residual_norms': List of residual norms after each cycle
                - 'total_time': Total computation time
                - 'converged': Boolean indicating if convergence was achieved
                - 'r': Block size used for RSP steps
                - 'p': Hyperpower order used for NS steps
                - 'T': Number of RSP steps per cycle
                
        Notes:
        ------
        - Currently supports only column variant (m >= n)
        - Initializes with X_0 = (1/||A||_F^2) * A^H
        - Performs T RSP steps followed by one NS hyperpower step per cycle
        - Monitors convergence using test sketch to avoid full residual computation
        - Combines benefits of randomized efficiency and exact convergence
        """
        m, n = A.shape
        if m < n:
            raise ValueError(
                "HybridRSPNewtonSchulz currently supports column variant (m >= n)"
            )
        # Init X = alpha A^H
        A_H = quat_hermitian(A)
        alpha = 1.0 / max(quat_frobenius_norm(A) ** 2, 1e-16)
        X = alpha * A_H
        # Test sketch Pi and A Pi
        real = np.random.randn(n, min(6, n))
        ii = np.random.randn(n, min(6, n))
        jj = np.random.randn(n, min(6, n))
        kk = np.random.randn(n, min(6, n))
        Pi = quaternion.as_quat_array(np.stack([real, ii, jj, kk], axis=-1))
        A_Pi = quat_matmat(A, Pi)
        Pi_norm = max(quat_frobenius_norm(Pi), 1e-30)

        iter_rsp = 0
        residuals = []
        t0 = time.time()
        while iter_rsp < self.max_iter:
            # T randomized steps
            for _ in range(self.T):
                X = self._rsp_step_column(A, X)
                iter_rsp += 1
                # Check convergence proxy occasionally
                if iter_rsp % 10 == 0:
                    proxy = quat_frobenius_norm(Pi - quat_matmat(X, A_Pi)) / Pi_norm
                    residuals.append(proxy)
                    if self.verbose:
                        print(f"[Hybrid] iter={iter_rsp} proxy={proxy:.2e}")
                    if proxy <= self.tol:
                        break
            # Hyperpower step
            X = self._ns_hyperpower_right(A, X)
            proxy = quat_frobenius_norm(Pi - quat_matmat(X, A_Pi)) / Pi_norm
            residuals.append(proxy)
            if self.verbose:
                print(f"[Hybrid] after-NS proxy={proxy:.2e}")
            if proxy <= self.tol:
                break

        info = {
            "iterations_rsp": iter_rsp,
            "residual_norms": residuals,
            "total_time": time.time() - t0,
            "converged": (residuals[-1] <= self.tol) if residuals else False,
            "r": self.r,
            "p": self.p,
            "T": self.T,
        }
        return X, info


class CGNEQSolver:
    """
    Conjugate Gradient on the Normal Equations (CGNE–Q) to solve XA = I_n in matrix form.

    Minimizes f(X) = 1/2 || X A - I_n ||_F^2 with quaternion-native operations.

    - Column case (full column rank): A in H^{m x n} with m >= n
    - Iterates stay in span{A^H} when initialized with X0 = alpha * A^H
    - Converges to A^† as ||I_n - X_k A||_F -> 0
    """

    def __init__(
        self,
        tol: float = 1e-6,
        max_iter: int = 500,
        verbose: bool = False,
        preconditioner_rank: int = 0,
        seed: int | None = None,
    ) -> None:
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.preconditioner_rank = max(0, preconditioner_rank)
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def _build_right_preconditioner(self, A: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Build a thin Nyström-type right preconditioner M ≈ (A A^H)^{-1}.
        
        Constructs M = Y (Y^H Y)^{-1} (Y^H Y)^{-1} Y^H, where Y = A Ω and Ω is a
        random sketch. This provides an approximate inverse for right preconditioning.
        
        Parameters:
        -----------
        A : np.ndarray
            Quaternion matrix of shape (m, n)
        n : int
            Number of columns in A
            
        Returns:
        --------
        tuple[np.ndarray, np.ndarray, np.ndarray] | None
            If preconditioner_rank > 0: (Y, G_inv, G_inv) where:
                Y : np.ndarray
                    Sketch Y = A @ Ω of shape (m, r)
                G_inv : np.ndarray
                    Inverse of G = Y^H @ Y of shape (r, r)
            If preconditioner_rank <= 0: None
            
        Notes:
        ------
        - Draws random sketch Ω of shape (n, r) where r = preconditioner_rank
        - Computes Y = A @ Ω and G = Y^H @ Y
        - Inverts G using Newton-Schulz iteration for small matrices
        - Returns None if preconditioner_rank is 0 or exceeds matrix dimensions
        - This is an internal helper method for CGNE-Q preconditioning
        """
        r = self.preconditioner_rank
        if r <= 0 or r > n:
            return None
        # Draw Ω (n x r)
        real = np.random.randn(n, r)
        ii = np.random.randn(n, r)
        jj = np.random.randn(n, r)
        kk = np.random.randn(n, r)
        Omega = quaternion.as_quat_array(np.stack([real, ii, jj, kk], axis=-1))
        # Y = A Ω (m x r)
        Y = quat_matmat(A, Omega)
        # G = Y^H Y (r x r)
        Y_H = quat_hermitian(Y)
        G = quat_matmat(Y_H, Y)
        # Invert G via small Newton–Schulz
        G_inv = RandomizedSketchProjectPseudoinverse._invert_quat_small(self, G, ns_iters=16)
        return (Y, G_inv, G_inv)

    def _apply_right_prec(self, Z: np.ndarray, prec: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> np.ndarray:
        """
        Apply right preconditioner Z @ M using cached representation.
        
        Applies the right preconditioner M to Z using the cached (Y, G_inv, G_inv)
        representation from _build_right_preconditioner.
        
        Parameters:
        -----------
        Z : np.ndarray
            Quaternion matrix to be preconditioned
        prec : tuple[np.ndarray, np.ndarray, np.ndarray] | None
            Preconditioner representation (Y, G_inv, G_inv) or None
            
        Returns:
        --------
        np.ndarray
            Preconditioned matrix Z @ M with same shape as Z
            
        Notes:
        ------
        - If prec is None, returns Z unchanged (no preconditioning)
        - Otherwise computes Z @ M = ((Z @ Y) @ G_inv) @ G_inv @ Y^H
        - This is an internal helper method for CGNE-Q preconditioning
        """
        if prec is None:
            return Z
        Y, Ginv1, Ginv2 = prec
        ZY = quat_matmat(Z, Y)          # (n x r)
        T = quat_matmat(ZY, Ginv1)      # (n x r)
        T = quat_matmat(T, Ginv2)       # (n x r)
        return quat_matmat(T, quat_hermitian(Y))  # (n x m)

    def compute(self, A: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Compute pseudoinverse using Conjugate Gradient on Normal Equations (CGNE–Q).
        
        Minimizes f(X) = 1/2 || X A - I_n ||_F^2 with quaternion-native operations.
        This method provides global convergence to the Moore-Penrose pseudoinverse
        by solving the normal equations via conjugate gradient.
        
        Parameters:
        -----------
        A : np.ndarray
            Quaternion matrix of shape (m, n) with m >= n (full column rank)
            
        Returns:
        --------
        tuple[np.ndarray, dict]
            X : np.ndarray
                Pseudoinverse of A, shape (n, m)
            info : dict
                Dictionary containing convergence information:
                - 'iterations': Number of CG iterations performed
                - 'residual_norms': List of relative residual norms per iteration
                - 'iteration_times': List of CPU times per iteration
                - 'total_time': Total computation time
                - 'converged': Boolean indicating if convergence was achieved
                - 'preconditioner_rank': Rank of preconditioner used (0 if none)
                
        Notes:
        ------
        - Requires m >= n (full column rank matrices)
        - Initializes with X_0 = (1/||A||_F^2) * A^H to stay in span{A^H}
        - Uses Fletcher-Reeves CG with exact line search
        - Supports optional thin Nyström-type right preconditioning
        - All operations are quaternion-native with no embeddings
        - Converges to A^† as ||I_n - X_k A||_F -> 0
        - Per iteration: one A multiply and one A^H multiply
        """
        m, n = A.shape
        if m < n:
            raise ValueError("CGNE–Q requires m >= n (full column rank)")

        I_n = quat_eye(n)
        Inorm = max(quat_frobenius_norm(I_n), 1e-30)
        A_H = quat_hermitian(A)

        # Safe alpha based on Frobenius norm
        alpha0 = 1.0 / max(quat_frobenius_norm(A) ** 2, 1e-16)
        X = alpha0 * A_H

        # Residuals and timings
        residual_norms: list[float] = []
        times: list[float] = []

        # Optional thin right preconditioner
        prec = self._build_right_preconditioner(A, n)

        # Initialize CGNE quantities
        R = I_n - quat_matmat(X, A)        # (n x n)
        Z = quat_matmat(R, A_H)            # (n x m)
        Zp = self._apply_right_prec(Z, prec)
        D = Zp.copy()

        for k in range(self.max_iter):
            t0 = time.time()
            W = quat_matmat(D, A)                      # (n x n)
            Wn = quat_frobenius_norm(W)
            if Wn <= 1e-20:
                break

            Zn = quat_frobenius_norm(Zp)
            alpha_k = (Zn * Zn) / (Wn * Wn)

            # Update X and residuals
            X = X + alpha_k * D
            R = R - alpha_k * W

            rnorm_rel = quat_frobenius_norm(R) / Inorm
            residual_norms.append(rnorm_rel)
            times.append(time.time() - t0)

            if self.verbose and (k < 5 or k % 10 == 0):
                print(f"[CGNE–Q] iter={k} rel_res={rnorm_rel:.3e}")

            if rnorm_rel <= self.tol:
                break

            # Next Z, direction update
            Z_new = quat_matmat(R, A_H)
            Zp_new = self._apply_right_prec(Z_new, prec)
            Zn_new = quat_frobenius_norm(Zp_new)
            beta_k = (Zn_new * Zn_new) / max(Zn * Zn, 1e-30)
            D = Zp_new + beta_k * D
            Z = Z_new
            Zp = Zp_new

        info = {
            'iterations': len(residual_norms),
            'residual_norms': residual_norms,
            'iteration_times': times,
            'total_time': sum(times),
            'converged': bool(residual_norms and residual_norms[-1] <= self.tol),
            'preconditioner_rank': self.preconditioner_rank,
        }
        return X, info
