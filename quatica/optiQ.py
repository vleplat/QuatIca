"""
Optimization solvers for quaternion Hermitian matrices (pure quaternion algebra).

This module provides an equality-constrained Newton method for the
log-det barrier formulation of quaternionic semidefinite programs (SDP):

  minimize <C, X> - mu * logdet(X)  subject to  A(X) = b,  X ≻ 0,  X ∈ H_n(H),

where <U, V> = Re tr(U^H V) is the real inner product on H_n(H), and A is a
linear map with Hermitian measurement matrices {H_i}.

All operations are quaternion-native using QuatIca utilities; no real/complex
embeddings are used.

References/Notes:
- Gradient/Hessian: ∇ logdet(X) = X^{-1},  ∇² logdet(X)[H] = - X^{-1} H X^{-1}
- Barrier Newton KKT via Schur complement with H^{-1}[W] = (1/mu) X W X
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import quaternion as nq

# Package-safe imports consistent with QuatIca style
try:
    from .utils import (
        quat_eye,
        quat_matmat,
        ishermitian,
        quat_frobenius_norm,
        real_contract,
    )
    from .decomp.eigen import (
        quaternion_eigenvalues,
        quaternion_eigendecomposition,
    )
    from .data_gen import generate_random_unitary_matrix
    from .solver import QGMRESSolver
except Exception:  # script context fallback
    from quatica.utils import (
        quat_eye,
        quat_matmat,
        ishermitian,
        quat_frobenius_norm,
        real_contract,
    )
    from quatica.decomp.eigen import (
        quaternion_eigenvalues,
        quaternion_eigendecomposition,
    )
    from quatica.data_gen import generate_random_unitary_matrix
    from quatica.solver import QGMRESSolver


# ------------------------------
# Low-level quaternion helpers
# ------------------------------

def qherm(A: np.ndarray) -> np.ndarray:
    """Hermitian projection: (A + A^H)/2."""
    return 0.5 * (A + qadj(A))

def qzeros(n: int, m: int) -> np.ndarray:
    """Return an n×m quaternion zero matrix."""
    return np.zeros((n, m), dtype=nq.quaternion)


def qeye(n: int) -> np.ndarray:
    """Return the n×n quaternion identity matrix."""
    return quat_eye(n)


def qadj(A: np.ndarray) -> np.ndarray:
    """Quaternion adjoint (conjugate transpose)."""
    # Note: this is the quaternion adjoint A^H = (conj A)^T.
    return A.conjugate().T


def qmm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Quaternion matrix multiply using QuatIca optimized matmul."""
    return quat_matmat(A, B)


def inner_real(U: np.ndarray, V: np.ndarray) -> float:
    """
    Real inner product <U, V>_R = Re tr(U^H V) using quaternion adjoint.

    Notes
    -----
    Must use the adjoint U^H, not the Hermitian part, to preserve adjointness
    and accurate Gram/Schur assembly.
    """
    UH = qadj(U)
    M = quat_matmat(UH, V)
    s = 0.0
    n = M.shape[0]
    for i in range(n):
        s += float(M[i, i].real)
    return s


# ------------------------------
# Constraint basis conditioning
# ------------------------------

def _build_orthonormal_ops(H_list: list[np.ndarray],
                          jitter: float = 1e-12,
                          *,
                          vectorized: bool = True) -> dict:
    """Construct orthonormalized (hat-space) constraint closures via real Gram Cholesky.

    Let <U,V>_R = Re tr(U^H V) on Hermitian quaternion matrices. Define the Gram matrix
      G_ij = <H_i, H_j>_R.
    With a Cholesky factorization G = R^T R (R upper-triangular), define hat operators:
      A_hat(X)  = R^{-T} A(X)
      AT_hat(y) = A^*(R^{-1} y)
      transform_b(b) = R^{-T} b.

    If constraints are independent, A_hat AT_hat = I_m.

    Performance note
    ----------------
    During Newton steps, A_hat is called O(m) times per iteration (to assemble the Schur complement).
    The default vectorized path precomputes the hat-basis matrices H_hat,i and evaluates A_hat(X) using
    a broadcasted elementwise multiply + reduction, avoiding Python loops over i and repeated calls to
    inner_real.
    """
    if len(H_list) == 0:
        raise ValueError("H_list must be non-empty.")
    H_list = [qherm(H) for H in H_list]
    m = len(H_list)

    # Gram matrix G (real)
    G = np.empty((m, m), dtype=float)
    for i, Hi in enumerate(H_list):
        for j, Hj in enumerate(H_list):
            G[i, j] = float(inner_real(Hi, Hj))

    # Cholesky with adaptive jitter
    try:
        R = np.linalg.cholesky(G).T  # G = R^T R, upper-triangular R
    except np.linalg.LinAlgError:
        scale = float(np.linalg.norm(G, 1) + 1.0)
        jj = max(float(jitter) * scale, 1e-15)
        R = np.linalg.cholesky(G + jj * np.eye(m)).T

    invR = np.linalg.solve(R, np.eye(m))

    def transform_b(b: np.ndarray) -> np.ndarray:
        # b_hat = R^{-T} b; solve R^T z = b
        b = np.asarray(b, dtype=float).reshape(-1)
        return np.linalg.solve(R.T, b)

    # Vectorized hat-space constraints: H_hat,i = Σ_j (R^{-T})_{i j} H_j = Σ_j invR[j,i] H_j
    if vectorized:
        try:
            n = int(H_list[0].shape[0])
            H_hat_list = []
            for i in range(m):
                Hi = qzeros(n, n)
                w = invR[:, i]
                for j in range(m):
                    if w[j] != 0.0:
                        Hi = Hi + float(w[j]) * H_list[j]
                H_hat_list.append(qherm(Hi))
            H_hat_stack = np.stack(H_hat_list, axis=0)
            H_hat_conj = np.conjugate(H_hat_stack)

            def A_hat(X: np.ndarray) -> np.ndarray:
                Xh = qherm(X)
                # A_hat_i(X) = Re sum_{ab} conj(H_hat_i[ab]) * X[ab]
                s = np.sum(H_hat_conj * Xh, axis=(1, 2))
                return nq.as_float_array(s)[..., 0].astype(float)

            def AT_hat(y: np.ndarray) -> np.ndarray:
                y = np.asarray(y, dtype=float).reshape(-1)
                Y = np.sum(H_hat_stack * y[:, None, None], axis=0)
                return qherm(Y)

            return dict(
                A_hat=A_hat,
                AT_hat=AT_hat,
                transform_b=transform_b,
                A_unscaled=None,
                R=R,
                H_hat_list=H_hat_list,
            )
        except Exception:
            # Fall back to loop-based implementations below
            pass

    def A_unscaled(X: np.ndarray) -> np.ndarray:
        Xh = qherm(X)
        return np.array([float(inner_real(Hi, Xh)) for Hi in H_list], dtype=float)

    def A_hat(X: np.ndarray) -> np.ndarray:
        rhs = A_unscaled(X)
        return np.linalg.solve(R.T, rhs)

    def AT_hat(y: np.ndarray) -> np.ndarray:
        y_raw = np.linalg.solve(R, np.asarray(y, dtype=float).reshape(-1))
        n = H_list[0].shape[0]
        Y = qzeros(n, n)
        for yi, Hi in zip(y_raw, H_list):
            if yi != 0.0:
                Y = Y + float(yi) * Hi
        return qherm(Y)

    return dict(A_hat=A_hat, AT_hat=AT_hat, transform_b=transform_b, A_unscaled=A_unscaled, R=R)


def eigvalsH(X: np.ndarray) -> np.ndarray:
    """Eigenvalues (real) of a Hermitian quaternion matrix X (with symmetrization)."""
    Xh = 0.5 * (X + qadj(X))
    vals = quaternion_eigenvalues(Xh)
    return np.real(np.asarray(vals, dtype=np.complex128))


def eighH(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Quaternion Hermitian eigendecomposition: X = V diag(lam) V^H.

    Returns
    -------
    lam : np.ndarray of shape (n,)
        Real eigenvalues in unspecified order.
    V : np.ndarray (dtype=quaternion)
        Unitary quaternion eigenvectors.
    """
    Xh = 0.5 * (X + qadj(X))
    lam_c, V = quaternion_eigendecomposition(Xh)
    lam = np.real(np.asarray(lam_c, dtype=np.complex128))
    return lam, V


def invH(X: np.ndarray) -> np.ndarray:
    """
    Inverse of a Hermitian positive definite quaternion matrix.

    Parameters
    ----------
    X : np.ndarray
        Hermitian PD matrix.

    Returns
    -------
    np.ndarray
        X^{-1} computed via eigen-decomposition in quaternion algebra.

    Notes
    -----
    This routine assumes X is strictly PD; it raises if any eigenvalue ≤ 0.
    """
    lam, V = eighH(X)
    if np.any(lam <= 0):
        raise ValueError("invH requires X ≻ 0")
    n = V.shape[0]
    Dinv = qzeros(n, n)
    for i in range(n):
        Dinv[i, i] = nq.quaternion(1.0 / lam[i], 0.0, 0.0, 0.0)
    return qmm(qmm(V, Dinv), qadj(V))


def logdetH(X: np.ndarray) -> float:
    """Return sum(log(lam_i)) for lam_i eigenvalues of PD quaternion X."""
    lam = eigvalsH(X)
    if np.any(lam <= 0):
        raise ValueError("logdetH requires X ≻ 0")
    return float(np.sum(np.log(lam)))


def sqrtH(X: np.ndarray) -> np.ndarray:
    """Hermitian square root via quaternion eigendecomposition."""
    lam, V = eighH(X)
    n = V.shape[0]
    Dh = qzeros(n, n)
    for i in range(n):
        Dh[i, i] = nq.quaternion(max(lam[i], 0.0) ** 0.5, 0, 0, 0)
    return qmm(qmm(V, Dh), qadj(V))


# ------------------------------
# Cached spectral factors for Hermitian PD matrices
# ------------------------------

@dataclass
class _PDCache:
    """Cache spectral information for a Hermitian positive definite quaternion matrix X.

    This cache is designed to drastically reduce repeated expensive eigendecompositions within Newton iterations.

    It provides:
      - invX      = X^{-1}
      - invsqrtX  = X^{-1/2}
      - logdet    = log det(X)
      - lam_min   = lambda_min(X)

    Notes
    -----
    For X = V diag(lam) V^H with lam>0:
      X^{-1}    = V diag(1/lam) V^H
      X^{-1/2}  = V diag(1/sqrt(lam)) V^H
      logdet(X) = sum log(lam)
    """

    lam: np.ndarray
    V: np.ndarray
    invX: np.ndarray
    invsqrtX: np.ndarray
    logdet: float
    lam_min: float

    @staticmethod
    def from_X(X: np.ndarray, *, floor: float = 0.0) -> "_PDCache":
        Xh = qherm(X)
        lam, V = eighH(Xh)
        lam = np.asarray(lam, dtype=float)
        if floor > 0.0:
            lam = np.maximum(lam, float(floor))
        if np.any(lam <= 0.0):
            raise ValueError("PDCache requires X ≻ 0 (all eigenvalues positive).")

        n = V.shape[0]
        Dinv = qzeros(n, n)
        Dinvh = qzeros(n, n)
        for i in range(n):
            li = float(lam[i])
            Dinv[i, i] = nq.quaternion(1.0 / li, 0.0, 0.0, 0.0)
            Dinvh[i, i] = nq.quaternion(1.0 / math.sqrt(li), 0.0, 0.0, 0.0)

        invX = qmm(qmm(V, Dinv), qadj(V))
        invsqrtX = qmm(qmm(V, Dinvh), qadj(V))
        logdet = float(np.sum(np.log(lam)))
        lam_min = float(np.min(lam))
        return _PDCache(lam=lam, V=V, invX=invX, invsqrtX=invsqrtX, logdet=logdet, lam_min=lam_min)


# ------------------------------
# Linear operator A and A*
# ------------------------------

@dataclass
class QuaternionSDPOperator:
    """
    Linear map A and its adjoint A* represented via Hermitian measurement matrices.

    A(X)_i = <H_i, X>_R = Re tr(H_i^H X)
    A*(y)   = sum_i y_i H_i

    Parameters
    ----------
    H_list : list[np.ndarray]
        List of Hermitian quaternion matrices (same shape n×n).
    """

    H_list: list[np.ndarray]
    def __post_init__(self):
        self.H_list = [qherm(H) for H in self.H_list]  # Hermitian projection (not adjoint)

    def A(self, X: np.ndarray) -> np.ndarray:
        return np.array([inner_real(H, X) for H in self.H_list], dtype=float)

    # def AT(self, y: np.ndarray) -> np.ndarray:
    #     n = self.H_list[0].shape[0]
    #     Y = qzeros(n, n)
    #     for yi, Hi in zip(y, self.H_list):
    #         if yi != 0.0:
    #             Y = Y + Hi * yi
    #     return Y
    def AT(self, y: np.ndarray) -> np.ndarray:
        n = self.H_list[0].shape[0]
        Y = qzeros(n, n)
        for yi, Hi in zip(y, self.H_list):
            if yi != 0.0:
                Y = Y + float(yi) * Hi
        return qherm(Y)


# Scaled operator that absorbs per-constraint normalization factors without touching b
@dataclass
class QuaternionSDPOperatorScaled:
    """
    Scaled constraint operator \tilde A with internal per-row scaling s_i.

    (\tilde A X)_i = <H_i, X> / s_i,
    \tilde A^*(y)   = \sum_i (y_i / s_i) H_i.
    """

    H_list: list[np.ndarray]
    s: np.ndarray

    @classmethod
    def from_raw(cls, H_list: list[np.ndarray]):
        scales = np.array([max(quat_frobenius_norm(H), 1e-12) for H in H_list], dtype=float)
        return cls(H_list=[qherm(H) for H in H_list], s=scales)

    def A(self, X: np.ndarray) -> np.ndarray:
        return np.array([inner_real(H, X) / si for H, si in zip(self.H_list, self.s)], dtype=float)

    def AT(self, y: np.ndarray) -> np.ndarray:
        n = self.H_list[0].shape[0]
        Y = qzeros(n, n)
        for yi, Hi, si in zip(y, self.H_list, self.s):
            if yi != 0.0:
                Y = Y + float(yi / si) * Hi
        return qherm(Y)

    def scale_b(self, b: np.ndarray) -> np.ndarray:
        """Return b̃ = b / s (elementwise), consistent with internal scaling."""
        return b / self.s

    def A_unscaled(self, X: np.ndarray) -> np.ndarray:
        """Return original (unscaled) A(X) with raw H_i."""
        return np.array([inner_real(H, X) for H in self.H_list], dtype=float)


# ------------------------------
# Barrier Newton components
# ------------------------------

@dataclass
class BarrierParams:
    """
    Parameters controlling the barrier Newton method.

    Parameters
    ----------
    mu : float
        Initial barrier parameter.
    mu_decay : float
        Geometric decay factor for mu per outer iteration.
    mu_min : float
        Minimum barrier value to stop.
    newton_tol : float
        Tolerance for inner Newton convergence (on combined residual).
    newton_maxit : int
        Max Newton iterations per barrier value.
    backtrack_beta : float
        Backtracking multiplicative shrink.
    backtrack_sigma : float
        Armijo parameter for sufficient decrease.
    posdef_eps : float
        Minimal eigenvalue threshold to accept PD candidate.
    """

    mu: float = 1.0
    mu_decay: float = 0.3
    mu_min: float = 1e-6
    newton_tol: float = 1e-8
    newton_maxit: int = 50
    backtrack_beta: float = 0.5
    backtrack_sigma: float = 1e-4
    posdef_eps: float = 1e-12


@dataclass
class SolverState:
    """Lightweight state container for solver bookkeeping."""

    X: np.ndarray
    y: np.ndarray
    r_p: np.ndarray
    r_d: np.ndarray
    obj: float
    logdet: float
    mu: float
    k_newton: int = 0


def hess_inv_apply(X: np.ndarray, mu: float, G: np.ndarray) -> np.ndarray:
    """
    Apply H^{-1}[G] for H[W] = mu * X^{-1} W X^{-1}.

    Parameters
    ----------
    X : np.ndarray
        Current PD iterate.
    mu : float
        Barrier parameter.
    G : np.ndarray
        Right-hand quaternion matrix.

    Returns
    -------
    np.ndarray
        (1/mu) X G X
    """
    return (1.0 / mu) * qmm(qmm(X, G), X)


def _min_eig(X: np.ndarray) -> float:
    """Minimal eigenvalue of Hermitian quaternion matrix."""
    return float(np.min(eigvalsH(X)))

# ------------------------------
# Path-following barrier/IPM ?
# ------------------------------
def _assemble_residuals(op: QuaternionSDPOperator,
                        C: np.ndarray,
                        b: np.ndarray,
                        X: np.ndarray,
                        y: np.ndarray,
                        mu: float,
                        *,
                        invX: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return primal and dual residuals (r_p, r_d).

    Notes
    -----
    We allow passing a cached inverse invX = X^{-1} to avoid repeated eigen-decompositions.
    """
    # Use r_p = b - A(X) to match A ΔX = r_p in Newton step
    r_p = b - op.A(X)
    if invX is None:
        invX = invH(X)
    r_d = C + op.AT(y) - (mu * invX)
    return r_p, r_d


def _min_eig(X: np.ndarray) -> float:
    """Minimal eigenvalue of Hermitian quaternion matrix."""
    return float(np.min(eigvalsH(X)))

def newton_step(op: QuaternionSDPOperator,
                C: np.ndarray,
                b: np.ndarray,
                state: SolverState,
                *,
                schur_solver: str = "dense",
                schur_precond: str = "none",
                cg_tol: float = 1e-10,
                cg_maxit: int = 500,
                precond_rank: int | None = None,
                precond_ridge_scale: float = 1e-6,
                precond_seed: int = 0) -> tuple[np.ndarray, np.ndarray, int, float]:
    X, mu = state.X, state.mu
    r_p, r_d = state.r_p, state.r_d
    m_dim = len(b)

    # Correct RHS for r_p = b - A(X):  (A H^{-1} A*) dy = -r_p - A(H^{-1} r_d)
    rhs = -r_p - op.A(hess_inv_apply(X, mu, r_d))

    def _schur_matvec(v: np.ndarray) -> np.ndarray:
        # v in R^m  ->  A( H^{-1}[ A*(v) ] ) in R^m
        return op.A(hess_inv_apply(X, mu, op.AT(v)))

    def _pcg_solve(
        matvec,
        bvec: np.ndarray,
        *,
        M_inv=None,
        tol: float = 1e-10,
        maxit: int = 500,
    ) -> tuple[np.ndarray, int, float]:
        """Basic (preconditioned) CG for SPD systems in R^m."""
        x = np.zeros_like(bvec)
        r = bvec - matvec(x)
        bnorm = float(np.linalg.norm(bvec)) + 1e-30
        rnorm = float(np.linalg.norm(r))
        if rnorm / bnorm <= tol:
            return x, 0, rnorm / bnorm

        z = M_inv(r) if M_inv is not None else r
        p = z.copy()
        rz = float(np.dot(r, z))

        it = 0
        for it in range(1, maxit + 1):
            Ap = matvec(p)
            denom = float(np.dot(p, Ap)) + 1e-30
            alpha = rz / denom
            x = x + alpha * p
            r = r - alpha * Ap
            rnorm = float(np.linalg.norm(r))
            if rnorm / bnorm <= tol:
                break
            z = M_inv(r) if M_inv is not None else r
            rz_new = float(np.dot(r, z))
            beta = rz_new / (rz + 1e-30)
            p = z + beta * p
            rz = rz_new
        return x, it, rnorm / bnorm

    schur_solver_lc = (schur_solver or "dense").lower()
    schur_precond_lc = (schur_precond or "none").lower()

    if schur_solver_lc not in {"dense", "cg"}:
        schur_solver_lc = "dense"
    if schur_precond_lc not in {"none", "diag", "nystrom"}:
        schur_precond_lc = "none"

    if schur_solver_lc == "cg":
        # Optional diagonal preconditioner: diag_j = (M e_j)_j
        M_inv = None
        if schur_precond_lc == "diag":
            diag = np.empty(m_dim, dtype=float)
            for j in range(m_dim):
                ej = np.zeros(m_dim, dtype=float)
                ej[j] = 1.0
                col = _schur_matvec(ej)
                dj = float(col[j])
                # Ensure positive and non-degenerate
                diag[j] = dj if dj > 1e-30 else max(abs(dj), 1e-12)
            M_inv = lambda r: r / diag

        elif schur_precond_lc == "nystrom":
            # Nyström / sketch-based SPD preconditioner for the Schur operator.
            # Build P^{-1} ≈ M^{-1} using r random probes (r << m) and a Woodbury solve.
            r = int(precond_rank) if precond_rank is not None else min(20, m_dim)
            r = max(1, min(r, m_dim))
            rng = np.random.default_rng(int(precond_seed))

            # Rademacher probe matrix Ω ∈ R^{m×r}
            Omega = rng.choice([-1.0, 1.0], size=(m_dim, r)).astype(float)

            # Y = M Ω  (matrix-free)
            Y = np.column_stack([_schur_matvec(Omega[:, j]) for j in range(r)])

            # B = Ω^T Y = Ω^T M Ω  (should be SPD)
            B = Omega.T @ Y
            B = 0.5 * (B + B.T)

            # Scale ridge δ from the typical quadratic form magnitude
            trB = float(np.trace(B))
            delta = float(precond_ridge_scale) * (trB / max(r, 1)) if trB > 0 else float(precond_ridge_scale)
            delta = max(delta, 1e-12)

            # T = B + (1/δ) Y^T Y
            G = Y.T @ Y
            T = B + (1.0 / delta) * G
            T = 0.5 * (T + T.T)

            # Cholesky with jitter
            try:
                L = np.linalg.cholesky(T)
            except np.linalg.LinAlgError:
                jitter = 1e-12 * (np.linalg.norm(T, 1) + 1.0)
                L = np.linalg.cholesky(T + jitter * np.eye(r))

            def _solve_T(u: np.ndarray) -> np.ndarray:
                w = np.linalg.solve(L, u)
                w = np.linalg.solve(L.T, w)
                return w

            def M_inv(v: np.ndarray) -> np.ndarray:
                # P^{-1} v = δ^{-1} v - δ^{-2} Y (T)^{-1} (Y^T v)
                u = Y.T @ v
                w = _solve_T(u)
                return (v / delta) - (Y @ w) / (delta * delta)

        dy, cg_iters, cg_res = _pcg_solve(
            _schur_matvec,
            rhs,
            M_inv=M_inv,
            tol=float(cg_tol),
            maxit=int(cg_maxit),
        )

        # If CG failed to reach tolerance, fall back to dense solve for robustness.
        if not np.isfinite(cg_res) or (cg_res > 10.0 * float(cg_tol)):
            schur_solver_lc = "dense"
        else:
            schur_res = float(cg_res)
    if schur_solver_lc == "dense":
        # Dense Schur M = A H^{-1} A*
        M = np.zeros((m_dim, m_dim), dtype=float)
        for j in range(m_dim):
            ej = np.zeros(m_dim, dtype=float)
            ej[j] = 1.0
            M[:, j] = _schur_matvec(ej)

        # Solve (adaptive jitter only if needed)
        delta_used = 0.0
        try:
            dy = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            delta_used = 1e-12 * (np.linalg.norm(M, 1) + 1.0)
            dy = np.linalg.solve(M + delta_used * np.eye(m_dim), rhs)

        denom = np.linalg.norm(rhs) + 1e-30
        schur_res = np.linalg.norm((M + delta_used * np.eye(m_dim)) @ dy - rhs) / denom
        cg_iters = 0
        cg_res = float(schur_res)

    # Back-substitute
    dX = -hess_inv_apply(X, mu, (r_d + op.AT(dy)))
    dX = qherm(dX)
    return dX, dy, int(cg_iters), float(cg_res)

def solve_barrier(A_list: list[np.ndarray],
                  b: np.ndarray,
                  C: np.ndarray,
                  X0: np.ndarray | None = None,
                  params: BarrierParams = BarrierParams(),
                  verbose: bool = True) -> SolverState:
    """
    Equality-constrained log-det barrier Newton method (quaternion-native), in hat space.

    Solves (for each mu stage):
        minimize  <C, X>_R - mu logdet(X)
        subject to A_hat(X) = b_hat,   X ≻ 0,

    with KKT residuals:
        r_p = b_hat - A_hat(X),
        r_d = C + A_hat^*(y) - mu X^{-1}.
    """
    n = C.shape[0]
    C = qherm(C)

    # Correct Hermitian projection for constraints
    H_list = [qherm(Ai) for Ai in A_list]

    # Orthonormalize constraints once and work in hat-space
    ops = _build_orthonormal_ops(H_list)
    Ahat = ops['A_hat']
    AThat = ops['AT_hat']
    b_hat = ops['transform_b'](b)

    class _HatOperator:
        def __init__(self, Ahat_fn, AThat_fn, H_list_ref):
            self._Ahat = Ahat_fn
            self._AThat = AThat_fn
            self.H_list = H_list_ref

        def A(self, X_: np.ndarray) -> np.ndarray:
            return self._Ahat(X_)

        def AT(self, y_: np.ndarray) -> np.ndarray:
            return self._AThat(y_)

    op = _HatOperator(Ahat, AThat, H_list)

    # Initialize X, y in hat coordinates
    X = qherm(X0) if X0 is not None else qeye(n)
    X = _ensure_spd(X, floor=params.posdef_eps)
    # optional: snap once at start
    X = qherm(X + AThat(b_hat - Ahat(X)))
    X = _ensure_spd(X, floor=params.posdef_eps)

    y = np.zeros(len(H_list), dtype=float)

    mu = float(params.mu)

    # Outer continuation loop: solve each mu stage once, including mu_min stage
    while True:
        # Warm-start y for this mu (hat space)
        try:
            y = op.A(mu * invH(X) - C)
        except Exception:
            pass

        if verbose:
            rp0, rd0 = _assemble_residuals(op, C, b_hat, X, y, mu)
            print(f"[barrier diag] mu={mu:.2e}  ||r_p||={np.linalg.norm(rp0):.2e}  ||r_d||_F={quat_frobenius_norm(rd0):.2e}")

        # Newton iterations at fixed mu
        for k in range(params.newton_maxit):
            cache = _PDCache.from_X(X)
            r_p, r_d = _assemble_residuals(op, C, b_hat, X, y, mu, invX=cache.invX)
            combo = math.sqrt(float(np.dot(r_p, r_p)) + float(quat_frobenius_norm(r_d) ** 2))
            if combo <= params.newton_tol:
                break

            state = SolverState(X=X, y=y, r_p=r_p, r_d=r_d,
                                obj=inner_real(C, X), logdet=cache.logdet, mu=mu, k_newton=k)

            dX, dy, cg_iters, cg_res = newton_step(op, C, b_hat, state)
            dX = qherm(dX)

            # Fraction-to-the-boundary cap for X
            try:
                Xmh = cache.invsqrtX
                B = qherm(qmm(qmm(Xmh, dX), Xmh))
                lam_min = float(np.min(eigvalsH(B)))
                alpha_max = 1.0 if lam_min >= 0 else min(1.0, 0.99 / (-lam_min))
            except Exception:
                alpha_max = 1.0
            t = min(0.99 * alpha_max, 1.0)

            # Update and (optional) snap after acceptance
            X = qherm(X + t * dX)
            y = y + t * dy

            # snap once per iteration (cheap in hat space)
            X = qherm(X + AThat(b_hat - Ahat(X)))
            X = _ensure_spd(X, floor=params.posdef_eps)

            if verbose:
                lam_min_X = _min_eig(X)
                print(
                    f"[mu={mu:.2e}] it={k:02d}  ||r_p||={np.linalg.norm(r_p):.2e}  "
                    f"||r_d||_F={quat_frobenius_norm(r_d):.2e}  "
                    f"lam_min(X)={lam_min_X:.3e}  f={inner_real(C, X):.6e}  "
                    f"logdet={logdetH(X):.6e}  t={t:.2e}  "
                    f"Schur_res={cg_res:.2e}"
                )

        # Decide next mu (ensure mu_min stage is run once)
        mu_next = max(float(params.mu_min), float(params.mu_decay) * mu)

        if mu_next == mu:
            # already at mu_min
            break

        mu = mu_next

    # Final snap
    X = qherm(X + AThat(b_hat - Ahat(X)))
    X = _ensure_spd(X, floor=params.posdef_eps)

    cache = _PDCache.from_X(X)
    r_p, r_d = _assemble_residuals(op, C, b_hat, X, y, mu, invX=cache.invX)
    return SolverState(X=X, y=y, r_p=r_p, r_d=r_d,
                       obj=inner_real(C, X), logdet=cache.logdet, mu=mu)

def solve_pd_mehrotra(H_list: list[np.ndarray],
                      b: np.ndarray,
                      C: np.ndarray,
                      X0: np.ndarray | None = None,
                      S0: np.ndarray | None = None,
                      y0: np.ndarray | None = None,
                      mu_init: float = 1.0,
                      beta_mu: float = 0.3,
                      eps_p: float = 1e-8,
                      eps_d: float = 1e-8,
                      eps_gap: float = 1e-8,
                      max_iter: int = 50,
                      cg_tol: float = 1e-10,
                      cg_maxit: int = 500,
                      verbose: bool = True,
                      fixed_mu: bool = False,
                      *,
                      schur_solver: str = "dense",
                      schur_precond: str = "none",
                      schur_precond_rank: int | None = None,
                      schur_precond_ridge_scale: float = 1e-6,
                      schur_precond_seed: int = 0,
                      ops: dict | None = None,
                      assume_hat: bool = False,
                      return_ops: bool = False) -> dict:
    """
    Barrier-path (μ-continuation) logdet solver in hat space. (Historical name: solve_pd_mehrotra)

    Current design (robust core):
      Solve the fixed-μ barrier KKT system (in hat space):
        Ahat(X) = b_hat,
        C + Ahat^*(y) - μ X^{-1} = 0.

    If fixed_mu=True:
      solve only for μ=mu_init.

    If fixed_mu=False:
      continuation on μ: μ <- max(beta_mu*μ, eps_gap),
      warm-starting from previous (X,y).
    """
    # Resolve hat operators and b_hat
    # Ensure constraint matrices are exactly Hermitian (prevents AT(y) non-Hermitian drift)
    Ahat, AThat, transform_b, b_hat = _resolve_ops(H_list, b, ops=ops, assume_hat=assume_hat)

    n = C.shape[0]
    # C = quat_hermitian(C)
    C = qherm(C)

    # Hat operator wrapper compatible with newton_step()
    class _HatOperator:
        def __init__(self, Ahat_fn, AThat_fn, H_list_ref):
            self._Ahat = Ahat_fn
            self._AThat = AThat_fn
            self.H_list = H_list_ref

        def A(self, X_: np.ndarray) -> np.ndarray:
            return self._Ahat(X_)

        def AT(self, y_: np.ndarray) -> np.ndarray:
            return self._AThat(y_)

    op = _HatOperator(Ahat, AThat, H_list)

    # Initialize X, y
    # X = quat_hermitian(X0) if X0 is not None else qeye(n)
    X = qherm(X0) if X0 is not None else qeye(n)
    X = _ensure_spd(X, floor=1e-12)

    y = y0.copy() if y0 is not None else np.zeros(int(b_hat.shape[0]), dtype=float)

    # Initialize mu
    mu = float(mu_init)
    mu_target = float(mu_init) if fixed_mu else float(eps_gap)

    # Ensure initial equality (cheap snap)
    # X = quat_hermitian(X + AThat(b_hat - Ahat(X)))
    X = qherm(X + AThat(b_hat - Ahat(X)))

    # Optional one-shot warm start for y
    if y0 is None:
        try:
            cache0 = _PDCache.from_X(X)
            y = Ahat(mu * cache0.invX - C)
        except Exception:
            pass

    history: list[dict] = []
    it_global = 0
    stage = 0

    # Outer μ-continuation loop
    while True:
        stage_converged = False

        # Inner Newton loop at fixed μ
        while it_global < max_iter:
            cache = _PDCache.from_X(X)
            r_p, r_d = _assemble_residuals(op, C, b_hat, X, y, mu, invX=cache.invX)
            rp = float(np.linalg.norm(r_p))
            rd = float(quat_frobenius_norm(r_d))
            combo = math.sqrt(rp * rp + rd * rd)

            if combo <= max(eps_p, eps_d):
                stage_converged = True
                lam_min_X = float(cache.lam_min)
                history.append(dict(
                    it=it_global, stage=stage, mu=float(mu),
                    rp=rp, rd=rd, gap=float(mu), t=0.0,
                    lam_min_X=lam_min_X,
                    cg_iters=0, cg_res=0.0,
                ))
                if verbose:
                    print(
                        f"[BarrierPath] stage={stage:02d} it={it_global:03d} mu={mu:.2e} "
                        f"CONVERGED  ||r_p||={rp:.2e} ||r_d||_F={rd:.2e} lam_min(X)={lam_min_X:.2e}"
                    )
                break

            state = SolverState(
                X=X, y=y, r_p=r_p, r_d=r_d,
                obj=inner_real(C, X),
                logdet=cache.logdet,
                mu=mu,
                k_newton=it_global
            )
            dX, dy, cg_iters, cg_res = newton_step(
                op, C, b_hat, state,
                schur_solver=schur_solver,
                schur_precond=schur_precond,
                cg_tol=cg_tol,
                cg_maxit=cg_maxit,
                precond_rank=schur_precond_rank,
                precond_ridge_scale=schur_precond_ridge_scale,
                precond_seed=schur_precond_seed,
            )
            # dX = qherm(dX)
            dX = qherm(dX)

            # Fraction-to-the-boundary cap for X
            # Use the Hermitian matrix B = X^{-1/2} dX X^{-1/2}. If λ_min(B) is known,
            # then X + t dX ≻ 0 is guaranteed whenever 1 + t λ_min(B) > 0.
            lam_min_B = None
            try:
                Xmh = cache.invsqrtX
                B = qherm(qmm(qmm(Xmh, dX), Xmh))
                lam_min_B = float(np.min(eigvalsH(B)))
                alpha_max = 1.0 if lam_min_B >= 0 else min(1.0, 0.99 / (-lam_min_B))
            except Exception:
                alpha_max = 1.0
            t0 = min(0.99 * alpha_max, 1.0)

            # Backtracking line search (Armijo on combined residual norm)
            # Note: we check Armijo on the unsnapped trial point, but require the snapped point to remain PD before accepting.
            backtrack_beta = 0.5
            backtrack_sigma = 1e-4
            posdef_eps = 1e-12
            t = t0

            for _ls in range(60):
                # Xcand = quat_hermitian(X + t * dX)
                Xcand = qherm(X + t * dX)
                # Fast PD check (avoid expensive eigendecomposition on Xcand):
                # If λ_min(B) is available for B = X^{-1/2} dX X^{-1/2}, then
                # X + t dX ≻ 0 whenever 1 + t λ_min(B) > 0.
                if lam_min_B is not None:
                    if 1.0 + t * lam_min_B <= posdef_eps:
                        t *= backtrack_beta
                        continue
                else:
                    if _min_eig(Xcand) <= posdef_eps:
                        t *= backtrack_beta
                        continue

                ycand = y + t * dy

                try:
                    rp_c, rd_c = _assemble_residuals(op, C, b_hat, Xcand, ycand, mu)
                    rp_c_n = float(np.linalg.norm(rp_c))
                    rd_c_n = float(quat_frobenius_norm(rd_c))
                    combo_c = math.sqrt(rp_c_n * rp_c_n + rd_c_n * rd_c_n)
                except Exception:
                    t *= backtrack_beta
                    continue

                if combo_c <= (1.0 - backtrack_sigma * t) * combo:
                    # Ensure the *snapped* candidate remains positive definite.
                    # (A post-snap SPD shift would generally break equality feasibility, e.g. under a trace constraint.)
                    Xsnap = qherm(Xcand + AThat(b_hat - Ahat(Xcand)))
                    if _min_eig(Xsnap) > posdef_eps:
                        Xcand = Xsnap
                        break
                    t *= backtrack_beta
                    continue
                t *= backtrack_beta

            # Accept step (Xcand is already snapped and PD)
            X = Xcand
            y = ycand

            lam_min_X = float(cache.lam_min)

            history.append(dict(
                it=it_global, stage=stage, mu=float(mu),
                rp=rp, rd=rd, gap=float(mu), t=float(t),
                lam_min_X=lam_min_X,
                cg_iters=int(cg_iters), cg_res=float(cg_res),
            ))

            if verbose:
                print(
                    f"[BarrierPath] stage={stage:02d} it={it_global:03d} mu={mu:.2e} "
                    f"||r_p||={rp:.2e} ||r_d||_F={rd:.2e} lam_min(X)={lam_min_X:.2e} "
                    f"t={t:.2e} CG_it={cg_iters} CG_res={cg_res:.2e}"
                )

            it_global += 1

        # End inner loop for this μ-stage

        if fixed_mu:
            break

        if it_global >= max_iter:
            break

        # Only continue μ if we actually converged the current stage
        if not stage_converged:
            break

        if mu <= mu_target:
            break

        # ---- Stage boundary: update μ and REFRESH (this is the right place) ----
        mu_next = max(beta_mu * mu, mu_target)
        if mu_next == mu:
            break

        mu = float(mu_next)
        stage += 1

        # Re-snap (cheap) and refresh y for the new μ (important for continuation)
        # X = quat_hermitian(X + AThat(b_hat - Ahat(X)))
        X = qherm(X + AThat(b_hat - Ahat(X)))
        try:
            cache0 = _PDCache.from_X(X)
            y = Ahat(mu * cache0.invX - C)
        except Exception:
            pass

    # Final snap + dual slack from barrier relation
    # X = quat_hermitian(X + AThat(b_hat - Ahat(X)))
    X = qherm(X + AThat(b_hat - Ahat(X)))
    # S = quat_hermitian(mu * invH(X))
    cache_final = _PDCache.from_X(X)
    S = qherm(mu * cache_final.invX)

    out = dict(
        X=X,
        S=S,
        y=y,
        history=history,
        rp_hat=float(np.linalg.norm(b_hat - Ahat(X))),
        mu=float(mu),
    )
    if return_ops:
        out["ops"] = dict(A_hat=Ahat, AT_hat=AThat, transform_b=transform_b)
    return out



# ---------------------------------------------------------------------
# Professional API names (preferred) + backward compatible aliases
# ---------------------------------------------------------------------

def solve_logdet_barrier_newton(
    H_list: list[np.ndarray],
    b: np.ndarray,
    C: np.ndarray,
    *,
    X0: np.ndarray | None = None,
    y0: np.ndarray | None = None,
    mu: float = 1.0,
    max_iter: int = 50,
    eps: float = 1e-8,
    verbose: bool = True,
    schur_solver: str = "dense",
    schur_precond: str = "none",
    schur_precond_rank: int | None = None,
    schur_precond_ridge_scale: float = 1e-6,
    schur_precond_seed: int = 0,
    cg_tol: float = 1e-10,
    cg_maxit: int = 500,
    ops: dict | None = None,
    assume_hat: bool = False,
    return_ops: bool = False,
) -> dict:
    """Solve the fixed-μ equality-constrained logdet barrier KKT system via Newton steps in hat-space."""
    return solve_pd_mehrotra(
        H_list, b, C,
        X0=X0, y0=y0,
        mu_init=float(mu),
        max_iter=int(max_iter),
        eps_p=float(eps), eps_d=float(eps), eps_gap=float(eps),
        cg_tol=float(cg_tol),
        cg_maxit=int(cg_maxit),
        fixed_mu=True,
        verbose=bool(verbose),
        schur_solver=schur_solver,
        schur_precond=schur_precond,
        schur_precond_rank=schur_precond_rank,
        schur_precond_ridge_scale=schur_precond_ridge_scale,
        schur_precond_seed=schur_precond_seed,
        ops=ops, assume_hat=assume_hat, return_ops=return_ops,
    )


def solve_logdet_barrier_path(
    H_list: list[np.ndarray],
    b: np.ndarray,
    C: np.ndarray,
    *,
    X0: np.ndarray | None = None,
    y0: np.ndarray | None = None,
    mu_init: float = 1.0,
    beta_mu: float = 0.3,
    mu_min: float = 1e-8,
    max_iter: int = 200,
    eps: float = 1e-8,
    verbose: bool = True,
    schur_solver: str = "dense",
    schur_precond: str = "none",
    schur_precond_rank: int | None = None,
    schur_precond_ridge_scale: float = 1e-6,
    schur_precond_seed: int = 0,
    cg_tol: float = 1e-10,
    cg_maxit: int = 500,
    ops: dict | None = None,
    assume_hat: bool = False,
    return_ops: bool = False,
) -> dict:
    """Barrier path-following (μ-continuation) for the logdet barrier SDP subproblem in hat-space."""
    return solve_pd_mehrotra(
        H_list, b, C,
        X0=X0, y0=y0,
        mu_init=float(mu_init),
        beta_mu=float(beta_mu),
        eps_p=float(eps), eps_d=float(eps), eps_gap=float(mu_min),
        max_iter=int(max_iter),
        cg_tol=float(cg_tol),
        cg_maxit=int(cg_maxit),
        fixed_mu=False,
        verbose=bool(verbose),
        schur_solver=schur_solver,
        schur_precond=schur_precond,
        schur_precond_rank=schur_precond_rank,
        schur_precond_ridge_scale=schur_precond_ridge_scale,
        schur_precond_seed=schur_precond_seed,
        ops=ops, assume_hat=assume_hat, return_ops=return_ops,
    )


# Backward-compatible alias with clearer intent
solve_barrier_path = solve_logdet_barrier_path

# ------------------------------
# Primal–Dual IPM (Mehrotra predictor–corrector) ?
# ------------------------------


# ------------------------------
# ADMM for fixed μ (closed-form X-prox, affine Z-projection)
# ------------------------------
def _prox_logdet_fro(W: np.ndarray, tau: float) -> np.ndarray:
    """
    Description
    -----------
    Compute the proximal operator of tau*(-logdet(X)) + 1/2||X - W||_F^2 over
    the cone of quaternion Hermitian positive definite matrices. The solution is
    closed-form via eigen-decomposition of the Hermitian input W.

    Parameters
    ----------
    W : np.ndarray
        Quaternion Hermitian matrix.
    tau : float
        Proximal parameter (> 0), typically tau = mu / rho in ADMM.

    Returns
    -------
    np.ndarray
        The proximal point X ≻ 0.

    Notes/References
    ----------------
    If W = Q diag(ω) Q^H, then X = Q diag(x) Q^H with x_i = 0.5(ω_i + sqrt(ω_i^2 + 4 tau)).
    """
    What = qherm(W)
    # Use Hermitian-safe eigendecomposition
    w, Q = eighH(What)
    x = 0.5 * (w + np.sqrt(w * w + 4.0 * tau))
    n = Q.shape[0]
    Xd = qzeros(n, n)
    for i in range(n):
        Xd[i, i] = nq.quaternion(float(x[i]), 0.0, 0.0, 0.0)
    return qmm(qmm(Q, Xd), qadj(Q))


def admm_barrier_fixed_mu(
    H_list: list[np.ndarray],
    b: np.ndarray,
    C: np.ndarray,
    mu: float,
    rho: float = 5.0,
    maxit: int = 300,
    abstol: float = 1e-6,
    reltol: float = 1e-5,
    X0: np.ndarray | None = None,
    Z0: np.ndarray | None = None,
    U0: np.ndarray | None = None,
    verbose: bool = True,
    alpha: float = 1.5,
    use_hat_projection: bool = False,
):
    """
    Description
    -----------
    Scaled ADMM for the fixed-μ barrier subproblem:
        minimize <C, X> - μ logdet(X)  subject to  A(X) = b,  X ≻ 0,
    using a consensus split with Z carrying only the equality constraints.

    Parameters
    ----------
    H_list : list[np.ndarray]
        Hermitian measurement matrices defining A and A*.
    b : np.ndarray
        Constraint right-hand side (R^m).
    C : np.ndarray
        Hermitian cost matrix.
    mu : float
        Barrier parameter (fixed for this solver call).
    rho : float
        ADMM penalty parameter.
    maxit : int
        Maximum ADMM iterations.
    abstol, reltol : float
        Absolute and relative stopping tolerances (Boyd criteria).
    X0, Z0, U0 : np.ndarray | None
        Optional initializations for primal/dual variables.
    verbose : bool
        If True, prints per-iteration diagnostics.

    Returns
    -------
    tuple
        (X, Z, U, history) where history contains norms and eigenvalue traces.

    Notes/References
    ----------------
    - Pure quaternion algebra via QuatIca utilities.
    - Z-projection is exact in orthonormalized coordinates Â (one line).
    - X-update is closed-form SPD prox of -μ logdet with Frobenius metric.
    """
    n = C.shape[0]
    # Prepare constraints once
    Hh = [qherm(H) for H in H_list]

    # Direct projector (original coordinates) via cached Gram Cholesky
    class EqualityProjectorDirect:
        def __init__(self, Hs: list[np.ndarray], ridge: float = 1e-12):
            mloc = len(Hs)
            G = np.empty((mloc, mloc), dtype=float)
            for i, Hi in enumerate(Hs):
                for j, Hj in enumerate(Hs):
                    G[i, j] = float(inner_real(Hi, Hj))
            if ridge > 0.0:
                G = G + ridge * np.eye(mloc)
            # Cholesky: G = R^T R
            R = np.linalg.cholesky(G).T
            self.R = R
            self.Hs = Hs

        def A_vec(self, X: np.ndarray) -> np.ndarray:
            return np.array([inner_real(Hi, X) for Hi in self.Hs], dtype=float)

        def AT(self, y: np.ndarray) -> np.ndarray:
            nloc = self.Hs[0].shape[0]
            Y = qzeros(nloc, nloc)
            for yi, Hi in zip(y, self.Hs):
                if yi != 0.0:
                    Y = Y + float(yi) * Hi
            return qherm(Y)

        def solve_Gy(self, rhs: np.ndarray) -> np.ndarray:
            # Solve R^T z = rhs; then R y = z
            z = np.linalg.solve(self.R.T, rhs)
            y = np.linalg.solve(self.R, z)
            return y

        def project(self, V: np.ndarray, bvec: np.ndarray) -> np.ndarray:
            rhs = bvec - self.A_vec(V)
            y = self.solve_Gy(rhs)
            return V + self.AT(y)

    proj_direct = EqualityProjectorDirect(Hh, ridge=1e-12)

    # Orthonormalized operator kept for optional use
    ops = _build_orthonormal_ops(Hh)
    Ahat = ops['A_hat']; AThat = ops['AT_hat']; A_unscaled = ops['A_unscaled']
    b_hat = ops['transform_b'](b)

    C = qherm(C)
    X = qherm(X0) if X0 is not None else qeye(n)
    Z = qherm(Z0) if Z0 is not None else X.copy()
    U = qherm(U0) if U0 is not None else qzeros(n, n)

    # Sanity check: projection should enforce A_unscaled(Z_test) ≈ b
    if verbose:
        Y_test = Z.copy()
        r0 = b_hat - Ahat(Y_test)
        Z_test = Y_test + AThat(r0)
        eq_test = float(np.linalg.norm(b - A_unscaled(Z_test)))
        print(f"[ADMM check] ||b - A_unscaled(Z_test)|| = {eq_test:.2e} (should be ~1e-12)")

    hist = {'pr': [], 'dr': [], 'eq': [], 'eq_hat': [], 'obj': [], 'lamX': [], 'lamZ': []}
    Z_prev = Z.copy()

    for k in range(maxit):
        # 1) X-update (closed-form prox) with tau = mu / rho
        W = qherm(Z - U - (1.0 / rho) * C)
        X = _prox_logdet_fro(W, tau=mu / rho)

        # 2) Z-update: project the consensus target (X + U)
        Z_old = Z
        V = qherm(X + U)
        if use_hat_projection:
            r = b_hat - Ahat(V)
            # Debug identity: Ahat(AThat(r)) should equal r
            if verbose and (k < 5):
                ident_err = float(np.linalg.norm(Ahat(AThat(r)) - r))
                print(f"    [proj] id_err=||ÂÂ* r - r||={ident_err:.2e}")
            Z = qherm(V + AThat(r))
        else:
            Z = qherm(proj_direct.project(V, b))
        # Immediate equality check in hat-coordinates
        if verbose and (k < 10 or k % 10 == 0):
            eq_hat_now = float(np.linalg.norm(b_hat - Ahat(Z)))
            print(f"    [proj] eq_hat_now={eq_hat_now:.2e}")
        
        # 3) Dual update
        if abs(alpha - 1.0) < 1e-12:
            U = qherm(U + (X - Z))
        else:
            # Over-relaxation applied in the dual only
            U = qherm(U + alpha * (X - Z) + (1.0 - alpha) * (Z_old - Z))

        # Residuals
        r_pr = X - Z
        pr = quat_frobenius_norm(r_pr)
        s = rho * quat_frobenius_norm(Z_prev - Z)
        Z_prev = Z.copy()

        # Equality residuals
        eq = float(np.linalg.norm(b - proj_direct.A_vec(Z)))
        eq_hat = float(np.linalg.norm(b_hat - Ahat(Z))) if use_hat_projection else float('nan')

        # Objective
        lamX = eigvalsH(X)
        if np.any(lamX <= 0):
            obj = float('nan')
        else:
            obj = float(inner_real(C, X) - mu * np.sum(np.log(lamX)))

        hist['pr'].append(pr); hist['dr'].append(s); hist['eq'].append(eq); hist['eq_hat'].append(eq_hat)
        hist['obj'].append(obj)
        hist['lamX'].append(float(np.min(lamX)))
        lamZ = eigvalsH(Z)
        hist['lamZ'].append(float(np.min(lamZ)))

        if verbose and (k < 10 or k % 10 == 0):
            print(
                f"[ADMM k={k:03d}]  ||r_pr||={pr:.2e}  ||r_du||={s:.2e}  ||A(Z)-b||={eq:.2e}  "
                + (f"||Â(Z)-b̂||={eq_hat:.2e}  " if use_hat_projection else "") +
                f"obj={obj:.6e}  lam_min(X)={hist['lamX'][-1]:.2e}  lam_min(Z)={hist['lamZ'][-1]:.2e}  rho={rho:.2e}  alpha={alpha:.2f}"
            )

        # Stopping (absolute/relative)
        nrmX = quat_frobenius_norm(X); nrmZ = quat_frobenius_norm(Z); nrmU = quat_frobenius_norm(U)
        eps_pr = math.sqrt(n * n) * abstol + reltol * max(nrmX, nrmZ)
        eps_dr = math.sqrt(n * n) * abstol + reltol * rho * nrmU
        if pr <= eps_pr and s <= eps_dr and eq <= max(1e-10, abstol):
            break

        # Optional adaptive rho (stabilize consensus)
        # Use a relatively aggressive balancing rule to avoid stalling in
        # ill-conditioned/random constraint regimes.
        ratio = 2.0
        if pr > ratio * max(s, 1e-30):
            rho *= 2.0
            U = U * 0.5
        elif s > ratio * max(pr, 1e-30):
            rho *= 0.5
            U = U * 2.0
        rho = max(rho, 1e-12)

    return X, Z, U, hist


def save_admm_history_plot(hist: dict, out_path: str) -> None:
    """
    Description
    -----------
    Save a semilog plot of ADMM residuals across iterations.

    Parameters
    ----------
    hist : dict
        History dictionary from admm_barrier_fixed_mu with keys 'pr','dr','eq'.
    out_path : str
        Path to save the figure (PNG). Parent directories will be created.

    Notes/References
    ----------------
    Figures are saved; not displayed interactively. Use validation_output directory.
    """
    import os
    import matplotlib.pyplot as plt

    pr = np.asarray(hist.get('pr', []), dtype=float)
    dr = np.asarray(hist.get('dr', []), dtype=float)
    eq = np.asarray(hist.get('eq', []), dtype=float)
    eq_hat = np.asarray(hist.get('eq_hat', []), dtype=float)
    it = np.arange(len(pr))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(7, 4))
    if len(pr):
        plt.semilogy(it, pr, label='||r_pr|| = ||X-Z||_F')
    if len(dr):
        plt.semilogy(it, dr, label='||r_du|| = rho ||Z_k - Z_{k+1}||_F')
    if len(eq):
        plt.semilogy(it, eq, label='||A(Z)-b|| (orig)')
    if len(eq_hat):
        plt.semilogy(it, eq_hat, label='||Â(Z)-b̂|| (hat)')
    plt.xlabel('iteration')
    plt.ylabel('norm (log scale)')
    plt.title('ADMM residuals (fixed μ)')
    plt.grid(True, which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_pdipm_history_plot(history: list, out_path: str) -> None:
    """
    Save iteration metrics for the barrier-path solver history.

    Plots semilog curves of ||r_p||, ||r_d||, gap/n, and step t over iterations.
    """
    import os
    import matplotlib.pyplot as plt
    it = np.arange(len(history))
    rp = np.asarray([h.get('rp', np.nan) for h in history], float)
    rd = np.asarray([h.get('rd', np.nan) for h in history], float)
    gap = np.asarray([h.get('gap', np.nan) for h in history], float)
    t = np.asarray([h.get('t', np.nan) for h in history], float)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(7, 4))
    if len(history):
        plt.semilogy(it, rp, label='||r_p||')
        plt.semilogy(it, rd, label='||r_d||')
        plt.semilogy(it, gap, label='gap/n')
    plt.xlabel('iteration')
    plt.ylabel('value (log scale)')
    plt.title('Barrier-path (logdet) metrics')
    plt.grid(True, which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()



# Clearer alias (historical name was save_pdipm_history_plot)
save_barrier_path_history_plot = save_pdipm_history_plot

def save_barrier_history_plot(history: list, out_path: str) -> None:
    """
    Save iteration metrics for primal barrier Newton history.

    Plots semilog curves of ||r_p||, ||r_d||, and step t over iterations.
    """
    import os
    import matplotlib.pyplot as plt
    it = np.arange(len(history))
    rp = np.asarray([h.get('rp', np.nan) for h in history], float)
    rd = np.asarray([h.get('rd', np.nan) for h in history], float)
    t = np.asarray([h.get('t', np.nan) for h in history], float)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(7, 4))
    if len(history):
        plt.semilogy(it, rp, label='||r_p||')
        plt.semilogy(it, rd, label='||r_d||')
    plt.xlabel('iteration')
    plt.ylabel('value (log scale)')
    plt.title('Primal barrier Newton metrics')
    plt.grid(True, which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ------------------------------
# Toy problem builders (for validations)
# ------------------------------

def dim_hermitian_quat(n: int) -> int:
    """Real dimension of H_n(H) = {X ∈ H^{n×n} : X = X^H} → 2 n^2 − n."""
    return 2 * (n ** 2) - n


def random_spd(n: int, lam_min: float = 0.5, lam_max: float = 2.0, seed: int | None = None) -> np.ndarray:
    """
    Generate a random quaternion SPD matrix via unitary diagonalization.

    Parameters
    ----------
    n : int
        Matrix size.
    lam_min, lam_max : float
        Bounds for positive eigenvalues.
    seed : int | None
        RNG seed.

    Returns
    -------
    np.ndarray
        Quaternion SPD matrix.
    """
    rng = np.random.default_rng(seed)
    Q = generate_random_unitary_matrix(n)
    lam = rng.uniform(lam_min, lam_max, size=n)
    D = qzeros(n, n)
    for i in range(n):
        D[i, i] = nq.quaternion(float(lam[i]), 0, 0, 0)
    return qmm(qmm(Q, D), qadj(Q))


def random_hermitian(n: int, seed: int | None = None) -> np.ndarray:
    """Random dense quaternion Hermitian matrix."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n, 4))
    A = nq.as_quat_array(a)
    # return quat_hermitian(A)
    return qherm(A)


def random_H_basis(n: int, seed: int | None = None) -> list[np.ndarray]:
    """
    Build a random (numerically independent) set of Hermitian directions H_i.
    For small n, sampling random Hermitians is sufficient for toy problems.
    """
    rng = np.random.default_rng(seed)
    m = dim_hermitian_quat(n)
    H_list: list[np.ndarray] = []
    for _ in range(m):
        H_list.append(random_hermitian(n, seed=rng.integers(1 << 31)))
    return H_list


def build_canonical_orthonormal_basis(n: int) -> list[np.ndarray]:
    """
    Construct a canonical orthonormal basis of quaternion-Hermitian matrices for H_n(H).

    Basis elements (orthonormal under <U,V>=Re tr(U^H V)):
    - Diagonals: D_k = E_{kk}
    - Off-diagonals (i<j): for u in {1, i, j, k}
        S^{(u)}_{ij} = (1/sqrt(2)) ( u E_{ij} + conj(u) E_{ji} )

    Returns
    -------
    list[np.ndarray]
        List of length 2 n^2 - n with dtype quaternion, orthonormal.
    """
    one = nq.quaternion(1.0, 0.0, 0.0, 0.0)
    qi  = nq.quaternion(0.0, 1.0, 0.0, 0.0)
    qj  = nq.quaternion(0.0, 0.0, 1.0, 0.0)
    qk  = nq.quaternion(0.0, 0.0, 0.0, 1.0)
    units = [one, qi, qj, qk]
    basis: list[np.ndarray] = []
    # Diagonals
    for k in range(n):
        D = qzeros(n, n)
        D[k, k] = one
        basis.append(D)
    # Off-diagonals
    s = 1.0 / (2.0 ** 0.5)
    for i in range(n):
        for j in range(i + 1, n):
            for u in units:
                S = qzeros(n, n)
                S[i, j] = s * u
                S[j, i] = s * u.conjugate()
                basis.append(S)
    return basis


def build_singleton_feasible_canonical(n: int = 3, m: int | None = None, seed: int | None = 0):
    """
    Build a singleton-feasible instance using a canonical orthonormal basis for constraints.

    Parameters
    ----------
    n : int
        Matrix size.
    m : int | None
        Number of constraints to use. If None, use full dimension (2 n^2 - n).
    seed : int | None
        RNG seed for X_star and for selecting a subset when m < d.

    Returns
    -------
    (H_list, b, C, X_star)
    """
    rng = np.random.default_rng(seed)
    basis = build_canonical_orthonormal_basis(n)
    d = len(basis)
    if m is None or m >= d:
        idxs = list(range(d))
    else:
        idxs = list(rng.choice(d, size=m, replace=False))
        idxs.sort()
    H_list = [basis[i] for i in idxs]
    X_star = random_spd(n, seed=seed)
    b = np.array([inner_real(H, X_star) for H in H_list], dtype=float)
    C = random_hermitian(n, seed=None if seed is None else seed + 12345)
    return H_list, b, C, X_star

def build_singleton_feasible(n: int = 3, seed: int | None = 0):
    """
    Regime A: Complete constraints that pin X uniquely.

    Returns
    -------
    (H_list, b, C, X_star)
    """
    X_star = random_spd(n, seed=seed)
    H_list = random_H_basis(n, seed=seed)
    b = np.array([inner_real(H, X_star) for H in H_list], dtype=float)
    C = random_hermitian(n, seed=seed + 1)
    return H_list, b, C, X_star


def build_certificate_optimum(n: int = 4, r: int = 2, seed: int | None = 0):
    """
    Regime B: Choose rank-r X_* and slack S_* on its complement.
    Set b = A(X_*), C = S_*, so that (X_*, S_*) is a certificate at optimum.

    Returns
    -------
    (H_list, b, C, X_star, S_star)
    """
    rng = np.random.default_rng(seed)
    Q = generate_random_unitary_matrix(n)
    D = qzeros(n, n)
    for i in range(r):
        D[i, i] = nq.quaternion(1.0, 0, 0, 0)
    X_star = qmm(qmm(Q, D), qadj(Q))

    S = qzeros(n, n)
    for i in range(r, n):
        S[i, i] = nq.quaternion(rng.uniform(0.5, 1.5), 0, 0, 0)
    S_star = qmm(qmm(Q, S), qadj(Q))

    H_list = [random_hermitian(n, seed=rng.integers(1 << 31)) for _ in range(2 * n + 2)]
    b = np.array([inner_real(H, X_star) for H in H_list], dtype=float)
    C = S_star
    return H_list, b, C, X_star, S_star


def build_central_mu_instance(n: int = 3,
                              m: int = 4,
                              mu: float = 1.0,
                              seed: int | None = 0,
                              basis: str = "canonical"):
    """
    Build a small, planted central-point instance for the fixed-μ barrier problem.

    The planted solution is X_star with y_star=0 and C = μ X_star^{-1}.

    Parameters
    ----------
    n : int
        Matrix size.
    m : int
        Number of constraints to use (subset of canonical basis if basis='canonical').
    mu : float
        Barrier parameter used to plant C.
    seed : int | None
        RNG seed.
    basis : str
        'canonical' for a subset of the canonical orthonormal basis; otherwise uses
        random Hermitian constraints orthonormalized via Gram–Cholesky.

    Returns
    -------
    (H_list, b, C, X_star, mu)
    """
    rng = np.random.default_rng(seed)
    X_star = random_spd(n, seed=seed)

    if basis == "canonical":
        B = build_canonical_orthonormal_basis(n)
        idx = sorted(rng.choice(len(B), size=m, replace=False))
        H_list = [B[i] for i in idx]
        b = np.array([inner_real(H, X_star) for H in H_list], dtype=float)
    else:
        # H_raw = [random_hermitian(n, seed=rng.integers(1 << 31)) for _ in range(m)]
        H_raw = [qherm(random_hermitian(n, seed=rng.integers(1 << 31))) for _ in range(m)]
        # Orthonormalize and transform b consistently using triangular solves
        G = np.empty((m, m), dtype=float)
        for i, Hi in enumerate(H_raw):
            for j, Hj in enumerate(H_raw):
                G[i, j] = inner_real(Hi, Hj)
        R = np.linalg.cholesky(G + 1e-12 * np.eye(m)).T  # G = R^T R
        # Build columns of R^{-1} by solving R x = e_j
        H_list = []
        for j in range(m):
            ej = np.zeros(m, dtype=float); ej[j] = 1.0
            col_j = np.linalg.solve(R, ej)  # this is (R^{-1})_{:,j}
            Hj_new = qzeros(n, n)
            for i, coef in enumerate(col_j):
                if coef != 0.0:
                    Hj_new = Hj_new + coef * H_raw[i]
            H_list.append(Hj_new)
        # b_on = R^{-T} b_raw, i.e., solve R^T z = b_raw
        b_raw = np.array([inner_real(H, X_star) for H in H_raw], dtype=float)
        b = np.linalg.solve(R.T, b_raw)

    C = mu * invH(X_star)
    return H_list, b, C, X_star, mu


def _ensure_spd(A: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Ensure Hermitian matrix A is strictly PD by shifting along identity if needed."""
    try:
        lam_min = float(np.min(eigvalsH(A)))
    except Exception:
        lam_min = -1.0
    if lam_min <= floor:
        A = qherm(A + (floor - lam_min) * qeye(A.shape[0]))
    return A


def _resolve_ops(H_list: list[np.ndarray], b: np.ndarray, ops: dict | None = None, assume_hat: bool = False):
    """Resolve or build hat-space closures (Ahat, AThat, transform_b, b_hat) once.

    If ops is provided, use its closures directly. Otherwise, build from raw H_list.
    Asserts Ahat(AThat(v)) ≈ v for a random test vector.
    """
    if ops is None:
        built = _build_orthonormal_ops([qherm(H) for H in H_list])
        Ahat = built['A_hat']
        AThat = built['AT_hat']
        transform_b = built['transform_b']
        b_hat = b if assume_hat else transform_b(b)
        source = "built"
    else:
        Ahat = ops['A_hat']
        AThat = ops['AT_hat']
        transform_b = ops.get('transform_b', lambda z: z)
        b_hat = b if assume_hat else transform_b(b)
        source = "provided"
    # Invariant check
    m_test = int(b_hat.shape[0])
    if m_test > 0:
        rng = np.random.default_rng(0)
        v = rng.standard_normal(m_test)
        ident_err = float(np.linalg.norm(Ahat(AThat(v)) - v))
        if ident_err > 1e-8:
            raise RuntimeError(f"Hat operator invariant failed ({source}): ||Ahat AThat v - v||={ident_err:.2e}")
    return Ahat, AThat, transform_b, b_hat


