import numpy as np
import quaternion
from scipy import sparse


def _rand_quat_matrix(rng: np.random.Generator, m: int, n: int) -> np.ndarray:
    arr = rng.standard_normal((m, n, 4))
    return quaternion.as_quat_array(arr)


def _quat_eye(n: int) -> np.ndarray:
    I = np.zeros((n, n), dtype=np.quaternion)
    np.fill_diagonal(I, quaternion.quaternion(1.0, 0.0, 0.0, 0.0))
    return I


def test_qgmres_reporting_dense_identity():
    from quatica.solver import QGMRESSolver

    rng = np.random.default_rng(0)
    n = 6
    A = _quat_eye(n)
    b = _rand_quat_matrix(rng, n, 1)

    tol = 1e-10
    solver = QGMRESSolver(tol=tol, max_iter=n, verbose=False)
    x, info = solver.solve(A, b)

    assert "residual_est" in info
    assert "residual_true_available" in info
    assert "residual_true" in info
    assert isinstance(info["converged"], bool)

    # Final residual field defines convergence
    assert info["converged"] == (info["residual"] < tol)

    # For A=I, true residual should be near machine precision
    assert info["residual_true_available"] is True
    assert info["residual_true"] is not None
    assert info["residual"] == info["residual_true"]
    assert info["residual_true"] <= 1e-12


def test_qgmres_reporting_sparse_identity_true_residual_available():
    from quatica.solver import QGMRESSolver
    from quatica.utils import SparseQuaternionMatrix

    rng = np.random.default_rng(1)
    n = 5

    # Sparse quaternion identity
    I = sparse.identity(n, format="csr", dtype=float)
    A = SparseQuaternionMatrix(I, sparse.csr_matrix((n, n)), sparse.csr_matrix((n, n)), sparse.csr_matrix((n, n)), (n, n))

    b = _rand_quat_matrix(rng, n, 1)
    tol = 1e-10
    solver = QGMRESSolver(tol=tol, max_iter=n, verbose=False)
    x, info = solver.solve(A, b)

    assert info["residual_true_available"] is True
    assert info["residual_true"] is not None
    assert info["converged"] == (info["residual"] < tol)
    assert info["residual_true"] <= 1e-12
