import numpy as np
import quaternion  # type: ignore
import pytest

from quatica.decomp.chol import chol_quat_dense, solve_chol_quat_dense
from quatica.utils import quat_eye, quat_hermitian, quat_matmat, quat_frobenius_norm


def _rand_quat_matrix(m: int, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Qf = rng.standard_normal((m, n, 4))
    return quaternion.as_quat_array(Qf)


def test_chol_quat_dense_reconstruction_and_solve():
    n = 8
    B = _rand_quat_matrix(n, n, seed=0)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        A = quat_matmat(B, quat_hermitian(B)) + 1.0 * quat_eye(n)

    L = chol_quat_dense(A, tol=1e-12, hermitianize=True, jitter=0.0)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        A_rec = quat_matmat(L, quat_hermitian(L))
    rel = quat_frobenius_norm(A - A_rec) / (quat_frobenius_norm(A) + 1e-30)
    assert float(rel) < 1e-10

    rng = np.random.default_rng(1)
    bf = rng.standard_normal((n, 4))
    b = quaternion.as_quat_array(bf)
    x = solve_chol_quat_dense(L, b)

    res = quat_frobenius_norm(quat_matmat(A, x.reshape(n, 1)).reshape(n) - b) / (
        quat_frobenius_norm(b) + 1e-30
    )
    assert float(res) < 1e-10


def test_chol_quat_dense_fails_on_non_hpd():
    # Hermitian but indefinite: diag([1, -1])
    A = np.zeros((2, 2, 4), dtype=float)
    A[0, 0, 0] = 1.0
    A[1, 1, 0] = -1.0
    Aq = quaternion.as_quat_array(A)

    with pytest.raises(np.linalg.LinAlgError):
        chol_quat_dense(Aq, tol=1e-12, hermitianize=True)

