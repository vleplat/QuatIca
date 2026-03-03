import numpy as np
import quaternion  # type: ignore

from quatica.decomp.schur import quaternion_schur
from quatica.utils import quat_frobenius_norm, quat_hermitian, quat_matmat


def _rand_quat_matrix(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return quaternion.as_quat_array(rng.standard_normal((n, n, 4)))


def test_schur_similarity_and_unitarity_residuals():
    # Non-Hermitian random case: the key invariants we want are
    # 1) similarity: A ≈ Q T Q^H
    # 2) unitarity: Q^H Q ≈ I
    n = 6
    A = _rand_quat_matrix(n, seed=7)

    Q, T = quaternion_schur(A, max_iter=800, tol=1e-10, shift="rayleigh")

    QT = quat_hermitian(Q)
    A_recon = quat_matmat(quat_matmat(Q, T), QT)

    sim = float(quat_frobenius_norm(A - A_recon) / (quat_frobenius_norm(A) + 1e-30))
    unit = float(
        quat_frobenius_norm(quat_matmat(QT, Q) - np.eye(n, dtype=np.quaternion))
    )

    assert sim <= 1e-6
    assert unit <= 1e-8

