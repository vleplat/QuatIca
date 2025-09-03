import numpy as np
import quaternion

from quatica import maxvol_submatrix_quat
from quatica.decomp.qsvd import classical_qsvd_full
from quatica.utils import quat_matmat, quat_hermitian


def quaternion_volume(B: np.ndarray) -> float:
    U, s, V = classical_qsvd_full(B)
    if len(s) == 0:
        return 0.0
    prod = 1.0
    for val in s[: min(B.shape)]:
        prod *= float(val)
    return float(prod)


def generate_quat_matrix(m: int, n: int) -> np.ndarray:
    real = np.random.randn(m, n)
    i = np.random.randn(m, n)
    j = np.random.randn(m, n)
    k = np.random.randn(m, n)
    return quaternion.as_quat_array(np.stack([real, i, j, k], axis=-1))


def test_maxvol_monotone_volume_small():
    np.random.seed(42)
    m, n, k = 12, 10, 4
    A = generate_quat_matrix(m, n)
    I, J, B, info = maxvol_submatrix_quat(A, k=k, tol=1e-8, max_sweeps=20, track_history=True)

    assert len(I) == k and len(J) == k

    # Check monotone non-decreasing volume across accepted swaps (history)
    vols = [quaternion_volume(Bi) for Bi in info.get("B_history", [])]
    for a, b in zip(vols, vols[1:]):
        assert b >= a - 1e-10


def test_maxvol_core_consistency():
    np.random.seed(0)
    m, n, k = 8, 9, 3
    A = generate_quat_matrix(m, n)
    I, J, B, info = maxvol_submatrix_quat(A, k=k, tol=1e-8, max_sweeps=10, track_history=False)
    # Consistency: B equals the extracted submatrix
    B_true = A[np.ix_(I, J)]
    diff = np.linalg.norm(quaternion.as_float_array(B - B_true))
    assert diff <= 1e-10


