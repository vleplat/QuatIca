import numpy as np
import quaternion

from quatica.utils import (
    induced_matrix_norm_1,
    induced_matrix_norm_inf,
    quat_abs_scalar,
)


def _slow_norm_1(A: np.ndarray) -> float:
    m, n = A.shape
    max_col_sum = 0.0
    for j in range(n):
        col_sum = 0.0
        for i in range(m):
            col_sum += quat_abs_scalar(A[i, j])
        max_col_sum = max(max_col_sum, col_sum)
    return float(max_col_sum)


def _slow_norm_inf(A: np.ndarray) -> float:
    m, n = A.shape
    max_row_sum = 0.0
    for i in range(m):
        row_sum = 0.0
        for j in range(n):
            row_sum += quat_abs_scalar(A[i, j])
        max_row_sum = max(max_row_sum, row_sum)
    return float(max_row_sum)


def _random_quat_matrix(rng: np.random.Generator, m: int, n: int) -> np.ndarray:
    comp = rng.standard_normal((m, n, 4))
    return quaternion.as_quat_array(comp)


def test_induced_matrix_norms_match_loop_definition():
    rng = np.random.default_rng(0)
    for (m, n) in [(1, 1), (2, 3), (5, 4), (10, 10)]:
        A = _random_quat_matrix(rng, m, n)
        assert np.isclose(induced_matrix_norm_1(A), _slow_norm_1(A), rtol=1e-12, atol=1e-12)
        assert np.isclose(induced_matrix_norm_inf(A), _slow_norm_inf(A), rtol=1e-12, atol=1e-12)


def test_induced_matrix_norms_empty_matrix_is_zero():
    A = np.empty((0, 0), dtype=np.quaternion)
    assert induced_matrix_norm_1(A) == 0.0
    assert induced_matrix_norm_inf(A) == 0.0

