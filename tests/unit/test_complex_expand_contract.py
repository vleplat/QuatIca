import numpy as np
import quaternion  # type: ignore
import pytest


from quatica.utils import complex_contract, complex_expand


def _rand_quat_matrix(m: int, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Qf = rng.standard_normal((m, n, 4))
    return quaternion.as_quat_array(Qf)


def test_complex_expand_contract_roundtrip_rectangular():
    Q = _rand_quat_matrix(5, 3, seed=1)
    M = complex_expand(Q)
    Q2 = complex_contract(M, 5, 3, check_structure=True, tol=1e-12)

    # Exact equality is not expected; components should match to floating error.
    assert np.allclose(quaternion.as_float_array(Q), quaternion.as_float_array(Q2), atol=1e-12)


def test_complex_contract_structure_check_raises():
    Q = _rand_quat_matrix(4, 2, seed=2)
    M = complex_expand(Q)
    M_bad = M.copy()
    M_bad[0, 0] += 1e-2  # break structure

    with pytest.raises(ValueError):
        complex_contract(M_bad, 4, 2, check_structure=True, tol=1e-10)

