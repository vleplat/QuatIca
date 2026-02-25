import numpy as np
import quaternion


def _rand_quat(m: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((m, n, 4))
    return quaternion.as_quat_array(arr)


def test_real_expand_contract_roundtrip_random():
    from quatica.utils import real_contract, real_expand

    for (m, n) in [(1, 1), (2, 3), (10, 7), (25, 25)]:
        Q = _rand_quat(m, n, seed=m * 100 + n)
        R = real_expand(Q)
        Q2 = real_contract(R, m, n)
        assert np.allclose(
            quaternion.as_float_array(Q),
            quaternion.as_float_array(Q2),
            atol=0.0,
            rtol=0.0,
        )


def test_real_expand_block_layout_single_entry():
    from quatica.utils import real_expand

    # Single quaternion entry should map to a 4x4 real block.
    q = quaternion.quaternion(1.0, 2.0, 3.0, 4.0)  # w=1, x=2, y=3, z=4
    Q = np.array([[q]], dtype=np.quaternion)
    R = real_expand(Q)
    expected = np.array(
        [
            [1.0, -2.0, -3.0, -4.0],
            [2.0, 1.0, -4.0, 3.0],
            [3.0, 4.0, 1.0, -2.0],
            [4.0, -3.0, 2.0, 1.0],
        ],
        dtype=float,
    )
    assert R.shape == (4, 4)
    assert np.allclose(R, expected, atol=0.0, rtol=0.0)

