import numpy as np
import quaternion  # type: ignore

from quatica.decomp.hessenberg import hessenbergize
from quatica.utils import quat_frobenius_norm, quat_hermitian, quat_matmat, real_contract, real_expand


def test_real_expand_scalar_block_convention_matches_first_column():
    rng = np.random.default_rng(0)
    q = quaternion.quaternion(
        float(rng.standard_normal()),
        float(rng.standard_normal()),
        float(rng.standard_normal()),
        float(rng.standard_normal()),
    )
    Q = np.array([[q]], dtype=np.quaternion)
    R = real_expand(Q)
    B = R[0:4, 0:4]

    # Convention used by Schur's _real_get_quat_entry: first column is [w,x,y,z]^T
    assert np.allclose(
        np.array([B[0, 0], B[1, 0], B[2, 0], B[3, 0]], dtype=float),
        np.array([q.w, q.x, q.y, q.z], dtype=float),
        atol=1e-12,
        rtol=0.0,
    )

    # Full round-trip check (should always hold)
    q2 = real_contract(B, 1, 1)[0, 0]
    assert np.allclose(
        np.array([q2.w, q2.x, q2.y, q2.z], dtype=float),
        np.array([q.w, q.x, q.y, q.z], dtype=float),
        atol=1e-12,
        rtol=0.0,
    )


def test_hessenbergize_similarity_convention():
    rng = np.random.default_rng(1)
    A = quaternion.as_quat_array(rng.standard_normal((6, 6, 4)))

    P, H = hessenbergize(A)

    # schur.py assumes (and hessenberg.py documents) H = P A P^H
    H2 = quat_matmat(quat_matmat(P, A), quat_hermitian(P))
    rel = float(quat_frobenius_norm(H - H2) / (quat_frobenius_norm(H) + 1e-30))
    assert rel < 1e-10

