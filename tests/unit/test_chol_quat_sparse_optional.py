import numpy as np
import quaternion  # type: ignore
import pytest
from scipy import sparse

from quatica.decomp.chol import chol_quat_sparse
from quatica.utils import SparseQuaternionMatrix


pytest.importorskip("sksparse.cholmod")


def _rand_sparse_lower(n: int, density: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    L = sparse.random(n, n, density=density, format="csr", random_state=rng)
    L = sparse.tril(L, k=0).tocsr()
    # Ensure positive diagonal
    L = L + sparse.diags(np.full(n, 1.0), format="csr")
    return L


def test_chol_quat_sparse_solve_residual():
    # Build quaternion sparse SPD A = L L^* with real-only components.
    n = 50
    Lr = _rand_sparse_lower(n, density=0.05, seed=1)
    zero = sparse.csr_matrix((n, n))
    Lq = SparseQuaternionMatrix(Lr, zero, zero, zero, (n, n))
    Aq = Lq @ Lq.conjugate().transpose()

    F = chol_quat_sparse(Aq, jitter=1e-12, ordering="cholmod")

    rng = np.random.default_rng(2)
    bf = rng.standard_normal((n, 4))
    b = quaternion.as_quat_array(bf)
    x = F.solve(b)

    # Residual: ||Ax - b|| / ||b||
    Ax = Aq @ x
    num = np.linalg.norm(quaternion.as_float_array(Ax) - quaternion.as_float_array(b))
    den = np.linalg.norm(quaternion.as_float_array(b)) + 1e-30
    assert float(num / den) < 1e-8

