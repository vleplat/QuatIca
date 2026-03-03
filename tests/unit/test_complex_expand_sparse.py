import numpy as np
from scipy import sparse

from quatica.utils import SparseQuaternionMatrix, complex_expand_sparse


def test_complex_expand_sparse_block_identities():
    rng = np.random.default_rng(0)
    n = 7
    density = 0.2

    def rs():
        return sparse.random(n, n, density=density, format="csr", random_state=rng)

    real = rs()
    i = rs()
    j = rs()
    k = rs()
    Aq = SparseQuaternionMatrix(real, i, j, k, (n, n))

    chi = complex_expand_sparse(Aq)
    assert chi.shape == (2 * n, 2 * n)
    chi = chi.tocsr()

    X = (real + 1j * i).tocsr()
    Y = (j + 1j * k).tocsr()

    # Top-left / top-right
    assert (chi[:n, :n] - X).nnz == 0
    assert (chi[:n, n:] - Y).nnz == 0

    # Bottom blocks
    assert (chi[n:, :n] - (-Y.conjugate())).nnz == 0
    assert (chi[n:, n:] - X.conjugate()).nnz == 0

