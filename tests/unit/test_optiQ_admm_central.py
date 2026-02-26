import os
import numpy as np
import pytest

from quatica import (
    build_central_mu_instance,
    admm_barrier_fixed_mu,
    save_admm_history_plot,
)
from quatica.utils import quat_frobenius_norm, quat_matmat


def _inner_real(U, V):
    UH = U.conjugate().T
    M = quat_matmat(UH, V)
    s = 0.0
    n = M.shape[0]
    for i in range(n):
        s += float(M[i, i].real)
    return s


@pytest.mark.parametrize("n,m,mu,seed,basis", [
    (3, 4, 1.0, 0, "canonical"),
    (5, 6, 1.0, 1, "canonical"),
    (10, 6, 1.0, 2, "canonical"),
    (10, 4, 1.0, 3, "random"),
])
def test_admm_central_point(n, m, mu, seed, basis):
    H_list, b, C, X_star, mu_used = build_central_mu_instance(n=n, m=m, mu=mu, seed=seed, basis=basis)
    # Direct projector (constraints are orthonormal in canonical or ON-random path)
    X, Z, U, hist = admm_barrier_fixed_mu(
        H_list, b, C, mu=mu_used, rho=10.0, maxit=400, alpha=1.2, verbose=False, use_hat_projection=False
    )
    out_dir = os.path.join("validation_output", "admm")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"central_{basis}_n{n}_m{m}_seed{seed}"
    save_admm_history_plot(hist, os.path.join(out_dir, f"residuals_{tag}.png"))

    # Equality (direct coordinates, quaternion-native)
    def A_on(X):
        return np.array([_inner_real(Hi, X) for Hi in H_list], dtype=float)
    eq = float(np.linalg.norm(b - A_on(Z)))

    # SPD and accuracy
    rel_err = quat_frobenius_norm(X - X_star) / max(1.0, quat_frobenius_norm(X_star))

    assert eq <= 1e-10
    assert rel_err <= 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
