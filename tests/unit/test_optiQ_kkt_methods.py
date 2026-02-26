import numpy as np
import pytest

from quatica import (
    BarrierParams,
    solve_barrier,
    solve_pd_mehrotra,
    build_central_mu_instance,
)
from quatica.utils import quat_frobenius_norm


@pytest.mark.parametrize("n,m,mu,seed,basis", [
    (3, 4, 1.0, 0, "canonical"),
    (5, 6, 1.0, 1, "canonical"),
])
def test_primal_barrier_on_central(n, m, mu, seed, basis):
    """
    Fixed-μ barrier problem: build_central_mu_instance plants (X_star, y_star=0)
    with C = μ X_star^{-1}. The barrier solver should recover X_star.
    """
    H_list, b, C, X_star, mu_used = build_central_mu_instance(
        n=n, m=m, mu=mu, seed=seed, basis=basis
    )

    # Identity start (same dtype as quaternion matrices)
    X0 = np.eye(n, dtype=H_list[0].dtype)

    # Keep μ fixed: one-stage barrier solve
    params = BarrierParams(
        mu=mu_used, mu_decay=0.5, mu_min=mu_used,
        newton_tol=1e-8, newton_maxit=80
    )

    state = solve_barrier(H_list, b, C, X0=X0, params=params, verbose=True)

    rel_err = quat_frobenius_norm(state.X - X_star) / max(1.0, quat_frobenius_norm(X_star))
    rp_norm = float(np.linalg.norm(state.r_p))

    print(f"[barrier] n={n} m={m} mu={mu_used:.2e} ||r_p||={rp_norm:.2e} rel_err={rel_err:.2e}")

    assert rp_norm <= 1e-6
    assert rel_err <= 1e-2


@pytest.mark.parametrize("n,m,mu,seed,basis", [
    (3, 4, 1.0, 2, "canonical"),
    (5, 6, 1.0, 3, "canonical"),
])
def test_primal_dual_mehrotra_on_central(n, m, mu, seed, basis):
    """
    IMPORTANT: With the current OptiQ implementation, solve_pd_mehrotra is a
    barrier-KKT Newton solver. For a planted central-μ instance, we must keep μ fixed.
    """
    H_list, b, C, X_star, mu_used = build_central_mu_instance(
        n=n, m=m, mu=mu, seed=seed, basis=basis
    )

    res = solve_pd_mehrotra(
        H_list, b, C,
        mu_init=mu_used,
        beta_mu=0.5,              # irrelevant when fixed_mu=True
        eps_p=1e-8, eps_d=1e-8, eps_gap=1e-8,
        max_iter=80,
        verbose=True,
        fixed_mu=True,            # <<< KEY FIX: do not run μ-continuation on a planted fixed-μ instance
    )

    X = res["X"]
    history = res.get("history", [])

    # Use rp_hat returned by the solver if present; else fall back to last history entry.
    rp_hat = res.get("rp_hat", None)
    if rp_hat is None:
        rp_hat = history[-1].get("rp", np.inf) if history else np.inf
    rp_hat = float(rp_hat)

    rel_err = quat_frobenius_norm(X - X_star) / max(1.0, quat_frobenius_norm(X_star))

    print(f"[pd/barrier] n={n} m={m} mu={mu_used:.2e} rp_hat={rp_hat:.2e} rel_err={rel_err:.2e}")

    assert rp_hat <= 1e-6
    assert rel_err <= 1e-2


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-q"])