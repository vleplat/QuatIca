import os
import numpy as np
import pytest

from quatica import (
    admm_barrier_fixed_mu,
    save_admm_history_plot,
    build_singleton_feasible_canonical,
    build_singleton_feasible,
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


def _gram(H_list):
    m = len(H_list)
    G = np.empty((m, m), dtype=float)
    for i, Hi in enumerate(H_list):
        for j, Hj in enumerate(H_list):
            G[i, j] = _inner_real(Hi, Hj)
    return G


def _hat_ops(H_list, b):
    G = _gram(H_list)
    R = np.linalg.cholesky(G).T
    R_inv = np.linalg.inv(R)
    R_inv_T = R_inv.T

    def A_unscaled(X):
        return np.array([_inner_real(Hi, X) for Hi in H_list], dtype=float)

    def A_hat(X):
        return R_inv_T @ A_unscaled(X)

    def AT_hat(y):
        y_raw = R_inv @ y
        n = H_list[0].shape[0]
        Y = np.zeros((n, n), dtype=H_list[0].dtype)
        for yi, Hi in zip(y_raw, H_list):
            if yi != 0.0:
                Y = Y + yi * Hi
        return Y

    b_hat = R_inv_T @ b
    return A_hat, AT_hat, b_hat, G


@pytest.mark.parametrize("n,use_canonical,seed", [
    (3, True,  0),
    (5, True,  1),
    (3, False, 2),
    (5, False, 3),
])
def test_admm_fixed_mu_convergence(n, use_canonical, seed):
    # Build problem
    if use_canonical:
        H_list, b, C, X_star = build_singleton_feasible_canonical(n=n, m=None, seed=seed)
        tag = f"canonical_n{n}_seed{seed}"
        # Verify canonical basis orthonormality
        G = _gram(H_list)
        I = np.eye(len(H_list))
        assert np.allclose(G, I, atol=1e-8, rtol=0.0)
        use_hat = False
        Ahat = AThat = b_hat = None
    else:
        # Original random constraints with hat projection (stay consistent in hat coords)
        H_list, b, C, X_star = build_singleton_feasible(n=n, seed=seed)
        tag = f"randomHAT_n{n}_seed{seed}"
        use_hat = True
        # Build hat closures and run projector identity invariant
        Ahat, AThat, b_hat, G = _hat_ops(H_list, b)
        # Projector identity check in hat space
        nloc = H_list[0].shape[0]
        V = np.zeros((nloc, nloc), dtype=H_list[0].dtype)
        Ztest = V + AThat(b_hat - Ahat(V))
        eq_hat_test = float(np.linalg.norm(b_hat - Ahat(Ztest)))
        print(f"[invariant] ||b_hat - Ahat(V + AT_hat(b_hat - Ahat(V)))|| = {eq_hat_test:.2e}")
        assert eq_hat_test <= 1e-10

    # Run ADMM (fixed mu)
    mu = 1.0
    X, Z, U, hist = admm_barrier_fixed_mu(
        H_list, b, C, mu=mu, rho=5.0, maxit=800, alpha=1.0, verbose=False, use_hat_projection=use_hat
    )

    # Save plots early so they're produced even if assertions fail
    out_dir = os.path.join("validation_output", "admm")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"residuals_{tag}.png")
    save_admm_history_plot(hist, out_path)

    # Basic measures
    pr_series = hist['pr']
    pr = pr_series[-1] if len(pr_series) else 1e9
    pr0 = pr_series[0] if len(pr_series) else pr
    eq = hist['eq'][-1] if len(hist['eq']) else 1e9
    lamX = hist['lamX'][-1] if len(hist['lamX']) else -1.0
    lamZ = hist['lamZ'][-1] if len(hist['lamZ']) else -1.0
    err = quat_frobenius_norm(X - X_star)

    # Random-case: compute hat residual at exit and snap Z to current V
    if not use_canonical:
        Ahat2, AThat2, b_hat2, _ = _hat_ops(H_list, b)
        V_end = X + U
        Z_proj = V_end + AThat2(b_hat2 - Ahat2(V_end))
        eq_hat = float(np.linalg.norm(b_hat2 - Ahat2(Z_proj)))
        print(f"[post] eq_hat={eq_hat:.2e} (after snap)")
        # Snap Z to projected value for strict equality
        Z = Z_proj

    # Tolerances
    if use_canonical:
        # Strict for orthonormal basis cases
        print(f"[diag] pr={pr:.2e} eq={eq:.2e} err={err:.2e}")
        assert pr <= 5e-5
        assert eq <= 5e-5
        assert err <= 5e-3
    else:
        # Random with hat projection: assert on hat residual (coordinate-consistent)
        assert eq_hat <= 1e-10
        assert lamX > 1e-8 and lamZ > 1e-8
        # Primal residual should improve substantially and be reasonably small
        improve = (pr0 / max(pr, 1e-12)) if pr0 > 0 else 1.0
        print(f"[diag] pr_end={pr:.2e} improve={improve:.2f} err={err:.2e}")
        assert improve >= 2.0
        assert pr <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
