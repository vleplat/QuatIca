import numpy as np
import quaternion as nq

from quatica import (
    solve_pd_mehrotra,
    build_central_mu_instance,
)
from quatica.utils import quat_frobenius_norm, quat_hermitian
from quatica.optiQ import _build_orthonormal_ops, qeye, eigvalsH, qherm, invH


def test_pdipm_central_small():
    n = 5
    H_list, b, C, X_star, mu_used = build_central_mu_instance(n=n, m=5, mu=1.0, seed=0, basis="blablabla")

    # Build hat closures and project a simple PD seed to exact feasibility
    ops = _build_orthonormal_ops(H_list)
    Ahat = ops['A_hat']; AThat = ops['AT_hat']; b_hat = ops['transform_b'](b)

    # Start from τ I, then project in hat-space to satisfy A(X)=b exactly
    tau = 1.0
    V = qeye(n) * tau
    X0 = quat_hermitian(V + AThat(b_hat - Ahat(V)))
    # Ensure PD: shift if needed
    lam_min = float(np.min(eigvalsH(X0)))
    if lam_min <= 1e-6:
        X0 = X0 + (1e-3 - lam_min) * qeye(n)

    # Dual start consistent with X0 and hat coordinates
    S0 = quat_hermitian(mu_used * invH(X0))
    lam_min_S = float(np.min(eigvalsH(S0)))
    if lam_min_S <= 1e-6:
        S0 = S0 + (1e-3 - lam_min_S) * qeye(n)
    # y0 such that C + A* y0 - S0 = 0 in hat coordinates
    y0 = Ahat(S0 - C)

    # Use PD-IPM Mehrotra with fixed mu = 1.0
    res = solve_pd_mehrotra(H_list, b_hat, C, X0=X0, S0=S0, y0=y0,
                            mu_init=mu_used, beta_mu=0.5,
                            eps_p=1e-8, eps_d=1e-8, eps_gap=1e-8,
                            max_iter=60, verbose=True, fixed_mu=True,
                            ops=ops, assume_hat=True)
    X = res['X']
    rp = np.linalg.norm(b_hat - Ahat(X))
    rel_err = quat_frobenius_norm(X - X_star) / max(1.0, quat_frobenius_norm(X_star))
    assert rp <= 1e-6
    assert rel_err <= 5e-2


if __name__ == "__main__":
    test_pdipm_central_small()
    print("PD-IPM central-point test completed")


