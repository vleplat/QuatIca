#!/usr/bin/env python3
"""
Quick OptiQ operator sanity checks (adjointness, Hermitian drift, hat-ops isometry,
and inverse quality). Prints diagnostics only.

Usage (from repo root):
  python tests/run_optiq_simple_checks.py

You can tweak n,m,basis,seed below.
"""
import os
import sys
import numpy as np

# Ensure local repo import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from quatica import build_central_mu_instance  # noqa: E402
from quatica.optiQ import (  # noqa: E402
    QuaternionSDPOperator,
    _build_orthonormal_ops,
    qeye,
    qmm,
    qadj,   # conjugate-transpose (adjoint)
    qherm,  # Hermitian projection
    inner_real,
    invH,
    eigvalsH,
)
from quatica.utils import quat_frobenius_norm  # noqa: E402


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    # ---- tweak here ----
    n = 20
    m = 20
    mu = 1.0
    seed = 0
    basis = "random"   # "canonical" or "random"
    rng = np.random.default_rng(123)

    print_header("Building central instance")
    H_list, b, C, X_star, mu_used = build_central_mu_instance(
        n=n, m=m, mu=mu, seed=seed, basis=basis
    )
    print(f"n={n}, m={m}, basis={basis}, seed={seed}, mu={mu_used}")

    # How Hermitian are the raw constraints?
    skews_raw = [quat_frobenius_norm(H - qadj(H)) for H in H_list]
    print("max ||Hi - Hi^H||_F (raw) =", float(np.max(skews_raw)))
    idx = int(np.argmax(skews_raw))
    print("worst i =", idx, "skew =", float(skews_raw[idx]))

    # Enforce Hermitian once (recommended)
    H_listH = [qherm(H) for H in H_list]

    # Operators
    op = QuaternionSDPOperator(H_list=H_listH)
    ops = _build_orthonormal_ops(H_listH)
    Ahat = ops["A_hat"]
    AThat = ops["AT_hat"]

    # Hermitian test matrix X
    X = qherm(X_star)

    # Random real y
    y = rng.standard_normal(m).astype(float)

    # ------------------------------------------------------------------------------
    print_header("1) Adjointness check: <A(X),y> == <X,AT(y)> (unscaled)")
    Ax = op.A(X)
    lhs = float(Ax @ y)
    rhs = float(inner_real(X, op.AT(y)))
    print(f"<A(X),y>        = {lhs:.16e}")
    print(f"<X,AT(y)>       = {rhs:.16e}")
    print(f"abs diff         = {abs(lhs - rhs):.3e}")

    # ------------------------------------------------------------------------------
    print_header("2) Hermitian drift check for AT(y), C, X")
    Y = op.AT(y)
    skew_Y = quat_frobenius_norm(Y - qadj(Y))
    print(f"||AT(y) - AT(y)^H||_F = {skew_Y:.3e}")

    skew_C = quat_frobenius_norm(C - qadj(C))
    skew_X = quat_frobenius_norm(X - qadj(X))
    print(f"||C - C^H||_F         = {skew_C:.3e}")
    print(f"||X - X^H||_F         = {skew_X:.3e}")

    # ------------------------------------------------------------------------------
    print_header("3) Hat-space isometry: Ahat(AThat(v)) ≈ v (should be near machine eps)")
    v = rng.standard_normal(m).astype(float)
    v_back = Ahat(AThat(v))
    err_iso = np.linalg.norm(v_back - v) / (np.linalg.norm(v) + 1e-30)
    print(f"relative ||Ahat(AThat(v)) - v|| = {err_iso:.3e}")

    # Also check hat-space adjointness:
    print_header("4) Hat adjointness: <Ahat(X),v> == <X,AThat(v)>")
    Axh = Ahat(X)
    lhs2 = float(Axh @ v)
    rhs2 = float(inner_real(X, AThat(v)))
    print(f"<Ahat(X),v>     = {lhs2:.16e}")
    print(f"<X,AThat(v)>    = {rhs2:.16e}")
    print(f"abs diff         = {abs(lhs2 - rhs2):.3e}")

    # ------------------------------------------------------------------------------
    print_header("5) invH(X) quality check: ||X*invH(X) - I||")
    Xinv = invH(X)
    I = qeye(n)
    inv_err = quat_frobenius_norm(qmm(X, Xinv) - I) / (quat_frobenius_norm(I) + 1e-30)
    inv_err2 = quat_frobenius_norm(qmm(Xinv, X) - I) / (quat_frobenius_norm(I) + 1e-30)
    print(f"rel ||X*Xinv - I||_F = {inv_err:.3e}")
    print(f"rel ||Xinv*X - I||_F = {inv_err2:.3e}")

    # ------------------------------------------------------------------------------
    print_header("6) Quick eig sanity: lam_min(X), lam_max(X)")
    lam = eigvalsH(X)
    print(f"lam_min(X) = {float(np.min(lam)):.6e}")
    print(f"lam_max(X) = {float(np.max(lam)):.6e}")

    # ------------------------------------------------------------------------------
    print_header("7) Optional: check Hermitian drift of AT_hat(v)")
    Yh = AThat(v)
    skew_Yh = quat_frobenius_norm(Yh - qadj(Yh))
    print(f"||AT_hat(v) - AT_hat(v)^H||_F = {skew_Yh:.3e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
