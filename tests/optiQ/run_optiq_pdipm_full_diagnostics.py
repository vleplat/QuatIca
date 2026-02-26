#!/usr/bin/env python3
"""
Full diagnostics for OptiQ log-det barrier solvers (fixed-μ Newton + μ-continuation).

This version is compatible with the *current* implementation you pasted:
OptiQ implements a log-det barrier Newton method in hat-space, optionally wrapped in a μ-continuation (barrier path-following) loop.

What it does:
  1) Builds a central-μ instance (build_central_mu_instance).
  2) Runs:
       - Barrier path run (μ-continuation) [μ decreases geometrically]
       - Fixed-μ Newton run (μ constant)
  3) Compares the fixed-μ and μ-continuation solutions at the final μ.
  4) Saves plots + report under:
       validation_output/optiQ/diagnostics/

Usage:
  python tests/run_optiq_pdipm_full_diagnostics.py
  python tests/run_optiq_pdipm_full_diagnostics.py --n 100 --m 100 --basis random --seed 0
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Ensure we import the local repo version (not a site-packages install)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from quatica import build_central_mu_instance  # noqa: E402
from quatica.optiQ import solve_barrier_path, solve_logdet_barrier_newton  # noqa: E402

from quatica.optiQ import (  # noqa: E402
    _build_orthonormal_ops,
    qherm,
    qeye,
    qmm,
    invH,
    eigvalsH,
    sqrtH,
    # barrier/KKT internals for one-step check
    SolverState,
    newton_step,
    _assemble_residuals,
)

from quatica.utils import quat_frobenius_norm  # noqa: E402


# -----------------------------
# Helpers for history keys
# -----------------------------
def mu_from_hist_entry(h: dict, fallback=np.nan) -> float:
    """Prefer 'mu' if present, else use 'gap' (older/barrier-style history)."""
    if "mu" in h and isinstance(h["mu"], (int, float, np.floating)):
        return float(h["mu"])
    if "gap" in h and isinstance(h["gap"], (int, float, np.floating)):
        return float(h["gap"])
    return float(fallback)


def extract_mu_series(hist: list[dict]) -> np.ndarray:
    return np.array([mu_from_hist_entry(h) for h in hist], dtype=float)


def get_numeric_history(history):
    """Keep only records with integer iteration index."""
    out = []
    for h in history:
        it = h.get("it", None)
        if isinstance(it, int):
            out.append(h)
    return out


def series(hist, key, default=np.nan):
    return np.array([float(h.get(key, default)) for h in hist], dtype=float)


# -----------------------------
# Metrics
# -----------------------------
def centrality_residual_std(X, S, mu):
    """Standard SDP centrality: || 0.5*(X S + S X) - mu I ||_F."""
    n = X.shape[0]
    XS = qmm(X, S)
    SX = qmm(S, X)
    return quat_frobenius_norm(0.5 * (XS + SX) - mu * qeye(n))


def barrier_like_residual(X, S, mu):
    """Barrier relation: ||S - mu X^{-1}||_F."""
    return quat_frobenius_norm(S - qherm(float(mu) * invH(X)))


def min_eig(A):
    return float(np.min(eigvalsH(qherm(A))))


# -----------------------------
# Plot helpers
# -----------------------------
def save_residual_plot(hist, out_path):
    it = series(hist, "it")
    rp = series(hist, "rp")
    rp_orig = series(hist, "rp_orig")
    rd = series(hist, "rd")
    gap = series(hist, "gap")  # often equals mu in barrier-style logging

    plt.figure()
    if np.all(np.isnan(rp_orig)):
        plt.semilogy(it, rp, label="||r_p|| (hat)")
    else:
        plt.semilogy(it, rp, label="||r_p|| (hat)")
        plt.semilogy(it, rp_orig, label="||r_p|| (orig)")
    plt.semilogy(it, rd, label="||r_d||_F")
    plt.semilogy(it, gap, label="gap (often == mu)")
    plt.xlabel("iteration")
    plt.ylabel("metric (semilogy)")
    plt.title("log-det barrier solver history: residuals + gap")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_steps_mu_plot(hist, out_path):
    it = series(hist, "it")
    mu = extract_mu_series(hist)
    t = series(hist, "t")
    cg_res = series(hist, "cg_res")
    cg_it = series(hist, "cg_iters")

    plt.figure()
    if not np.all(np.isnan(mu)):
        plt.semilogy(it, mu, label="mu (or gap)")
    if not np.all(np.isnan(t)):
        plt.semilogy(it, t, label="t")
    if not np.all(np.isnan(cg_res)):
        plt.semilogy(it, cg_res, label="CG_res")
    if not np.all(np.isnan(cg_it)):
        plt.semilogy(it, np.maximum(cg_it, 1e-16), label="CG_iters")
    plt.xlabel("iteration")
    plt.ylabel("value (semilogy)")
    plt.title("log-det barrier solver: mu / step / CG info")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_eigs_plot(hist, out_path):
    it = series(hist, "it")
    lamx = series(hist, "lam_min_X")

    plt.figure()
    if not np.all(np.isnan(lamx)):
        plt.semilogy(it, lamx, label="lam_min(X)")
    plt.xlabel("iteration")
    plt.ylabel("value (semilogy)")
    plt.title("log-det barrier solver: interior eigenvalue of X")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# One-step linearization check (BARRIER KKT NEWTON)
# -----------------------------
def one_step_linearization_report_barrier(H_list, b, C, mu0):
    """
    Reconstruct one Newton step for the barrier KKT system and measure
    which linearized equations are satisfied.

    For residuals:
      r_p = b_hat - Ahat(X)
      r_d = C + Ahat^*(y) - mu X^{-1}

    Newton system (as in newton_step docstring):
      H dX + A^* dy = -r_d
      A dX         = -r_p
    where H[W] = mu X^{-1} W X^{-1}.
    """
    ops = _build_orthonormal_ops([qherm(H) for H in H_list])
    Ahat = ops["A_hat"]
    AThat = ops["AT_hat"]
    b_hat = ops["transform_b"](b)

    n = C.shape[0]
    C = qherm(C)

    # Mimic solver init
    X = qherm(qeye(n))
    X = qherm(X + AThat(b_hat - Ahat(X)))
    if min_eig(X) < 1e-12:
        X = qherm(X + (1e-12 - min_eig(X)) * qeye(n))

    y = np.zeros(int(b_hat.shape[0]), dtype=float)
    try:
        y = Ahat(mu0 * invH(X) - C)
    except Exception:
        pass

    # Hat operator wrapper for newton_step
    class _HatOp:
        def __init__(self, Ahat_fn, AThat_fn, H_list_ref):
            self._A = Ahat_fn
            self._AT = AThat_fn
            self.H_list = H_list_ref

        def A(self, X_):
            return self._A(X_)

        def AT(self, y_):
            return self._AT(y_)

    op = _HatOp(Ahat, AThat, H_list)

    # Residuals exactly as solver uses
    r_p, r_d = _assemble_residuals(op, C, b_hat, X, y, mu0)

    state = SolverState(
        X=X,
        y=y,
        r_p=r_p,
        r_d=r_d,
        obj=0.0,
        logdet=0.0,
        mu=mu0,
        k_newton=0
    )
    dX, dy, cg_iters, cg_res = newton_step(op, C, b_hat, state)
    dX = qherm(dX)

    # Check linearized equations
    # (1) primal: A dX = r_p
    primal_lin = float(np.linalg.norm(op.A(dX) - r_p))

    # (2) stationarity: H dX + AT dy = -r_d
    Xinv = invH(X)
    HdX = qherm(mu0 * qmm(qmm(Xinv, dX), Xinv))
    stat_lin = float(quat_frobenius_norm(HdX + op.AT(dy) + r_d))

    return {
        "primal_lin ||A(dX) - r_p||": primal_lin,
        "stationarity_lin ||H(dX)+AT(dy)+r_d||_F": stat_lin,
        "||r_p||": float(np.linalg.norm(r_p)),
        "||r_d||_F": float(quat_frobenius_norm(r_d)),
        "cg_iters": int(cg_iters),
        "cg_res": float(cg_res),
        "lam_min(X)": float(min_eig(X)),
    }


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--basis", type=str, default="random", choices=["canonical", "random"])
    parser.add_argument("--max_iter", type=int, default=80)
    parser.add_argument("--beta_mu", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=1e-8)
    args = parser.parse_args()

    out_dir = os.path.join(_PROJECT_ROOT, "validation_output", "optiQ", "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "report.txt")

    def log(line, f):
        print(line)
        f.write(line + "\n")

    with open(report_path, "w", encoding="utf-8") as f:
        log("=== OptiQ full diagnostics ===", f)
        log(f"Using solve_barrier_path from: {solve_barrier_path.__code__.co_filename}", f)
        log(f"Using solve_logdet_barrier_newton from: {solve_logdet_barrier_newton.__code__.co_filename}", f)
        log(f"n={args.n}, m={args.m}, mu0={args.mu}, beta_mu={args.beta_mu}, eps={args.eps}, max_iter={args.max_iter}", f)
        log(f"seed={args.seed}, basis={args.basis}", f)

        log("\n[1] Building central-μ instance...", f)
        H_list, b, C, X_star, mu_used = build_central_mu_instance(
            n=args.n, m=args.m, mu=args.mu, seed=args.seed, basis=args.basis
        )
        # H_list = [qherm(H) for H in H_list]
        log(f"build_central_mu_instance returned mu={mu_used}", f)

        # One-step Newton consistency check (barrier KKT)
        log("\n[2] One-step linearization check (barrier KKT Newton)", f)
        rep = one_step_linearization_report_barrier(H_list, b, C, mu_used)
        for k, v in rep.items():
            if isinstance(v, float):
                log(f"  {k}: {v:.3e}", f)
            else:
                log(f"  {k}: {v}", f)

        # Build ops and b_hat for controlled hat-space calls
        ops = _build_orthonormal_ops([qherm(H) for H in H_list])
        b_hat = ops["transform_b"](b)

        # PD path run (should decrease μ if your log-det barrier solver implements continuation)
        log("\n[3] Running solve_barrier_path (μ-continuation) in hat space...", f)
        res_path = solve_barrier_path(
            H_list, b_hat, C,
            mu_init=mu_used,
            beta_mu=args.beta_mu,
            mu_min=args.eps,
            eps=args.eps,
            max_iter=args.max_iter,
            verbose=False,
            ops=ops,
            assume_hat=True,
            return_ops=False,
        )
        hist_path = get_numeric_history(res_path.get("history", []))
        if len(hist_path) == 0:
            log("  No numeric history produced; aborting plots.", f)
            return

        mu_hist = extract_mu_series(hist_path)
        log(f"  mu(first 10): {[f'{x:.2e}' for x in mu_hist[:10]]}", f)
        log(f"  mu(last  10): {[f'{x:.2e}' for x in mu_hist[-10:]]}", f)

        Xp, Sp, yp = res_path["X"], res_path["S"], res_path["y"]
        mu_last = float(mu_hist[-1]) if np.isfinite(mu_hist[-1]) else float(mu_used)

        # Final diagnostics on PD/path
        cent_path = centrality_residual_std(Xp, Sp, mu_last)
        barr_path = barrier_like_residual(Xp, Sp, mu_last)
        lamx_path = min_eig(Xp)
        lams_path = min_eig(Sp)

        log("\n  Path final diagnostics:", f)
        log(f"    mu_last={mu_last:.3e}", f)
        log(f"    centrality_std ||0.5*(XS+SX)-muI||_F = {cent_path:.3e}", f)
        log(f"    barrier_like   ||S - mu X^-1||_F     = {barr_path:.3e}", f)
        log(f"    lam_min(X)={lamx_path:.3e}  lam_min(S)={lams_path:.3e}", f)

        # Plots
        log("\n  Saving plots...", f)
        save_residual_plot(hist_path, os.path.join(out_dir, "path_residuals.png"))
        save_steps_mu_plot(hist_path, os.path.join(out_dir, "path_steps_mu.png"))
        save_eigs_plot(hist_path, os.path.join(out_dir, "path_eigs.png"))
        log("    saved: path_residuals.png / path_steps_mu.png / path_eigs.png", f)

        # Fixed-μ run (central point at mu_used)
        log("\n[4] Running solve_logdet_barrier_newton (fixed-μ) in hat space...", f)
        res_fix = solve_logdet_barrier_newton(
            H_list, b_hat, C,
            mu=mu_used,
            eps=args.eps,
            max_iter=args.max_iter,
            verbose=False,
            ops=ops,
            assume_hat=True,
            return_ops=False,
        )
        Xf, Sf, yf = res_fix["X"], res_fix["S"], res_fix["y"]
        hist_fix = get_numeric_history(res_fix.get("history", []))
        if len(hist_fix) > 0:
            mu_fix = mu_from_hist_entry(hist_fix[-1], fallback=mu_used)
        else:
            mu_fix = float(mu_used)

        cent_fix = centrality_residual_std(Xf, Sf, mu_fix)
        barr_fix = barrier_like_residual(Xf, Sf, mu_fix)

        log("  Fixed-μ diagnostics:", f)
        log(f"    mu_fixed={mu_fix:.3e}", f)
        log(f"    centrality_std = {cent_fix:.3e}", f)
        log(f"    barrier_like   = {barr_fix:.3e}", f)
        # Compare fixed-μ and path solutions (at the final μ)
        rel_X = quat_frobenius_norm(Xf - Xp) / max(1.0, quat_frobenius_norm(Xp))
        log("\n[5] Fixed vs path comparison:", f)
        log(f"    ||X_fixed - X_path||_F / ||X_path||_F = {rel_X:.3e}", f)

        log("\nDone. Please share:", f)
        log(f"  - report: {report_path}", f)
        log(f"  - plots in: {out_dir}", f)


if __name__ == "__main__":
    main()
