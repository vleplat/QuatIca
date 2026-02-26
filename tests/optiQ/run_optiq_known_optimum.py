#!/usr/bin/env python3
"""Known-optimum (trace-bounded) OptiQ regression run + visual diagnostics.

This script constructs an SDP with a certified primal optimum X_star and runs the
fixed-\mu log-det barrier Newton solver over a decreasing \mu schedule
(warm-starting each stage). It then saves:

- objective vs \mu
- relative error vs \mu
- eigenvalue min/max vs \mu
- side-by-side 5×5 block comparisons of quaternion components between X_star and X(\mu_min)
- full-matrix magnitude heatmaps |X_star|, |X(\mu_min)|, and |X_star - X(\mu_min)|

Usage (from project root):
  python tests/run_optiq_known_optimum.py
"""
import os
import sys
import argparse
import numpy as np
import quaternion as nq  # type: ignore

# Optional: plotting is only used for saving figures
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_project_root(start_dir: str) -> str:
    """Find repository root by walking up until pyproject.toml is found."""
    cur = start_dir
    while True:
        if os.path.isfile(os.path.join(cur, "pyproject.toml")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise RuntimeError("Could not locate project root (pyproject.toml not found).")
        cur = parent


PROJECT_ROOT = _find_project_root(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# Ensure the script directory is importable (for optiq_visualization.py)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from quatica.optiQ import (  # noqa: E402
    solve_logdet_barrier_newton,
    qzeros, qeye, qmm, qadj, qherm,
    inner_real, eigvalsH, eighH,
    random_hermitian,
    _build_orthonormal_ops,
)

from optiq_visualization import (  # noqa: E402
    save_quaternion_block_comparison,
    save_quaternion_abs_comparison,
)

def save_png_and_pdf(fig, png_path: str, dpi: int = 200) -> None:
    """Save the same figure as PNG and PDF using the same basename."""
    root, ext = os.path.splitext(png_path)
    if ext.lower() != ".png":
        raise ValueError(f"Expected a .png path, got: {png_path}")
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(root + ".pdf")


def diag_real_quat(d: np.ndarray):
    n = d.size
    D = qzeros(n, n)
    for i in range(n):
        D[i, i] = nq.quaternion(float(d[i]), 0.0, 0.0, 0.0)
    return D


def normF(A) -> float:
    return float(np.sqrt(max(inner_real(A, A), 0.0)))


def lam_minmax(X):
    lam = np.sort(eigvalsH(qherm(X)))
    return float(lam[0]), float(lam[-1])


def build_instance(n, m_extra, rank, seed, eps_slater=1e-2):
    """
    Known optimum SDP (trace-bounded):

      minimize <C,X>   s.t. Ahat(X)=b_hat,  X ⪰ 0

    Construction:
      - X_star PSD rank=rank, objective 0
      - C = S_star PSD complementary to X_star
      - constraints include trace(X)=rank to prevent blow-up in ker(C)
      - plus m_extra random trace-free constraints (optional) to reduce degeneracy
    """
    rng = np.random.default_rng(seed)

    # Eigenbasis V
    Hr = random_hermitian(n, seed=int(rng.integers(1 << 31)))
    _, V = eighH(Hr)

    # X_star and C=S_star complementary
    dx = np.zeros(n); dx[:rank] = 1.0
    ds = np.zeros(n); ds[rank:] = 1.0
    X_star = qherm(qmm(qmm(V, diag_real_quat(dx)), qadj(V)))
    C = qherm(qmm(qmm(V, diag_real_quat(ds)), qadj(V)))  # = S_star

    # Build constraints:
    # (1) trace constraint: <I, X> = rank
    I = qeye(n)
    H0 = (1.0 / np.sqrt(n)) * I  # normalized for conditioning
    b0 = inner_real(H0, X_star)  # = rank / sqrt(n)

    # (2) m_extra random trace-free constraints (optional)
    ii = inner_real(I, I)
    H_raw = [H0]
    for _ in range(m_extra):
        H = random_hermitian(n, seed=int(rng.integers(1 << 31)))
        alpha = inner_real(H, I) / ii
        H = qherm(H - float(alpha) * I)  # trace-free
        H_raw.append(H)

    # Orthonormalize -> hat ops
    ops = _build_orthonormal_ops([qherm(H) for H in H_raw])
    Ahat = ops["A_hat"]

    # b_hat built from X_star (so X_star feasible)
    b_hat = Ahat(X_star)

    # Slater point: keep trace fixed, add small isotropic mass
    X_feas = qherm((1.0 - eps_slater) * X_star + eps_slater * (rank / n) * I)

    return H_raw, ops, b_hat, C, X_star, X_feas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--rank", type=int, default=10)
    ap.add_argument("--m_extra", type=int, default=19, help="extra constraints besides trace (total m = 1+m_extra)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps_slater", type=float, default=1e-2)

    ap.add_argument("--mu0", type=float, default=1.0)
    ap.add_argument("--mu_min", type=float, default=1e-8)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--newton", type=int, default=120)
    ap.add_argument("--tol", type=float, default=1e-10)

    ap.add_argument("--block", type=int, default=5, help="top-left block size for component visualization")
    args = ap.parse_args()

    H_list, ops, b_hat, C, X_star, X0 = build_instance(
        args.n, args.m_extra, args.rank, args.seed, args.eps_slater
    )
    Ahat = ops["A_hat"]

    print("=== Instance sanity ===")
    print("m total =", 1 + args.m_extra)
    print("||Ahat(X_star)-b|| =", np.linalg.norm(Ahat(X_star) - b_hat))
    print("||Ahat(X0)-b||     =", np.linalg.norm(Ahat(X0) - b_hat), " (Slater)")
    print("<C,X_star>         =", inner_real(C, X_star))
    lmin, lmax = lam_minmax(X_star)
    print("eig(X_star) min/max:", lmin, lmax)
    lminC, lmaxC = lam_minmax(C)
    print("eig(C)      min/max:", lminC, lmaxC)
    print("||X_star||_F =", normF(X_star), " ||C||_F =", normF(C))

    mu = float(args.mu0)
    stage = 0
    X = X0

    mus, objs, rels, lmins, lmaxs = [], [], [], [], []

    print("\n=== Stage runs (fixed-\mu solves) ===")
    while True:
        res = solve_logdet_barrier_newton(
            H_list, b_hat, C,
            X0=X,
            y0=None,
            mu=mu,
            eps=args.tol,
            max_iter=args.newton,
            verbose=False,
            ops=ops,
            assume_hat=True,
        )
        X = res["X"]

        rp = float(np.linalg.norm(b_hat - Ahat(X)))
        obj = float(inner_real(C, X))
        rel = normF(X - X_star) / max(1.0, normF(X_star))
        lmin, lmax = lam_minmax(X)

        print(f"[stage {stage:02d}] mu={mu:.2e} rp_hat={rp:.2e} obj={obj:.3e} "
              f"relerr={rel:.3e} lam_min={lmin:.3e} lam_max={lmax:.3e}")

        mus.append(mu); objs.append(obj); rels.append(rel); lmins.append(lmin); lmaxs.append(lmax)

        if mu <= args.mu_min * (1.0 + 1e-15):
            break
        mu = max(args.beta * mu, args.mu_min)
        stage += 1

    out_dir = os.path.join(PROJECT_ROOT, "validation_output", "optiQ", "known_optimum_trace")
    os.makedirs(out_dir, exist_ok=True)

    mus = np.array(mus, dtype=float)

    # --- curve plots ---
    plt.figure()
    plt.loglog(mus, np.maximum(objs, 1e-300), marker="o")
    plt.gca().invert_xaxis()
    plt.xlabel("mu")
    plt.ylabel("<C, X(mu)>")
    plt.grid(True, which="both", ls=":")
    plt.title("Objective vs mu (trace-bounded known optimum)")
    plt.tight_layout()
    save_png_and_pdf(plt.gcf(), os.path.join(out_dir, "obj_vs_mu.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.loglog(mus, np.maximum(rels, 1e-300), marker="o")
    plt.gca().invert_xaxis()
    plt.xlabel("mu")
    plt.ylabel("||X(mu)-X_star||_F / ||X_star||_F")
    plt.grid(True, which="both", ls=":")
    plt.title("Distance to X_star vs mu")
    plt.tight_layout()
    save_png_and_pdf(plt.gcf(), os.path.join(out_dir, "relerr_vs_mu.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.loglog(mus, np.maximum(lmins, 1e-300), marker="o", label="lam_min")
    plt.gca().invert_xaxis()
    plt.loglog(mus, np.maximum(lmaxs, 1e-300), marker="o", label="lam_max")
    plt.xlabel("mu")
    plt.ylabel("eigenvalues")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.title("lam_min/lam_max vs mu")
    plt.tight_layout()
    save_png_and_pdf(plt.gcf(), os.path.join(out_dir, "eigs_vs_mu.png"), dpi=200)
    plt.close()

    # --- matrix visualizations at final stage ---
    mu_final = float(mus[-1])
    label_right = f"X(mu={mu_final:.1e})"

    save_quaternion_block_comparison(
        X_star, X,
        block=int(args.block),
        labels=("X_star", label_right),
        title=f"Top-left {args.block}×{args.block} block: X_star vs X(mu_final)",
        save_path=os.path.join(out_dir, f"block_{args.block}x{args.block}_components.png"),
        annotate=True,
    )
    save_quaternion_block_comparison(
        X_star, X,
        block=int(args.block),
        labels=("X_star", label_right),
        title=f"Top-left {args.block}×{args.block} block: X_star vs X(mu_final)",
        save_path=os.path.join(out_dir, f"block_{args.block}x{args.block}_components.pdf"),
        annotate=True,
    )

    save_quaternion_abs_comparison(
        X_star, X,
        labels=("X_star", label_right),
        title="Magnitude heatmaps (full matrix)",
        save_path=os.path.join(out_dir, "abs_heatmaps_full.png"),
    )
    save_quaternion_abs_comparison(
        X_star, X,
        labels=("X_star", label_right),
        title="Magnitude heatmaps (full matrix)",
        save_path=os.path.join(out_dir, "abs_heatmaps_full.pdf"),
    )

    print("Saved plots under:", out_dir)


if __name__ == "__main__":
    main()
