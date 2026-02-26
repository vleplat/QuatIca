#!/usr/bin/env python3
"""Known-optimum (trace-bounded) OptiQ regression run + visual diagnostics + Schur-solver benchmark.

This script constructs an SDP with a certified primal optimum X_star and runs the
fixed-μ log-det barrier Newton solver over a decreasing μ schedule (warm-starting each stage).

It also benchmarks alternative Schur complement solvers:
  - dense       : explicit Schur assembly + dense solve (baseline)
  - cg          : matrix-free Schur matvec + PCG in R^m (no preconditioner)
  - cg_nystrom  : PCG with Nyström sketch-based preconditioner (recommended for CG experiments)
  - cg_diag     : PCG with simple diagonal preconditioner (optional)

Saved outputs (under validation_output/optiQ/known_optimum_trace/):
  - obj_vs_mu.{png,pdf}            : <C, X(μ)> vs μ
  - relerr_vs_mu.{png,pdf}         : ||X(μ)-X_star||_F / ||X_star||_F vs μ
  - eigs_vs_mu.{png,pdf}           : λ_min(X(μ)), λ_max(X(μ)) vs μ
  - runtime_vs_mu.{png,pdf}        : stage runtime vs μ for each Schur solver
  - runtime_totals.{png,pdf}       : total runtime per Schur solver (bar chart)
  - abs_heatmaps_full.{png,pdf}    : |X_star|, |X(μ_min)|, |X_star - X(μ_min)|
  - block_kxk_components.{png,pdf} : component-wise k×k blocks (Re,i,j,k)
  - benchmark_summary.txt          : textual summary

Usage (from project root):
  python tests/optiQ/run_optiq_known_optimum_benchmark.py --benchmark
"""

import os
import sys
import argparse
import time
import numpy as np
import quaternion as nq  # type: ignore
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

# Ensure the script directory is importable (in case local helpers exist)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from quatica.optiQ import (  # noqa: E402
    solve_logdet_barrier_newton,
    qzeros, qeye, qmm, qadj, qherm,
    inner_real, eigvalsH, eighH,
    random_hermitian,
    _build_orthonormal_ops,
)

# Prefer the module inside quatica/, but allow local import for convenience.
try:  # noqa: E402
    from quatica.optiq_visualization import (
        save_quaternion_block_comparison,
        save_quaternion_abs_comparison,
    )
except Exception:  # pragma: no cover
    from optiq_visualization import (  # type: ignore
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
    _ = inner_real(H0, X_star)   # b0 (not used explicitly; we work in hat space)
    H_raw = [H0]

    # (2) m_extra random trace-free constraints (optional)
    ii = inner_real(I, I)
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


def _pretty_tag(schur_solver: str, schur_precond: str) -> str:
    if schur_solver == "dense":
        return "dense"
    if schur_solver == "cg" and schur_precond == "none":
        return "cg"
    if schur_solver == "cg" and schur_precond == "diag":
        return "cg_diag"
    if schur_solver == "cg" and schur_precond == "nystrom":
        return "cg_nystrom"
    return f"{schur_solver}_{schur_precond}"


def _run_mu_schedule(
    H_list, ops, b_hat, C, X_star, X0,
    mu0, mu_min, beta, newton_max, tol,
    *,
    schur_solver="dense",
    schur_precond="none",
    schur_precond_rank=None,
    schur_precond_ridge_scale=1e-6,
    schur_precond_seed=0,
    cg_tol=1e-10,
    cg_maxit=1000,
):
    """Run fixed-μ solves over a decreasing μ schedule; return metrics and timings."""
    Ahat = ops["A_hat"]
    mu = float(mu0)
    stage = 0
    X = X0

    mus = []
    objs = []
    rels = []
    lmins = []
    lmaxs = []
    rp_hats = []
    stage_times = []
    cg_iters_last = []
    cg_res_last = []

    tag = _pretty_tag(schur_solver, schur_precond)

    while True:
        t0 = time.perf_counter()
        res = solve_logdet_barrier_newton(
            H_list, b_hat, C,
            X0=X,
            y0=None,
            mu=mu,
            eps=tol,
            max_iter=newton_max,
            verbose=False,
            ops=ops,
            assume_hat=True,
            schur_solver=schur_solver,
            schur_precond=schur_precond,
            schur_precond_rank=schur_precond_rank,
            schur_precond_ridge_scale=schur_precond_ridge_scale,
            schur_precond_seed=schur_precond_seed,
            cg_tol=cg_tol,
            cg_maxit=cg_maxit,
        )
        t1 = time.perf_counter()
        X = res["X"]

        # Stage metrics
        rp = float(np.linalg.norm(b_hat - Ahat(X)))
        obj = float(inner_real(C, X))
        rel = normF(X - X_star) / max(1.0, normF(X_star))
        lmin, lmax = lam_minmax(X)

        # Pull CG diagnostics (from last Newton iteration recorded)
        hist = res.get("history", [])
        if len(hist) > 0:
            cg_it = int(hist[-1].get("cg_iters", 0))
            cg_rs = float(hist[-1].get("cg_res", 0.0))
        else:
            cg_it, cg_rs = 0, 0.0

        mus.append(mu)
        objs.append(obj)
        rels.append(rel)
        lmins.append(lmin)
        lmaxs.append(lmax)
        rp_hats.append(rp)
        stage_times.append(float(t1 - t0))
        cg_iters_last.append(cg_it)
        cg_res_last.append(cg_rs)

        print(
            f"[{tag}] [stage {stage:02d}] mu={mu:.2e} time={stage_times[-1]:.3f}s "
            f"rp_hat={rp:.2e} obj={obj:.3e} relerr={rel:.3e} "
            f"lam_min={lmin:.3e} lam_max={lmax:.3e} "
            f"cg_it={cg_it} cg_res={cg_rs:.1e}"
        )

        if mu <= mu_min * (1.0 + 1e-15):
            break
        mu = max(beta * mu, mu_min)
        stage += 1

    out = {
        "tag": tag,
        "X_final": X,
        "mus": np.array(mus, dtype=float),
        "objs": np.array(objs, dtype=float),
        "rels": np.array(rels, dtype=float),
        "lmins": np.array(lmins, dtype=float),
        "lmaxs": np.array(lmaxs, dtype=float),
        "rp_hats": np.array(rp_hats, dtype=float),
        "stage_times": np.array(stage_times, dtype=float),
        "cg_iters_last": np.array(cg_iters_last, dtype=int),
        "cg_res_last": np.array(cg_res_last, dtype=float),
        "total_time": float(np.sum(stage_times)),
    }
    return out


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
    ap.add_argument("--benchmark", action="store_true", help="run Schur-solver benchmark (dense vs CG variants)")

    # CG knobs
    ap.add_argument("--cg_tol", type=float, default=1e-10)
    ap.add_argument("--cg_maxit", type=int, default=1000)

    # Nyström preconditioner knobs
    ap.add_argument("--nystrom_rank", type=int, default=20)
    ap.add_argument("--nystrom_ridge", type=float, default=1e-6)
    ap.add_argument("--nystrom_seed", type=int, default=0)

    # Optional include diag CG (mostly for sanity)
    ap.add_argument("--include_diag", action="store_true", help="also benchmark cg_diag")

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

    out_dir = os.path.join(PROJECT_ROOT, "validation_output", "optiQ", "known_optimum_trace")
    os.makedirs(out_dir, exist_ok=True)

    print("\n=== Stage runs (fixed-μ solves) ===")

    configs = [("dense", dict(schur_solver="dense", schur_precond="none"))]
    if args.benchmark:
        configs += [
            ("cg", dict(schur_solver="cg", schur_precond="none")),
            ("cg_nystrom", dict(
                schur_solver="cg",
                schur_precond="nystrom",
                schur_precond_rank=int(args.nystrom_rank),
                schur_precond_ridge_scale=float(args.nystrom_ridge),
                schur_precond_seed=int(args.nystrom_seed),
            )),
        ]
        if args.include_diag:
            configs += [("cg_diag", dict(schur_solver="cg", schur_precond="diag"))]

    results = {}
    for name, kwargs in configs:
        print(f"\n--- Running config: {name} ---")
        results[name] = _run_mu_schedule(
            H_list, ops, b_hat, C, X_star, X0,
            args.mu0, args.mu_min, args.beta, args.newton, args.tol,
            cg_tol=args.cg_tol,
            cg_maxit=args.cg_maxit,
            **kwargs,
        )

    # Use dense run as reference for figures
    ref = results["dense"]
    mus = ref["mus"]

    # --- curve plots (reference run) ---
    plt.figure()
    plt.loglog(mus, np.maximum(ref["objs"], 1e-300), marker="o")
    plt.gca().invert_xaxis()
    plt.xlabel("mu")
    plt.ylabel("<C, X(mu)>")
    plt.grid(True, which="both", ls=":")
    plt.title("Objective vs mu (trace-bounded known optimum)")
    plt.tight_layout()
    save_png_and_pdf(plt.gcf(), os.path.join(out_dir, "obj_vs_mu.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.loglog(mus, np.maximum(ref["rels"], 1e-300), marker="o")
    plt.gca().invert_xaxis()
    plt.xlabel("mu")
    plt.ylabel("||X(mu)-X_star||_F / ||X_star||_F")
    plt.grid(True, which="both", ls=":")
    plt.title("Distance to X_star vs mu")
    plt.tight_layout()
    save_png_and_pdf(plt.gcf(), os.path.join(out_dir, "relerr_vs_mu.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.loglog(mus, np.maximum(ref["lmins"], 1e-300), marker="o", label="lam_min")
    plt.gca().invert_xaxis()
    plt.loglog(mus, np.maximum(ref["lmaxs"], 1e-300), marker="o", label="lam_max")
    plt.xlabel("mu")
    plt.ylabel("eigenvalues")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.title("lam_min/lam_max vs mu")
    plt.tight_layout()
    save_png_and_pdf(plt.gcf(), os.path.join(out_dir, "eigs_vs_mu.png"), dpi=200)
    plt.close()

    # --- benchmark plots ---
    if args.benchmark:
        plt.figure()
        for name, rr in results.items():
            plt.loglog(rr["mus"], np.maximum(rr["stage_times"], 1e-12), marker="o", label=rr["tag"])
        plt.gca().invert_xaxis()
        plt.xlabel("mu")
        plt.ylabel("stage runtime (s)")
        plt.grid(True, which="both", ls=":")
        plt.legend()
        plt.title("OptiQ stage runtime vs mu (Schur solver benchmark)")
        plt.tight_layout()
        save_png_and_pdf(plt.gcf(), os.path.join(out_dir, "runtime_vs_mu.png"), dpi=200)
        plt.close()

        # Total runtime bar
        plt.figure()
        names = list(results.keys())
        totals = [results[k]["total_time"] for k in names]
        plt.bar(names, totals)
        plt.ylabel("total runtime (s)")
        plt.title("Total runtime (same μ schedule)")
        plt.tight_layout()
        save_png_and_pdf(plt.gcf(), os.path.join(out_dir, "runtime_totals.png"), dpi=200)
        plt.close()

        # Write a small text summary
        summary_path = os.path.join(out_dir, "benchmark_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("OptiQ Schur-solver benchmark summary\n")
            f.write(f"n={args.n}, m={1+args.m_extra}, rank={args.rank}, seed={args.seed}\n")
            f.write(f"mu0={args.mu0}, mu_min={args.mu_min}, beta={args.beta}, newton={args.newton}, tol={args.tol}\n")
            f.write(f"cg_tol={args.cg_tol}, cg_maxit={args.cg_maxit}\n")
            f.write(f"nystrom_rank={args.nystrom_rank}, nystrom_ridge={args.nystrom_ridge}, nystrom_seed={args.nystrom_seed}\n\n")
            for k in names:
                rr = results[k]
                f.write(f"[{rr['tag']}] total_time={rr['total_time']:.6f}s\n")
                f.write(f"  final_mu={rr['mus'][-1]:.3e}\n")
                f.write(f"  final_obj={rr['objs'][-1]:.6e}\n")
                f.write(f"  final_relerr={rr['rels'][-1]:.6e}\n")
                f.write(f"  final_rp_hat={rr['rp_hats'][-1]:.6e}\n")
                f.write(f"  final_cg_it={int(rr['cg_iters_last'][-1])}  final_cg_res={float(rr['cg_res_last'][-1]):.3e}\n\n")
        print("Wrote benchmark summary:", summary_path)

    # --- matrix visualizations at final stage (reference dense run) ---
    mu_final = float(ref["mus"][-1])
    X_final = ref["X_final"]
    label_right = f"X(mu={mu_final:.1e})"

    save_quaternion_block_comparison(
        X_star, X_final,
        block=int(args.block),
        labels=("X_star", label_right),
        title=f"Top-left {args.block}×{args.block} block: X_star vs X(mu_final)",
        save_path=os.path.join(out_dir, f"block_{args.block}x{args.block}_components.png"),
        annotate=True,
    )
    save_quaternion_block_comparison(
        X_star, X_final,
        block=int(args.block),
        labels=("X_star", label_right),
        title=f"Top-left {args.block}×{args.block} block: X_star vs X(mu_final)",
        save_path=os.path.join(out_dir, f"block_{args.block}x{args.block}_components.pdf"),
        annotate=True,
    )

    save_quaternion_abs_comparison(
        X_star, X_final,
        labels=("X_star", label_right),
        title="Magnitude heatmaps (full matrix)",
        save_path=os.path.join(out_dir, "abs_heatmaps_full.png"),
    )
    save_quaternion_abs_comparison(
        X_star, X_final,
        labels=("X_star", label_right),
        title="Magnitude heatmaps (full matrix)",
        save_path=os.path.join(out_dir, "abs_heatmaps_full.pdf"),
    )

    print("Saved plots under:", out_dir)


if __name__ == "__main__":
    main()
