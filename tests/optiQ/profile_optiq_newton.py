#!/usr/bin/env python3
"""
Profile OptiQ (fixed-μ log-det barrier Newton) on the certified known-optimum instance.

Goal
----
For a single μ (or a short μ list), run solve_logdet_barrier_newton and record wall-clock time
spent in key OptiQ routines (and optionally full cProfile stats).

This is meant to answer: "Where is the time going?" before further optimizations.

Usage (from repo root)
----------------------
# 1) Lightweight manual timers (recommended first)
python tests/optiQ/profile_optiq_newton.py --mode timers --mu 1e-6 --n 40 --m_extra 39 --newton 40

# 2) Full Python-level profiling (cProfile)
python tests/optiQ/profile_optiq_newton.py --mode cprofile --mu 1e-6 --n 40 --m_extra 39 --newton 40

# 3) Both
python tests/optiQ/profile_optiq_newton.py --mode both --mu 1e-6 --n 40 --m_extra 39 --newton 40

Outputs are saved under:
  validation_output/optiQ/profile/
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import cProfile
import pstats
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, Tuple

import numpy as np
import quaternion as nq  # type: ignore


# ----------------------------
# Locate repo root
# ----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _find_project_root(start_dir: str) -> str:
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


# ----------------------------
# Imports from QuatIca
# ----------------------------
import quatica.optiQ as optiq  # noqa: E402


# ----------------------------
# Timing utilities
# ----------------------------
@dataclass
class _Stat:
    total: float = 0.0
    calls: int = 0

class TimerRegistry:
    def __init__(self) -> None:
        self.stats: Dict[str, _Stat] = {}

    def wrap(self, label: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        stats = self.stats.setdefault(label, _Stat())

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                stats.total += dt
                stats.calls += 1

        wrapped.__name__ = getattr(fn, "__name__", "wrapped")
        wrapped.__doc__ = getattr(fn, "__doc__", None)
        return wrapped

    def report(self, *, top: int = 25) -> str:
        rows = []
        for k, s in self.stats.items():
            if s.calls <= 0:
                continue
            rows.append((s.total, k, s.calls, s.total / s.calls))
        rows.sort(reverse=True, key=lambda x: x[0])

        out = []
        out.append(f"{'label':40s}  {'total(s)':>10s}  {'calls':>8s}  {'avg(ms)':>10s}")
        out.append("-" * 76)
        for total, k, calls, avg in rows[:top]:
            out.append(f"{k:40s}  {total:10.4f}  {calls:8d}  {1000*avg:10.3f}")
        if len(rows) > top:
            out.append(f"... ({len(rows)-top} more)")
        return "\n".join(out)


# ----------------------------
# Known-optimum instance builder (same construction as your demo)
# ----------------------------
def _diag_real_quat(d: np.ndarray) -> np.ndarray:
    n = int(d.size)
    D = optiq.qzeros(n, n)
    for i in range(n):
        D[i, i] = nq.quaternion(float(d[i]), 0.0, 0.0, 0.0)
    return D

def build_known_optimum_instance(
    n: int,
    m_extra: int,
    rank: int,
    seed: int,
    eps_slater: float = 1e-2,
) -> Tuple[list[np.ndarray], dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    Hr = optiq.random_hermitian(n, seed=int(rng.integers(1 << 31)))
    _, V = optiq.eighH(Hr)

    dx = np.zeros(n); dx[:rank] = 1.0
    ds = np.zeros(n); ds[rank:] = 1.0
    X_star = optiq.qherm(optiq.qmm(optiq.qmm(V, _diag_real_quat(dx)), optiq.qadj(V)))
    C = optiq.qherm(optiq.qmm(optiq.qmm(V, _diag_real_quat(ds)), optiq.qadj(V)))

    I = optiq.qeye(n)
    H0 = (1.0 / np.sqrt(n)) * I
    H_raw = [H0]

    ii = float(optiq.inner_real(I, I))
    for _ in range(int(m_extra)):
        H = optiq.random_hermitian(n, seed=int(rng.integers(1 << 31)))
        alpha = float(optiq.inner_real(H, I)) / ii
        H = optiq.qherm(H - float(alpha) * I)
        H_raw.append(H)

    ops = optiq._build_orthonormal_ops([optiq.qherm(H) for H in H_raw])
    Ahat = ops["A_hat"]
    b_hat = Ahat(X_star)

    X0 = optiq.qherm((1.0 - eps_slater) * X_star + eps_slater * (rank / n) * I)
    return H_raw, ops, b_hat, C, X_star, X0


# ----------------------------
# Profiling runner
# ----------------------------
def run_once(
    *,
    n: int,
    m_extra: int,
    rank: int,
    seed: int,
    eps_slater: float,
    mu: float,
    newton_max: int,
    tol: float,
    schur_solver: str,
    schur_precond: str,
    cg_tol: float,
    cg_maxit: int,
    nystrom_rank: int,
    nystrom_ridge: float,
    nystrom_seed: int,
    mode: str,
    out_dir: str,
    top: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    H_list, ops, b_hat, C, X_star, X0 = build_known_optimum_instance(
        n=n, m_extra=m_extra, rank=rank, seed=seed, eps_slater=eps_slater
    )

    # ----------------------------
    # Manual timers via monkeypatching (module-local only)
    # ----------------------------
    timer = TimerRegistry()
    originals: Dict[str, Any] = {}

    def patch(name: str, obj: Any) -> None:
        originals[name] = getattr(optiq, name)
        setattr(optiq, name, obj)

    def patch_numpy_attr(attr: str, new_fn: Any) -> None:
        # patch optiq.np.linalg.* used inside optiq module only
        originals[f"np.linalg.{attr}"] = getattr(optiq.np.linalg, attr)
        setattr(optiq.np.linalg, attr, new_fn)

    if mode in ("timers", "both"):
        # Wrap heavy OptiQ-level functions (present in both old & new versions)
        for name in [
            "qmm", "qadj", "qherm", "inner_real",
            "eighH", "eigvalsH",
            "invH", "sqrtH", "logdetH",
            "_assemble_residuals",
            "newton_step",
            "_build_orthonormal_ops",
        ]:
            if hasattr(optiq, name):
                fn = getattr(optiq, name)
                if callable(fn):
                    patch(name, timer.wrap(f"optiq.{name}", fn))

        # If caching exists in your current OptiQ, time it explicitly
        if hasattr(optiq, "_PDCache"):
            try:
                PDCache = getattr(optiq, "_PDCache")
                if hasattr(PDCache, "from_X"):
                    PDCache_from_X = PDCache.from_X
                    # wrap the classmethod/staticmethod
                    def _wrapped_from_X(*args: Any, **kwargs: Any) -> Any:
                        return timer.wrap("optiq._PDCache.from_X", PDCache_from_X)(*args, **kwargs)
                    PDCache.from_X = _wrapped_from_X  # type: ignore[attr-defined]
            except Exception:
                pass

        # Wrap module-local numpy solves
        patch_numpy_attr("solve", timer.wrap("np.linalg.solve", optiq.np.linalg.solve))
        patch_numpy_attr("cholesky", timer.wrap("np.linalg.cholesky", optiq.np.linalg.cholesky))
        patch_numpy_attr("norm", timer.wrap("np.linalg.norm", optiq.np.linalg.norm))

        # Wrap hat-ops closures (constraint applications)
        for k in ["A_hat", "AT_hat", "transform_b"]:
            if k in ops and callable(ops[k]):
                ops[k] = timer.wrap(f"ops.{k}", ops[k])

    # ----------------------------
    # Run solver (optionally under cProfile)
    # ----------------------------
    def _call_solver() -> Dict[str, Any]:
        return optiq.solve_logdet_barrier_newton(
            H_list, b_hat, C,
            X0=X0,
            y0=None,
            mu=float(mu),
            eps=float(tol),
            max_iter=int(newton_max),
            verbose=False,
            ops=ops,
            assume_hat=True,
            schur_solver=schur_solver,
            schur_precond=schur_precond,
            schur_precond_rank=int(nystrom_rank),
            schur_precond_ridge_scale=float(nystrom_ridge),
            schur_precond_seed=int(nystrom_seed),
            cg_tol=float(cg_tol),
            cg_maxit=int(cg_maxit),
        )

    tag = f"n{n}_m{1+m_extra}_mu{mu:.0e}_{schur_solver}_{schur_precond}"
    prof_path = os.path.join(out_dir, f"cprofile_{tag}.prof")
    prof_txt = os.path.join(out_dir, f"cprofile_{tag}.txt")
    timers_txt = os.path.join(out_dir, f"timers_{tag}.txt")

    if mode in ("cprofile", "both"):
        pr = cProfile.Profile()
        pr.enable()
        res = _call_solver()
        pr.disable()
        pr.dump_stats(prof_path)

        with open(prof_txt, "w", encoding="utf-8") as f:
            ps = pstats.Stats(pr, stream=f).strip_dirs().sort_stats("cumtime")
            ps.print_stats(top)

        print(f"[cProfile] wrote: {prof_path}")
        print(f"[cProfile] wrote: {prof_txt}")
    else:
        res = _call_solver()

    # ----------------------------
    # Report timers
    # ----------------------------
    if mode in ("timers", "both"):
        rep = timer.report(top=top)
        with open(timers_txt, "w", encoding="utf-8") as f:
            f.write("OptiQ manual timers report\n")
            f.write(f"tag: {tag}\n")
            f.write(f"n={n}, m={1+m_extra}, rank={rank}, mu={mu}\n")
            f.write(f"solver: {schur_solver}, precond: {schur_precond}\n\n")
            f.write(rep + "\n")
        print(rep)
        print(f"[timers] wrote: {timers_txt}")

    # Quick sanity print
    Ahat = ops["A_hat"]
    rp = float(np.linalg.norm(b_hat - Ahat(res["X"])))
    obj = float(optiq.inner_real(C, res["X"]))
    print(f"Sanity: rp_hat={rp:.3e}  obj={obj:.6e}  iters={len(res.get('history', []))}")

    # Restore monkeypatches
    if mode in ("timers", "both"):
        for name, orig in originals.items():
            if name.startswith("np.linalg."):
                _, attr = name.split(".", 1)
                setattr(optiq.np.linalg, attr, orig)
            else:
                setattr(optiq, name, orig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--rank", type=int, default=20)
    ap.add_argument("--m_extra", type=int, default=39)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps_slater", type=float, default=1e-2)

    ap.add_argument("--mu", type=float, default=1e-6)
    ap.add_argument("--newton", type=int, default=40)
    ap.add_argument("--tol", type=float, default=1e-10)

    ap.add_argument("--schur_solver", type=str, default="dense", choices=["dense", "cg"])
    ap.add_argument("--schur_precond", type=str, default="none", choices=["none", "diag", "nystrom"])

    ap.add_argument("--cg_tol", type=float, default=1e-10)
    ap.add_argument("--cg_maxit", type=int, default=1000)

    ap.add_argument("--nystrom_rank", type=int, default=20)
    ap.add_argument("--nystrom_ridge", type=float, default=1e-6)
    ap.add_argument("--nystrom_seed", type=int, default=0)

    ap.add_argument("--mode", type=str, default="timers", choices=["timers", "cprofile", "both"])
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--outdir", type=str, default=os.path.join(PROJECT_ROOT, "validation_output", "optiQ", "profile"))

    args = ap.parse_args()

    run_once(
        n=args.n,
        m_extra=args.m_extra,
        rank=args.rank,
        seed=args.seed,
        eps_slater=args.eps_slater,
        mu=args.mu,
        newton_max=args.newton,
        tol=args.tol,
        schur_solver=args.schur_solver,
        schur_precond=args.schur_precond,
        cg_tol=args.cg_tol,
        cg_maxit=args.cg_maxit,
        nystrom_rank=args.nystrom_rank,
        nystrom_ridge=args.nystrom_ridge,
        nystrom_seed=args.nystrom_seed,
        mode=args.mode,
        out_dir=args.outdir,
        top=args.top,
    )


if __name__ == "__main__":
    main()
