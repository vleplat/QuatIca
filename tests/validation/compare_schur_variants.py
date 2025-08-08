"""
Compare quaternion Schur variants using the unified API with plots.

Variants compared:
 - "rayleigh"     : Pure QR with Rayleigh shift (stable, slower)
 - "implicit+AED" : Quaternion implicit QR with aggressive early deflation (fast)

Outputs:
 - Convergence plots and Schur T real-component visualizations in validation_output/.

Usage (after venv activation):
  python tests/validation/compare_schur_variants.py --sizes 50 --iters 1500 --tol 1e-10 --tag rand
  python tests/validation/compare_schur_variants.py --sizes 50 --iters 1000 --tol 1e-10 --hermitian --tag herm
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import List

import numpy as np
import matplotlib

# Non-interactive backend for saving figures
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import quaternion  # type: ignore


# Robust sys.path setup when run directly
import sys
root = os.getcwd()
if root not in sys.path:
    sys.path.insert(0, root)
core_path = os.path.join(root, "core")
if core_path not in sys.path:
    sys.path.insert(0, core_path)

from core.decomp.schur import quaternion_schur_unified
from core.data_gen import create_test_matrix
from core.utils import (
    quat_matmat,
    quat_hermitian,
    quat_frobenius_norm,
    quat_eye,
)


def _extract_curve(diag: dict) -> List[float]:
    return [d.get("max_subdiag", float("nan")) for d in diag.get("iterations", [])]


## Note: all algorithmic variants are provided by core.decomp.schur (unified API).


def run_variants(A: np.ndarray, n: int, iters: int, tol: float, out_dir: Path, tag: str = "") -> None:
    # Variants to compare (unified API)
    variants = [("rayleigh", "rayleigh"), ("implicit+AED", "aed")]

    curves = {}
    Ts = {}
    for name, vkey in variants:
        t0 = os.times()[4]
        Q, T, diag = quaternion_schur_unified(A, variant=vkey, max_iter=iters, tol=tol, return_diagnostics=True)
        t1 = os.times()[4]
        cpu = t1 - t0
        curve = _extract_curve(diag)
        sim = quat_frobenius_norm(quat_matmat(quat_hermitian(Q), quat_matmat(A, Q)) - T)
        unit = quat_frobenius_norm(quat_matmat(quat_hermitian(Q), Q) - np.eye(n, dtype=np.quaternion))
        final = curve[-1] if curve else float("nan")
        curves[name] = (curve, cpu, sim, unit, final)
        Ts[name] = T
        print(
            f"n={n:3d} | {name:13s} | ran={len(curve):4d}/{iters} | cpu={cpu:6.2f}s | final_sub={final:.3e} | sim={sim:.3e} | unit={unit:.3e}"
        )

    # Plot
    out_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(7, 4))
    for name, (curve, *_rest) in curves.items():
        if curve:
            plt.semilogy(curve, label=name)
    plt.title(f"Quaternion Schur (Lead 2) variants — n={n}")
    plt.xlabel("iteration")
    plt.ylabel("max |subdiag|")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    suffix = f"_{tag}" if tag else ""
    fname = out_dir / f"schur_lead2_variants{suffix}_n{n}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    print(f"saved plot: {fname}")

    # Visualize Schur matrices (real component) for each variant
    import quaternion as _q
    for name, T in Ts.items():
        Tarr = _q.as_float_array(T)[:, :, 0]
        plt.figure(figsize=(5, 4))
        im = plt.imshow(Tarr, cmap='viridis', aspect='auto')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(f"Schur T (real) — {name}, n={n}")
        plt.xlabel("column"); plt.ylabel("row")
        tname = out_dir / f"schur_T{suffix}_n{n}_{name}_real.png"
        plt.tight_layout(); plt.savefig(tname, dpi=300); plt.close()
        print(f"saved T visualization: {tname}")


def main():
    parser = argparse.ArgumentParser(description="Compare quaternion Schur variants")
    parser.add_argument("--sizes", nargs="*", type=int, default=[10, 20, 30, 50])
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--hermitian", action="store_true", help="Use A = B^H @ B (Hermitian)")
    parser.add_argument("--tag", type=str, default="", help="Optional tag to append to output filenames")
    # keep script simple: no balancing knobs
    args = parser.parse_args()

    out_dir = Path("validation_output")
    for n in args.sizes:
        if args.hermitian:
            B = create_test_matrix(n, n)
            A = quat_matmat(quat_hermitian(B), B)
            tag = args.tag or "herm"
        else:
            A = create_test_matrix(n, n)
            tag = args.tag or "rand"
        run_variants(A, n=n, iters=args.iters, tol=args.tol, out_dir=out_dir, tag=tag)
    print("Done.")


if __name__ == "__main__":
    main()


