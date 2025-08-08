#!/usr/bin/env python3
"""
Analyze Schur QR diagnostics: run quaternion_schur with diagnostics, save JSON, and plot
subdiagonal max per iteration and active window size with deflation markers.

Usage:
  python tests/validation/analyze_schur_diagnostics.py --sizes 12 16 20 --seeds 1 7 --max_iter 6000 --tol 1e-12
"""
import os
import json
import argparse
import numpy as np
import quaternion  # type: ignore
import matplotlib.pyplot as plt

from core.decomp.schur import quaternion_schur


def run_case(n: int, seed: int, max_iter: int, tol: float, shift: str = "wilkinson"):
    rng = np.random.default_rng(seed)
    A = quaternion.as_quat_array(rng.standard_normal((n, n, 4)))
    Q, T, diag = quaternion_schur(
        A, max_iter=max_iter, tol=tol, shift=shift, verbose=False, return_diagnostics=True
    )
    return Q, T, diag


def extract_series(diag: dict):
    iters = [entry["iter"] for entry in diag["iterations"]]
    m_active = [entry["m_active"] for entry in diag["iterations"]]
    subdiag_max = [entry["subdiag_max"] for entry in diag["iterations"]]
    deflated_any = [len(entry.get("deflated_indices", [])) > 0 for entry in diag["iterations"]]
    return iters, m_active, subdiag_max, deflated_any


def plot_diagnostics(n: int, seed: int, diag: dict, out_dir: str):
    iters, m_active, subdiag_max, deflated_any = extract_series(diag)
    if not iters:
        return None
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(iters, subdiag_max, label="max subdiag")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("max |subdiag|")
    ax[0].grid(True, which="both", ls=":")
    # Mark deflation events
    for it, df in zip(iters, deflated_any):
        if df:
            ax[0].axvline(it, color="red", alpha=0.2)

    ax[1].plot(iters, m_active, label="active size", color="tab:green")
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("m_active")
    ax[1].grid(True, ls=":")

    fig.suptitle(f"Schur diagnostics n={n}, seed={seed}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"schur_diag_n{n}_seed{seed}.png")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[12, 16, 20])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 7])
    parser.add_argument("--max_iter", type=int, default=6000)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--shift", type=str, default="wilkinson", choices=["wilkinson", "rayleigh", "double"])
    parser.add_argument("--out_dir", type=str, default="validation_output")
    args = parser.parse_args()

    for n in args.sizes:
        for seed in args.seeds:
            print(f"Running n={n}, seed={seed}...")
            Q, T, diag = run_case(n, seed, args.max_iter, args.tol, args.shift)
            # Save JSON
            os.makedirs(args.out_dir, exist_ok=True)
            json_path = os.path.join(args.out_dir, f"schur_diag_n{n}_seed{seed}.json")
            with open(json_path, "w") as f:
                json.dump(diag, f, indent=2)
            # Plot
            plot_path = plot_diagnostics(n, seed, diag, args.out_dir)
            print(f"  Saved JSON: {json_path}")
            if plot_path:
                print(f"  Saved plot: {plot_path}")


if __name__ == "__main__":
    main()


