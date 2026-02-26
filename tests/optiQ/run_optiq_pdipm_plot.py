#!/usr/bin/env python3
"""
Run OptiQ's fixed-\mu log-det barrier Newton solver on a central-point instance,
capture history, and save a convergence plot.

Usage (from project root):
  python tests/run_optiq_pdipm_plot.py

Output:
  validation_output/optiQ/logdet_newton_convergence.png
"""
import os
import sys

# Ensure we import the local repo version (not an installed site-packages one)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from quatica import build_central_mu_instance  # noqa: E402
from quatica.optiQ import solve_logdet_barrier_newton, save_pdipm_history_plot  # noqa: E402

OUT_DIR = os.path.join(_PROJECT_ROOT, "validation_output", "optiQ")
OUT_PATH = os.path.join(OUT_DIR, "logdet_newton_convergence.png")


def main():
    print("Using solve_logdet_barrier_newton from:", solve_logdet_barrier_newton.__code__.co_filename)

    print("Building central-\mu instance (n=100, m=100, basis=random)...")
    H_list, b, C, X_star, mu = build_central_mu_instance(
        n=100, m=100, mu=1.0, seed=0, basis="random"
    )

    print("Running fixed-\mu log-det barrier Newton (verbose=False)...")
    res = solve_logdet_barrier_newton(
        H_list, b, C,
        mu=mu,
        eps=1e-8,
        max_iter=80,
        verbose=False,
    )

    history = res.get("history", [])
    if not history:
        print("No history entries; skipping plot.")
        return

    h0 = history[0]
    print("History keys example:", sorted(h0.keys()))
    if "mu" in h0:
        mus = [h.get("mu") for h in history if isinstance(h.get("mu"), (int, float))]
        if mus:
            print(f"mu in history: first={mus[0]:.3e} last={mus[-1]:.3e}")
    elif "gap" in h0:
        gaps = [h.get("gap") for h in history if isinstance(h.get("gap"), (int, float))]
        if gaps:
            print(f"gap (often == mu here): first={gaps[0]:.3e} last={gaps[-1]:.3e}")

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Saving convergence plot to {OUT_PATH} ...")
    save_pdipm_history_plot(history, OUT_PATH)
    print(f"Done. Open {OUT_PATH} to view ||r_p||, ||r_d||, gap/n, and step t.")


if __name__ == "__main__":
    main()
