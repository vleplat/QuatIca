#!/usr/bin/env python3
"""
Quaternion Schur decomposition demo (stable variants only).

This demo focuses on the Schur implementations that are currently validated by unit
tests in this repository:

- `variant="rayleigh"`: pure quaternion QR iteration with a real Rayleigh shift
- `variant="implicit"`: pure quaternion implicit bulge-chase with real Rayleigh shift

It avoids experimental variants (e.g. AED / double-shift surrogates) so that the demo
is a reliable reference for users and for release notes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import quaternion  # type: ignore


def _find_project_root(start: str) -> str:
    cur = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(cur, "quatica")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(start)
        cur = parent


# Ensure `import quatica` works when running this script directly.
PROJECT_ROOT = _find_project_root(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from quatica.decomp.schur import quaternion_schur_unified
from quatica.utils import quat_eye, quat_frobenius_norm, quat_hermitian, quat_matmat


def _quat_abs_matrix(A: np.ndarray) -> np.ndarray:
    Af = quaternion.as_float_array(A)
    return np.sqrt(np.sum(Af * Af, axis=-1))


def _random_complex_unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    X = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Qc, _ = np.linalg.qr(X)
    return Qc


def _complex_to_quat_xsubfield(C: np.ndarray) -> np.ndarray:
    m, n = C.shape
    Q = np.empty((m, n), dtype=np.quaternion)
    for i in range(m):
        for j in range(n):
            Q[i, j] = quaternion.quaternion(float(np.real(C[i, j])), float(np.imag(C[i, j])), 0.0, 0.0)
    return Q


def _build_diagonal_complex_quat(vals: np.ndarray) -> np.ndarray:
    n = vals.shape[0]
    S = np.zeros((n, n), dtype=np.quaternion)
    for i, lam in enumerate(vals):
        S[i, i] = quaternion.quaternion(float(np.real(lam)), float(np.imag(lam)), 0.0, 0.0)
    return S


def _save_abs_heatmap(M: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.6), constrained_layout=True)
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, label="|·|")
    ax.set_title(title)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_demo(n: int, variant: str, seed: int, outdir: Path) -> None:
    rng = np.random.default_rng(seed)
    outdir.mkdir(parents=True, exist_ok=True)

    print("### Quaternion Schur demo")
    print(f"- size: n={n}")
    print(f"- variant: {variant!r} (stable)")
    print(f"- seed: {seed}")
    print(f"- outputs: {outdir}")
    print()
    print("Available Schur APIs (library):")
    print("- `quaternion_schur(A, ...)`: real-embedded implicit single-shift QR (legacy/real-embedded)")
    print("- `quaternion_schur_unified(A, variant=...)`: quaternion-only variants")
    print("  - stable: 'rayleigh', 'implicit'")
    print("  - advanced/experimental: 'aed', 'ds', 'experimental' (not shown here)")
    print()

    # -------------------------------------------------------------------------
    # Case 1: Hermitian positive definite (should yield nearly diagonal T)
    # -------------------------------------------------------------------------
    print("## Case 1: Hermitian HPD matrix")
    B = quaternion.as_quat_array(rng.standard_normal((n, n, 4)))
    A1 = quat_matmat(quat_hermitian(B), B) + 1.0 * quat_eye(n)

    Q1, T1, d1 = quaternion_schur_unified(A1, variant=variant, max_iter=3000, tol=1e-10, return_diagnostics=True)
    QT1 = quat_hermitian(Q1)
    sim1 = float(quat_frobenius_norm(quat_matmat(QT1, quat_matmat(A1, Q1)) - T1) / (quat_frobenius_norm(A1) + 1e-30))
    unit1 = float(quat_frobenius_norm(quat_matmat(QT1, Q1) - quat_eye(n)))

    print(f"- converged: {d1.get('converged', False)}  iterations: {d1.get('iterations_run', None)}")
    print(f"- similarity residual ||Q^H A Q - T||/||A||: {sim1:.2e}")
    print(f"- unitarity residual ||Q^H Q - I||:         {unit1:.2e}")

    _save_abs_heatmap(_quat_abs_matrix(T1), f"|T| (Hermitian, variant={variant})", outdir / "T_abs_case1_hermitian.png")
    _save_abs_heatmap(_quat_abs_matrix(A1 - quat_matmat(quat_matmat(Q1, T1), QT1)), f"|A - Q T Q^H| (Hermitian)", outdir / "recon_abs_case1_hermitian.png")
    print(f"- saved: {outdir / 'T_abs_case1_hermitian.png'}")
    print(f"- saved: {outdir / 'recon_abs_case1_hermitian.png'}")
    print()

    # -------------------------------------------------------------------------
    # Case 2: Synthetic diagonalizable in x-axis complex subfield
    # -------------------------------------------------------------------------
    print("## Case 2: Synthetic diagonalizable (x-axis complex subfield)")
    Uc = _random_complex_unitary(n, rng)
    P = _complex_to_quat_xsubfield(Uc)
    vals = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    S = _build_diagonal_complex_quat(vals)
    A2 = quat_matmat(quat_matmat(P, S), quat_hermitian(P))

    Q2, T2, d2 = quaternion_schur_unified(A2, variant=variant, max_iter=6000, tol=1e-10, return_diagnostics=True)
    QT2 = quat_hermitian(Q2)
    sim2 = float(quat_frobenius_norm(quat_matmat(QT2, quat_matmat(A2, Q2)) - T2) / (quat_frobenius_norm(A2) + 1e-30))
    unit2 = float(quat_frobenius_norm(quat_matmat(QT2, Q2) - quat_eye(n)))

    print(f"- converged: {d2.get('converged', False)}  iterations: {d2.get('iterations_run', None)}")
    print(f"- similarity residual ||Q^H A Q - T||/||A||: {sim2:.2e}")
    print(f"- unitarity residual ||Q^H Q - I||:         {unit2:.2e}")

    _save_abs_heatmap(_quat_abs_matrix(T2), f"|T| (synthetic diag, variant={variant})", outdir / "T_abs_case2_synthetic.png")
    _save_abs_heatmap(_quat_abs_matrix(A2 - quat_matmat(quat_matmat(Q2, T2), QT2)), f"|A - Q T Q^H| (synthetic diag)", outdir / "recon_abs_case2_synthetic.png")
    print(f"- saved: {outdir / 'T_abs_case2_synthetic.png'}")
    print(f"- saved: {outdir / 'recon_abs_case2_synthetic.png'}")
    print()

    print("Done.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=10, help="Matrix size (default: 10)")
    ap.add_argument("--variant", type=str, default="rayleigh", choices=["rayleigh", "implicit"], help="Stable Schur variant to demo.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="validation_output/schur_demo", help="Where to save figures.")
    ap.add_argument("--no-display", action="store_true", help="Use a non-interactive backend (safe in CI/headless).")
    args = ap.parse_args()

    if args.no_display:
        matplotlib.use("Agg")

    run_demo(n=int(args.n), variant=str(args.variant), seed=int(args.seed), outdir=Path(args.outdir))


if __name__ == "__main__":
    main()
