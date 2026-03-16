#!/usr/bin/env python3
"""
Paper-ready benchmark: quaternion pseudoinverse via Newton–Schulz variants.

This script benchmarks and compares:
- NS (gamma=1): classical Newton–Schulz
- NS (gamma=1/2): damped Newton–Schulz
- Higher-Order NS (3rd): third-order Newton–Schulz variant

It produces an aggregated dashboard figure (PNG), individual PDF panels, and exports
raw benchmark data (CSV + NPZ) for reproducibility.

Run via:
  python run_analysis.py ns_pinv_bench [--no-display] [--out-dir DIR] ...
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Keep benchmark output clean for paper runs.
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _find_repo_root(start: str) -> str:
    here = os.path.abspath(start)
    while True:
        if os.path.exists(os.path.join(here, "pyproject.toml")) and os.path.isdir(
            os.path.join(here, "quatica")
        ):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        here = parent


def _ensure_quatica_importable() -> None:
    root = _find_repo_root(os.path.dirname(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_quatica_importable()

import matplotlib

import quaternion  # type: ignore

from quatica.solver import (
    HigherOrderNewtonSchulzPseudoinverse,
    NewtonSchulzPseudoinverse,
)
from quatica.utils import quat_eye, quat_frobenius_norm, quat_hermitian, quat_matmat


def _setup_matplotlib_style(*, headless: bool) -> None:
    if headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            # LaTeX-like typography without requiring a LaTeX install
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.titlesize": 16,
            # Paper-friendly exports
            "lines.linewidth": 2.2,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_axis_as_pdf(fig, ax, path_pdf: Path, *, pad: float = 0.02) -> None:
    path_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_tightbbox(renderer).expanded(1.0 + pad, 1.0 + pad)
    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(path_pdf, bbox_inches=bbox_inches, facecolor="white")


def _export_dashboard_and_panels(fig, axes: dict[str, Any], out_dir: Path, *, base: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.35)
    panels_dir = out_dir
    for key, ax in axes.items():
        _save_axis_as_pdf(fig, ax, panels_dir / f"{base}__{key}.pdf", pad=0.03)


def _parse_sizes(values: list[str] | None) -> list[tuple[int, int]]:
    if not values:
        return [(40, 25), (60, 40), (100, 60), (160, 100)]
    out: list[tuple[int, int]] = []
    for v in values:
        v = v.lower().replace(" ", "")
        if "x" not in v:
            raise ValueError(f"Invalid size '{v}'. Expected format like 100x60.")
        a, b = v.split("x", 1)
        out.append((int(a), int(b)))
    return out


def _parse_int_list(values: str | None, *, default: list[int]) -> list[int]:
    if values is None:
        return default
    values = values.strip()
    if not values:
        return []
    return [int(x) for x in values.split(",")]


def _random_quat_matrix(m: int, n: int) -> np.ndarray:
    data = np.random.randn(m, n, 4)
    return quaternion.as_quat_array(data)


def _diag_real_as_quat(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=float).reshape(-1)
    n = d.size
    A = np.zeros((n, n, 4), dtype=float)
    A[np.arange(n), np.arange(n), 0] = d
    return quaternion.as_quat_array(A)


def _ill_conditioned_family(m: int, n: int, *, cond: float = 1e4) -> np.ndarray:
    """
    Build a rectangular ill-conditioned quaternion matrix with controlled singular values.

    Construction: A = U Σ V^H, where U and V are quaternion unitary (square) and
    Σ has logarithmically decaying real singular values.
    """
    r = min(m, n)
    try:
        from quatica.decomp.qsvd import qr_qua
    except Exception:
        # Fallback to real orthogonal lift (still stable, but purely real quaternions)
        U_real, _ = np.linalg.qr(np.random.randn(m, m))
        V_real, _ = np.linalg.qr(np.random.randn(n, n))
        U = quaternion.as_quat_array(np.stack([U_real, np.zeros_like(U_real), np.zeros_like(U_real), np.zeros_like(U_real)], axis=-1))
        V = quaternion.as_quat_array(np.stack([V_real, np.zeros_like(V_real), np.zeros_like(V_real), np.zeros_like(V_real)], axis=-1))
    else:
        U, _ = qr_qua(_random_quat_matrix(m, m))
        V, _ = qr_qua(_random_quat_matrix(n, n))

    s = np.logspace(0.0, -np.log10(cond), r)
    Sigma = _diag_real_as_quat(s)
    U_r = U[:, :r]
    V_r = V[:, :r]
    return quat_matmat(quat_matmat(U_r, Sigma), quat_hermitian(V_r))


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str


METHODS: list[MethodSpec] = [
    MethodSpec("ns_g1", r"NS ($\gamma=1$)"),
    MethodSpec("ns_g12", r"NS ($\gamma=1/2$)"),
    MethodSpec("hon3", r"Higher-Order NS (3rd)"),
]


def _compute_final_metrics(A: np.ndarray, X: np.ndarray) -> dict[str, float]:
    AX = quat_matmat(A, X)
    XA = quat_matmat(X, A)
    E1 = float(quat_frobenius_norm(quat_matmat(AX, A) - A))
    E2 = float(quat_frobenius_norm(quat_matmat(XA, X) - X))
    E3 = float(quat_frobenius_norm(AX - quat_hermitian(AX)))
    E4 = float(quat_frobenius_norm(XA - quat_hermitian(XA)))
    return {"E1": E1, "E2": E2, "E3": E3, "E4": E4}


def _iters_to_threshold(history: Iterable[float], tol: float) -> float:
    for k, v in enumerate(history, start=1):
        if np.isfinite(v) and v <= tol:
            return float(k)
    return float("nan")


def _run_one(A: np.ndarray, *, max_iter: int, tol: float) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    # NS gamma=1
    ns1 = NewtonSchulzPseudoinverse(gamma=1.0, max_iter=max_iter, tol=tol, verbose=False, compute_residuals=True)
    t0 = time.time()
    X1, res1, _cov1 = ns1.compute(A)
    t1 = time.time() - t0
    out["ns_g1"] = {"X": X1, "residuals": res1, "time": t1}

    # NS gamma=1/2
    ns12 = NewtonSchulzPseudoinverse(gamma=0.5, max_iter=max_iter, tol=tol, verbose=False, compute_residuals=True)
    t0 = time.time()
    X12, res12, _cov12 = ns12.compute(A)
    t12 = time.time() - t0
    out["ns_g12"] = {"X": X12, "residuals": res12, "time": t12}

    # Higher-order 3rd
    hon = HigherOrderNewtonSchulzPseudoinverse(max_iter=max_iter, tol=tol, verbose=False)
    t0 = time.time()
    Xh, resh, times_h = hon.compute(A)
    th = time.time() - t0
    out["hon3"] = {"X": Xh, "residuals": resh, "time": th, "times_per_iter": times_h}

    return out


def _make_family_matrix(family: str, m: int, n: int) -> np.ndarray:
    if family == "dense_random":
        return _random_quat_matrix(m, n)
    if family == "ill_conditioned":
        return _ill_conditioned_family(m, n, cond=1e4)
    raise ValueError(f"Unknown family: {family}")


def run_benchmark(
    *,
    sizes: list[tuple[int, int]],
    seeds: list[int],
    max_iter: int,
    tol: float,
    out_dir: Path,
    no_display: bool,
) -> None:
    _setup_matplotlib_style(headless=no_display)
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    families = ["dense_random", "ill_conditioned"]

    rows: list[dict[str, Any]] = []
    histories: dict[str, Any] = {}

    rep_family = "dense_random"
    rep_seed = seeds[0] if seeds else 0
    # Prefer the canonical representative size if present, otherwise pick a middle size.
    rep_size = (100, 60) if (100, 60) in sizes else sizes[len(sizes) // 2]

    for family in families:
        for (m, n) in sizes:
            for seed in seeds:
                np.random.seed(seed)
                A = _make_family_matrix(family, m, n)

                # Run all methods on the same A
                results = _run_one(A, max_iter=max_iter, tol=tol)

                for spec in METHODS:
                    r = results[spec.key]
                    X = r["X"]
                    res = r["residuals"]
                    Ehist = list(res.get("AXA-A", []))
                    final = _compute_final_metrics(A, X)
                    iters = int(len(Ehist))
                    # Success flag used for the "success-rate" plot.
                    # For paper clarity and comparability across variants (and consistent with the
                    # iterations-to-threshold plot), we define success based on the primary
                    # Moore–Penrose residual E1 only.
                    #
                    # Note: Higher-order NS (3rd) implementation in `quatica.solver` uses E1 for its
                    # internal stopping criterion as well; using max(E1..E4) here can therefore make
                    # it appear to "fail" even when it met its own convergence target.
                    success = bool(np.isfinite(final["E1"]) and final["E1"] <= tol)
                    it_to_tol = _iters_to_threshold(Ehist, 1e-8)

                    rows.append(
                        {
                            "family": family,
                            "m": m,
                            "n": n,
                            "size_label": f"{m}×{n}",
                            "seed": seed,
                            "method": spec.label,
                            "method_key": spec.key,
                            "time_s": float(r["time"]),
                            "iterations": iters,
                            "success": success,
                            "E1": final["E1"],
                            "E2": final["E2"],
                            "E3": final["E3"],
                            "E4": final["E4"],
                            "Emax": float(max(final.values())),
                            "iters_to_E1_le_1e-8": it_to_tol,
                        }
                    )

                    if family == rep_family and (m, n) == rep_size and seed == rep_seed:
                        histories[f"rep__{spec.key}__E1"] = np.asarray(Ehist, dtype=float)

    # Export raw data
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("This benchmark requires pandas (already used elsewhere in QuatIca).") from e
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ns_pinv_benchmark_results.csv", index=False)
    np.savez(out_dir / "ns_pinv_benchmark_histories.npz", **histories)

    # Aggregate plots: build dashboard
    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.30)

    ax_conv = fig.add_subplot(gs[0, 0])
    ax_time = fig.add_subplot(gs[0, 1])
    ax_it2 = fig.add_subplot(gs[1, 0])
    ax_box = fig.add_subplot(gs[1, 1])
    ax_succ = fig.add_subplot(gs[2, 0])
    ax_blank = fig.add_subplot(gs[2, 1])
    ax_blank.axis("off")

    colors = {"ns_g1": "#d62728", "ns_g12": "#1f77b4", "hon3": "#2ca02c"}

    # 1) Representative convergence plot (E1^(k))
    for spec in METHODS:
        y = histories.get(f"rep__{spec.key}__E1", None)
        if y is None:
            continue
        ax_conv.semilogy(np.arange(1, len(y) + 1), y, label=spec.label, color=colors[spec.key])
    ax_conv.set_title(r"Representative convergence ($E_1^{(k)}$)")
    ax_conv.set_xlabel("Iteration")
    ax_conv.set_ylabel(r"$E_1^{(k)}=\|A X_k A - A\|_F$")
    ax_conv.grid(True, ls=":")
    if ax_conv.get_legend_handles_labels()[0]:
        ax_conv.legend(frameon=False)

    # Helper groupby
    size_order = [f"{m}×{n}" for (m, n) in sizes]
    df["size_label"] = pd.Categorical(df["size_label"], categories=size_order, ordered=True)

    # 2) Scalability plot: mean runtime vs size (error bars), per method (dense_random only)
    df_time = df[df["family"] == "dense_random"].copy()
    for spec in METHODS:
        d = df_time[df_time["method_key"] == spec.key]
        means = d.groupby("size_label", observed=True)["time_s"].mean()
        stds = d.groupby("size_label", observed=True)["time_s"].std()
        x = np.arange(len(size_order))
        y = np.array([means.get(lbl, np.nan) for lbl in size_order], dtype=float)
        e = np.array([stds.get(lbl, np.nan) for lbl in size_order], dtype=float)
        ax_time.errorbar(
            x,
            y,
            yerr=e,
            marker="o",
            markersize=7,
            capsize=5,
            label=spec.label,
            color=colors[spec.key],
        )
    ax_time.set_title("Scalability (dense random): runtime vs size")
    ax_time.set_xlabel("Size")
    ax_time.set_ylabel("Runtime (s)")
    ax_time.set_xticks(np.arange(len(size_order)))
    ax_time.set_xticklabels([rf"${lbl}$" for lbl in size_order], rotation=0)
    ax_time.grid(True, ls=":")
    ax_time.legend(frameon=False)

    # 3) Iterations-to-threshold plot (E1 <= 1e-8)
    df_it = df[df["family"] == "dense_random"].copy()
    for spec in METHODS:
        d = df_it[df_it["method_key"] == spec.key]
        means = d.groupby("size_label", observed=True)["iters_to_E1_le_1e-8"].mean()
        stds = d.groupby("size_label", observed=True)["iters_to_E1_le_1e-8"].std()
        x = np.arange(len(size_order))
        y = np.array([means.get(lbl, np.nan) for lbl in size_order], dtype=float)
        e = np.array([stds.get(lbl, np.nan) for lbl in size_order], dtype=float)
        ax_it2.errorbar(
            x,
            y,
            yerr=e,
            marker="o",
            markersize=7,
            capsize=5,
            label=spec.label,
            color=colors[spec.key],
        )
    ax_it2.set_title(r"Iterations to reach $E_1\leq 10^{-8}$ (dense random)")
    ax_it2.set_xlabel("Size")
    ax_it2.set_ylabel("Iterations")
    ax_it2.set_xticks(np.arange(len(size_order)))
    ax_it2.set_xticklabels([rf"${lbl}$" for lbl in size_order])
    ax_it2.grid(True, ls=":")
    ax_it2.legend(frameon=False)

    # 4) Final residual distribution (boxplot of E1 across all runs)
    box_data = []
    box_labels = []
    for spec in METHODS:
        vals = df[df["method_key"] == spec.key]["E1"].values
        vals = vals[np.isfinite(vals)]
        box_data.append(vals)
        box_labels.append(spec.label)
    try:
        bp = ax_box.boxplot(box_data, tick_labels=box_labels, patch_artist=True, showfliers=True)
    except TypeError:
        bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=True)
    for patch, spec in zip(bp["boxes"], METHODS):
        patch.set_facecolor(colors[spec.key])
        patch.set_alpha(0.65)
    ax_box.set_yscale("log")
    ax_box.set_title(r"Final $E_1$ distribution (all runs)")
    ax_box.set_ylabel(r"$E_1=\|A X A - A\|_F$")
    ax_box.grid(True, ls=":", axis="y")

    # 5) Success-rate plot (% successful runs), per method
    succ_rates = []
    for spec in METHODS:
        d = df[df["method_key"] == spec.key]
        rate = 100.0 * float(d["success"].mean()) if len(d) else 0.0
        succ_rates.append(rate)
    bars = ax_succ.bar(box_labels, succ_rates, color=[colors[s.key] for s in METHODS], alpha=0.85, edgecolor="black", linewidth=0.8)
    for bar, rate in zip(bars, succ_rates):
        ax_succ.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1.5, f"{rate:.0f}\\%", ha="center", va="bottom", fontsize=11)
    ax_succ.set_ylim(0, 105)
    ax_succ.set_title("Success rate (all runs)")
    ax_succ.set_ylabel(r"Success rate (\%)")
    ax_succ.grid(True, ls=":", axis="y")

    # (No global suptitle: easier to place panels directly into paper layouts.)
    base = "ns_pinv_performance_report"
    _export_dashboard_and_panels(
        fig,
        axes={
            "convergence_example": ax_conv,
            "time_vs_size": ax_time,
            "iterations_to_tol": ax_it2,
            "final_residual_boxplot": ax_box,
            "success_rate": ax_succ,
        },
        out_dir=out_dir,
        base=base,
    )
    plt.close(fig)

    if not no_display:
        # Regenerate a lightweight view (optional)
        try:
            img = matplotlib.image.imread(out_dir / f"{base}.png")
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis("off")
            plt.show()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Paper-ready NS pseudoinverse benchmark (QuatIca).")
    p.add_argument("--sizes", nargs="*", default=None, help="Sizes like 40x25 60x40 100x60 160x100")
    p.add_argument("--seeds", default=None, help="Comma-separated seeds, e.g. 0,1,2,3,4")
    p.add_argument("--max-iter", type=int, default=120, help="Maximum iterations per run")
    p.add_argument("--tol", type=float, default=1e-8, help="Tolerance for success based on E1 (and used by solvers where applicable)")
    p.add_argument("--out-dir", default="validation_output", help="Output directory")
    p.add_argument("--no-display", action="store_true", help="Headless mode (no GUI)")
    args = p.parse_args(argv)

    sizes = _parse_sizes(args.sizes)
    seeds = _parse_int_list(args.seeds, default=[0, 1, 2, 3, 4])
    root = Path(_find_repo_root(os.path.dirname(__file__)))
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    run_benchmark(
        sizes=sizes,
        seeds=seeds,
        max_iter=args.max_iter,
        tol=args.tol,
        out_dir=out_dir,
        no_display=args.no_display,
    )


if __name__ == "__main__":
    main()
