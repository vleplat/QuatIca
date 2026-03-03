#!/usr/bin/env python3
"""
Benchmark dense quaternion matmul: current vs legacy implementation.

Compares:
  1) legacy_quat_matmat_dense: historical code path (as kept in utils.py comments)
  2) quat_matmat_new_default: current quatica.utils.quat_matmat(A, B)
  3) quat_matmat_new_precomp: current quatica.utils.quat_matmat with precomputed components

The benchmark reports wall-clock and CPU time and writes a CSV summary under:
  validation_output/quat_matmat_benchmark/
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import quaternion
import matplotlib.pyplot as plt

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

from quatica.utils import quat_matmat


def rand_quat(m: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    comp = rng.standard_normal((m, n, 4))
    return quaternion.as_quat_array(comp)


def legacy_quat_matmat_dense(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Legacy dense x dense implementation (from previous commented code)."""
    A_comp = quaternion.as_float_array(A)
    B_comp = quaternion.as_float_array(B)
    Aw, Ax, Ay, Az = A_comp[..., 0], A_comp[..., 1], A_comp[..., 2], A_comp[..., 3]
    Bw, Bx, By, Bz = B_comp[..., 0], B_comp[..., 1], B_comp[..., 2], B_comp[..., 3]
    Cw = Aw @ Bw - Ax @ Bx - Ay @ By - Az @ Bz
    Cx = Aw @ Bx + Ax @ Bw + Ay @ Bz - Az @ By
    Cy = Aw @ By - Ax @ Bz + Ay @ Bw + Az @ Bx
    Cz = Aw @ Bz + Ax @ By - Ay @ Bx + Az @ Bw
    C = np.stack([Cw, Cx, Cy, Cz], axis=-1)
    return quaternion.as_quat_array(C)


@dataclass
class Timing:
    wall_s: float
    cpu_s: float


def time_call(fn, *args, repeats: int) -> list[Timing]:
    out: list[Timing] = []
    for _ in range(repeats):
        t0w = time.perf_counter()
        t0c = time.process_time()
        _ = fn(*args)
        t1c = time.process_time()
        t1w = time.perf_counter()
        out.append(Timing(wall_s=t1w - t0w, cpu_s=t1c - t0c))
    return out


def stats(xs: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(xs, dtype=float)
    return float(np.mean(arr)), float(np.std(arr)), float(np.min(arr))


def save_png_and_pdf(fig, png_path: str, dpi: int = 200) -> None:
    root, ext = os.path.splitext(png_path)
    if ext.lower() != ".png":
        raise ValueError(f"Expected .png path, got: {png_path}")
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(root + ".pdf")


def make_plots(rows: list[dict[str, float | int | str]], out_dir: str, ts: str) -> None:
    methods = ["legacy", "new_default", "new_precomp"]
    labels = {
        "legacy": "legacy",
        "new_default": "new default",
        "new_precomp": "new + precomp",
    }
    colors = {
        "legacy": "#444444",
        "new_default": "#1f77b4",
        "new_precomp": "#2ca02c",
    }

    sizes = sorted({int(r["size_n"]) for r in rows})

    def series(method: str, key: str) -> np.ndarray:
        vals = []
        for n in sizes:
            row = next(r for r in rows if int(r["size_n"]) == n and str(r["method"]) == method)
            vals.append(float(row[key]))
        return np.asarray(vals, dtype=float)

    # 1) Wall time (mean) vs size
    plt.figure()
    for m in methods:
        y_ms = 1e3 * series(m, "wall_mean_s")
        plt.plot(sizes, y_ms, marker="o", label=labels[m], color=colors[m])
    plt.xlabel("matrix size n (n×n @ n×n)")
    plt.ylabel("wall time mean (ms)")
    plt.title("Quaternion matmul benchmark (wall-clock)")
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    save_png_and_pdf(plt.gcf(), os.path.join(out_dir, f"quat_matmat_wall_time_{ts}.png"))
    plt.close()

    # 2) CPU time (mean) vs size
    plt.figure()
    for m in methods:
        y_ms = 1e3 * series(m, "cpu_mean_s")
        plt.plot(sizes, y_ms, marker="o", label=labels[m], color=colors[m])
    plt.xlabel("matrix size n (n×n @ n×n)")
    plt.ylabel("CPU time mean (ms)")
    plt.title("Quaternion matmul benchmark (CPU)")
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    save_png_and_pdf(plt.gcf(), os.path.join(out_dir, f"quat_matmat_cpu_time_{ts}.png"))
    plt.close()

    # 3) Speedup vs legacy
    plt.figure()
    for m in ["new_default", "new_precomp"]:
        s = series(m, "speedup_vs_legacy")
        plt.plot(sizes, s, marker="o", label=labels[m], color=colors[m])
    plt.axhline(1.0, color="#888888", ls="--", lw=1.0)
    plt.xlabel("matrix size n (n×n @ n×n)")
    plt.ylabel("speedup vs legacy (x)")
    plt.title("Speedup over legacy quaternion matmul")
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    save_png_and_pdf(plt.gcf(), os.path.join(out_dir, f"quat_matmat_speedup_{ts}.png"))
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[64, 128, 256], help="Square sizes n for n×n @ n×n")
    ap.add_argument("--warmup", type=int, default=2, help="Warmup calls per method/size")
    ap.add_argument("--repeats", type=int, default=7, help="Timed repeats per method/size")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = os.path.join("validation_output", "quat_matmat_benchmark")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(out_dir, f"quat_matmat_old_vs_new_{ts}.csv")

    print("=== Quaternion matmul benchmark (dense) ===")
    print(f"sizes={args.sizes} warmup={args.warmup} repeats={args.repeats} seed={args.seed}")
    print("")

    rows: list[dict[str, float | int | str]] = []

    for n in args.sizes:
        A = rand_quat(n, n, seed=args.seed + n)
        B = rand_quat(n, n, seed=args.seed + 10_000 + n)

        # Warmups
        for _ in range(args.warmup):
            _ = legacy_quat_matmat_dense(A, B)
            _ = quat_matmat(A, B)
            A_comp = quaternion.as_float_array(A)
            B_comp = quaternion.as_float_array(B)
            _ = quat_matmat(A, B, A_comp=A_comp, B_comp=B_comp)

        # Timings
        t_old = time_call(legacy_quat_matmat_dense, A, B, repeats=args.repeats)
        t_new = time_call(quat_matmat, A, B, repeats=args.repeats)
        A_comp = quaternion.as_float_array(A)
        B_comp = quaternion.as_float_array(B)
        t_new_pre = time_call(
            lambda X, Y: quat_matmat(X, Y, A_comp=A_comp, B_comp=B_comp),
            A,
            B,
            repeats=args.repeats,
        )

        # Numerical check against legacy
        C_old = legacy_quat_matmat_dense(A, B)
        C_new = quat_matmat(A, B)
        C_pre = quat_matmat(A, B, A_comp=A_comp, B_comp=B_comp)
        err_new = float(np.max(np.abs(quaternion.as_float_array(C_old - C_new))))
        err_pre = float(np.max(np.abs(quaternion.as_float_array(C_old - C_pre))))

        old_w_mean, old_w_std, old_w_min = stats([t.wall_s for t in t_old])
        old_c_mean, old_c_std, old_c_min = stats([t.cpu_s for t in t_old])
        new_w_mean, new_w_std, new_w_min = stats([t.wall_s for t in t_new])
        new_c_mean, new_c_std, new_c_min = stats([t.cpu_s for t in t_new])
        pre_w_mean, pre_w_std, pre_w_min = stats([t.wall_s for t in t_new_pre])
        pre_c_mean, pre_c_std, pre_c_min = stats([t.cpu_s for t in t_new_pre])

        spd_new = old_w_mean / max(new_w_mean, 1e-12)
        spd_pre = old_w_mean / max(pre_w_mean, 1e-12)

        print(f"n={n:4d} | old={old_w_mean*1e3:8.2f} ms | new={new_w_mean*1e3:8.2f} ms "
              f"(x{spd_new:5.2f}) | new_precomp={pre_w_mean*1e3:8.2f} ms (x{spd_pre:5.2f})")
        print(f"       max|old-new|={err_new:.3e}  max|old-new_precomp|={err_pre:.3e}")

        rows.extend(
            [
                {
                    "size_n": n,
                    "method": "legacy",
                    "wall_mean_s": old_w_mean,
                    "wall_std_s": old_w_std,
                    "wall_min_s": old_w_min,
                    "cpu_mean_s": old_c_mean,
                    "cpu_std_s": old_c_std,
                    "cpu_min_s": old_c_min,
                    "speedup_vs_legacy": 1.0,
                    "max_abs_diff_vs_legacy": 0.0,
                },
                {
                    "size_n": n,
                    "method": "new_default",
                    "wall_mean_s": new_w_mean,
                    "wall_std_s": new_w_std,
                    "wall_min_s": new_w_min,
                    "cpu_mean_s": new_c_mean,
                    "cpu_std_s": new_c_std,
                    "cpu_min_s": new_c_min,
                    "speedup_vs_legacy": spd_new,
                    "max_abs_diff_vs_legacy": err_new,
                },
                {
                    "size_n": n,
                    "method": "new_precomp",
                    "wall_mean_s": pre_w_mean,
                    "wall_std_s": pre_w_std,
                    "wall_min_s": pre_w_min,
                    "cpu_mean_s": pre_c_mean,
                    "cpu_std_s": pre_c_std,
                    "cpu_min_s": pre_c_min,
                    "speedup_vs_legacy": spd_pre,
                    "max_abs_diff_vs_legacy": err_pre,
                },
            ]
        )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "size_n",
                "method",
                "wall_mean_s",
                "wall_std_s",
                "wall_min_s",
                "cpu_mean_s",
                "cpu_std_s",
                "cpu_min_s",
                "speedup_vs_legacy",
                "max_abs_diff_vs_legacy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    make_plots(rows, out_dir, ts)

    print("")
    print(f"Saved CSV: {out_csv}")
    print(f"Saved plots: {out_dir}")


if __name__ == "__main__":
    main()

