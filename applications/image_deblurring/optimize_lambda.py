#!/usr/bin/env python3
"""
Lambda Optimization for Image Deblurring

This script performs a line search to find the optimal lambda value for each
image and size combination, using SNR=30dB.
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Optional, Sequence


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _script_path() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "script_image_deblurring.py"))


def _output_dir() -> str:
    return os.path.join(_repo_root(), "output_figures")


def run_single_experiment(
    image_name, size, snr, lam, ns_iters=12, *, psf_radius: int = 4, psf_sigma: float = 1.0
):
    """Run a single deblurring experiment and return results"""
    print(f"  Testing λ={lam:.3f}...", end=" ", flush=True)

    os.makedirs(_output_dir(), exist_ok=True)
    metrics_path = os.path.join(
        _output_dir(), f"_metrics_opt_{image_name}_{size}_{int(snr)}dB_lam{lam:.6g}.json"
    )

    # Build command (machine-readable output)
    cmd = [
        sys.executable,
        _script_path(),
        "--image",
        image_name,
        "--size",
        str(size),
        "--lam",
        str(lam),
        "--snr",
        str(snr),
        "--psf_radius",
        str(int(psf_radius)),
        "--psf_sigma",
        str(float(psf_sigma)),
        "--ns_mode",
        "fftT",
        "--ns_iters",
        str(ns_iters),
        "--fftT_order",
        "2",
        "--metrics_json",
        metrics_path,
    ]

    # Run the experiment
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_repo_root())

    if result.returncode != 0:
        print("❌ FAILED")
        return None

    if not os.path.exists(metrics_path):
        print("❌ MISSING METRICS JSON")
        return None

    with open(metrics_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    results = {
        "psf_radius": int(payload.get("psf_radius", psf_radius)),
        "psf_sigma": float(payload.get("psf_sigma", psf_sigma)),
        "qslst_fft": {
            "psnr": float(payload["qslst_fft"]["psnr"]),
            "ssim": float(payload["qslst_fft"]["ssim"]),
            "time": float(payload["qslst_fft"]["time_s"]),
        },
        "ns_fft": {
            "psnr": float(payload["ns"]["psnr"]),
            "ssim": float(payload["ns"]["ssim"]),
            "time": float(payload["ns"]["time_s"]),
        },
    }
    print(f"✅ PSNR={results['qslst_fft']['psnr']:.2f}dB, SSIM={results['qslst_fft']['ssim']:.3f}")
    return results


def optimize_lambda_for_image_size(
    image_name,
    size,
    snr=30,
    ns_iters=12,
    *,
    psf_radius: int = 4,
    psf_sigma: float = 1.0,
    lambda_values: Optional[Sequence[float]] = None,
):
    """Find optimal lambda for a specific image and size"""
    print(f"\n🔍 Optimizing λ for {image_name} at size {size}x{size}")
    print("=" * 60)

    # Define lambda search range (logarithmic scale)
    # Start with a reasonable range based on typical values
    if lambda_values is None:
        lambda_values = [
            0.001,
            0.002,
            0.005,
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
        ]

    best_lambda = None
    best_psnr = -float("inf")
    best_ssim = -float("inf")
    best_results = None
    all_results = []

    print(f"Testing {len(lambda_values)} λ values: {lambda_values}")
    print("-" * 60)

    for lam in lambda_values:
        results = run_single_experiment(
            image_name,
            size,
            snr,
            lam,
            ns_iters,
            psf_radius=int(psf_radius),
            psf_sigma=float(psf_sigma),
        )

        if results is not None:
            psnr = results["qslst_fft"]["psnr"]
            ssim = results["qslst_fft"]["ssim"]

            all_results.append(
                {
                    "lambda": lam,
                    "psnr": psnr,
                    "ssim": ssim,
                    "time": results["qslst_fft"]["time"],
                }
            )

            # Update best if this is better (prioritize PSNR, then SSIM)
            if psnr > best_psnr or (psnr == best_psnr and ssim > best_ssim):
                best_lambda = lam
                best_psnr = psnr
                best_ssim = ssim
                best_results = results.copy()
                best_results["lambda"] = lam

    if best_lambda is not None:
        print(f"\n🏆 Best λ = {best_lambda:.3f}")
        print(f"   PSNR = {best_psnr:.2f} dB")
        print(f"   SSIM = {best_ssim:.3f}")
        print(f"   Time = {best_results['qslst_fft']['time']:.3f} s")

        # Create optimization plot
        create_lambda_optimization_plot(image_name, size, all_results, best_lambda)

        return {
            "image": image_name,
            "size": size,
            "psf_radius": int(psf_radius),
            "psf_sigma": float(psf_sigma),
            "best_lambda": best_lambda,
            "best_psnr": best_psnr,
            "best_ssim": best_ssim,
            "best_time": best_results["qslst_fft"]["time"],
            "all_results": all_results,
            "qslst_results": best_results["qslst_fft"],
            "ns_results": best_results["ns_fft"],
        }
    else:
        print("❌ No valid results found")
        return None


def create_lambda_optimization_plot(image_name, size, all_results, best_lambda):
    """Create a plot showing lambda optimization results"""
    try:
        import matplotlib.pyplot as plt

        lambdas = [r["lambda"] for r in all_results]
        psnrs = [r["psnr"] for r in all_results]
        ssims = [r["ssim"] for r in all_results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # PSNR plot
        ax1.semilogx(lambdas, psnrs, "b-o", linewidth=2, markersize=6)
        ax1.axvline(
            x=best_lambda,
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Best λ={best_lambda:.3f}",
        )
        ax1.set_xlabel("λ (Regularization Parameter)", fontsize=12)
        ax1.set_ylabel("PSNR (dB)", fontsize=12)
        ax1.set_title(
            f"PSNR vs λ - {image_name} ({size}×{size})", fontsize=14, fontweight="bold"
        )
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # SSIM plot
        ax2.semilogx(lambdas, ssims, "g-s", linewidth=2, markersize=6)
        ax2.axvline(
            x=best_lambda,
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Best λ={best_lambda:.3f}",
        )
        ax2.set_xlabel("λ (Regularization Parameter)", fontsize=12)
        ax2.set_ylabel("SSIM", fontsize=12)
        ax2.set_title(
            f"SSIM vs λ - {image_name} ({size}×{size})", fontsize=14, fontweight="bold"
        )
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        # Save plot
        output_dir = _output_dir()
        os.makedirs(output_dir, exist_ok=True)
        plot_file = f"{output_dir}/lambda_optimization_{image_name}_{size}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"   📊 Optimization plot saved: {plot_file}")

    except ImportError:
        print("   ⚠️  matplotlib not available, skipping plot")


def generate_optimized_latex_table(all_optimizations):
    """Generate LaTeX table with optimized lambda values"""
    print("\n📋 Generating optimized LaTeX table...")
    print("=" * 80)

    print("\\begin{table}[ht!]")
    print("\\centering")
    print(
        "\\caption{Image deblurring: FFT--NS--Q vs.\\ QSLST--FFT with optimized $\\lambda$ on $N\\times N$ subimages from Kodak images (SNR = 30 dB).}"
    )
    print("\\label{tab:deblur-results-optimized}")
    print("\\begin{tabular}{lccccc}")
    print("\\hline")
    print("$N$ & $\\lambda^*$ & Method & CPU time (s) & PSNR (dB) & SSIM \\\\")
    print("\\hline")

    # Group results by image
    for image in ["kodim16", "kodim20"]:
        print(f"\\multicolumn{{6}}{{l}}{{\\textit{{{image}}}}} \\\\")
        image_results = [r for r in all_optimizations if r["image"] == image]

        for result in sorted(image_results, key=lambda x: x["size"]):
            size = result["size"]
            best_lambda = result["best_lambda"]
            qslst = result["qslst_results"]
            ns = result["ns_results"]

            print(
                f"{size} & {best_lambda:.3f} & QSLST--FFT & {qslst['time']:6.3f} & {qslst['psnr']:6.2f} & {qslst['ssim']:5.3f} \\\\"
            )
            print(
                f"    &        & FFT--NS--Q & {ns['time']:6.3f} & {ns['psnr']:6.2f} & {ns['ssim']:5.3f} \\\\"
            )
            print("\\hline")

    print("\\end{tabular}")
    print("\\end{table}")
    print("=" * 80)


def main():
    """Main optimization function"""
    parser = argparse.ArgumentParser(description="Optimize lambda for image deblurring (QSLST-FFT).")
    parser.add_argument(
        "--images",
        type=str,
        default="kodim16,kodim20",
        help="Comma-separated image names (default: kodim16,kodim20).",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="32,64,128,256,400,512",
        help="Comma-separated square sizes N (default: 32,64,128,256,400,512).",
    )
    parser.add_argument("--snr", type=float, default=30.0, help="SNR in dB (default: 30).")
    parser.add_argument("--ns-iters", type=int, default=12, help="NS iterations (default: 12).")
    parser.add_argument(
        "--lambda-values",
        type=str,
        default="0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0,5.0,10.0",
        help="Comma-separated lambda grid to test (default: 0.001..10.0).",
    )
    parser.add_argument(
        "--psf-radius",
        "--psf_radius",
        type=int,
        default=4,
        help="Gaussian PSF radius r forwarded to the deblurring driver (default: 4).",
    )
    parser.add_argument(
        "--psf-sigma",
        "--psf_sigma",
        type=float,
        default=1.0,
        help="Gaussian PSF sigma forwarded to the deblurring driver (default: 1.0).",
    )
    args = parser.parse_args()

    print("🚀 Starting Lambda Optimization for Image Deblurring")
    print("=" * 80)
    print("Parameters:")
    print(f"  - Images: {args.images}")
    print(f"  - Sizes: {args.sizes}")
    print(f"  - SNR: {args.snr:g} dB")
    print(f"  - PSF: Gaussian (radius={args.psf_radius}, sigma={args.psf_sigma})")
    print(f"  - Lambda grid: {args.lambda_values}")
    print(f"  - NS iterations: {args.ns_iters}")
    print("=" * 80)

    # Create output directory
    os.makedirs(_output_dir(), exist_ok=True)

    # Configuration
    images = [x.strip() for x in args.images.split(",") if x.strip()]
    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    snr = float(args.snr)
    ns_iters = int(args.ns_iters)

    all_optimizations = []

    # Run optimization for each image and size
    lambda_values = [float(x.strip()) for x in str(args.lambda_values).split(",") if x.strip()]
    for image in images:
        for size in sizes:
            result = optimize_lambda_for_image_size(
                image,
                size,
                snr,
                ns_iters,
                psf_radius=int(args.psf_radius),
                psf_sigma=float(args.psf_sigma),
                lambda_values=lambda_values,
            )
            if result:
                all_optimizations.append(result)

    # Save results to JSON
    results_file = os.path.join(_output_dir(), "lambda_optimization_results.json")
    with open(results_file, "w") as f:
        json.dump(all_optimizations, f, indent=2)

    # Generate optimized LaTeX table
    generate_optimized_latex_table(all_optimizations)

    # Print summary
    print("\n🎉 Lambda optimization completed!")
    print("📁 Results saved to: output_figures/")
    print(f"📊 JSON results: {results_file}")
    print("📋 Optimized LaTeX table generated above")
    print("📈 Optimization plots: output_figures/lambda_optimization_*.png")

    # Print best lambda summary
    print("\n📊 Best Lambda Summary:")
    print("-" * 50)
    for image in images:
        print(f"\n{image}:")
        image_results = [r for r in all_optimizations if r["image"] == image]
        for result in sorted(image_results, key=lambda x: x["size"]):
            print(
                f"  N={result['size']:3d}: λ={result['best_lambda']:.3f} → PSNR={result['best_psnr']:.2f}dB, SSIM={result['best_ssim']:.3f}"
            )


if __name__ == "__main__":
    main()
