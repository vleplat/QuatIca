#!/usr/bin/env python3
"""
Image Deblurring Benchmark Script for Report

This script runs the image deblurring experiments according to the report requirements:
- Compare FFT-NS-Q vs QSLST-FFT on kodim16 and kodim20
- Test sizes: 32, 64, 128, 256, 400, 512
- Blur kernel: Gaussian (radius=4, sigma=1.0) - 9×9 kernel
- Noise: 30 dB SNR
- Regularization: Optimized λ per image/size
- Generate side-by-side comparison plots for N=128
"""

import argparse
import json
import os
import subprocess
import sys
import time


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _script_path() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "script_image_deblurring.py"))


def _output_dir() -> str:
    return os.path.join(_repo_root(), "output_figures")


def _load_optimized_lambdas(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    out: dict = {}
    for r in rows:
        image = r.get("image")
        size = int(r.get("size"))
        lam = float(r.get("best_lambda"))
        out.setdefault(image, {})[size] = lam
    return out


def run_deblur_experiment(
    image_name,
    size,
    snr=30,
    lam=0.1,
    ns_iters=12,
    *,
    psf_radius: int = 4,
    psf_sigma: float = 1.0,
):
    """Run a single deblurring experiment and return parsed results (JSON-based)."""
    print(f"\n{'=' * 60}")
    print(f"Running experiment: {image_name} at size {size}x{size}")
    print(
        f"Parameters: SNR={snr}dB, λ={lam}, NS_iters={ns_iters}, PSF(radius={psf_radius}, sigma={psf_sigma})"
    )
    print(f"{'=' * 60}")

    out_dir = _output_dir()
    os.makedirs(out_dir, exist_ok=True)

    # Build command
    metrics_path = os.path.join(
        out_dir,
        f"_metrics_{image_name}_{size}_{int(snr)}dB_lam{lam:.6g}.json",
    )
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
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_repo_root())
    end_time = time.time()

    if result.returncode != 0:
        print(f"❌ Experiment failed for {image_name} size {size}")
        print(f"Error: {result.stderr}")
        return None

    print(f"✅ Experiment completed in {end_time - start_time:.2f}s")

    if not os.path.exists(metrics_path):
        print("❌ Experiment did not write expected metrics JSON.")
        print("Stdout:")
        print(result.stdout)
        print("Stderr:")
        print(result.stderr)
        return None

    with open(metrics_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Keep the public result shape stable for downstream plotting scripts.
    results = {
        "image": image_name,
        "size": int(size),
        "snr": float(snr),
        "lam": float(lam),
        "ns_iters": int(ns_iters),
        "psf_radius": int(payload.get("psf_radius", psf_radius)),
        "psf_sigma": float(payload.get("psf_sigma", psf_sigma)),
        "qslst_fft": {
            "time": float(payload["qslst_fft"]["time_s"]),
            "psnr": float(payload["qslst_fft"]["psnr"]),
            "ssim": float(payload["qslst_fft"]["ssim"]),
            "rel_err": float(payload["qslst_fft"]["rel_err"]),
        },
        "ns_fft": {
            "time": float(payload["ns"]["time_s"]),
            "psnr": float(payload["ns"]["psnr"]),
            "ssim": float(payload["ns"]["ssim"]),
            "rel_err": float(payload["ns"]["rel_err"]),
        },
        "meta": {
            "ns_mode": payload.get("ns_mode"),
            "fftT_order": payload.get("fftT_order"),
            "snr_measured_db": payload.get("snr_measured_db"),
            "validate_bccb_rel_mismatch": payload.get("validate_bccb_rel_mismatch"),
            "saved_files": payload.get("saved_files", []),
        },
    }
    return results


def create_comparison_plot(image_name, size=128, *, snr=30, ns_mode="fftT"):
    """Create side-by-side comparison plot for the specified image and size"""
    print(f"\n🎨 Creating comparison plot for {image_name} at size {size}x{size}")

    # Expected output files with unique names
    output_dir = _output_dir()
    base_name = f"{image_name}_{size}"
    clean_file = f"{output_dir}/deblur_input_clean_{base_name}.png"
    observed_file = (
        f"{output_dir}/deblur_observed_blurred_{base_name}.png"
        if snr is None
        else f"{output_dir}/deblur_observed_blur_noise_{int(snr)}dB_{base_name}.png"
    )
    qslst_file = f"{output_dir}/deblur_qslst_fft_{base_name}.png"
    ns_file = f"{output_dir}/deblur_ns_{ns_mode}_{base_name}.png"

    # Check if files exist
    required_files = [clean_file, observed_file, qslst_file, ns_file]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False

    # Create comparison plot
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Load images
    clean_img = mpimg.imread(clean_file)
    observed_img = mpimg.imread(observed_file)
    qslst_img = mpimg.imread(qslst_file)
    ns_img = mpimg.imread(ns_file)

    # Plot images
    axes[0].imshow(clean_img)
    axes[0].set_title("Clean Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(observed_img)
    axes[1].set_title(
        "Noisy + Blurred\n({} dB SNR)".format("blur only" if snr is None else int(snr)),
        fontsize=14,
        fontweight="bold",
    )
    axes[1].axis("off")

    axes[2].imshow(qslst_img)
    axes[2].set_title("QSLST-FFT\nRecovery", fontsize=14, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(ns_img)
    axes[3].set_title("FFT-NS-Q\nRecovery", fontsize=14, fontweight="bold")
    axes[3].axis("off")

    plt.tight_layout()

    # Save plot
    plot_filename = f"{output_dir}/deblur_comparison_{image_name}_{size}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"✅ Comparison plot saved: {plot_filename}")
    return True


def generate_latex_table(results):
    """Generate LaTeX table from benchmark results"""
    print("\n📋 Generating LaTeX table for paper...")
    print("=" * 80)

    print("\\begin{table}[ht!]")
    print("\\centering")
    print(
        "\\caption{Image deblurring: FFT--NS--Q vs.\\ QSLST--FFT on $N\\times N$ subimages from Kodak images.}"
    )
    print("\\label{tab:deblur-results}")
    print("\\begin{tabular}{lccccc}")
    print("\\hline")
    print(
        "$N$ & $\\lambda_{\\text{opt}}$ & Method & CPU time (s) & PSNR (dB) & SSIM \\\\"
    )
    print("\\hline")

    # Group results by image and size
    for image in ["kodim16", "kodim20"]:
        print(f"\\multicolumn{{6}}{{l}}{{\\textit{{{image}}}}} \\\\")
        for size in [32, 64, 128, 256, 400, 512]:
            # Find results for this image and size
            result = None
            for r in results:
                if r["image"] == image and r["size"] == size:
                    result = r
                    break

            if result:
                qslst = result["qslst_fft"]
                ns = result["ns_fft"]
                lam = result["lam"]

                print(
                    f"{size} & {lam:5.3f} & QSLST--FFT & {qslst['time']:6.3f} & {qslst['psnr']:6.2f} & {qslst['ssim']:5.3f} \\\\"
                )
                print(
                    f"    &         & FFT--NS--Q & {ns['time']:6.3f} & {ns['psnr']:6.2f} & {ns['ssim']:5.3f} \\\\"
                )
                print("\\hline")

    print("\\end{tabular}")
    print("\\end{table}")
    print("=" * 80)


def main():
    """Main benchmark execution"""
    parser = argparse.ArgumentParser(
        description="Run the image deblurring benchmark (QSLST-FFT vs FFT-NS-Q)."
    )
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
        "--default-lam",
        type=float,
        default=0.05,
        help="Fallback λ when optimized λ is unavailable (default: 0.05).",
    )
    parser.add_argument(
        "--force-lam",
        type=float,
        default=None,
        help=(
            "Force a single λ for all runs (overrides optimized lambdas and --default-lam). "
            "Example: --force-lam 1e-1"
        ),
    )
    parser.add_argument(
        "--lambda-json",
        type=str,
        default=os.path.join(_output_dir(), "lambda_optimization_results.json"),
        help="Path to lambda optimization JSON (default: output_figures/lambda_optimization_results.json).",
    )
    parser.add_argument(
        "--compare-size",
        type=int,
        default=128,
        help="Size N at which to generate side-by-side comparison plots (default: 128).",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip generating side-by-side comparison plots.",
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
    parser.add_argument(
        "--out-json",
        type=str,
        default=os.path.join(_output_dir(), "deblur_benchmark_results.json"),
        help="Path to save JSON results (default: output_figures/deblur_benchmark_results.json).",
    )
    args = parser.parse_args()

    print("🚀 Starting Image Deblurring Benchmark for Report")
    print("=" * 80)
    print("Parameters:")
    print(f"  - Images: {args.images}")
    print(f"  - Sizes: {args.sizes}")
    print(
        f"  - Blur: Gaussian (radius={int(args.psf_radius)}, sigma={float(args.psf_sigma)}) - "
        f"{2*int(args.psf_radius)+1}×{2*int(args.psf_radius)+1} kernel"
    )
    print(f"  - Noise: {args.snr:g} dB SNR")
    if args.force_lam is not None:
        print(f"  - Regularization: forced λ={float(args.force_lam)} (all runs)")
    else:
        print("  - Regularization: Optimized λ per image/size (fallback to --default-lam)")
    print(f"  - NS iterations: {args.ns_iters}")
    print("=" * 80)

    # Create output directory
    os.makedirs(_output_dir(), exist_ok=True)

    # Experiment configuration with optimized lambda values
    images = [x.strip() for x in args.images.split(",") if x.strip()]
    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    snr = float(args.snr)
    ns_iters = int(args.ns_iters)

    # Optimized lambda values from lambda optimization
    if args.force_lam is None:
        if os.path.exists(args.lambda_json):
            optimized_lambdas = _load_optimized_lambdas(args.lambda_json)
        else:
            print(f"⚠️  Optimized lambda JSON not found at: {args.lambda_json}")
            print("    Falling back to embedded defaults (may drift from latest optimization).")
            optimized_lambdas = {
                "kodim16": {
                    32: 0.020,
                    64: 0.050,
                    128: 0.050,
                    256: 0.050,
                    400: 0.050,
                    512: 0.050,
                },
                "kodim20": {
                    32: 0.020,
                    64: 0.020,
                    128: 0.020,
                    256: 0.050,
                    400: 0.050,
                    512: 0.050,
                },
            }
    else:
        optimized_lambdas = None

    all_results = []

    # Run experiments with optimized lambda values
    for image in images:
        for size in sizes:
            if args.force_lam is not None:
                lam = float(args.force_lam)
            else:
                lam = (
                    float(optimized_lambdas.get(image, {}).get(size, args.default_lam))
                    if optimized_lambdas is not None
                    else float(args.default_lam)
                )
            result = run_deblur_experiment(
                image,
                size,
                snr,
                lam,
                ns_iters,
                psf_radius=int(args.psf_radius),
                psf_sigma=float(args.psf_sigma),
            )
            if result:
                all_results.append(result)

    # Create comparison plots
    if not args.skip_comparison:
        for image in images:
            create_comparison_plot(image, args.compare_size, snr=snr, ns_mode="fftT")

    # Generate LaTeX table
    generate_latex_table(all_results)

    # Save results to JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n🎉 Benchmark completed successfully!")
    print("📁 Results saved to: output_figures/")
    print(f"📊 JSON results: {args.out_json}")
    print("📋 LaTeX table generated above")
    print("🎨 Comparison plots: output_figures/deblur_comparison_*.png")


if __name__ == "__main__":
    main()
