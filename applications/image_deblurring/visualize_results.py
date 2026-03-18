#!/usr/bin/env python3
"""
Visualize Image Deblurring Benchmark Results

This script creates performance plots from the JSON results file.
"""

import json
import os

import matplotlib.pyplot as plt


def load_results(json_file):
    """Load results from JSON file"""
    with open(json_file, "r") as f:
        return json.load(f)


def create_performance_plots(results):
    """Create performance comparison plots"""

    # Separate data by image
    kodim16_data = [r for r in results if r["image"] == "kodim16"]
    kodim20_data = [r for r in results if r["image"] == "kodim20"]

    def _sizes_for(data):
        return sorted({int(r["size"]) for r in data})

    def extract_metrics_by_size(data, metric, sizes):
        """Return values aligned with `sizes` (sorted), filling missing with NaN."""
        qslst_map = {int(r["size"]): r["qslst_fft"].get(metric, float("nan")) for r in data}
        ns_map = {int(r["size"]): r["ns_fft"].get(metric, float("nan")) for r in data}
        qslst_values = [qslst_map.get(int(s), float("nan")) for s in sizes]
        ns_values = [ns_map.get(int(s), float("nan")) for s in sizes]
        return qslst_values, ns_values

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: CPU Time (kodim16)
    sizes_16 = _sizes_for(kodim16_data)
    qslst_time_16, ns_time_16 = extract_metrics_by_size(kodim16_data, "time", sizes_16)
    axes[0, 0].plot(
        sizes_16, qslst_time_16, "b-o", linewidth=2, markersize=8, label="QSLST-FFT"
    )
    axes[0, 0].plot(
        sizes_16, ns_time_16, "r-s", linewidth=2, markersize=8, label="FFT-NS-Q"
    )
    axes[0, 0].set_xlabel("Image Size N", fontsize=12)
    axes[0, 0].set_ylabel("CPU Time (s)", fontsize=12)
    axes[0, 0].set_title("CPU Time - kodim16", fontsize=14, fontweight="bold")
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")

    # Plot 2: PSNR (kodim16)
    qslst_psnr_16, ns_psnr_16 = extract_metrics_by_size(kodim16_data, "psnr", sizes_16)
    axes[0, 1].plot(
        sizes_16, qslst_psnr_16, "b-o", linewidth=2, markersize=8, label="QSLST-FFT"
    )
    axes[0, 1].plot(
        sizes_16, ns_psnr_16, "r-s", linewidth=2, markersize=8, label="FFT-NS-Q"
    )
    axes[0, 1].set_xlabel("Image Size N", fontsize=12)
    axes[0, 1].set_ylabel("PSNR (dB)", fontsize=12)
    axes[0, 1].set_title("PSNR - kodim16", fontsize=14, fontweight="bold")
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale("log")

    # Plot 3: SSIM (kodim16)
    qslst_ssim_16, ns_ssim_16 = extract_metrics_by_size(kodim16_data, "ssim", sizes_16)
    axes[0, 2].plot(
        sizes_16, qslst_ssim_16, "b-o", linewidth=2, markersize=8, label="QSLST-FFT"
    )
    axes[0, 2].plot(
        sizes_16, ns_ssim_16, "r-s", linewidth=2, markersize=8, label="FFT-NS-Q"
    )
    axes[0, 2].set_xlabel("Image Size N", fontsize=12)
    axes[0, 2].set_ylabel("SSIM", fontsize=12)
    axes[0, 2].set_title("SSIM - kodim16", fontsize=14, fontweight="bold")
    axes[0, 2].legend(fontsize=11)
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xscale("log")

    # Plot 4: CPU Time (kodim20)
    sizes_20 = _sizes_for(kodim20_data)
    qslst_time_20, ns_time_20 = extract_metrics_by_size(kodim20_data, "time", sizes_20)
    axes[1, 0].plot(
        sizes_20, qslst_time_20, "b-o", linewidth=2, markersize=8, label="QSLST-FFT"
    )
    axes[1, 0].plot(
        sizes_20, ns_time_20, "r-s", linewidth=2, markersize=8, label="FFT-NS-Q"
    )
    axes[1, 0].set_xlabel("Image Size N", fontsize=12)
    axes[1, 0].set_ylabel("CPU Time (s)", fontsize=12)
    axes[1, 0].set_title("CPU Time - kodim20", fontsize=14, fontweight="bold")
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")

    # Plot 5: PSNR (kodim20)
    qslst_psnr_20, ns_psnr_20 = extract_metrics_by_size(kodim20_data, "psnr", sizes_20)
    axes[1, 1].plot(
        sizes_20, qslst_psnr_20, "b-o", linewidth=2, markersize=8, label="QSLST-FFT"
    )
    axes[1, 1].plot(
        sizes_20, ns_psnr_20, "r-s", linewidth=2, markersize=8, label="FFT-NS-Q"
    )
    axes[1, 1].set_xlabel("Image Size N", fontsize=12)
    axes[1, 1].set_ylabel("PSNR (dB)", fontsize=12)
    axes[1, 1].set_title("PSNR - kodim20", fontsize=14, fontweight="bold")
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale("log")

    # Plot 6: SSIM (kodim20)
    qslst_ssim_20, ns_ssim_20 = extract_metrics_by_size(kodim20_data, "ssim", sizes_20)
    axes[1, 2].plot(
        sizes_20, qslst_ssim_20, "b-o", linewidth=2, markersize=8, label="QSLST-FFT"
    )
    axes[1, 2].plot(
        sizes_20, ns_ssim_20, "r-s", linewidth=2, markersize=8, label="FFT-NS-Q"
    )
    axes[1, 2].set_xlabel("Image Size N", fontsize=12)
    axes[1, 2].set_ylabel("SSIM", fontsize=12)
    axes[1, 2].set_title("SSIM - kodim20", fontsize=14, fontweight="bold")
    axes[1, 2].legend(fontsize=11)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xscale("log")

    plt.tight_layout()
    return fig


def create_summary_table(results):
    """Create a summary table of results"""
    print("\n📊 SUMMARY TABLE")
    print("=" * 80)
    print(
        f"{'Image':<10} {'Size':<6} {'Method':<12} {'Time(s)':<8} {'PSNR(dB)':<10} {'SSIM':<8}"
    )
    print("-" * 80)

    for image in ["kodim16", "kodim20"]:
        image_data = [r for r in results if r["image"] == image]
        for result in sorted(image_data, key=lambda x: x["size"]):
            size = result["size"]
            qslst = result["qslst_fft"]
            ns = result["ns_fft"]

            print(
                f"{image:<10} {size:<6} {'QSLST-FFT':<12} {qslst['time']:<8.3f} {qslst['psnr']:<10.2f} {qslst['ssim']:<8.3f}"
            )
            print(
                f"{'':<10} {size:<6} {'FFT-NS-Q':<12} {ns['time']:<8.3f} {ns['psnr']:<10.2f} {ns['ssim']:<8.3f}"
            )
            print("-" * 80)


def main():
    """Main visualization function"""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    json_file = os.path.join(repo_root, "output_figures", "deblur_benchmark_results.json")

    if not os.path.exists(json_file):
        print(f"❌ JSON file not found: {json_file}")
        print(
            "Please run the benchmark first: python applications/image_deblurring/run_deblur_benchmark.py"
        )
        return

    print("📊 Loading benchmark results...")
    results = load_results(json_file)

    print(f"✅ Loaded {len(results)} benchmark results")

    # Create summary table
    create_summary_table(results)

    # Create performance plots
    print("\n🎨 Creating performance plots...")
    fig = create_performance_plots(results)

    # Save plots
    plot_file = os.path.join(repo_root, "output_figures", "deblur_performance_analysis.png")
    fig.savefig(plot_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"✅ Performance plots saved: {plot_file}")
    # Simple computed summary (avoid hard-coded claims).
    print("\n📈 Summary (computed):")
    for image in ["kodim16", "kodim20"]:
        data = [r for r in results if r.get("image") == image]
        if not data:
            continue
        sizes = sorted({int(r["size"]) for r in data})
        q_psnr = {int(r["size"]): float(r["qslst_fft"]["psnr"]) for r in data}
        n_psnr = {int(r["size"]): float(r["ns_fft"]["psnr"]) for r in data}
        q_t = {int(r["size"]): float(r["qslst_fft"]["time"]) for r in data}
        n_t = {int(r["size"]): float(r["ns_fft"]["time"]) for r in data}

        deltas_psnr = [q_psnr[s] - n_psnr[s] for s in sizes if s in q_psnr and s in n_psnr]
        ratios_time = [q_t[s] / (n_t[s] + 1e-30) for s in sizes if s in q_t and s in n_t]
        if deltas_psnr:
            print(
                f"  - {image}: ΔPSNR(QSLST-FFT − FFT-NS-Q): mean={sum(deltas_psnr)/len(deltas_psnr):.2f} dB "
                f"(min={min(deltas_psnr):.2f}, max={max(deltas_psnr):.2f})"
            )
        if ratios_time:
            print(
                f"  - {image}: time ratio (QSLST-FFT / FFT-NS-Q): mean={sum(ratios_time)/len(ratios_time):.2f} "
                f"(min={min(ratios_time):.2f}, max={max(ratios_time):.2f})"
            )


if __name__ == "__main__":
    main()
