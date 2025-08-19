#!/usr/bin/env python3
"""
Main runner script for Quaternion Linear Algebra Analysis

This script allows users to run various analysis scripts from the main directory.
Usage examples:
    python run_analysis.py cifar10
    python run_analysis.py pseudoinverse
    python run_analysis.py multiple_images
    python run_analysis.py test_newton
"""

import os
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_analysis.py <script_name>")
        print("\nAvailable scripts:")
        print(
            "  tutorial - Quaternion basics tutorial with visualizations (recommended to start here)"
        )
        print("  demo - Core functionality demo (alternative comprehensive overview)")
        print("  qgmres - Q-GMRES solver test (basic functionality)")
        print(
            "  qgmres_bench - Q-GMRES comprehensive performance benchmark with LU preconditioning (new!)"
        )
        print("  lorenz_signal - Lorenz attractor signal processing with Q-GMRES")
        print(
            "                Note: Use --num_points <N> to control resolution/execution time"
        )
        print(
            "                Examples: --num_points 100 (fast), --num_points 500 (high quality)"
        )
        print("  lorenz_benchmark - Lorenz attractor method comparison benchmark")
        print(
            "                    Compares Q-GMRES vs Newton-Schulz performance and accuracy"
        )
        print("  cifar10         - CIFAR-10 pseudoinverse analysis")
        print("  pseudoinverse   - Single image pseudoinverse analysis")
        print("  multiple_images - Multiple images pseudoinverse analysis")
        # print("  test_newton     - Test deep linear solver")  # Disabled - not working properly
        print("  image_completion - Real image completion example")
        print("  image_deblurring - Quaternion image deblurring (QSLST vs NS/HON)")
        print(
            "                    Usage: image_deblurring [--size N] [--lam LAMBDA] [--snr SNR_DB]"
        )
        print(
            "                           Optional NS args: --ns_mode {dense,sparse,fftT,tikhonov_aug}"
        )
        print(
            "                                            --ns_iters K  --fftT_order {2,3}"
        )
        print("                    Recommended:")
        print(
            "                      image_deblurring --size 64 --lam 1e-3 --snr 40 --ns_mode fftT --fftT_order 3 --ns_iters 12"
        )
        print(
            "  deblur_benchmark - Comprehensive image deblurring benchmark with LaTeX table generation"
        )
        print(
            "                     Runs multiple sizes (32,64,128,256,400,512) on kodim16/kodim20 with optimized lambdas"
        )
        print(
            "                     Generates performance plots and publication-ready LaTeX tables"
        )
        print("  synthetic       - Synthetic image completion (controlled experiments)")
        print("  synthetic_matrices - Test pseudoinverse on synthetic matrices")
        print(
            "  eigenvalue_test - Eigenvalue decomposition test (tridiagonalization and eigendecomposition)"
        )
        print(
            "  ns_compare      - Compare NS vs Higher-Order NS (saves plots to output_figures)"
        )
        print(
            "  schur_demo      - Quaternion Schur decomposition demo with comprehensive comparison"
        )
        print("                    Usage: schur_demo [matrix_size] (default: 10)")
        print(
            "                    Examples: schur_demo 10 (fast), schur_demo 25 (comprehensive)"
        )
        print(
            "  jupyter_test    - Test Jupyter notebook setup and verify environment configuration"
        )
        print(
            "  pinv_bench      - Benchmark pseudoinverse methods (NS, HON, RSP-Q, Hybrid, CGNE–Q)"
        )
        return

    script_name = sys.argv[1]

    # Map script names to file paths
    script_map = {
        "tutorial": "tests/tutorial_quaternion_basics.py",
        "demo": "QuatIca_Core_Functionality_Demo.py",
        "qgmres": "tests/QGMRES/test_qgmres_solver.py",
        "qgmres_bench": "tests/QGMRES/benchmark_qgmres_preconditioner.py",
        "lorenz_signal": "applications/signal_processing/lorenz_attractor_qgmres.py",
        "lorenz_benchmark": "applications/signal_processing/benchmark_lorenz_methods.py",
        "cifar10": "tests/pseudoinverse/analyze_cifar10_pseudoinverse.py",
        "pseudoinverse": "tests/pseudoinverse/analyze_pseudoinverse.py",
        "multiple_images": "tests/pseudoinverse/analyze_multiple_images_pseudoinverse.py",
        "test_newton": "tests/unit/test_simple_newton.py",  # Disabled - not working properly
        "image_completion": "applications/image_completion/script_real_image_completion.py",
        "image_deblurring": "applications/image_deblurring/script_image_deblurring.py",
        "deblur_benchmark": "applications/image_deblurring/run_deblur_benchmark.py",
        "synthetic": "applications/image_completion/script_synthetic_image_completion.py",  # Matrix completion on synthetic images
        "synthetic_matrices": "tests/pseudoinverse/script_synthetic_matrices.py",  # Pseudoinverse test on synthetic matrices
        "eigenvalue_test": "tests/decomp/eigenvalue_demo.py",  # Eigenvalue decomposition test
        "ns_compare": "tests/unit/test_ns_vs_higher_order_compare.py",  # NS vs Higher-Order NS comparison
        "schur_demo": "tests/schur_demo.py",  # Quaternion Schur decomposition demo
        "jupyter_test": "tests/test_jupyter_setup.py",  # Jupyter setup verification
        "pinv_bench": "tests/unit/benchmark_pseudoinverse_methods.py",  # Pseudoinverse benchmark
    }

    if script_name not in script_map:
        print(f"Unknown script: {script_name}")
        print("Available scripts:", list(script_map.keys()))
        return

    # Check if script is disabled
    if script_name == "test_newton":
        print(
            "❌ test_newton is currently disabled (deep linear solver not working properly)"
        )
        print("The code is still available in quatica/solver.py for future development")
        return

    script_path = script_map[script_name]

    if not os.path.exists(script_path):
        print(f"Script not found: {script_path}")
        return

    print(f"Running: {script_path}")
    print("=" * 50)

    # Run the script
    try:
        # Special handling for files in root directory
        if script_name in ["demo", "tutorial"]:
            script_dir = "."  # Root directory
            cmd = [sys.executable, script_path]
        elif script_name == "jupyter_test":
            script_dir = "."  # Root directory for jupyter_test
            cmd = [sys.executable, script_path]
        else:
            # Change to the script's directory for proper relative path handling
            script_dir = os.path.dirname(script_path)
            cmd = [sys.executable, os.path.basename(script_path)]

        # Pass through additional arguments for specific scripts
        if script_name == "lorenz_signal" and len(sys.argv) > 2:
            # Pass all remaining arguments to the Lorenz script
            cmd.extend(sys.argv[2:])
            print(f"Additional arguments passed to Lorenz script: {sys.argv[2:]}")
        elif script_name == "schur_demo" and len(sys.argv) > 2:
            # Pass matrix size argument to the Schur demo script
            cmd.extend(sys.argv[2:])
            print(f"Matrix size argument passed to Schur demo: {sys.argv[2:]}")
        elif script_name == "image_deblurring" and len(sys.argv) > 2:
            # Pass optional args to image deblurring script (e.g., --size, --lam, --snr)
            cmd.extend(sys.argv[2:])
            print(f"Additional arguments passed to image_deblurring: {sys.argv[2:]}")
        elif script_name == "deblur_benchmark" and len(sys.argv) > 2:
            # Pass optional args to deblur benchmark script
            cmd.extend(sys.argv[2:])
            print(f"Additional arguments passed to deblur_benchmark: {sys.argv[2:]}")

        subprocess.run(cmd, cwd=script_dir, check=True)
        print("\n" + "=" * 50)
        print("Script completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nScript failed with exit code: {e.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
