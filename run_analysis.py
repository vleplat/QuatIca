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

import sys
import os
import subprocess

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_analysis.py <script_name>")
        print("\nAvailable scripts:")
        print("  tutorial - Quaternion basics tutorial (recommended to start here)")
        print("  qgmres - Q-GMRES solver test (new!)")
        print("  lorenz_signal - Lorenz attractor signal processing with Q-GMRES (new!)")
        print("                Note: Use --num_points <N> to control resolution/execution time")
        print("                Examples: --num_points 100 (fast), --num_points 500 (high quality)")
        print("  lorenz_benchmark - Lorenz attractor method comparison benchmark (new!)")
        print("                    Compares Q-GMRES vs Newton-Schulz performance and accuracy")
        print("  cifar10         - CIFAR-10 pseudoinverse analysis")
        print("  pseudoinverse   - Single image pseudoinverse analysis")
        print("  multiple_images - Multiple images pseudoinverse analysis")
        # print("  test_newton     - Test deep linear solver")  # Disabled - not working properly
        print("  image_completion - Real image completion example")
        print("  synthetic       - Synthetic image completion (controlled experiments)")
        print("  synthetic_matrices - Test pseudoinverse on synthetic matrices")
        print("  eigenvalue_test - Eigenvalue decomposition test (tridiagonalization and eigendecomposition)")
        return
    
    script_name = sys.argv[1]
    
    # Map script names to file paths
    script_map = {
        'tutorial': 'tests/unit/tutorial_quaternion_basics.py',
        'qgmres': 'tests/QGMRES/test_qgmres_solver.py',
        'lorenz_signal': 'applications/signal_processing/lorenz_attractor_qgmres.py',
        'lorenz_benchmark': 'applications/signal_processing/benchmark_lorenz_methods.py',
        'cifar10': 'tests/pseudoinverse/analyze_cifar10_pseudoinverse.py',
        'pseudoinverse': 'tests/pseudoinverse/analyze_pseudoinverse.py',
        'multiple_images': 'tests/pseudoinverse/analyze_multiple_images_pseudoinverse.py',
        'test_newton': 'tests/unit/test_simple_newton.py',  # Disabled - not working properly
        'image_completion': 'applications/image_completion/script_real_image_completion.py',
        'synthetic': 'applications/image_completion/script_synthetic_image_completion.py',  # Matrix completion on synthetic images
        'synthetic_matrices': 'tests/pseudoinverse/script_synthetic_matrices.py',  # Pseudoinverse test on synthetic matrices
        'eigenvalue_test': 'tests/decomp/eigenvalue_demo.py'  # Eigenvalue decomposition test
    }
    
    if script_name not in script_map:
        print(f"Unknown script: {script_name}")
        print("Available scripts:", list(script_map.keys()))
        return
    
    # Check if script is disabled
    if script_name == 'test_newton':
        print("❌ test_newton is currently disabled (deep linear solver not working properly)")
        print("The code is still available in core/solver.py for future development")
        return
    
    script_path = script_map[script_name]
    
    if not os.path.exists(script_path):
        print(f"Script not found: {script_path}")
        return
    
    print(f"Running: {script_path}")
    print("="*50)
    
    # Run the script
    try:
        # Change to the script's directory for proper relative path handling
        script_dir = os.path.dirname(script_path)
        
        # Build command with arguments
        cmd = [sys.executable, os.path.basename(script_path)]
        
        # Pass through additional arguments for specific scripts
        if script_name == 'lorenz_signal' and len(sys.argv) > 2:
            # Pass all remaining arguments to the Lorenz script
            cmd.extend(sys.argv[2:])
            print(f"Additional arguments passed to Lorenz script: {sys.argv[2:]}")
        
        result = subprocess.run(cmd, cwd=script_dir, check=True)
        print("\n" + "="*50)
        print("Script completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nScript failed with exit code: {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main() 