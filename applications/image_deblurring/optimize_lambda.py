#!/usr/bin/env python3
"""
Lambda Optimization for Image Deblurring

This script performs a line search to find the optimal lambda value for each
image and size combination, using SNR=30dB.
"""

import os
import sys
import subprocess
import time
import json
import numpy as np
from pathlib import Path

def run_single_experiment(image_name, size, snr, lam, ns_iters=12):
    """Run a single deblurring experiment and return results"""
    print(f"  Testing λ={lam:.3f}...", end=" ", flush=True)
    
    # Build command
    cmd = [
        sys.executable, 
        "applications/image_deblurring/script_image_deblurring.py",
        "--image", image_name,
        "--size", str(size),
        "--lam", str(lam),
        "--snr", str(snr),
        "--ns_mode", "fftT",
        "--ns_iters", str(ns_iters),
        "--fftT_order", "2"
    ]
    
    # Run the experiment
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    
    if result.returncode != 0:
        print("❌ FAILED")
        return None
    
    # Parse results from output
    output_lines = result.stdout.split('\n')
    results = {}
    
    for line in output_lines:
        if 'QSLST (FFT):' in line:
            try:
                # Extract PSNR value
                psnr_start = line.find('PSNR=') + 5
                psnr_end = line.find('dB', psnr_start)
                psnr = float(line[psnr_start:psnr_end])
                
                # Extract SSIM value
                ssim_start = line.find('SSIM=') + 5
                ssim_end = line.find(' ', ssim_start)
                ssim = float(line[ssim_start:ssim_end])
                
                # Extract time value
                time_start = line.find('time=') + 5
                time_end = line.find('s', time_start)
                time_val = float(line[time_start:time_end])
                
                results['qslst_fft'] = {
                    'psnr': psnr,
                    'ssim': ssim,
                    'time': time_val
                }
            except (ValueError, IndexError) as e:
                print("❌ PARSE ERROR")
                return None
        elif 'NS (T^-1, FFT, order-2):' in line:
            try:
                # Extract PSNR value
                psnr_start = line.find('PSNR=') + 5
                psnr_end = line.find('dB', psnr_start)
                psnr = float(line[psnr_start:psnr_end])
                
                # Extract SSIM value
                ssim_start = line.find('SSIM=') + 5
                ssim_end = line.find(' ', ssim_start)
                ssim = float(line[ssim_start:ssim_end])
                
                # Extract time value
                time_start = line.find('time=') + 5
                time_end = line.find('s', time_start)
                time_val = float(line[time_start:time_end])
                
                results['ns_fft'] = {
                    'psnr': psnr,
                    'ssim': ssim,
                    'time': time_val
                }
            except (ValueError, IndexError) as e:
                print("❌ PARSE ERROR")
                return None
    
    if 'qslst_fft' in results and 'ns_fft' in results:
        print(f"✅ PSNR={results['qslst_fft']['psnr']:.2f}dB, SSIM={results['qslst_fft']['ssim']:.3f}")
        return results
    else:
        print("❌ MISSING RESULTS")
        return None

def optimize_lambda_for_image_size(image_name, size, snr=30, ns_iters=12):
    """Find optimal lambda for a specific image and size"""
    print(f"\n🔍 Optimizing λ for {image_name} at size {size}x{size}")
    print("=" * 60)
    
    # Define lambda search range (logarithmic scale)
    # Start with a reasonable range based on typical values
    lambda_values = [
        0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0
    ]
    
    best_lambda = None
    best_psnr = -float('inf')
    best_ssim = -float('inf')
    best_results = None
    all_results = []
    
    print(f"Testing {len(lambda_values)} λ values: {lambda_values}")
    print("-" * 60)
    
    for lam in lambda_values:
        results = run_single_experiment(image_name, size, snr, lam, ns_iters)
        
        if results is not None:
            psnr = results['qslst_fft']['psnr']
            ssim = results['qslst_fft']['ssim']
            
            all_results.append({
                'lambda': lam,
                'psnr': psnr,
                'ssim': ssim,
                'time': results['qslst_fft']['time']
            })
            
            # Update best if this is better (prioritize PSNR, then SSIM)
            if psnr > best_psnr or (psnr == best_psnr and ssim > best_ssim):
                best_lambda = lam
                best_psnr = psnr
                best_ssim = ssim
                best_results = results.copy()
                best_results['lambda'] = lam
    
    if best_lambda is not None:
        print(f"\n🏆 Best λ = {best_lambda:.3f}")
        print(f"   PSNR = {best_psnr:.2f} dB")
        print(f"   SSIM = {best_ssim:.3f}")
        print(f"   Time = {best_results['qslst_fft']['time']:.3f} s")
        
        # Create optimization plot
        create_lambda_optimization_plot(image_name, size, all_results, best_lambda)
        
        return {
            'image': image_name,
            'size': size,
            'best_lambda': best_lambda,
            'best_psnr': best_psnr,
            'best_ssim': best_ssim,
            'best_time': best_results['qslst_fft']['time'],
            'all_results': all_results,
            'qslst_results': best_results['qslst_fft'],
            'ns_results': best_results['ns_fft']
        }
    else:
        print("❌ No valid results found")
        return None

def create_lambda_optimization_plot(image_name, size, all_results, best_lambda):
    """Create a plot showing lambda optimization results"""
    try:
        import matplotlib.pyplot as plt
        
        lambdas = [r['lambda'] for r in all_results]
        psnrs = [r['psnr'] for r in all_results]
        ssims = [r['ssim'] for r in all_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # PSNR plot
        ax1.semilogx(lambdas, psnrs, 'b-o', linewidth=2, markersize=6)
        ax1.axvline(x=best_lambda, color='r', linestyle='--', alpha=0.7, label=f'Best λ={best_lambda:.3f}')
        ax1.set_xlabel('λ (Regularization Parameter)', fontsize=12)
        ax1.set_ylabel('PSNR (dB)', fontsize=12)
        ax1.set_title(f'PSNR vs λ - {image_name} ({size}×{size})', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # SSIM plot
        ax2.semilogx(lambdas, ssims, 'g-s', linewidth=2, markersize=6)
        ax2.axvline(x=best_lambda, color='r', linestyle='--', alpha=0.7, label=f'Best λ={best_lambda:.3f}')
        ax2.set_xlabel('λ (Regularization Parameter)', fontsize=12)
        ax2.set_ylabel('SSIM', fontsize=12)
        ax2.set_title(f'SSIM vs λ - {image_name} ({size}×{size})', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        output_dir = "output_figures"
        os.makedirs(output_dir, exist_ok=True)
        plot_file = f"{output_dir}/lambda_optimization_{image_name}_{size}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
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
    print("\\caption{Image deblurring: FFT--NS--Q vs.\\ QSLST--FFT with optimized $\\lambda$ on $N\\times N$ subimages from Kodak images (SNR = 30 dB).}")
    print("\\label{tab:deblur-results-optimized}")
    print("\\begin{tabular}{lccccc}")
    print("\\hline")
    print("$N$ & $\\lambda^*$ & Method & CPU time (s) & PSNR (dB) & SSIM \\\\")
    print("\\hline")
    
    # Group results by image
    for image in ['kodim16', 'kodim20']:
        print(f"\\multicolumn{{6}}{{l}}{{\\textit{{{image}}}}} \\\\")
        image_results = [r for r in all_optimizations if r['image'] == image]
        
        for result in sorted(image_results, key=lambda x: x['size']):
            size = result['size']
            best_lambda = result['best_lambda']
            qslst = result['qslst_results']
            ns = result['ns_results']
            
            print(f"{size} & {best_lambda:.3f} & QSLST--FFT & {qslst['time']:6.3f} & {qslst['psnr']:6.2f} & {qslst['ssim']:5.3f} \\\\")
            print(f"    &        & FFT--NS--Q & {ns['time']:6.3f} & {ns['psnr']:6.2f} & {ns['ssim']:5.3f} \\\\")
            print("\\hline")
    
    print("\\end{tabular}")
    print("\\end{table}")
    print("=" * 80)

def main():
    """Main optimization function"""
    print("🚀 Starting Lambda Optimization for Image Deblurring")
    print("=" * 80)
    print("Parameters:")
    print("  - Images: kodim16, kodim20")
    print("  - Sizes: 32, 64, 128, 256, 400, 512")
    print("  - SNR: 30 dB")
    print("  - Lambda range: 0.001 to 10.0 (logarithmic)")
    print("  - NS iterations: 12")
    print("=" * 80)
    
    # Create output directory
    os.makedirs("output_figures", exist_ok=True)
    
    # Configuration
    images = ['kodim16', 'kodim20']
    sizes = [32, 64, 128, 256, 400, 512]
    snr = 30
    ns_iters = 12
    
    all_optimizations = []
    
    # Run optimization for each image and size
    for image in images:
        for size in sizes:
            result = optimize_lambda_for_image_size(image, size, snr, ns_iters)
            if result:
                all_optimizations.append(result)
    
    # Save results to JSON
    results_file = "output_figures/lambda_optimization_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_optimizations, f, indent=2)
    
    # Generate optimized LaTeX table
    generate_optimized_latex_table(all_optimizations)
    
    # Print summary
    print(f"\n🎉 Lambda optimization completed!")
    print(f"📁 Results saved to: output_figures/")
    print(f"📊 JSON results: {results_file}")
    print(f"📋 Optimized LaTeX table generated above")
    print(f"📈 Optimization plots: output_figures/lambda_optimization_*.png")
    
    # Print best lambda summary
    print(f"\n📊 Best Lambda Summary:")
    print("-" * 50)
    for image in images:
        print(f"\n{image}:")
        image_results = [r for r in all_optimizations if r['image'] == image]
        for result in sorted(image_results, key=lambda x: x['size']):
            print(f"  N={result['size']:3d}: λ={result['best_lambda']:.3f} → PSNR={result['best_psnr']:.2f}dB, SSIM={result['best_ssim']:.3f}")

if __name__ == '__main__':
    main()
