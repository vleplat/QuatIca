#!/usr/bin/env python3
"""
Image Deblurring Benchmark Script for Report

This script runs the image deblurring experiments according to the report requirements:
- Compare FFT-NS-Q vs QSLST-FFT on kodim16 and kodim20
- Test sizes: 32, 64, 128
- Blur kernel: Gaussian (radius=2, sigma=1.0)
- Noise: 40 dB SNR
- Regularization: λ=0.1
- Generate side-by-side comparison plots for N=128
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path

def run_deblur_experiment(image_name, size, snr=40, lam=0.1, ns_iters=12):
    """Run a single deblurring experiment"""
    print(f"\n{'='*60}")
    print(f"Running experiment: {image_name} at size {size}x{size}")
    print(f"Parameters: SNR={snr}dB, λ={lam}, NS_iters={ns_iters}")
    print(f"{'='*60}")
    
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
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    end_time = time.time()
    
    if result.returncode != 0:
        print(f"❌ Experiment failed for {image_name} size {size}")
        print(f"Error: {result.stderr}")
        return None
    
    print(f"✅ Experiment completed in {end_time - start_time:.2f}s")
    
    # Parse results from output
    output_lines = result.stdout.split('\n')
    results = {}
    
    for line in output_lines:
        if 'QSLST (FFT):' in line:
            # Parse: QSLST (FFT):    PSNR=XX.XXdB  SSIM=X.XXX  RelErr=X.XXXe-X  time=X.XXXs
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
                print(f"Warning: Could not parse QSLST line: {line}")
                print(f"Error: {e}")
        elif 'NS (T^-1, FFT, order-2):' in line:
            # Parse: NS (T^{-1}, FFT, order-2):        PSNR=XX.XXdB    SSIM=X.XXX    RelErr=X.XXXe-X    time=X.XXXs
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
                print(f"Warning: Could not parse NS line: {line}")
                print(f"Error: {e}")
    
    results['image'] = image_name
    results['size'] = size
    results['snr'] = snr
    results['lam'] = lam
    results['ns_iters'] = ns_iters
    
    return results

def create_comparison_plot(image_name, size=128):
    """Create side-by-side comparison plot for the specified image and size"""
    print(f"\n🎨 Creating comparison plot for {image_name} at size {size}x{size}")
    
    # Expected output files with unique names
    output_dir = "output_figures"
    base_name = f"{image_name}_{size}"
    clean_file = f"{output_dir}/deblur_input_clean_{base_name}.png"
    observed_file = f"{output_dir}/deblur_observed_blur_noise_{40}dB_{base_name}.png"
    qslst_file = f"{output_dir}/deblur_qslst_fft_{base_name}.png"
    ns_file = f"{output_dir}/deblur_ns_{base_name}.png"
    
    # Check if files exist
    required_files = [clean_file, observed_file, qslst_file, ns_file]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    # Create comparison plot
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Load images
    clean_img = mpimg.imread(clean_file)
    observed_img = mpimg.imread(observed_file)
    qslst_img = mpimg.imread(qslst_file)
    ns_img = mpimg.imread(ns_file)
    
    # Plot images
    axes[0].imshow(clean_img)
    axes[0].set_title('Clean Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(observed_img)
    axes[1].set_title('Noisy + Blurred\n(40 dB SNR)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(qslst_img)
    axes[2].set_title('QSLST-FFT\nRecovery', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    axes[3].imshow(ns_img)
    axes[3].set_title('FFT-NS-Q\nRecovery', fontsize=14, fontweight='bold')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{output_dir}/deblur_comparison_{image_name}_{size}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Comparison plot saved: {plot_filename}")
    return True

def generate_latex_table(results):
    """Generate LaTeX table from benchmark results"""
    print("\n📋 Generating LaTeX table for paper...")
    print("=" * 80)
    
    print("\\begin{table}[ht!]")
    print("\\centering")
    print("\\caption{Image deblurring: FFT--NS--Q vs.\\ QSLST--FFT on $N\\times N$ subimages from Kodak images.}")
    print("\\label{tab:deblur-results}")
    print("\\begin{tabular}{lccccc}")
    print("\\hline")
    print("$N$ & $\\lambda_{\\text{opt}}$ & Method & CPU time (s) & PSNR (dB) & SSIM \\\\")
    print("\\hline")
    
    # Group results by image and size
    for image in ['kodim16', 'kodim20']:
        print(f"\\multicolumn{{6}}{{l}}{{\\textit{{{image}}}}} \\\\")
        for size in [32, 64, 128, 256, 400, 512]:
            # Find results for this image and size
            result = None
            for r in results:
                if r['image'] == image and r['size'] == size:
                    result = r
                    break
            
            if result:
                qslst = result['qslst_fft']
                ns = result['ns_fft']
                lam = result['lam']
                
                print(f"{size} & {lam:5.3f} & QSLST--FFT & {qslst['time']:6.3f} & {qslst['psnr']:6.2f} & {qslst['ssim']:5.3f} \\\\")
                print(f"    &         & FFT--NS--Q & {ns['time']:6.3f} & {ns['psnr']:6.2f} & {ns['ssim']:5.3f} \\\\")
                print("\\hline")
    
    print("\\end{tabular}")
    print("\\end{table}")
    print("=" * 80)

def main():
    """Main benchmark execution"""
    print("🚀 Starting Image Deblurring Benchmark for Report")
    print("=" * 80)
    print("Parameters:")
    print("  - Images: kodim16, kodim20")
    print("  - Sizes: 32, 64, 128, 256, 400, 512")
    print("  - Blur: Gaussian (radius=2, sigma=1.0)")
    print("  - Noise: 30 dB SNR")
    print("  - Regularization: Optimized λ per image/size")
    print("  - NS iterations: 12")
    print("=" * 80)
    
    # Create output directory
    os.makedirs("output_figures", exist_ok=True)
    
    # Experiment configuration with optimized lambda values
    images = ['kodim16', 'kodim20']
    sizes = [32, 64, 128, 256, 400, 512]
    snr = 30
    ns_iters = 12
    
    # Optimized lambda values from lambda optimization
    optimized_lambdas = {
        'kodim16': {32: 0.020, 64: 0.050, 128: 0.050, 256: 0.050, 400: 0.050, 512: 0.050},
        'kodim20': {32: 0.020, 64: 0.020, 128: 0.020, 256: 0.050, 400: 0.050, 512: 0.050}
    }
    
    all_results = []
    
    # Run experiments with optimized lambda values
    for image in images:
        for size in sizes:
            lam = optimized_lambdas[image][size]
            result = run_deblur_experiment(image, size, snr, lam, ns_iters)
            if result:
                all_results.append(result)
    
    # Create comparison plots for N=128
    for image in images:
        create_comparison_plot(image, 128)
    
    # Generate LaTeX table
    generate_latex_table(all_results)
    
    # Save results to JSON
    results_file = "output_figures/deblur_benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n🎉 Benchmark completed successfully!")
    print(f"📁 Results saved to: output_figures/")
    print(f"📊 JSON results: {results_file}")
    print(f"📋 LaTeX table generated above")
    print(f"🎨 Comparison plots: output_figures/deblur_comparison_*.png")

if __name__ == '__main__':
    main()
