# Image Deblurring with Quaternion Methods

Comprehensive guide to quaternion-based image deblurring using QSLST and Newton-Schulz methods.

## 🎯 Overview

QuatIca provides state-of-the-art quaternion methods for image deblurring, comparing:

- **QSLST-FFT**: Fast FFT-based Tikhonov regularization
- **Newton-Schulz (NS)**: Iterative baselines (FFT inverse-iteration on \(T=|\hat h|^2+\lambda\), augmented system, or explicit blur-matrix pseudoinverse)
- **QSLST-Matrix** *(optional)*: Explicit matrix-based Tikhonov reference for small sizes
- **Higher-Order NS (HON)** *(optional)*: Higher-order pseudoinverse baseline (only in some modes)

## 🚀 Quick Start

### Basic Deblurring
```bash
# Default parameters (32×32, fast)
python run_analysis.py image_deblurring

# High quality (64×64, recommended)
python run_analysis.py image_deblurring --size 64 --lam 1e-1 --snr 40 --ns_mode fftT --fftT_order 3 --ns_iters 12
```

### Parameter Options
```bash
# Image size
--size 32          # 32×32 grid (fast)
--size 64          # 64×64 grid (recommended)
--size 128         # 128×128 grid (high quality)

# Regularization
--lam 1e-3         # Tikhonov parameter (default: 1e-3)
--lam 1e-1         # Stronger regularization
--lam 1e-5         # Lighter regularization

# Noise level (optional; omit --snr for a blur-only observation)
--snr 30           # 30 dB signal-to-noise ratio
--snr 40           # 40 dB SNR
--snr 50           # 50 dB SNR (less noise)

# Blur kernel (Gaussian PSF)
--psf_radius 4      # radius r (kernel size = (2r+1)×(2r+1); default 4 → 9×9)
--psf_sigma 1.0     # sigma (increase for stronger blur; default 1.0)

# Newton-Schulz options
--ns_mode fftT     # FFT-based (recommended)
--ns_mode dense    # Dense matrix method
--ns_mode sparse   # Sparse matrix method
--ns_mode tikhonov_aug  # Augmented Tikhonov system (exact but slow for dense)

# FFT solver settings
--fftT_order 2     # Quadratic convergence (Newton-Schulz)
--fftT_order 3     # Cubic convergence (Higher-Order NS)
--ns_iters 12      # Number of iterations

# Validation / reproducibility (direct script)
--run_qslst_matrix  # compute explicit matrix QSLST reference (slower)
--validate_bccb     # check BCCB matrix vs FFT blur convention
--metrics_json PATH # write machine-readable metrics (for wrappers)
```

## 🔬 Algorithm Details

### Problem Formulation

**Image Deblurring as Linear System:**
- **Clean image**: x (unknown)
- **Observed image**: b (blurred + noise)
- **Blur operator**: A (convolution matrix)
- **Goal**: Solve Ax = b for x

**Tikhonov Regularization:**
```
minimize ||Ax - b||² + λ||x||²
```

Solution: `x_λ = (A^T A + λI)^(-1) A^T b`

### Method Comparison

| Method | Speed | Accuracy | Memory | Best For |
|--------|-------|----------|---------|----------|
| **QSLST-FFT** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **Production use** |
| **QSLST-Matrix** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Small images, validation |
| **NS (fftT)** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | **General purpose** |
| **HON (fftT)** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | **High accuracy** |

### QSLST-FFT Algorithm

**Why it's fast**: FFT diagonalizes the circulant blur matrix A^T A

```python
# Frequency domain solution
for each frequency (u,v):
    X̂[u,v] = conj(Ĥ[u,v]) * B̂[u,v] / (|Ĥ[u,v]|² + λ)
```

**Complexity**: O(N log N) vs O(N³) for direct methods

### Newton-Schulz FFT Methods

**Inverse Newton-Schulz**: Solves T^(-1) where T = A^T A + λI

```python
# Iteration: Y ← Y(2I - TY) (order-2)
# Iteration: Y ← Y(I + R + R²), R = I - TY (order-3)
```

**Per-frequency implementation**: Parallelizable across frequency bins

## 📊 Performance Analysis

### Execution Time (64×64 image)

| Method | Time | Relative Speed |
|--------|------|----------------|
| QSLST-FFT | ~0.1s | 1× (baseline) |
| NS (fftT, order-2) | ~0.3s | 3× |
| HON (fftT, order-3) | ~0.5s | 5× |
| QSLST-Matrix | ~2.5s | 25× |

### Quality Metrics

**PSNR (Peak Signal-to-Noise Ratio)**: Higher is better
- **Excellent**: >40 dB
- **Good**: 30-40 dB
- **Acceptable**: 20-30 dB

**SSIM (Structural Similarity)**: Higher is better (0-1 range)
- **Excellent**: >0.95
- **Good**: 0.85-0.95
- **Acceptable**: 0.70-0.85

## 🎛️ Parameter Tuning Guide

### Regularization Parameter (λ)

**Effect of λ**:
- **Too small** (λ < 1e-5): Amplifies noise, overfitting
- **Optimal** (λ ≈ 1e-3): Balances smoothing and detail preservation
- **Too large** (λ > 1e-1): Over-smoothing, loss of details

**Tuning strategy**:
```bash
# Test different values
python run_analysis.py image_deblurring --lam 1e-5  # Less smoothing
python run_analysis.py image_deblurring --lam 1e-3  # Balanced (recommended)
python run_analysis.py image_deblurring --lam 1e-1  # More smoothing
```

### Newton-Schulz Iterations

**Convergence behavior**:
- **Order-2**: Linear convergence, needs ~12-20 iterations
- **Order-3**: Cubic convergence, needs ~8-12 iterations

**Optimal settings**:
```bash
# Balanced accuracy/speed
--ns_mode fftT --fftT_order 3 --ns_iters 12

# Maximum accuracy
--ns_mode fftT --fftT_order 3 --ns_iters 20

# Fast testing
--ns_mode fftT --fftT_order 2 --ns_iters 10
```

## 🖼️ Output Analysis

### Generated Files (in `output_figures/`)

For a single run (e.g. `python run_analysis.py image_deblurring ...`), outputs include:

1. **`deblur_input_clean_<image>_<N>.png`**: clean resized RGB input
2. **`deblur_blurred_<image>_<N>.png`**: blur-only (before noise injection)
3. **`deblur_observed_blurred_<image>_<N>.png`** or **`deblur_observed_blur_noise_<SNR>dB_<image>_<N>.png`**: observed image
4. **`deblur_qslst_fft_<image>_<N>.png`**: QSLST-FFT reconstruction
5. **`deblur_ns_<mode>_<image>_<N>.png`**: baseline reconstruction (mode-dependent)
6. **`deblur_comparison_grid_<image>_<N>.png`**: 2×2 comparison grid (Clean / Observed / QSLST-FFT / baseline)

### Reading the Results

**Comparison Grid Layout**:
```
[Clean]    [Observed]
[QSLST-FFT] [baseline]
```

**Metrics Table**:
- **PSNR**: Image quality (higher = better)
- **SSIM**: Structural similarity (closer to 1 = better)
- **Time**: Processing time (lower = faster)

## 🔧 Advanced Usage

### Custom Blur Kernels

The demo uses a Gaussian PSF by default. You can increase blur strength from the CLI:

```bash
python run_analysis.py image_deblurring --size 64 --lam 1e-1 --snr 40 --psf_radius 6 --psf_sigma 2.0
```

If you want a different PSF family (e.g. motion blur), the PSF builders live in `quatica/qslst.py`
(`build_psf_gaussian`, `build_psf_motion`). You can adapt the demo to call them, keeping the same
FFT-based blur and restoration pipeline.

### Batch Processing
```bash
# Process multiple images
for size in 32 64 128; do
    python applications/image_deblurring/script_image_deblurring.py \
        --size $size --lam 1e-3 --snr 40
done
```

### Integration with Other Tools

```python
# Use QuatIca deblurring in your pipeline
from quatica.qslst import qslst_restore_fft
from quatica.utils import rgb_to_quat, quat_to_rgb

# Your image processing pipeline
def deblur_image(rgb_image, blur_kernel, lambda_reg=1e-3):
    quat_image = rgb_to_quat(rgb_image)
    restored_quat = qslst_restore_fft(quat_image, blur_kernel, lambda_reg)
    return quat_to_rgb(restored_quat)
```

## 📈 Benchmark Results

Benchmark results depend on the blur/noise/regularization regime. For paper-ready aggregated results,
use the benchmark wrapper and the visualization script:

```bash
python run_analysis.py deblur_benchmark
python applications/image_deblurring/visualize_results.py
```

This produces:
- `output_figures/deblur_benchmark_results.json`
- `output_figures/deblur_comparison_<image>_<N>.png` (side-by-side panels at the chosen comparison size)
- `output_figures/deblur_performance_analysis.png` (runtime/PSNR/SSIM curves)

## 🎯 Recommendations

### For Production Use
```bash
# Recommended settings for production
python run_analysis.py image_deblurring \
    --size 64 \
    --lam 1e-3 \
    --snr 40 \
    --ns_mode fftT \
    --fftT_order 3 \
    --ns_iters 12
```

### For Research/Validation
```bash
# Maximum accuracy settings
python run_analysis.py image_deblurring \
    --size 128 \
    --lam 1e-3 \
    --snr 50 \
    --ns_mode fftT \
    --fftT_order 3 \
    --ns_iters 20
```

### For Fast Prototyping
```bash
# Quick testing settings
python run_analysis.py image_deblurring \
    --size 32 \
    --lam 1e-3 \
    --ns_mode fftT \
    --fftT_order 2 \
    --ns_iters 10
```

## 🔍 Further Reading

- **QSLST Paper**: Fei, W., Tang, J., & Shan, M. (2025). Quaternion special least squares with Tikhonov regularization method in image restoration.
- **Newton-Schulz Methods**: Classical iterative methods for matrix inversion
- **FFT Deconvolution**: Fast frequency-domain approaches to image restoration

## 🎓 Next Steps

1. **Try different parameters**: Experiment with λ, image sizes, noise levels
2. **Test on your images**: Replace the default image with your own
3. **Explore other applications**: Check out [Image Completion](../examples.md#image-processing)
4. **Understand the theory**: Dive into the [API documentation](../api/qslst.md)
