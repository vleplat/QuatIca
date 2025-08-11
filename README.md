# QuatIca: Quaternion Linear Algebra Library

<div align="center">
  <img src="Logo.png" alt="QuatIca Logo" width="250">
</div>

**A comprehensive Python library for Numerical Linear Algebra with Quaternions**

## 🤔 What is QuatIca?

**QuatIca** is a Python library that extends traditional linear algebra to work with **quaternions** - a mathematical system that extends complex numbers to 4D space. Think of it as "linear algebra on steroids" for 3D and 4D data.

### **🎯 What are Quaternions?**
- **Complex numbers** work in 2D (real + imaginary)
- **Quaternions** work in 4D (real + 3 imaginary components: i, j, k)
- **Perfect for**: 3D rotations, color images (RGB), 4D signals, and more
- **Why useful**: Can represent complex relationships in data that regular matrices can't

### **🚀 What Can You Do With QuatIca?**
- **Matrix Operations**: Multiply, invert, and analyze quaternion matrices
 - **Matrix Decompositions**: QR decomposition, Q-SVD (full and truncated), **Randomized Q-SVD**, **LU decomposition**, **Hessenberg form (upper Hessenberg reduction)**, **Schur decomposition**, and **Eigenvalue Decomposition** for quaternion matrices
 - **Linear System Solving**: Solve quaternion systems A*x = b using Q-GMRES (iterative Krylov subspace method) with **LU preconditioning** for enhanced convergence
 - **Pseudoinverse Computation**: Newton–Schulz methods including a higher-order (third-order) variant with cubic local convergence
- **Image Processing**: Complete missing pixels in images using quaternion math
- **Signal Analysis**: Process 3D/4D signals with quaternion algebra
- **Data Science**: Extract complex patterns from multi-dimensional data

### **🧪 Preview: Quaternion Tensor Algebra (Experimental)**
- We added a preview of quaternion tensor tools (order-3) laying groundwork for tensor decompositions (e.g., HOSVD, TT, Tucker):
  - Tensor Frobenius-like norm, entrywise |T| (quaternion magnitude)
  - Mode-n unfolding and folding for order-3 tensors
- See the notebook section "13. Preview: Quaternion Tensor Algebra and Decompositions" in `QuatIca_Core_Functionality_Demo.ipynb`.
- Utilities live in `core/tensor.py`; unit tests in `tests/unit/test_tensor_quaternion_basics.py`.

## 🌟 Motivation and Acknowledgments

QuatIca was inspired by the pioneering work in quaternion linear algebra, particularly the **QTFM (Quaternion Toolbox for MATLAB)** developed by Stephen J. Sangwine and Nicolas Le Bihan. Their comprehensive MATLAB implementation demonstrated the power and potential of quaternion-based numerical methods.

Recognizing the growing importance of Python in scientific computing and the need for robust quaternion tools in the Python ecosystem, we developed QuatIca to bring these capabilities to Python users. We extend our sincere gratitude to Sangwine and Le Bihan for providing the inspiration that drove us to create this library.

Our goal is to continue advancing the field of quaternion linear algebra while making these powerful tools accessible to the broader Python community, from researchers and engineers to students and practitioners across diverse domains.

## ⚠️ CRITICAL PERFORMANCE INFORMATION

**numpy Version Requirement:**
- **REQUIRED**: numpy >= 2.3.2 for optimal performance
- **CRITICAL**: numpy 2.3.2 provides **10-15x speedup** for quaternion matrix operations compared to 2.2.6
- **WARNING**: Using older numpy versions will result in significantly slower performance

**Package Performance Warnings:**
- **opencv-python** and **tqdm** cause **3x performance degradation** and are NOT included in requirements.txt
- These packages pull in heavy dependencies that affect numpy performance
- If you need these for matrix completion features, install them separately but be aware of the performance cost

**Performance Benchmarks (Newton-Schulz Pseudoinverse Computation on 800x1000 matrices):**
- Dense matrices: ~16 seconds with numpy 2.3.2 (vs minutes/hours with 2.2.6)
- Sparse matrices: ~9 seconds with numpy 2.3.2
- Small matrices (200x200): ~0.4 seconds

## 📋 System Requirements

### **Minimum Requirements:**
- **Python**: 3.8 or higher (3.9+ recommended)
- **RAM**: 4GB minimum, 8GB recommended for large matrices
- **Storage**: 500MB free space
- **OS**: Windows, macOS, or Linux

### **Recommended:**
- **Python**: 3.9 or 3.10
- **RAM**: 16GB for large-scale analysis
- **CPU**: Multi-core processor for faster computation





## 🚀 Quick Start Guide

### **🎯 For Complete Beginners (Step-by-Step)**

#### **Step 1: Install Python**
If you don't have Python installed:
1. Go to [python.org](https://python.org)
2. Download Python 3.9 or higher
3. Install with default settings
4. Verify: Open terminal/command prompt and type `python --version`

#### **Step 2: Download QuatIca**
```bash
# Clone the repository (if you have git)
git clone https://github.com/vleplat/QuatIca.git
cd QuatIca

# OR download as ZIP and extract to a folder
```

#### **Step 3: Set Up Environment**
```bash
# Create a virtual environment (isolated Python environment)
python3 -m venv quatica

# Activate the environment
# On Mac/Linux:
source quatica/bin/activate
# On Windows:
quatica\Scripts\activate

# You should see (quatica) at the start of your command line
```

#### **Alternative: Docker Setup (For Advanced Users)**
For maximum reproducibility, you can also use Docker:

```bash
# Build the Docker image
docker build -t quatica .

# Run the container
docker run -it --rm quatica

# Or run a specific script
docker run -it --rm quatica python run_analysis.py tutorial
```

*Note: Docker setup is optional. The virtual environment approach above works perfectly for most users.*

#### **Step 4: Install Dependencies**
```bash
# Install all required packages
pip install -r requirements.txt

# This might take a few minutes - be patient!
```

#### **Step 5: Verify Installation**
```bash
# Test if everything works
python run_analysis.py

# You should see a list of available scripts
```

### **🎯 Super Simple: Run Any Analysis with One Command!**

The library provides a **super easy** way to run any analysis script. Just use `run_analysis.py`:

```bash
# 🚀 The Magic Command:
python run_analysis.py <script_name>
```

#### **📋 Available Scripts (Choose One):**

| Script Name | What It Does | Best For |
|-------------|--------------|----------|
| `tutorial` | **🎓 Quaternion Basics Tutorial** - Complete introduction with visualizations | **🚀 START HERE!** Learn the framework |
| `qgmres` | **Q-GMRES Solver Test** - Tests the iterative Krylov subspace solver | **Linear system solving** with quaternions |
| `qgmres_bench` | **🚀 Q-GMRES Performance Benchmark** - Comprehensive preconditioner benchmarking | **Algorithm performance** and LU preconditioning analysis |
| `lorenz_signal` | **Lorenz Attractor Signal Processing** - 3D signal processing with Q-GMRES | **Signal processing** applications |
| `lorenz_benchmark` | **🏆 Method Comparison Benchmark** - Q-GMRES vs Newton-Schulz performance comparison | **Algorithm selection** and performance analysis |
| `ns_compare` | **NS vs Higher-Order NS** - Compares pseudoinverse solvers, saves residual/time plots | **Pseudoinverse** benchmarking |
| `pseudoinverse` | **Single Image Analysis** - Analyzes one image (kodim16.png) | Understanding pseudoinverse structure |
| `multiple_images` | **Multi-Image Analysis** - Compares multiple small images | Pattern comparison across images |
| `image_completion` | **Image Completion Demo** - Fills missing pixels in real images | **Practical application** |
| `image_deblurring` | **Quaternion Image Deblurring** - QSLST (Algorithm 2) vs NS/HON with FFT specialization | **Image restoration** |
| `synthetic` | **Synthetic Image Completion** - Matrix completion on generated test images | Controlled experiments |
| `synthetic_matrices` | **Synthetic Matrix Pseudoinverse Test** - Tests pseudoinverse on various matrix types | Algorithm validation |
| `eigenvalue_test` | **🔬 Eigenvalue Decomposition Test** - Tests tridiagonalization and eigendecomposition | **Matrix analysis** and eigenvalue computation |
| `schur_demo` | **🎯 Quaternion Schur Decomposition Demo** - Comprehensive comparison of rayleigh vs aed variants | **Matrix decomposition** and algorithm comparison |

#### **🎯 Quick Examples:**

```bash
# 🚀 START HERE: Learn the framework with interactive tutorial
python run_analysis.py tutorial

# Test Q-GMRES linear system solver
python run_analysis.py qgmres

# Test Q-GMRES with LU preconditioning benchmark
python run_analysis.py qgmres_bench

# Process 3D signals with Lorenz attractor (default quality)
python run_analysis.py lorenz_signal

# Process 3D signals with Lorenz attractor (fast testing)
python run_analysis.py lorenz_signal --num_points 100

# Compare Q-GMRES vs Newton-Schulz methods
python run_analysis.py lorenz_benchmark


# See image completion in action
python run_analysis.py image_completion

# Quaternion image deblurring (with optional parameters)
python run_analysis.py image_deblurring --size 32 --lam 1e-3
# Optional: add noise SNR in dB
python run_analysis.py image_deblurring --size 32 --lam 1e-3 --snr 30

# Test matrix completion on synthetic images
python run_analysis.py synthetic

# Test pseudoinverse on synthetic matrices
python run_analysis.py synthetic_matrices

# Test Schur decomposition with comprehensive comparison
python run_analysis.py schur_demo

# Test Schur decomposition with custom matrix size
python run_analysis.py schur_demo 15

# Compare Newton–Schulz variants (saves plots to output_figures)
python run_analysis.py ns_compare

# Get help and see all options
python run_analysis.py
```

#### **🚀 Quick Reference - Most Common Commands:**

```bash
# 🎓 Learn the framework (START HERE)
python run_analysis.py tutorial

# ⚡ Test Q-GMRES solver
python run_analysis.py qgmres

# 🌪️ Lorenz attractor (fast testing)
python run_analysis.py lorenz_signal --num_points 100

# 🌪️ Lorenz attractor (default quality)
python run_analysis.py lorenz_signal

# 🌪️ Lorenz attractor (high quality)
python run_analysis.py lorenz_signal --num_points 500

# 🏆 Method comparison benchmark
python run_analysis.py lorenz_benchmark

# 🎯 Advanced analysis with real data
python run_analysis.py cifar10

# 🖼️ Image completion demo
python run_analysis.py image_completion
```

#### **📊 What You Get:**

- **All plots saved** in `output_figures/` directory
- **Detailed analysis** printed to console
- **No need to navigate directories** - everything works from main folder

## 📁 Project Structure

```
QuatIca/
├── core/                    # Core library files
│   ├── solver.py           # Main algorithms (pseudoinverse computation, Q-GMRES with LU preconditioning)
│   ├── utils.py            # Quaternion operations, utilities, and power iteration
│   ├── data_gen.py         # Matrix generation functions
│   ├── visualization.py    # Plotting and visualization tools
│   ├── tensor.py           # Quaternion tensor utilities (norms, |T|, unfold/fold)
│   └── decomp/             # Matrix decomposition algorithms
│       ├── __init__.py
│       ├── qsvd.py         # QR and Q-SVD implementations
│       ├── eigen.py         # Eigenvalue decomposition for Hermitian matrices
│       ├── LU.py           # LU decomposition with partial pivoting
│       ├── tridiagonalize.py # Tridiagonalization using Householder transformations
│       ├── hessenberg.py    # Upper Hessenberg reduction using Householder similarity
│       └── schur.py        # Schur decomposition (experimental)
├── tests/
│   ├── tutorial_quaternion_basics.py  # 🎓 Interactive tutorial with visualizations
│   ├── schur_demo.py       # 🎯 Comprehensive Schur decomposition demo with algorithm comparison
│   ├── unit/               # Unit tests for core functionality
│   │   ├── test_tensor_quaternion_basics.py       # Quaternion tensor basics (norms, |T|, unfold/fold)
│   │   ├── test_schur_synthetic.py                 # Synthetic Schur (|T| visuals saved)
│   │   ├── test_schur_power_synthetic.py           # Schur vs power-iteration transversal check
│   │   └── [See tests/unit/README.md for complete list]
│   ├── QGMRES/             # Q-GMRES solver tests
│   │   ├── test_qgmres_solver.py         # Main Q-GMRES solver tests
│   │   ├── test_qgmres_large.py          # Large-scale Q-GMRES performance tests
│   │   ├── benchmark_qgmres_preconditioner.py # Comprehensive Q-GMRES LU preconditioning benchmark
│   │   └── benchmark_qgmres_accuracy.py   # Q-GMRES accuracy investigation and analysis
│   ├── pseudoinverse/      # Pseudoinverse analysis scripts
│   │   ├── analyze_pseudoinverse.py      # Single image pseudoinverse analysis
│   │   ├── analyze_multiple_images_pseudoinverse.py # Multiple images analysis
│   │   └── script_synthetic_matrices.py  # Synthetic matrices testing
│   ├── decomp/             # Matrix decomposition tests
│   │   ├── test_qsvd.py    # QR and Q-SVD unit tests
│   │   ├── test_eigen.py   # Eigenvalue decomposition unit tests
│   │   ├── test_LU.py      # LU decomposition unit tests
│   │   ├── test_tridiagonalize.py # Tridiagonalization unit tests
│   │   ├── eigenvalue_demo.py # Demonstration of eigenvalue decomposition
│   │   └── test_hessenberg.py  # Hessenberg reduction unit tests
├── applications/
│   ├── image_completion/   # Image processing applications
│   │   ├── script_real_image_completion.py    # Real image completion
│   │   ├── script_synthetic_image_completion.py # Synthetic image completion
│   │   └── script_small_image_completion.py   # Small image completion
│   └── signal_processing/  # Signal processing applications
│       ├── lorenz_attractor_qgmres.py    # Lorenz attractor Q-GMRES application
│       └── benchmark_lorenz_methods.py   # Q-GMRES vs Newton-Schulz benchmark
├── data/                   # Sample data and datasets
│   ├── images/            # Sample images for testing
│   └── cifar-10-batches-py/ # CIFAR-10 dataset
├── References_and_SuppMat/ # Research papers and supplementary materials
├── output_figures/        # Generated plots and visualizations (auto-created)
├── validation_output/     # Validation plots and analysis figures (auto-created)
├── requirements.txt       # Python dependencies
├── run_analysis.py       # Easy-to-use script runner
├── QuatIca_Core_Functionality_Demo.py     # Interactive demo script testing all core functionality
├── QuatIca_Core_Functionality_Demo.ipynb  # Jupyter notebook version for interactive exploration
└── README_Demo.md                         # Documentation for demo files
```

## 📊 What Each Script Produces

### **🎓 `tutorial` - Complete Framework Introduction**
- **What it is**: Interactive tutorial with beautiful visualizations
- **Perfect for**: Learning the framework from scratch
- **Duration**: ~2-3 minutes with visualizations
- **Output**: 7+ visualization files in `output_figures/`:
  - Matrix component heatmaps (real, i, j, k components)
  - Convergence plots showing Newton-Schulz algorithm performance
  - Performance scaling analysis across different matrix sizes
  - Creative tutorial summary flowchart
- **Covers**:
  - Creating dense and sparse quaternion matrices
  - **Custom matrix creation** (including Pauli matrices example)
  - Basic matrix operations (multiplication, norms)
  - Advanced pseudoinverse computation
  - Solution verification (`||A*x - b||_F` analysis)
  - Linear system solving with quaternions
  - Performance benchmarking
  - Best practices and key takeaways

### **⚡ `qgmres` - Q-GMRES Linear System Solver**
- **What it is**: Comprehensive test suite for the Q-GMRES iterative solver
- **Perfect for**: Testing linear system solving with quaternions
- **Duration**: ~1-2 minutes
- **Output**: Detailed analysis and convergence plots in `output_figures/`:
  - Q-GMRES convergence analysis for different matrix sizes
  - Performance comparison with pseudoinverse method
  - Accuracy verification on dense, sparse, and ill-conditioned matrices
- **Covers**:
  - Basic Q-GMRES functionality (3x3 to 15x15 systems)
  - Convergence testing across different matrix sizes
  - Sparse matrix support verification
  - Ill-conditioned system handling
  - Solution accuracy comparison with pseudoinverse
  - Performance analysis and timing

### **🌪️ `lorenz_signal` - Lorenz Attractor Signal Processing**
- **What it is**: 3D signal processing application using Q-GMRES with adaptive LU preconditioning
- **Perfect for**: Signal processing and dynamical systems analysis
- **Duration**: Configurable via `--num_points` parameter
- **Performance**: Automatic LU preconditioning for large systems (≥200 points) for enhanced convergence
- **Output**: 6+ high-resolution visualization files in `output_figures/`:
  - `lorenz_observed_components.png` - Noisy signal components (x, y, z)
  - `lorenz_observed_trajectory.png` - 3D Lorenz attractor with noise
  - `lorenz_reconstructed_components.png` - Cleaned signal components
  - `lorenz_reconstructed_trajectory.png` - Reconstructed 3D trajectory
  - `lorenz_rhs_components.png` - Right-hand side components
  - `lorenz_rhs_trajectory.png` - RHS 3D trajectory
  - `lorenz_residual_history.png` - Q-GMRES convergence plot

#### **🎛️ Parameter Configuration:**
The script accepts command-line arguments to control resolution and execution time:

```bash
# Fast testing (100 points, ~30 seconds)
python run_analysis.py lorenz_signal --num_points 100

# Balanced performance (200 points, ~75 seconds) - DEFAULT
python run_analysis.py lorenz_signal

# High resolution (500 points, ~5-10 minutes)
python run_analysis.py lorenz_signal --num_points 500

# Research quality (1000 points, ~20-30 minutes)
python run_analysis.py lorenz_signal --num_points 1000

# Save plots without displaying them
python run_analysis.py lorenz_signal --no_show
```

#### **📊 Performance Guide:**
| Points | Execution Time | Resolution | Solver Method | Use Case |
|--------|----------------|------------|---------------|----------|
| 100 | ~30 seconds | Low | Standard Q-GMRES | Fast testing, development |
| 200 | ~20 seconds* | Good | **LU Preconditioned Q-GMRES** | **Default, balanced performance** |
| 500 | ~2-5 minutes* | High | **LU Preconditioned Q-GMRES** | Publication quality |
| 1000 | ~8-15 minutes* | Very High | **LU Preconditioned Q-GMRES** | Research, detailed analysis |

**\*Significantly improved with LU preconditioning (5-10x faster for large systems)**

#### **🔬 What It Covers:**
- **Lorenz attractor signal generation** with configurable resolution
- **Time simulation**: 10-second simulation window (parameter `T` in script)
- **Noise addition and signal corruption** simulation
- **Quaternion matrix construction** for signal filtering
- **Q-GMRES-based signal reconstruction** with convergence analysis
- **3D trajectory visualization** (classic butterfly pattern)
- **Time series analysis** of signal components
- **Performance scaling** with different system sizes

#### **⏰ Time Parameter Configuration:**
The script simulates the Lorenz attractor for **10 seconds** by default. To modify the simulation time:
1. **Open the script**: `applications/signal_processing/lorenz_attractor_qgmres.py`
2. **Find line ~140**: `T, delta, seed = 10.0, 1.0, 0`
3. **Change the first value**: `T = 20.0` for 20 seconds, `T = 5.0` for 5 seconds
4. **Run the script** with your desired `num_points` parameter

**Note**: Longer simulation times require more `num_points` for good resolution.

### **🏆 `lorenz_benchmark` - Method Comparison Benchmark**
- **What it is**: Comprehensive performance comparison between Q-GMRES and Newton-Schulz methods
- **Perfect for**: Understanding method trade-offs and choosing the right algorithm
- **Duration**: ~5-10 minutes (comprehensive testing)
- **Output**: 2 high-quality analysis files in `output_figures/`:
  - `lorenz_benchmark_performance.png` - Performance comparison plots (4 subplots)
  - `lorenz_trajectory_comparison.png` - 3D trajectory reconstruction comparison

#### **📊 Benchmark Results:**
The benchmark tests both methods across different problem sizes (50-200 points) and provides:

**Performance Metrics:**
- Computational time comparison
- Iteration count analysis
- Solution accuracy (residual norms)
- Time vs accuracy trade-off analysis

**Visualization:**
- Side-by-side 3D trajectory reconstructions
- Clean signal vs reconstructed signal comparison
- Method performance across different problem sizes

#### **🎯 Key Findings:**
- **Newton-Schulz is ~100x faster** than Q-GMRES on average
- **Newton-Schulz is ~270x more accurate** than Q-GMRES on average
- **Newton-Schulz scales better** with problem size
- **Q-GMRES shows inconsistent accuracy** across different problem sizes

**Usage:**
```bash
# Run the complete benchmark
python run_analysis.py lorenz_benchmark
```

<!-- ### **🎯 `cifar10` - Most Comprehensive Analysis**
- **Input**: 250 CIFAR-10 images (50 per class from 5 classes)
- **Output**: 8 detailed plots in `output_figures/`:
  - `pixel_reconstruction_filters.png` - How each pixel is reconstructed
  - `spectral_analysis.png` - Singular value analysis
  - `pseudoinverse_manifold.png` - Phase and magnitude visualization
  - `channel_correlations.png` - Color channel relationships
  - `class_average_filters.png` - Class-specific reconstruction filters
  - `pca_analysis.png` - PCA and t-SNE analysis
  - `class_spectral_analysis.png` - Class-specific spectral patterns
  - `sample_images_verification.png` - Sample images for verification -->

### **🖼️ `pseudoinverse` - Single Image Analysis**
- **Input**: kodim16.png image
- **Output**: 4 analysis plots:
  - `pseudoinverse_component_analysis.png`
  - `reconstruction_error_map.png`
  - `pseudoinverse_filter_bank.png`
  - `pseudoinverse_distributions_interpreted.png`

### **🔄 `image_completion` - Practical Application**
- **Input**: Real RGB images with missing pixels
- **Output**: Completed images and PSNR metrics
- **Shows**: How quaternion matrix completion works in practice

### **🧪 `synthetic` - Controlled Experiments**
### **🖼️ `image_deblurring` - Quaternion Image Deblurring (QSLST vs NS/HON)**
- **What it is**: Compares QSLST (Algorithm 2) with our NS variants on restoring a blurred/noisy image. Two QSLST paths are provided: a literal matrix-based implementation and an efficient FFT specialization for convolution with periodic boundary (BCCB).
- **Why it matters**: The target solution is Tikhonov-regularized: x_λ = (A^T A + λI)^{-1} A^T b. FFT diagonalization yields an O(N log N) filter; NS can match it via an augmented system or via inverse-NS on T in the frequency domain.
- **Output**: Side-by-side grid with Clean, Observed, QSLST-FFT, QSLST-Matrix, NS, and HON panels, including PSNR/SSIM and timing. Images saved to `output_figures/`.
- **Usage**:
  ```bash
  python run_analysis.py image_deblurring
  # Options (when running the script directly):
  #   --size 32                # grid size (default 32)
  #   --lam 1e-3               # Tikhonov lambda (default 1e-3)
  #   --snr 30                 # optional AWGN SNR in dB
  #   --ns_mode {dense,sparse,fftT,tikhonov_aug}
  #   --ns_iters K             # iterations for fftT solver
  #   --fftT_order {2,3}       # 2=Newton–Schulz, 3=Halley (cubic)
  ```

#### Problem formulation
- Blur operator A is real (2D convolution, periodic boundary), b is the observed quaternion image (RGB mapped to q=(0,R,G,B)).
- Tikhonov-regularized LS: minimize ||A x − b||^2 + λ||x||^2 ⇒ (A^T A + λI) x = A^T b.
- With real A, quaternion components decouple; FFT diagonalizes A^T A.

#### Methods compared
- QSLST-FFT: X̂ = conj(Ĥ) B̂ / (|Ĥ|^2 + λ) per frequency.
- QSLST-Matrix: T = A^T A + λI; x = T^+ A^T b per component.
- NS (fftT, order-2): inverse-NS on T: y ← y (2 − T y); x ≈ y A^T b.
- HON-NS (fftT, order-3): cubic inverse-NS per frequency: y ← y (1 + r + r^2), r = 1 − T y.
- NS (tikhonov_aug): augmented C = [A; √λ I], y = [b; 0]; x = C^† y (exact but slower when dense).

#### Examples
```bash
# FFT inverse-NS (order-2)
python run_analysis.py image_deblurring --size 32 --lam 1e-3 --snr 30 \
  --ns_mode fftT --fftT_order 2 --ns_iters 14

# FFT inverse-NS (order-3)
python run_analysis.py image_deblurring --size 32 --lam 1e-3 --snr 30 \
  --ns_mode fftT --fftT_order 3 --ns_iters 10

# Augmented NS (dense; small sizes only)
python run_analysis.py image_deblurring --size 32 --lam 1e-3 --snr 30 --ns_mode tikhonov_aug
```

#### Recommended default test
```bash
python run_analysis.py image_deblurring --size 64 --lam 1e-3 --snr 40 --ns_mode fftT --fftT_order 3 --ns_iters 12
```

#### Parameters
- --size N: resize `data/images/kodim16.png` to N×N (default 32).
- --lam λ: Tikhonov regularization (default 1e-3).
- --snr dB: add AWGN at the given SNR.
- --ns_mode: fftT (recommended), tikhonov_aug (exact), dense/sparse (unregularized reference).
- --ns_iters: iterations for fftT (12–20 typical).
- --fftT_order: 2 (quadratic) or 3 (cubic; fewer iterations).

- **Input**: Generated 16×16 test images with known patterns
- **Output**: Matrix completion results and PSNR evolution
- **Shows**: Algorithm performance on controlled, reproducible test cases

### **🔬 `synthetic_matrices` - Algorithm Validation**
- **Input**: Various synthetic matrices (dense, sparse, ill-conditioned) + validation example from literature
- **Output**: Pseudoinverse computation results, timing, accuracy validation, and interactive plots
- **Shows**: Algorithm performance on different matrix types, including known theoretical result from Huang et al. (2015)
- **Note**: Generates interactive plots (not saved to files) for convergence analysis

### **🎯 `schur_demo` - Comprehensive Schur Decomposition Analysis**
- **What it is**: Educational demo comparing rayleigh vs aed variants for quaternion Schur decomposition
- **Perfect for**: Understanding algorithm behavior and performance differences
- **Duration**: ~2-5 minutes (depends on matrix size)
- **Input**: Configurable matrix size (default: 10, can specify up to 25+)
- **Output**: Comprehensive analysis and comparison tables:
  - Convergence results for both rayleigh and aed variants
  - Performance comparison across different matrix types
  - Educational insights about algorithm selection
  - Detailed eigenvalue analysis and structure verification
- **Covers**:
  - **Hermitian matrices**: Guaranteed diagonal Schur form
  - **Random matrices**: Gaussian, skew-symmetric, ill-conditioned, pure imaginary
  - **Synthetic construction**: Upper triangular and diagonal test cases
  - **Algorithm comparison**: rayleigh vs aed variant performance
  - **Educational insights**: When to use which algorithm
- **Usage**:
  ```bash
  # Default size (10x10) - fast testing
  python run_analysis.py schur_demo
  
  # Custom size - comprehensive analysis
  python run_analysis.py schur_demo 15
  
  # Large size - full performance analysis
  python run_analysis.py schur_demo 25
  ```



## 🔬 Core Functionality

### Quaternion Matrix Operations

```python
import quaternion
from core.utils import (
    quat_matmat,
    quat_frobenius_norm,
    quat_eye,
    matrix_norm,              # unified interface: 'fro', 1, 2, inf
    induced_matrix_norm_1,    # max column sum of |A_ij|
    induced_matrix_norm_inf   # max row sum of |A_ij|
)
from core.utils import power_iteration, power_iteration_nonhermitian  # see notes below
from core.solver import NewtonSchulzPseudoinverse, HigherOrderNewtonSchulzPseudoinverse

# Create quaternion matrices
A = quaternion.as_quat_array(...)
B = quaternion.as_quat_array(...)

# Compute the Frobenius norm of matrix A
norm_A = quat_frobenius_norm(A)
print(f"Frobenius norm of matrix A: {norm_A:.6f}")

# Matrix multiplication
C = quat_matmat(A, B)

# Compute pseudoinverse (baseline damped Newton–Schulz)
ns_solver = NewtonSchulzPseudoinverse(gamma=0.5)
A_pinv_ns, ns_residuals, ns_metrics = ns_solver.compute(A)

# Compute pseudoinverse (higher-order third-order Newton–Schulz, cubic local rate)
hon_solver = HigherOrderNewtonSchulzPseudoinverse()
A_pinv_hon, hon_residuals, hon_metrics = hon_solver.compute(A)

# Solve linear system A*x = b using Q-GMRES
from core.solver import QGMRESSolver

# Create Q-GMRES solver without preconditioning
qgmres_solver = QGMRESSolver(tol=1e-6, max_iter=100, verbose=False, preconditioner='none')

# Solve the system
x, info = qgmres_solver.solve(A, b)
print(f"Solution found in {info['iterations']} iterations")
print(f"Final residual: {info['residual']:.2e}")

# Create Q-GMRES solver with LU preconditioning for enhanced convergence
qgmres_solver_lu = QGMRESSolver(tol=1e-6, max_iter=100, verbose=False, preconditioner='left_lu')
x_prec, info_prec = qgmres_solver_lu.solve(A, b)
print(f"With LU preconditioning: {info_prec['iterations']} iterations")
print(f"Preconditioned residual: {info_prec['residual']:.2e}")
```

#### Matrix Norms (Quaternion Matrices)

```python
import numpy as np
import quaternion
from core.utils import matrix_norm, quat_frobenius_norm, induced_matrix_norm_1, induced_matrix_norm_inf

# Random quaternion matrix (m x n)
m, n = 4, 5
A = quaternion.as_quat_array(np.random.randn(m, n, 4))

# Frobenius norm
nf = matrix_norm(A, 'fro')          # same as quat_frobenius_norm(A)

# Induced 1-norm (max column sum of |A_ij|)
n1 = matrix_norm(A, 1)              # or induced_matrix_norm_1(A)

# Induced infinity-norm (max row sum of |A_ij|)
ninf = matrix_norm(A, np.inf)       # or induced_matrix_norm_inf(A)

# Spectral 2-norm (largest singular value) — square matrices
B = quaternion.as_quat_array(np.random.randn(5, 5, 4))
n2 = matrix_norm(B, 2)              # via quaternion SVD (costlier)

print(nf, n1, ninf, n2)
```



### Matrix Generation

#### **🎲 Random Matrix Generation**
```python
from core.data_gen import create_test_matrix, create_sparse_quat_matrix

# Generate random dense matrix
X = create_test_matrix(m=100, n=50, rank=20)

# Generate sparse matrix
X_sparse = create_sparse_quat_matrix(m=100, n=50, density=0.1)
```

#### **🔧 Creating Custom Quaternion Matrices**

**Step-by-Step Guide to Building Your Own Quaternion Matrices:**

##### **1. Understanding Quaternion Format**
Quaternion matrices in QuatIca use the format: `[real, i, j, k]` components
- **Real component**: Scalar part (index 0)
- **i component**: First imaginary part (index 1) 
- **j component**: Second imaginary part (index 2)
- **k component**: Third imaginary part (index 3)

##### **2. Example: Pauli Matrices in Quaternion Format**

**What are Pauli Matrices?**
Pauli matrices are fundamental 2×2 matrices in quantum mechanics:
- **σ₁ (sigma_x)**: `[[0, 1], [1, 0]]` - represents spin-x measurement
- **σ₂ (sigma_y)**: `[[0, -i], [i, 0]]` - represents spin-y measurement  
- **σ₃ (sigma_z)**: `[[1, 0], [0, -1]]` - represents spin-z measurement
- **σ₀ (identity)**: `[[1, 0], [0, 1]]` - identity matrix

**Building Pauli Matrices as Quaternion Matrices (Correct Method):**

```python
import numpy as np
import quaternion

def create_pauli_matrices_quaternion():
    """Create Pauli matrices in quaternion format using the correct pattern"""
    
    # Step 1: Create numpy arrays with shape (rows, cols, 4) for [real, i, j, k]
    # Each matrix is 2x2, so we need (2, 2, 4) arrays
    sigma_0_array = np.zeros((2, 2, 4), dtype=float)
    sigma_x_array = np.zeros((2, 2, 4), dtype=float)
    sigma_y_array = np.zeros((2, 2, 4), dtype=float)
    sigma_z_array = np.zeros((2, 2, 4), dtype=float)
    
    # Step 2: Fill the components [real, i, j, k]
    # sigma_0 (identity): [1, 0, 0, 0] for diagonal, [0, 0, 0, 0] for off-diagonal
    sigma_0_array[0, 0, 0] = 1.0  # real component of (0,0)
    sigma_0_array[1, 1, 0] = 1.0  # real component of (1,1)
    
    # sigma_x: [0, 0, 0, 0] for diagonal, [0, 0, 0, 0] for (0,1), [1, 0, 0, 0] for (1,0)
    sigma_x_array[0, 1, 0] = 1.0  # real component of (0,1)
    sigma_x_array[1, 0, 0] = 1.0  # real component of (1,0)
    
    # sigma_y: [0, 0, 0, 0] for diagonal, [0, -1, 0, 0] for (0,1), [0, 1, 0, 0] for (1,0)
    sigma_y_array[0, 1, 1] = -1.0  # i component of (0,1)
    sigma_y_array[1, 0, 1] = 1.0   # i component of (1,0)
    
    # sigma_z: [1, 0, 0, 0] for (0,0), [-1, 0, 0, 0] for (1,1)
    sigma_z_array[0, 0, 0] = 1.0   # real component of (0,0)
    sigma_z_array[1, 1, 0] = -1.0  # real component of (1,1)
    
    # Step 3: Convert to quaternion arrays using the correct pattern
    # Pattern: quaternion.as_quat_array(array.reshape(-1, 4)).reshape(rows, cols)
    sigma_0 = quaternion.as_quat_array(sigma_0_array.reshape(-1, 4)).reshape(2, 2)
    sigma_x = quaternion.as_quat_array(sigma_x_array.reshape(-1, 4)).reshape(2, 2)
    sigma_y = quaternion.as_quat_array(sigma_y_array.reshape(-1, 4)).reshape(2, 2)
    sigma_z = quaternion.as_quat_array(sigma_z_array.reshape(-1, 4)).reshape(2, 2)
    
    return sigma_0, sigma_x, sigma_y, sigma_z

# Usage example
sigma_0, sigma_x, sigma_y, sigma_z = create_pauli_matrices_quaternion()

print("Pauli matrices in quaternion format:")
print(f"σ₀ (identity):\n{sigma_0}")
print(f"σ₁ (sigma_x):\n{sigma_x}")
print(f"σ₂ (sigma_y):\n{sigma_y}")
print(f"σ₃ (sigma_z):\n{sigma_z}")
```

##### **3. General Steps for Custom Matrices**

```python
# Step 1: Create numpy array with shape (rows, cols, 4) for [real, i, j, k]
matrix_quat = np.zeros((rows, cols, 4), dtype=float)

# Step 2: Fill the components element by element
for i in range(rows):
    for j in range(cols):
        matrix_quat[i, j, 0] = real_part[i, j]      # Real component
        matrix_quat[i, j, 1] = i_part[i, j]         # i component  
        matrix_quat[i, j, 2] = j_part[i, j]         # j component
        matrix_quat[i, j, 3] = k_part[i, j]         # k component

# Step 3: Convert to quaternion array using the correct pattern
# Pattern: quaternion.as_quat_array(array.reshape(-1, 4)).reshape(rows, cols)
quat_matrix = quaternion.as_quat_array(matrix_quat.reshape(-1, 4)).reshape(rows, cols)
```

##### **4. Common Patterns**

```python
# Real matrix: [real, 0, 0, 0]
real_matrix = np.zeros((2, 2, 4))
real_matrix[0, 0, 0] = 1.0  # (0,0) real component
real_matrix[0, 1, 0] = 2.0  # (0,1) real component
real_matrix[1, 0, 0] = 3.0  # (1,0) real component
real_matrix[1, 1, 0] = 4.0  # (1,1) real component
real_quat = quaternion.as_quat_array(real_matrix.reshape(-1, 4)).reshape(2, 2)

# Complex matrix: [real, i, 0, 0]  
complex_matrix = np.zeros((2, 2, 4))
complex_matrix[0, 0, 0] = 1.0  # (0,0) real part
complex_matrix[0, 0, 1] = 0.0  # (0,0) i part
complex_matrix[0, 1, 0] = 0.0  # (0,1) real part
complex_matrix[0, 1, 1] = 1.0  # (0,1) i part
complex_matrix[1, 0, 0] = 0.0  # (1,0) real part
complex_matrix[1, 0, 1] = -1.0 # (1,0) i part
complex_matrix[1, 1, 0] = 1.0  # (1,1) real part
complex_matrix[1, 1, 1] = 0.0  # (1,1) i part
complex_quat = quaternion.as_quat_array(complex_matrix.reshape(-1, 4)).reshape(2, 2)

# Pure quaternion: [0, i, j, k]
pure_quat = np.zeros((2, 2, 4))
pure_quat[0, 0, 1] = 0.0  # (0,0) i part
pure_quat[0, 0, 2] = 0.0  # (0,0) j part
pure_quat[0, 0, 3] = 1.0  # (0,0) k part
pure_quat[0, 1, 1] = 1.0  # (0,1) i part
pure_quat[0, 1, 2] = 0.0  # (0,1) j part
pure_quat[0, 1, 3] = 0.0  # (0,1) k part
pure_quat[1, 0, 1] = 1.0  # (1,0) i part
pure_quat[1, 0, 2] = 0.0  # (1,0) j part
pure_quat[1, 0, 3] = 0.0  # (1,0) k part
pure_quat[1, 1, 1] = 0.0  # (1,1) i part
pure_quat[1, 1, 2] = 0.0  # (1,1) j part
pure_quat[1, 1, 3] = -1.0 # (1,1) k part
pure_quat_result = quaternion.as_quat_array(pure_quat.reshape(-1, 4)).reshape(2, 2)
```

## 🔧 Matrix Decompositions Included

QuatIca provides robust implementations of fundamental matrix decompositions for quaternion matrices:

**📖 For a comprehensive overview of all decomposition methods, algorithms, and usage recommendations, see [`core/decomp/README.md`](core/decomp/README.md).**

### **QR Decomposition**
```python
from core.decomp.qsvd import qr_qua

# QR decomposition of quaternion matrix
Q, R = qr_qua(X_quat)
# X_quat = Q @ R, where Q has orthonormal columns and R is upper triangular
```

### **Quaternion SVD (Q-SVD)**
```python
from core.decomp.qsvd import classical_qsvd, classical_qsvd_full

# Truncated Q-SVD for low-rank approximation
U, s, V = classical_qsvd(X_quat, R)
# X_quat ≈ U @ diag(s) @ V^H

# Full Q-SVD for complete decomposition
U_full, s_full, V_full = classical_qsvd_full(X_quat)
# X_quat = U_full @ Σ @ V_full^H
```

**Features:**
- ✅ **Mathematically validated** with comprehensive tests
- ✅ **Perfect reconstruction** at full rank
- ✅ **Monotonic error decrease** with increasing rank
- ✅ **Robust across matrix sizes** (tested on 4×3 to 8×6 matrices)
- ✅ **Production-ready** with 10/10 tests passing

### **Randomized Q-SVD (Fast Approximation)**
```python
from core.decomp.qsvd import rand_qsvd

# Fast randomized Q-SVD for large matrices
U, s, V = rand_qsvd(X_quat, R, oversample=10, n_iter=2)
# X_quat ≈ U @ diag(s) @ V^H (approximate, rank-R)
```

**Features:**
- ✅ **Fast approximation** for large matrices
- ✅ **Configurable accuracy** via power iterations and oversampling
- ✅ **Memory efficient** compared to full Q-SVD
- ✅ **Production-ready** with comprehensive test suite
- ✅ **Based on Gaussian sketching** with power iterations

### **Eigenvalue Decomposition (Hermitian Matrices)**
```python
from core.decomp import quaternion_eigendecomposition, quaternion_eigenvalues, quaternion_eigenvectors

# Full eigendecomposition: A = V @ diag(λ) @ V^H
eigenvalues, eigenvectors = quaternion_eigendecomposition(A_quat)
# A_quat @ eigenvectors[:, i] = eigenvalues[i] * eigenvectors[:, i]

# Extract only eigenvalues
eigenvals = quaternion_eigenvalues(A_quat)

# Extract only eigenvectors
eigenvecs = quaternion_eigenvectors(A_quat)
```

**Features:**
- ✅ **Hermitian matrices only** - specialized for real eigenvalues
- ✅ **Tridiagonalization approach** - efficient Householder transformations
- ✅ **High accuracy** - residuals < 10^-15
- ✅ **Production-ready** with 15/15 tests passing
- ✅ **Based on MATLAB QTFM** - follows established mathematical approach

### Power Iteration (Hermitian vs Non-Hermitian)
```python
from core.utils import power_iteration, power_iteration_nonhermitian

# Hermitian case (recommended): returns dominant eigenvector and a real eigenvalue estimate
v_dom, lambda_real = power_iteration(A_hermitian, return_eigenvalue=True, verbose=False)

# General (non-Hermitian) case (experimental): complex eigenvalue in a fixed complex subfield
q_vec, lambda_complex, residuals = power_iteration_nonhermitian(
    A_general,
    max_iterations=3000,
    eig_tol=1e-12,
    res_tol=1e-10,
    seed=0,
    return_vector=True,
)
```
Notes:
- Use `power_iteration` for Hermitian quaternion matrices; eigenvalues are real and convergence behavior matches theory.
- If `power_iteration` is applied to non-Hermitian matrices, the returned scalar (when requested) is a real magnitude-based Rayleigh-quotient heuristic (not a true complex eigenvalue).
- For general (non-Hermitian) quaternion matrices, use `power_iteration_nonhermitian` (experimental). It maps to a 2n×2n complex adjoint in a fixed complex subfield and returns a complex eigenvalue along with a quaternion eigenvector approximation. Residual `||Mv - λ v||_2` is available for convergence diagnostics.

See the demo section “Non-Hermitian Complex Power Iteration (Experimental)” in `QuatIca_Core_Functionality_Demo.py` / `.ipynb` for a complete example and residual plots.

### **LU Decomposition (Gaussian Elimination with Partial Pivoting)**
```python
from core.decomp import quaternion_lu

# LU decomposition with permutation: P @ A = L @ U
L, U, P = quaternion_lu(A_quat, return_p=True)
# A_quat = (P^T @ L) @ U, where L is lower triangular with unit diagonal
# U is upper triangular, P is permutation matrix

# LU decomposition without permutation: A = L @ U
L_perm, U_perm = quaternion_lu(A_quat, return_p=False)
# A_quat = L_perm @ U_perm, where L_perm is permuted version of L
```

**Features:**
- ✅ **Partial pivoting** - numerically stable for ill-conditioned matrices
- ✅ **Two output modes** - with/without permutation matrix
- ✅ **Perfect reconstruction** - P*A = L*U or A = L*U depending on mode
- ✅ **Production-ready** with comprehensive test suite
- ✅ **Based on MATLAB QTFM** 
- ✅ **Handles rectangular matrices** - works for m×n matrices

### **Tridiagonalization (Householder Transformations)**
```python
from core.decomp import tridiagonalize

# Tridiagonalize Hermitian matrix: P @ A @ P^H = B
P, B = tridiagonalize(A_quat)
# B is real tridiagonal with same eigenvalues as A
# P is unitary transformation matrix
```

**Features:**
- ✅ **Householder transformations** - numerically stable
- ✅ **Real tridiagonal output** - efficient for eigenvalue computation
- ✅ **Unitary transformations** - preserves eigenvalues
- ✅ **Production-ready** with 13/13 tests passing
- ✅ **Recursive algorithm** - handles matrices of any size

### **Hessenberg Form (Upper Hessenberg Reduction)**
```python
from core.decomp.hessenberg import hessenbergize, is_hessenberg
from core.utils import quat_hermitian, quat_matmat

# Reduce a general quaternion matrix to Hessenberg form
P, H = hessenbergize(X_quat)
# Verify: H = P @ X_quat @ P^H and H is upper Hessenberg
```

**Features:**
- ✅ **Householder similarity transforms** - numerically stable
- ✅ **Unitarity preserved** - P is unitary (P^H P = I)
- ✅ **Structure** - H has zeros strictly below the first subdiagonal
- ✅ **Works for general (non-Hermitian) matrices**

### **📊 Visualization and Validation**

QuatIca includes a comprehensive visualization package for validating and demonstrating the correctness of our implementations:

#### **Q-SVD Reconstruction Error Analysis**
```bash
# Generate convincing visualizations of Q-SVD validation
python tests/validation/qsvd_reconstruction_analysis.py
```

This creates professional-quality plots showing:
- **Perfect monotonicity**: Reconstruction error decreases as rank increases
- **Perfect reconstruction**: Full rank achieves 0.000000 error
- **Consistent behavior**: Same patterns across different matrix sizes
- **Mathematical correctness**: Our Q-SVD follows proper SVD principles

**Generated plots:**
- `qsvd_reconstruction_error_vs_rank.png` - Detailed analysis for each matrix size
- `qsvd_relative_error_summary.png` - Summary with log scale convergence

#### **Eigenvalue Decomposition Testing**
```bash
# Test eigenvalue decomposition functionality
python run_analysis.py eigenvalue_test

# Run comprehensive unit tests
python -m pytest tests/decomp/test_eigen.py -v
python -m pytest tests/decomp/test_tridiagonalize.py -v
```

This validates:
- **Eigenvalue accuracy**: A @ v = λ @ v for each eigenpair
- **Hermitian properties**: Real eigenvalues for Hermitian matrices
- **Tridiagonalization**: P @ A @ P^H = B transformation
- **Numerical stability**: High precision with residuals < 10^-15

#### **Why This Visualization is Convincing**
1. **Mathematical Validation**: Shows expected SVD behavior
2. **Visual Proof**: Clear graphs demonstrate monotonicity
3. **Comprehensive Testing**: Multiple matrix sizes tested
4. **Quantitative Results**: Exact error values provided
5. **Professional Quality**: High-resolution plots suitable for presentations

## 📊 Analysis and Visualization

The library includes comprehensive analysis tools:

- **Pseudoinverse Analysis**: Study the structure and properties of quaternion pseudoinverses
- **Q-GMRES Solver**: Iterative Krylov subspace method for solving quaternion linear systems A*x = b
- **Class-aware Analysis**: Analyze pseudoinverses with respect to data classes (e.g., CIFAR-10)
- **Spectral Analysis**: Examine singular value distributions and spectral properties
- **Visualization**: Generate detailed plots of matrix properties, reconstruction filters, and more

## 🎯 Core Functionality Demo Files

### **📋 `QuatIca_Core_Functionality_Demo.py` - Interactive Core Functionality Tests**
- **What it is**: Comprehensive Python script testing all 16 core functionality areas
- **Perfect for**: Verifying that all README code examples work correctly
- **Duration**: ~30 seconds
- **Output**: Detailed verification of all core functions with numerical accuracy metrics
- **Covers**:
  - Basic matrix operations (creation, multiplication, norms)
  - QR decomposition with reconstruction verification
  - Quaternion SVD (Q-SVD) - both truncated and full
  - Randomized Q-SVD for large matrix approximation
  - Eigenvalue decomposition for Hermitian matrices
  - LU decomposition with partial pivoting
  - Tridiagonalization using Householder transformations
  - Pseudoinverse computation using Newton-Schulz
  - Linear system solving with Q-GMRES
  - Matrix component visualization
  - Determinant and rank computation
  - Power iteration for dominant eigenvectors
  - Hessenberg form reduction
  - Advanced eigenvalue methods (Hermitian and synthetic cases)
  - Schur decomposition with synthetic validation
  - Tensor operations (Frobenius norm, unfolding/folding)

**Usage:**
```bash
python QuatIca_Core_Functionality_Demo.py
```

### **📓 `QuatIca_Core_Functionality_Demo.ipynb` - Jupyter Notebook Version**
- **What it is**: Interactive Jupyter notebook version of the core functionality tests
- **Perfect for**: Step-by-step exploration and learning
- **Features**: 
  - Cell-by-cell execution for detailed understanding
  - Interactive visualizations
  - Easy modification and experimentation
  - Educational comments and explanations

**Usage:**
```bash
jupyter notebook QuatIca_Core_Functionality_Demo.ipynb
```

### **📖 `README_Demo.md` - Demo Documentation**
- **What it is**: Detailed documentation explaining how to use the demo files
- **Perfect for**: Understanding the demo structure and troubleshooting
- **Contains**: Usage instructions, expected outputs, and troubleshooting tips

## 🎯 Applications

### Image Processing
- **Matrix Completion**: Fill in missing pixels in images
- **Image Inpainting**: Reconstruct damaged or occluded regions
- **Feature Analysis**: Study quaternion representations of image features

### Signal Processing
- **Quaternion Signal Analysis**: Process 3D/4D signals using quaternion algebra
- **Spectral Analysis**: Analyze frequency domain properties
- **Filter Design**: Design quaternion-based filters

### Data Science
- **Dimensionality Reduction**: Use quaternion PCA and factorizations
- **Clustering**: Apply quaternion-based clustering algorithms
- **Feature Engineering**: Create quaternion-based features

## 🔧 Advanced Usage

### Direct Script Execution (Optional)

**💡 Tip: Use the runner script above - it's much easier!**

If you need to run scripts directly or modify them:

```bash
# From main directory
python tests/pseudoinverse/analyze_cifar10_pseudoinverse.py
python applications/image_completion/script_real_image_completion.py
```

### Running Tests

```bash
# Run pseudoinverse analysis
python tests/pseudoinverse/analyze_cifar10_pseudoinverse.py
```



### Adding New Features

1. Add core functionality to `core/` directory
2. Create tests in `tests/unit/`
3. Add analysis scripts in `tests/pseudoinverse/`
4. Update `run_analysis.py` for new scripts

## 📚 References

- **Quaternion Pseudoinverse**: Huang, L., Wang, Q.-W., & Zhang, Y. (2015). The Moore–Penrose inverses of matrices over quaternion polynomial rings. Linear Algebra and its Applications, 475, 45-61.
- **Q-GMRES Solver**: Jia, Z., & Ng, M. K. (2021). Structure Preserving Quaternion Generalized Minimal Residual Method. SIAM Journal on Matrix Analysis and Applications (SIMAX), 42(2), 1-25.
- **Advanced Q-SVD Method**: Ma, R.-R., & Bai, Z.-J. (2018). A Structure-Preserving One-Sided Jacobi Method for Computing the SVD of a Quaternion Matrix. arXiv preprint arXiv:1811.08671.
- **QSLST Image Restoration**: Fei, W., Tang, J., & Shan, M. (2025). Quaternion special least squares with Tikhonov regularization method in image restoration. Numerical Algorithms. doi: 10.1007/s11075-025-02187-6.
- **Pass-Efficient Randomized Algorithms**: Ahmadi-Asl, S., Nobakht Kooshkghazi, M., & Leplat, V. (2025). Pass-efficient Randomized Algorithms for Low-rank Approximation of Quaternion Matrices. arXiv preprint arXiv:2507.13731.
- **Newton-Schulz Algorithm**: Newton's method for matrix inversion and pseudoinverse computation

## 🚀 Upcoming Features (Coming Soon!)

### **🔬 Advanced Quaternion Matrix Algorithms**

QuatIca is actively being extended with cutting-edge algorithms from recent research. The following features will be released soon:

#### **📊 Efficient Q-SVD Computation**
- **High-performance SVD** for quaternion matrices based on **Ma & Bai (2018)**
- **Structure-preserving one-sided Jacobi method** for computing Q-SVD
- **Optimized memory usage** for large-scale matrices
- **Parallel computation** support for multi-core systems

#### **🎵 Advanced Signal Processing Tools**
- **Quaternion Fourier Transform** for 3D/4D signal analysis
- **Frequency domain processing** with quaternion algebra
- **Spectral analysis** for multi-dimensional signals
- **Filter design** and signal reconstruction capabilities

#### **🔢 Numerical Linear Algebra (NLA) Tools**
- **Schur Decomposition** using QR algorithm for quaternion matrices
- **Eigenvalue computation** via iterative QR method
- **Matrix diagonalization** for non-Hermitian quaternion matrices
- **Structured eigenvalue problems** with quaternion arithmetic

#### **🔗 Quaternion Tensor Decompositions (Preview to Full Release)**
- **Tensor models**: HOSVD, Tucker, Tensor-Train (TT) adapted to quaternion tensors
- **Core utilities**: tensor norms, entrywise magnitudes, mode-n unfolding/folding (already available in `core/tensor.py`)
- **Demos & tests**: Notebook preview section and `tests/unit/test_tensor_quaternion_basics.py`

**Stay tuned for these exciting new features!** 🚀



## 🔧 Troubleshooting

### **Common Issues and Solutions:**

#### **❌ "Command not found: python"**
- **Solution**: Install Python from [python.org](https://python.org)
- **Alternative**: Try `python3` instead of `python`

#### **❌ "pip: command not found"**
- **Solution**: Python comes with pip. Try `python -m pip` instead of `pip`
- **Alternative**: Install pip separately: `python -m ensurepip --upgrade`

#### **❌ "Permission denied" when installing packages**
- **Solution**: Use virtual environment (see installation steps above)
- **Alternative**: Add `--user` flag: `pip install --user -r requirements.txt`

#### **❌ "numpy version too old"**
- **Solution**: Upgrade numpy: `pip install --upgrade numpy>=2.3.2`
- **Check version**: `python -c "import numpy; print(numpy.__version__)"`

#### **❌ "Script not found"**
- **Solution**: Make sure you're in the correct directory (`QuatIca`)
- **Check**: Run `ls` or `dir` to see if `run_analysis.py` exists

#### **❌ "Import error"**
- **Solution**: Activate virtual environment: `source quatica/bin/activate` (Mac/Linux) or `quatica\Scripts\activate` (Windows)
- **Check**: You should see `(quatica)` at the start of your command line

#### **❌ "Memory error"**
- **Solution**: Close other applications to free RAM
- **Alternative**: Use smaller datasets or reduce matrix sizes in scripts

#### **❌ "Slow performance"**
- **Check numpy version**: Must be >= 2.3.2 for optimal performance
- **Solution**: `pip install --upgrade numpy>=2.3.2`

#### **❌ "No visualizations appear"**
- **Solution**: Make sure matplotlib backend is working: `python -c "import matplotlib.pyplot as plt; plt.plot([1,2,3]); plt.show()"`
- **Alternative**: Check if `output_figures/` directory exists and has write permissions
- **Note**: Visualizations are automatically saved to `output_figures/` directory

#### **⚠️ "DeprecationWarning about seaborn"**
- **This is normal**: The warning about seaborn date parsing is harmless and doesn't affect functionality
- **Solution**: Can be ignored - it's a known issue with seaborn and will be fixed in future versions

### **🔍 Verification Steps:**

After installation, run these commands to verify everything works:

```bash
# 1. Check Python version
python --version

# 2. Check numpy version (should be >= 2.3.2)
python -c "import numpy; print(f'numpy: {numpy.__version__}')"

# 3. Check if virtual environment is active (should see (quatica))
# If not, activate it: source quatica/bin/activate

# 4. Test the runner script
python run_analysis.py

# 5. Run the tutorial (recommended first step)
python run_analysis.py tutorial

# 6. Run a simple test
python run_analysis.py pseudoinverse
```

## 🤝 Contributing

This library is designed to be a comprehensive framework for quaternion linear algebra. Contributions are welcome for:

- New quaternion matrix operations
- Additional factorization algorithms
- Performance optimizations
- New applications and examples
- Documentation improvements

## 📄 License

This project is licensed under the **CC0 1.0 Universal** license - a public domain dedication that allows you to use, modify, and distribute this software freely for any purpose, including commercial use, without any restrictions.

**Key Points:**
- ✅ **Public Domain**: You can use this software for any purpose
- ✅ **No Attribution Required**: You don't need to credit the original authors
- ✅ **Commercial Use**: You can use it in commercial projects
- ✅ **Modification**: You can modify and distribute your changes
- ✅ **No Warranty**: The software is provided "as-is" without any warranties

**Full License Text:** See [`LICENSE.txt`](LICENSE.txt) for the complete license terms.

**Why CC0?** This license promotes the ideal of a free culture and encourages the further production of creative, cultural, and scientific works by allowing maximum freedom of use and redistribution.

## 📧 Support and Contact

For questions, bug reports, or contributions, please contact:  
**v dot leplat [at] innopolis dot ru**

We welcome feedback, collaboration opportunities, and contributions to the QuatIca project.

---

**QuatIca**: Empowering quaternion-based numerical linear algebra for modern applications.