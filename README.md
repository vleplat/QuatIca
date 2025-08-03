# QuatIca: Quaternion Linear Algebra Library

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
- **Linear System Solving**: Solve quaternion systems A*x = b using Q-GMRES (iterative Krylov subspace method)
- **Image Processing**: Complete missing pixels in images using quaternion math
- **Signal Analysis**: Process 3D/4D signals with quaternion algebra
- **Data Science**: Extract complex patterns from multi-dimensional data

## ⚠️ CRITICAL PERFORMANCE INFORMATION

**numpy Version Requirement:**
- **REQUIRED**: numpy >= 2.3.2 for optimal performance
- **CRITICAL**: numpy 2.3.2 provides **10-15x speedup** for quaternion matrix operations compared to 2.2.6
- **WARNING**: Using older numpy versions will result in significantly slower performance

**Package Performance Warnings:**
- **opencv-python** and **tqdm** cause **3x performance degradation** and are NOT included in requirements.txt
- These packages pull in heavy dependencies that affect numpy performance
- If you need these for matrix completion features, install them separately but be aware of the performance cost

**Performance Benchmarks (800x1000 matrices):**
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
git clone <repository-url>
cd QuatIca

# OR download as ZIP and extract to a folder
```

#### **Step 3: Set Up Environment**
```bash
# Create a virtual environment (isolated Python environment)
python3 -m venv venv

# Activate the environment
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# You should see (venv) at the start of your command line
```

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
| `lorenz_signal` | **Lorenz Attractor Signal Processing** - 3D signal processing with Q-GMRES | **Signal processing** applications |
| `cifar10` | **CIFAR-10 Image Analysis** - Analyzes 250 images with class insights | **Advanced analysis** with real data |
| `pseudoinverse` | **Single Image Analysis** - Analyzes one image (kodim16.png) | Understanding pseudoinverse structure |
| `multiple_images` | **Multi-Image Analysis** - Compares multiple small images | Pattern comparison across images |
| `image_completion` | **Image Completion Demo** - Fills missing pixels in real images | **Practical application** |
| `synthetic` | **Synthetic Image Completion** - Matrix completion on generated test images | Controlled experiments |
| `synthetic_matrices` | **Synthetic Matrix Pseudoinverse Test** - Tests pseudoinverse on various matrix types | Algorithm validation |

#### **🎯 Quick Examples:**

```bash
# 🚀 START HERE: Learn the framework with interactive tutorial
python run_analysis.py tutorial

# Test Q-GMRES linear system solver
python run_analysis.py qgmres

# Process 3D signals with Lorenz attractor
python run_analysis.py lorenz_signal

# Advanced analysis with real data
python run_analysis.py cifar10

# See image completion in action
python run_analysis.py image_completion

# Test matrix completion on synthetic images
python run_analysis.py synthetic

# Test pseudoinverse on synthetic matrices
python run_analysis.py synthetic_matrices

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
│   ├── solver.py           # Main algorithms (pseudoinverse computation, Q-GMRES, Q-SVD)
│   ├── utils.py            # Quaternion operations and utilities
│   ├── data_gen.py         # Matrix generation functions
│   └── visualization.py    # Plotting and visualization tools
├── tests/
│   ├── unit/               # Unit tests and tutorial
│   │   ├── tutorial_quaternion_basics.py  # 🎓 Interactive tutorial
│   │   ├── test_qgmres_accuracy.py       # Q-GMRES accuracy tests
│   │   ├── test_qgmres_basics.py         # Q-GMRES basic functionality tests
│   │   ├── test_qgmres_debug.py          # Q-GMRES debug tests
│   │   └── test_qgmres_simple.py         # Q-GMRES simple tests
│   ├── QGMRES/             # Q-GMRES solver tests
│   │   ├── test_qgmres_solver.py         # Main Q-GMRES solver tests
│   │   └── test_qgmres_large.py          # Large-scale Q-GMRES performance tests
│   └── pseudoinverse/      # Pseudoinverse analysis scripts
├── applications/
│   ├── image_completion/   # Image processing applications
│   └── signal_processing/  # Signal processing applications
│       └── lorenz_attractor_qgmres.py    # Lorenz attractor Q-GMRES application

├── data/                   # Sample data and datasets
│   ├── images/            # Sample images for testing
│   └── cifar-10-batches-py/ # CIFAR-10 dataset
├── output_figures/        # Generated plots and visualizations (auto-created)
├── requirements.txt       # Python dependencies
└── run_analysis.py       # Easy-to-use script runner
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
- **What it is**: 3D signal processing application using Q-GMRES
- **Perfect for**: Signal processing and dynamical systems analysis
- **Duration**: Configurable via `--num_points` parameter
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
| Points | Execution Time | Resolution | Use Case |
|--------|----------------|------------|----------|
| 100 | ~30 seconds | Low | Fast testing, development |
| 200 | ~75 seconds | Good | **Default, balanced performance** |
| 500 | ~5-10 minutes | High | Publication quality |
| 1000 | ~20-30 minutes | Very High | Research, detailed analysis |

#### **🔬 What It Covers:**
- **Lorenz attractor signal generation** with configurable resolution
- **Noise addition and signal corruption** simulation
- **Quaternion matrix construction** for signal filtering
- **Q-GMRES-based signal reconstruction** with convergence analysis
- **3D trajectory visualization** (classic butterfly pattern)
- **Time series analysis** of signal components
- **Performance scaling** with different system sizes

### **🎯 `cifar10` - Most Comprehensive Analysis**
- **Input**: 250 CIFAR-10 images (50 per class from 5 classes)
- **Output**: 8 detailed plots in `output_figures/`:
  - `pixel_reconstruction_filters.png` - How each pixel is reconstructed
  - `spectral_analysis.png` - Singular value analysis
  - `pseudoinverse_manifold.png` - Phase and magnitude visualization
  - `channel_correlations.png` - Color channel relationships
  - `class_average_filters.png` - Class-specific reconstruction filters
  - `pca_analysis.png` - PCA and t-SNE analysis
  - `class_spectral_analysis.png` - Class-specific spectral patterns
  - `sample_images_verification.png` - Sample images for verification

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
- **Input**: Generated 16×16 test images with known patterns
- **Output**: Matrix completion results and PSNR evolution
- **Shows**: Algorithm performance on controlled, reproducible test cases

### **🔬 `synthetic_matrices` - Algorithm Validation**
- **Input**: Various synthetic matrices (dense, sparse, ill-conditioned) + validation example from literature
- **Output**: Pseudoinverse computation results, timing, accuracy validation, and interactive plots
- **Shows**: Algorithm performance on different matrix types, including known theoretical result from Huang et al. (2015)
- **Note**: Generates interactive plots (not saved to files) for convergence analysis

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
- **Solution**: Activate virtual environment: `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows)
- **Check**: You should see `(venv)` at the start of your command line

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

# 3. Check if virtual environment is active (should see (venv))
# If not, activate it: source venv/bin/activate

# 4. Test the runner script
python run_analysis.py

# 5. Run the tutorial (recommended first step)
python run_analysis.py tutorial

# 6. Run a simple test
python run_analysis.py pseudoinverse
```

## 🔬 Core Functionality

### Quaternion Matrix Operations

```python
import quaternion
from core.utils import quat_matmat, quat_frobenius_norm, quat_eye
from core.solver import NewtonSchulzPseudoinverse

# Create quaternion matrices
A = quaternion.as_quat_array(...)
B = quaternion.as_quat_array(...)

# Matrix multiplication
C = quat_matmat(A, B)

# Compute pseudoinverse
solver = NewtonSchulzPseudoinverse()
A_pinv, residuals, covariances = solver.compute(A)

# Solve linear system A*x = b using Q-GMRES
from core.solver import QGMRESSolver

# Create Q-GMRES solver
qgmres_solver = QGMRESSolver(tol=1e-6, max_iter=100, verbose=False)

# Solve the system
x, info = qgmres_solver.solve(A, b)
print(f"Solution found in {info['iterations']} iterations")
print(f"Final residual: {info['residual']:.2e}")
```



### Matrix Generation

```python
from core.data_gen import create_test_matrix, create_sparse_quat_matrix

# Generate random dense matrix
X = create_test_matrix(m=100, n=50, rank=20)

# Generate sparse matrix
X_sparse = create_sparse_quat_matrix(m=100, n=50, density=0.1)
```

## 📊 Analysis and Visualization

The library includes comprehensive analysis tools:

- **Pseudoinverse Analysis**: Study the structure and properties of quaternion pseudoinverses
- **Q-GMRES Solver**: Iterative Krylov subspace method for solving quaternion linear systems A*x = b
- **Class-aware Analysis**: Analyze pseudoinverses with respect to data classes (e.g., CIFAR-10)
- **Spectral Analysis**: Examine singular value distributions and spectral properties
- **Visualization**: Generate detailed plots of matrix properties, reconstruction filters, and more

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
python tests/unit/test_simple_newton.py
python applications/image_completion/script_real_image_completion.py
```

### Running Tests

```bash
# Run unit tests
python tests/unit/test_simple_newton.py

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
- **Newton-Schulz Algorithm**: Newton's method for matrix inversion and pseudoinverse computation

## 🤝 Contributing

This library is designed to be a comprehensive framework for quaternion linear algebra. Contributions are welcome for:

- New quaternion matrix operations
- Additional factorization algorithms
- Performance optimizations
- New applications and examples
- Documentation improvements

## 📄 License

[Add your license information here]

---

**QuatIca**: Empowering quaternion-based numerical linear algebra for modern applications.