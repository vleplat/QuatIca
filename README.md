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
| `cifar10` | **CIFAR-10 Image Analysis** - Analyzes 250 images with class insights | **Start here!** Most comprehensive analysis |
| `pseudoinverse` | **Single Image Analysis** - Analyzes one image (kodim16.png) | Understanding pseudoinverse structure |
| `multiple_images` | **Multi-Image Analysis** - Compares multiple small images | Pattern comparison across images |
| `image_completion` | **Image Completion Demo** - Fills missing pixels in real images | **Practical application** |
| `synthetic` | **Synthetic Image Completion** - Matrix completion on generated test images | Controlled experiments |
| `synthetic_matrices` | **Synthetic Matrix Pseudoinverse Test** - Tests pseudoinverse on various matrix types | Algorithm validation |

#### **🎯 Quick Examples:**

```bash
# Start with the most comprehensive analysis
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

#### **📊 What You Get:**

- **All plots saved** in `output_figures/` directory
- **Detailed analysis** printed to console
- **No need to navigate directories** - everything works from main folder

## 📁 Project Structure

```
QuatIca/
├── core/                    # Core library files
│   ├── solver.py           # Deep linear solver and pseudoinverse algorithms
│   ├── utils.py            # Quaternion operations and utilities
│   ├── data_gen.py         # Matrix generation functions
│   └── visualization.py    # Plotting and visualization tools
├── tests/
│   ├── unit/               # Unit tests for core functionality
│   └── pseudoinverse/      # Pseudoinverse analysis scripts
├── applications/
│   └── image_completion/   # Image processing applications
├── scripts/                # Utility scripts
├── data/                   # Sample data and datasets
│   ├── images/            # Sample images for testing
│   └── cifar-10-batches-py/ # CIFAR-10 dataset
├── output_figures/        # Generated plots and visualizations
├── requirements.txt       # Python dependencies
└── run_analysis.py       # Easy-to-use script runner
```

## 📊 What Each Script Produces

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

# 5. Run a simple test
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
```

### Deep Linear Networks

```python
from core.solver import DeepLinearNewtonSchulz

# Create deep linear network
solver = DeepLinearNewtonSchulz(
    layers=[input_dim, hidden_dim, output_dim],
    gamma=0.2,
    max_iter=100,
    random_init=False
)

# Solve for factors W1, W2 such that X @ W1 @ W2 ≈ I
W1, W2, errors = solver.solve(X)
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
- **Newton-Schulz Algorithm**: Newton's method for matrix inversion and pseudoinverse computation
- **Deep Linear Networks**: Multi-layer matrix factorizations for complex matrix operations

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