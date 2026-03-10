# QuatIca: Quaternion Linear Algebra Library

<div align="center">
  <img src="https://raw.githubusercontent.com/vleplat/QuatIca/main/Logo.jpg" alt="QuatIca Logo" width="250">
</div>

**Numerical linear algebra for quaternions — fast, practical, and well‑tested.**

## 📚 Documentation

**📖 [Complete Documentation](https://vleplat.github.io/QuatIca/)** - Comprehensive guides, API reference, and examples

**Quick Links:**

- **[Getting Started](https://vleplat.github.io/QuatIca/getting-started/)** - Setup and installation guide
- **[Examples](https://vleplat.github.io/QuatIca/examples/)** - Copy-paste commands and code snippets
- **[API Reference](https://vleplat.github.io/QuatIca/api/utils/)** - Complete function documentation
- **[Troubleshooting](https://vleplat.github.io/QuatIca/troubleshooting/)** - Common issues and solutions

## 🚀 Try it Online - Colab Demos

**No installation required! Try QuatIca directly in your browser:**

### 🎯 **Getting Started**
| Demo | Description | Link |
|------|-------------|------|
| **🔬 Core Functionality Demo** | Test all major features including matrix operations, decompositions, and advanced algorithms without any setup. | [Open in Colab](https://colab.research.google.com/drive/1LQMnpGdSiWZsXjZQrMp1BmVT9Uzt_CKM?usp=sharing) |

### 🖼️ **Image Processing Applications**
| Demo | Description | Link |
|------|-------------|------|
| **Image Completion** | Fill missing pixels in real images using quaternion matrix completion algorithms. | [Open in Colab](https://colab.research.google.com/drive/1-LB6T6caPmayvtcWcIMNXQgiVI16HOQy?usp=sharing) |
| **Image Deblurring (Technical)** | Reproducible benchmarks comparing FFT–NS–Q and QSLST–FFT on Kodak images with λ-optimization, PSNR/SSIM reporting, and LaTeX tables. | [Open in Colab](https://colab.research.google.com/drive/1cm8oW5PhPNtI1lfNDYMvNm9xgJLFgO7n?usp=sharing) |
| **Image Deblurring (Visual)** | Same as above but with beautiful visual examples and side-by-side comparisons. | [Open in Colab](https://colab.research.google.com/drive/1vqWvo2ilFZcCnic5VDlf0z-KPzA8J17f?usp=sharing) |

### 📊 **Algorithm Benchmarks**
| Demo | Description | Link |
|------|-------------|------|
| **Pseudoinverse Methods Comparison** | Comprehensive benchmark of NS (γ=1), HON (3rd), RSP-Q (col), Hybrid RSP+NS, and CGNE–Q with runtime and accuracy analysis. | [Open in Colab](https://colab.research.google.com/drive/1H2a4M64RS5GNLzv1FcP-kviTg3qvlez6?usp=sharing) |
| **Q-GMRES Performance Analysis** | Statistically robust comparison of Q-GMRES with and without LU preconditioning across multiple matrix types, producing publication-ready dashboards. | [Open in Colab](https://colab.research.google.com/drive/1re00YLCsXZtiG9tIgB1UPvMcXN8VkO70?usp=sharing) |

### 🔬 **Research Applications**
| Demo | Description | Link |
|------|-------------|------|
| **Lorenz Attractor Benchmark** | Q-GMRES vs Newton–Schulz comparison on quaternion linear systems from the Lorenz attractor, with runtime, iterations, and residual accuracy analysis. | [Open in Colab](https://colab.research.google.com/drive/1T_vMBDgRK3LT0uemIuqRUNURV_uHWYuz?usp=sharing) |
| **OptiQ Known-Optimum SDP (Trace-Bounded)** | End-to-end OptiQ validation on a synthetic quaternion Hermitian SDP instance with a **certified known optimum** and explicit dual certificate, solved via the log-det barrier path (μ-continuation). Includes hat-space whitening and optional benchmarking of Schur-complement solvers (dense vs CG/PCG). | [Open in Colab](https://colab.research.google.com/drive/1RX6PkJfCdvOJ03wFi8sWEIMfybh5Ul8q?usp=sharing) |


## ⚡ Quick Start (2 minutes)

### Installation

```bash
pip install quatica
```

### Basic Usage

```python
import numpy as np
import quaternion
from quatica.data_gen import create_test_matrix
from quatica.utils import quat_matmat, quat_frobenius_norm
from quatica.decomp.qsvd import qr_qua, classical_qsvd
from quatica.solver import NewtonSchulzPseudoinverse, QGMRESSolver

# Create quaternion matrices
A = create_test_matrix(4, 3)
B = create_test_matrix(3, 2)

# Basic operations
C = quat_matmat(A, B)  # Matrix multiplication
norm_A = quat_frobenius_norm(A)  # Frobenius norm

# Advanced: Pseudoinverse computation
ns_solver = NewtonSchulzPseudoinverse(gamma=0.5)
A_pinv, residuals, metrics = ns_solver.compute(A)

# Decompositions
Q, R = qr_qua(A)  # QR decomposition
U, s, V = classical_qsvd(A, R=2)  # SVD decomposition

# Linear system solving
A = create_test_matrix(3, 3)
x_true = create_test_matrix(3, 1)
b = quat_matmat(A, x_true)
qgmres_solver = QGMRESSolver(tol=1e-6, max_iter=100)
x, info = qgmres_solver.solve(A, b)
print("Solution x shape:", x.shape)
print("Convergence info:", info)

print("✅ QuatIca is working!")
```

### 🎓 Getting Started

After installing QuatIca, you can start coding immediately. For comprehensive examples and tutorials, clone the [GitHub repository](https://github.com/vleplat/QuatIca) and use the interactive tutorial:

```bash
# Comprehensive tutorial with visualizations (recommended for beginners)
python run_analysis.py tutorial

# Core functionality demo (comprehensive overview)
python run_analysis.py demo

# Q-GMRES solver introduction
python run_analysis.py qgmres
```

## 🤔 What is QuatIca?

QuatIca brings modern numerical linear algebra to quaternion matrices and tensors:

- **Matrix Operations**: Multiplication, norms, basic operations optimized for quaternions
- **Factorizations**: QR, LU, SVD, eigendecomposition, Hessenberg, tridiagonal, **Cholesky**, **Schur**
- **Pseudoinverse**: Newton–Schulz method with higher-order variants
- **Linear Solvers**: Q-GMRES with LU preconditioning
- **Applications**: Image processing, signal processing, computer vision

### Key Features

- **🚀 Performance**: Optimized quaternion operations with NumPy backend
- **🧪 Well-tested**: Comprehensive test suite with >100 unit tests
- **📚 Documented**: Complete API documentation with examples
- **🔬 Research-ready**: Implements latest algorithms from quaternion linear algebra literature
- **🎯 Practical**: Real-world applications in image processing and signal analysis

## 📖 Core Functionality

### Matrix Operations
```python
from quatica.data_gen import create_test_matrix
from quatica.utils import quat_matmat, quat_frobenius_norm

# Basic operations
A = create_test_matrix(5, 4)
B = create_test_matrix(4, 3)
C = quat_matmat(A, B)
norm = quat_frobenius_norm(A)
```

### Matrix Decompositions
```python
from quatica.decomp.qsvd import qr_qua, classical_qsvd_full
from quatica.decomp.eigen import quaternion_eigendecomposition
from quatica.decomp import quaternion_lu
from quatica.utils import quat_hermitian

# QR decomposition
Q, R = qr_qua(A)

# SVD decomposition
U, s, V = classical_qsvd_full(A)

# Eigendecomposition (Hermitian matrices)
# Create a Hermitian matrix A = B^H @ B
A = create_test_matrix(4, 3)
A_H = quat_hermitian(A)
A_herm = quat_matmat(A_H, A)
eigenvals, eigenvecs = quaternion_eigendecomposition(A_herm)

# LU decomposition
L, U, P = quaternion_lu(A)
```

### Pseudoinverse and Linear Systems
```python
from quatica.solver import NewtonSchulzPseudoinverse, QGMRESSolver

# Newton-Schulz pseudoinverse
ns_solver = NewtonSchulzPseudoinverse(gamma=0.5)
A_pinv, residuals, metrics = ns_solver.compute(A)

# Q-GMRES solver
qgmres_solver = QGMRESSolver(tol=1e-6, max_iter=100, restart=20)
x, info = qgmres_solver.solve(A, b)
```

### Advanced Algorithms
```python
from quatica.decomp.qsvd import rand_qsvd
from quatica.utils import power_iteration_nonhermitian
from quatica.decomp.schur import quaternion_schur_unified

# Build a square test matrix for the advanced routines
from quatica.data_gen import create_test_matrix

A = create_test_matrix(12, 12)

# Randomized SVD (rank-R approximation)
U_rand, s_rand, V_rand = rand_qsvd(A, R=10, oversample=10, n_iter=2)

# Power iteration for a dominant eigenpair (non-Hermitian, experimental)
q_vec, lambda_complex, residuals = power_iteration_nonhermitian(
    A,
    max_iterations=100,
    eig_tol=1e-12,
    res_tol=1e-10,
    seed=0,
    return_vector=True,
)

# Schur decomposition (stable variants: 'rayleigh' or 'implicit')
Q, T = quaternion_schur_unified(A, variant="rayleigh")
```

## 🏗️ Applications

QuatIca excels in various real-world applications. Explore comprehensive examples and demos:

### 🚀 **Interactive Demos**

See the **Colab demo links** at the top of this page to run QuatIca directly in your browser.

### 📂 **GitHub Repository Examples**

Clone the [QuatIca repository](https://github.com/vleplat/QuatIca) for complete application examples:

**Image Processing Applications:**
- **Quaternion Image Deblurring** - Compare QSLST vs Newton-Schulz methods
- **Image Completion** - Matrix completion techniques for missing pixels  
- **Deblurring Benchmarks** - Comprehensive performance analysis

**Signal Processing Applications:**
- **Lorenz Attractor Analysis** - Q-GMRES solver for dynamical systems
- **Quaternion Signal Processing** - Multi-channel signal analysis
- **Method Comparisons** - Performance benchmarks across algorithms

**Research & Analysis Tools:**
- **Pseudoinverse Analysis** - Single and multi-image studies
- **CIFAR-10 Analysis** - Large-scale image dataset processing
- **Synthetic Matrix Validation** - Controlled experiments and verification

### 📋 **Quick Access**

```bash
# Clone the repository
git clone https://github.com/vleplat/QuatIca.git
cd QuatIca

# See all available applications
python run_analysis.py

# Run specific examples
python run_analysis.py image_deblurring --size 64
python run_analysis.py lorenz_signal --num_points 500
```

## 🔬 Research Applications

QuatIca is designed for researchers working with:

- **Computer Vision**: Color image processing, stereo vision, 3D rotations
- **Signal Processing**: Quaternion-valued signals, spatial-temporal analysis
- **Robotics**: Orientation estimation, SLAM, sensor fusion
- **Graphics**: 3D rotations, animations, geometric transformations
- **Machine Learning**: Quaternion neural networks, geometric deep learning

## 🏆 Performance

QuatIca is optimized for performance:

- **Efficient quaternion operations** using numpy-quaternion backend
- **Randomized algorithms** for large-scale problems (rand_qsvd, power iteration)
- **Optimized solvers** with preconditioning (Q-GMRES with LU)
- **Memory-efficient** implementations for large matrices

### Benchmarks

- **Q-SVD**: Up to 6x faster than full decomposition for low-rank matrices
- **Newton-Schulz**: Quadratic convergence for pseudoinverse computation
- **Q-GMRES**: Competitive with specialized quaternion solvers

## 📦 Installation Requirements

- **Python**: ≥3.11
- **NumPy**: ≥2.3.2
- **numpy-quaternion**: ≥2024.0.10
- **SciPy**: ≥1.16.1
- **matplotlib**: ≥3.10.3
- **scikit-learn**: ≥1.5.0
- **seaborn**: ≥0.13.0

## 🧪 Validation

QuatIca is thoroughly validated:

- **100+ unit tests** covering all major functions
- **Numerical accuracy** verified against theoretical results
- **Performance benchmarks** comparing different algorithms
- **Literature validation** against published research results

## 📄 Citation

If you use QuatIca in your research, please cite:

```bibtex
@software{quatica2025,
  title   = {QuatIca: Quaternion Linear Algebra Library},
  author  = {Leplat, Valentin and Pan, Junjun and Ahmadi-Asl, Salman and Ouerdane, Henni and Ng, Michael},
  year    = {2025},
  version = {v1.0.0},
  doi     = {10.5281/zenodo.16910158},
  url     = {https://github.com/vleplat/QuatIca},
  note    = {Numerical linear algebra for quaternions}
}
```

## 🤝 Contributing

Contributions are welcome! Please see our [GitHub repository](https://github.com/vleplat/QuatIca) for:

- **Issues**: Bug reports and feature requests
- **Pull Requests**: Code contributions and improvements
- **Documentation**: Help improve docs and examples
- **Testing**: Add test cases and benchmarks

## 📜 License

QuatIca is licensed under the **MIT License** (as of v1.0.0).

Earlier versions were released under **CC0 1.0 Universal** and remain available under those terms.

See [LICENSE](https://github.com/vleplat/QuatIca/blob/main/LICENSE) for details.

## 🔗 Links

- **📖 [Documentation](https://vleplat.github.io/QuatIca/)**
- **🐙 [GitHub Repository](https://github.com/vleplat/QuatIca)**
- **🔬 [Colab Demo](https://colab.research.google.com/drive/1LQMnpGdSiWZsXjZQrMp1BmVT9Uzt_CKM?usp=sharing)**
- **📊 [PyPI Package](https://pypi.org/project/quatica/)**

---

**Made with ❤️ for the quaternion computing community**
