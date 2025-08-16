# QuatIca: Quaternion Linear Algebra Library

<div align="center">
  <img src="https://raw.githubusercontent.com/vleplat/QuatIca/main/Logo.png" alt="QuatIca Logo" width="250">
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

- **🔬 Core Functionality Demo** → [Open in Colab](https://colab.research.google.com/drive/1LQMnpGdSiWZsXjZQrMp1BmVT9Uzt_CKM?usp=sharing)
  
  Test all major features including matrix operations, decompositions, and advanced algorithms without any setup.

## ⚡ Quick Start (2 minutes)

### Installation

```bash
pip install quatica
```

### Basic Usage

```python
import quatica
import numpy as np
from numpy import quaternion

# Create quaternion matrices
A = quatica.create_test_matrix(4, 3, density=0.8)
B = quatica.create_test_matrix(3, 2, density=0.8)

# Basic operations
C = quatica.quat_matmat(A, B)  # Matrix multiplication
norm_A = quatica.quat_frobenius_norm(A)  # Frobenius norm

# Advanced: Pseudoinverse computation
A_pinv = quatica.NewtonSchulzPseudoinverse(A, max_iters=50)

# Decompositions
Q, R = quatica.quat_qr(A)  # QR decomposition
U, s, V = quatica.quat_svd(A)  # SVD decomposition

# Linear system solving
x = quatica.quat_gmres(A, b)  # Q-GMRES solver

print("✅ QuatIca is working!")
```

## 🤔 What is QuatIca?

QuatIca brings modern numerical linear algebra to quaternion matrices and tensors:

- **Matrix Operations**: Multiplication, norms, basic operations optimized for quaternions
- **Factorizations**: QR, LU, SVD, eigendecomposition, Hessenberg, tridiagonal
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
# Basic operations
A = quatica.create_test_matrix(5, 4)
B = quatica.create_test_matrix(4, 3)
C = quatica.quat_matmat(A, B)
norm = quatica.quat_frobenius_norm(A)
```

### Matrix Decompositions
```python
# QR decomposition
Q, R = quatica.quat_qr(A)

# SVD decomposition
U, s, V = quatica.quat_svd(A)

# Eigendecomposition (Hermitian matrices)
eigenvals, eigenvecs = quatica.quat_eig(A)

# LU decomposition
L, U, P = quatica.quat_lu(A)
```

### Pseudoinverse and Linear Systems
```python
# Newton-Schulz pseudoinverse
A_pinv = quatica.NewtonSchulzPseudoinverse(A, max_iters=50)

# Q-GMRES solver
x, info = quatica.quat_gmres(A, b, restart=20, maxiter=100)
```

### Advanced Algorithms
```python
# Randomized SVD
U_rand, s_rand, V_rand = quatica.rand_qsvd(A, rank=10, n_iter=2)

# Power iteration for dominant eigenvector
eigenval, eigenvec = quatica.power_iteration_nonhermitian(A, max_iter=100)

# Schur decomposition
Q, T = quatica.quat_schur(A, variant='rayleigh')
```

## 🏗️ Applications

### Image Processing
```python
# Image deblurring with quaternion representation
import quatica.applications.image_deblurring as deblur

# Load and process image
result = deblur.quaternion_deblur(image_path, lambda_reg=0.1)
```

### Signal Processing
```python
# Quaternion-based signal analysis
import quatica.applications.signal_processing as signal

# Process quaternion-valued signals
processed_signal = signal.quaternion_filter(quaternion_signal)
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
@software{quatica2024,
  title={QuatIca: Quaternion Linear Algebra Library},
  author={Valentin Leplat and Dmitry Beresnev},
  year={2024},
  url={https://github.com/vleplat/QuatIca},
  doi={10.5281/zenodo.XXXXXXX}
}
```

## 🤝 Contributing

Contributions are welcome! Please see our [GitHub repository](https://github.com/vleplat/QuatIca) for:

- **Issues**: Bug reports and feature requests
- **Pull Requests**: Code contributions and improvements
- **Documentation**: Help improve docs and examples
- **Testing**: Add test cases and benchmarks

## 📜 License

CC0 1.0 Universal (Public Domain) - see [LICENSE](https://github.com/vleplat/QuatIca/blob/main/LICENSE.txt) for details.

## 🔗 Links

- **📖 [Documentation](https://vleplat.github.io/QuatIca/)**
- **🐙 [GitHub Repository](https://github.com/vleplat/QuatIca)**
- **🔬 [Colab Demo](https://colab.research.google.com/drive/1LQMnpGdSiWZsXjZQrMp1BmVT9Uzt_CKM?usp=sharing)**
- **📊 [PyPI Package](https://pypi.org/project/quatica/)**

---

**Made with ❤️ for the quaternion computing community**
