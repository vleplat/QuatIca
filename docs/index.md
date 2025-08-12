# QuatIca Documentation

Welcome to QuatIca - a comprehensive quaternion linear algebra library for Python. This documentation provides everything you need to get started with quaternion matrix operations, decompositions, and real-world applications.

## 🚀 Quick Navigation

### New to QuatIca?
- **[Getting Started](getting-started.md)** - Complete setup guide from installation to first run
- **[Examples](examples.md)** - Copy-paste commands and code snippets
- **[Tutorial](getting-started.md#first-examples)** - Interactive learning with `python run_analysis.py tutorial`

### Applications
- **[Image Deblurring](applications/image_deblurring.md)** - QSLST vs Newton-Schulz methods
- **[Image Completion](applications/image_completion.md)** - Advanced restoration using quaternion matrices
- **[Lorenz Attractor Filtering](applications/lorenz_attractor.md)** - Quaternion signal processing with chaotic systems
- **[Matrix Analysis](examples.md#matrix-decompositions)** - Pseudoinverse and decomposition benchmarks

### Reference
- **[API Documentation](api/utils.md)** - Complete function reference
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

## ⚡ What is QuatIca?

QuatIca brings modern numerical linear algebra to quaternion matrices and tensors:

- **Matrix Operations**: Multiplication, norms, Hermitian conjugate, sparse support
- **Decompositions**: QR, SVD, LU, eigendecomposition, Schur, Hessenberg
- **Solvers**: Newton-Schulz pseudoinverse, Q-GMRES with LU preconditioning
- **Applications**: Image deblurring, image completion, quaternion signal processing

### Key Features
- ⚡ **Fast**: numpy≥2.3.2 provides 10-15x speedup for quaternion operations
- 🧪 **Tested**: 194 passing tests with comprehensive validation
- 🎯 **Practical**: Real-world applications with saved outputs
- 📊 **Visual**: Rich plotting and analysis tools

## 🎯 Quick Examples

### 2-Minute Setup
```bash
# Create environment and install
python3 -m venv quatica && source quatica/bin/activate
pip install -r requirements.txt

# Run interactive tutorial
python run_analysis.py tutorial
```

### Common Tasks
```bash
# Image processing
python run_analysis.py image_deblurring

# Signal processing  
python run_analysis.py lorenz_signal

# Linear system solving
python run_analysis.py qgmres

# Performance benchmarking
python run_analysis.py lorenz_benchmark
```

### Basic Code
```python
import numpy as np
import quaternion
from core.utils import quat_matmat, quat_frobenius_norm
from core.solver import NewtonSchulzPseudoinverse

# Create quaternion matrices
A = quaternion.as_quat_array(np.random.randn(4, 4, 4))
B = quaternion.as_quat_array(np.random.randn(4, 4, 4))

# Matrix operations
C = quat_matmat(A, B)
norm_A = quat_frobenius_norm(A)

# Pseudoinverse computation
solver = NewtonSchulzPseudoinverse()
A_pinv, residuals, _ = solver.compute(A)
```

## 🔬 Research Applications

QuatIca supports cutting-edge research in:

- **Quaternion Signal Processing**: 3D/4D signal analysis
- **Image Restoration**: Deblurring, completion, inpainting
- **Matrix Theory**: Quaternion decompositions and eigenanalysis
- **Numerical Methods**: Iterative solvers and preconditioning

### Recent Algorithms
- **QSLST**: Quaternion Special Least Squares with Tikhonov regularization
- **Q-GMRES**: Generalized Minimal Residual method for quaternion systems
- **Randomized Q-SVD**: Fast approximation for large quaternion matrices
- **Higher-Order Newton-Schulz**: Cubic convergence for pseudoinverse computation

## 📊 Performance Highlights

| Operation | Matrix Size | Time | Accuracy |
|-----------|-------------|------|----------|
| **Q-SVD** | 100×100 | ~0.5s | Machine precision |
| **Pseudoinverse** | 200×200 | ~0.4s | Residual < 10⁻¹⁵ |
| **Q-GMRES** | 500×500 | ~2.0s | Converges in <50 iterations |
| **Image Deblurring** | 64×64 | ~0.1s | >35 dB PSNR |

## 🎓 Learning Path

### Beginners
1. **[Getting Started](getting-started.md)** - Setup and verification
2. **Tutorial**: `python run_analysis.py tutorial` - Interactive learning
3. **[Examples](examples.md)** - Copy-paste code snippets

### Intermediate Users
1. **[API Reference](api/utils.md)** - Function documentation
2. **Applications**: Try image deblurring, completion, and Lorenz attractor filtering
3. **Custom matrices**: Learn quaternion matrix creation patterns

### Advanced Users
1. **Algorithm comparison**: Benchmark different methods
2. **Performance optimization**: Large-scale problems
3. **Research applications**: Extend with new algorithms

## 🔗 External Resources

- **GitHub Repository**: [https://github.com/vleplat/QuatIca](https://github.com/vleplat/QuatIca)
- **Issue Tracker**: [https://github.com/vleplat/QuatIca/issues](https://github.com/vleplat/QuatIca/issues)
- **QTFM (MATLAB)**: Original inspiration for quaternion linear algebra
- **Research Papers**: References included in function documentation

## 📧 Support

- **Documentation**: Complete guides and API reference on this site
- **Community**: GitHub Issues for questions and bug reports
- **Contact**: v dot leplat [at] innopolis dot ru
- **License**: CC0 1.0 Universal (public domain)

---

**Ready to start?** Head to [Getting Started](getting-started.md) for installation, or dive into [Examples](examples.md) for immediate copy-paste commands!


