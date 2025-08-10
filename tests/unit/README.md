# 🧪 Unit Tests Directory

This directory contains comprehensive unit tests for all core QuatIca functionality. Each test file focuses on specific components and provides detailed validation of the library's capabilities.

---

## 📋 Test Files Overview

### **🎓 Tutorial and Learning**
| File | Purpose | What It Covers |
|------|---------|----------------|
| `../tutorial_quaternion_basics.py` | **Interactive Tutorial** | Complete introduction to quaternion matrices, operations, and basic concepts with visualizations |

### **🔧 Core Matrix Operations**
| File | Purpose | What It Tests |
|------|---------|---------------|
| `test_basic_algebra.py` | **Basic Algebra Operations** | Matrix operations, ishermitian function, determinant computation |
| (removed legacy deep/cur tests) |  |  |
| `test_normQsparse.py` | **Matrix Norms** | Quaternion matrix norm computations and comparisons |
| (removed: `test_simple_newton.py`) |  |  |
| `test_tensor_quaternion_basics.py` | **Tensor Basics** | Order-3 quaternion tensors: Frobenius norm, entrywise |T|, mode-n unfold/fold |

### **⚡ Q-GMRES Solver Tests**
| File | Purpose | What It Tests |
|------|---------|---------------|
| `test_qgmres_accuracy.py` | **Q-GMRES Accuracy** | Precision and accuracy of Q-GMRES solver |
| `test_qgmres_basics.py` | **Q-GMRES Basics** | Fundamental Q-GMRES functionality |
| `test_qgmres_debug.py` | **Q-GMRES Debug** | Debugging and troubleshooting Q-GMRES |
| `test_qgmres_preconditioner.py` | **Q-GMRES Preconditioning** | LU-based left preconditioning for enhanced convergence |
| `test_qgmres_simple.py` | **Q-GMRES Simple** | Simple Q-GMRES test cases |
| `test_qgmres_large.py` | **Q-GMRES Large Scale** | Performance on large matrices |

### **📊 Matrix Decomposition Tests**
| File | Purpose | What It Tests |
|------|---------|---------------|
| `test_rand_qsvd.py` | **Randomized Q-SVD** | Randomized singular value decomposition |
| `test_pass_eff_qsvd.py` | **Pass-Efficient Q-SVD** | Memory-efficient Q-SVD implementation |
| `test_qsvd_reconstruction_analysis.py` | **Q-SVD Reconstruction Analysis** | Validates monotonic error decrease and perfect full-rank reconstruction; generates plots in `validation_output/` |
| `test_schur_synthetic.py` | **Synthetic Schur (unitary similarity)** | Builds `A = P S P^H` with complex subfield; validates similarity/unitarity; saves |T| heatmaps in `validation_output/` |
| `test_schur_power_synthetic.py` | **Schur vs Power-Iteration (transversal)** | Synthetic `A = P S P^H` (diagonal S); compares Schur max-eigenvalue (diag T) with non-Hermitian power-iteration eigenvalue (up to conjugation); asserts close match |

### **🔬 Advanced Algorithms**
| File | Purpose | What It Tests |
|------|---------|---------------|
| `test_power_iteration_simple.py` | **Power Iteration** | Dominant eigenvector computation using power iteration |
| `test_power_iteration_nonhermitian_validation.py` | **Non-Hermitian Power Iteration (validation)** | Hermitian: real eigenvalue matches classical power iteration and `quaternion_eigendecomposition`; Complex embedding: eigenvalue matches NumPy spectrum (up to conjugation); asserts on adjoint residuals; no figures |
| `test_rand.py` | **Matrix Rank** | Rank computation for quaternion matrices |
| `test_rand_unitary.py` | **Random Unitary Matrices** | Generation and validation of unitary matrices |
| `test_compare_schur_variants.py` | **Schur Variants (Rayleigh, AED, DS)** | Compares convergence and |T| visualizations; saves plots to `validation_output/` |
| `test_compare_schur_experimental.py` | **Experimental Schur (windowed AED/Francis-DS)** | Runs experimental routines; saves plots to `validation_output/` |
| `test_experimental_power_iteration.py` | **Complex Power Iteration (Experimental)** | Estimates complex eigenvalues via complex adjoint mapping; residual plots to `validation_output/` |

### **📈 Determinant and Special Functions**
| File | Purpose | What It Tests |
|------|---------|---------------|
| `test_det_demonstration.py` | **Determinant Demo** | Demonstration of Dieudonné determinant computation |
| `test_ns_vs_higher_order_compare.py` | **NS vs Higher-Order NS** | Comparison of pseudoinverse solvers with residual/time plots |

---

## 🎯 Test Categories

### **🧪 Core Functionality Tests**
These tests validate the fundamental operations that form the backbone of QuatIca:

- **Matrix Operations**: Basic algebra, norms, rank computation
- **Linear Systems**: Q-GMRES solver, deep linear solvers
- **Matrix Decompositions**: QR, SVD, eigenvalue decomposition
- **Special Functions**: Determinant, unitary matrix generation

### **⚡ Performance Tests**
These tests ensure algorithms perform well across different scenarios:

- **Scalability**: Large matrix performance
- **Accuracy**: Precision and numerical stability
- **Convergence**: Algorithm convergence behavior
- **Memory Efficiency**: Resource usage optimization

### **🔬 Algorithm Validation Tests**
These tests validate specific algorithmic implementations:

- **Randomized Methods**: rand_qsvd, pass_eff_qsvd
- **Iterative Methods**: Power iteration, Newton-Schulz
- **Decomposition Methods**: CUR, eigenvalue decomposition

### **📊 Application Tests**
These tests validate real-world applications:

- **Image Processing**: Real image matrix operations
- **Signal Processing**: Signal analysis with quaternions
- **Data Analysis**: CIFAR-10 dataset processing

### **🎨 Visualization Tests**
These tests validate enhanced visualization capabilities:

| File | Purpose | What It Tests |
|------|---------|---------------|
| `test_visualization_enhanced.py` | **Enhanced Visualization** | Matrix absolute value visualization, tensor slice visualization, Schur structure analysis, convergence comparison plots |

- **Matrix Visualization**: Absolute value heatmaps, component visualization
- **Tensor Visualization**: 3D tensor slice analysis, mode-specific views  
- **Structure Analysis**: Schur form structure, triangular matrix analysis
- **Convergence Plots**: Algorithm comparison, performance visualization

---

## 🚀 Running Tests

### **Run All Unit Tests**
```bash
# From the project root
python -m pytest tests/unit/

# Or run specific test files
python tests/unit/test_qgmres_basics.py
python tests/unit/test_rand_qsvd.py
python tests/unit/test_pass_eff_qsvd.py
```

### **Run with Verbose Output**
```bash
python -m pytest tests/unit/ -v
```

### **Run Specific Test Categories**
```bash
# Q-GMRES tests only
python -m pytest tests/unit/test_qgmres_*.py

# Q-SVD tests only
python -m pytest tests/unit/test_*qsvd*.py

# Core algebra tests
python -m pytest tests/unit/test_basic_algebra.py
```

---

## 📊 Test Coverage

### **✅ Fully Covered Components**
- **Q-GMRES Solver**: All variants tested (basic, accuracy, debug, preconditioner, large-scale)
- **Q-SVD Methods**: Classical, randomized, and pass-efficient implementations
- **Matrix Operations**: Basic algebra, norms, rank, determinant
- **Tensor Basics**: Entrywise |T|, Frobenius norm, mode-n unfolding/folding
- **Power Iteration**: Dominant eigenvector computation
- **Unitary Matrices**: Generation and validation

### **🔧 Test Quality**
- **Unit Tests**: Individual function testing
- **Integration Tests**: Component interaction testing
- **Performance Tests**: Scalability and efficiency validation
- **Accuracy Tests**: Numerical precision verification
- **Edge Cases**: Boundary condition handling

---

## 🎯 Key Test Files for Different Use Cases

### **🚀 For Beginners**
- `../tutorial_quaternion_basics.py` - Start here to learn the framework (with visualizations)

### **🔧 For Developers**
- `test_basic_algebra.py` - Core matrix operations
- `test_qgmres_basics.py` - Linear system solving
- `test_qgmres_preconditioner.py` - Enhanced Q-GMRES with preconditioning
- `test_rand_qsvd.py` - Matrix decomposition

### **📊 For Performance Analysis**
- `test_qgmres_large.py` - Large-scale performance
- `test_rand_qsvd.py` - Randomized algorithm performance
- `test_pass_eff_qsvd.py` - Memory-efficient algorithms

### **🔬 For Algorithm Validation**
- `test_power_iteration_simple.py` - Power iteration method
- `test_power_iteration_nonhermitian_validation.py` - Non-Hermitian power iteration validation
- `test_tensor_quaternion_basics.py` - Quaternion tensor basics (norms, |T|, unfold/fold)
- `test_schur_synthetic.py` - Synthetic Schur unitary-similarity validation with |T| plots
- `test_schur_power_synthetic.py` - Schur vs Power-Iteration eigenvalue agreement (transversal)
- `test_rank.py` - Rank computation
- `test_rand_unitary.py` - Unitary matrix generation

---

*This directory provides comprehensive testing coverage for all QuatIca functionality. Each test file is designed to be run independently and provides detailed output for debugging and validation.* 