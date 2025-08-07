# 🧪 Unit Tests Directory

This directory contains comprehensive unit tests for all core QuatIca functionality. Each test file focuses on specific components and provides detailed validation of the library's capabilities.

---

## 📋 Test Files Overview

### **🎓 Tutorial and Learning**
| File | Purpose | What It Tests |
|------|---------|---------------|
| `tutorial_quaternion_basics.py` | **Interactive Tutorial** | Complete introduction to quaternion matrices, operations, and basic concepts |

### **🔧 Core Matrix Operations**
| File | Purpose | What It Tests |
|------|---------|---------------|
| `test_basic_algebra.py` | **Basic Algebra Operations** | Matrix operations, ishermitian function, determinant computation |
| `test_cur_decomposition.py` | **CUR Decomposition** | Matrix decomposition using CUR method |
| `test_deep_linear_comprehensive.py` | **Deep Linear Systems** | Comprehensive testing of linear system solving |
| `test_deep_linear_solver.py` | **Deep Linear Solver** | Advanced linear system solving algorithms |
| `test_initialization_sensitivity.py` | **Initialization Sensitivity** | How different initializations affect algorithm convergence |
| `test_normQsparse.py` | **Matrix Norms** | Quaternion matrix norm computations and comparisons |
| `test_simple_newton.py` | **Newton Method** | Newton-Schulz iteration for matrix operations |

### **⚡ Q-GMRES Solver Tests**
| File | Purpose | What It Tests |
|------|---------|---------------|
| `test_qgmres_accuracy.py` | **Q-GMRES Accuracy** | Precision and accuracy of Q-GMRES solver |
| `test_qgmres_basics.py` | **Q-GMRES Basics** | Fundamental Q-GMRES functionality |
| `test_qgmres_debug.py` | **Q-GMRES Debug** | Debugging and troubleshooting Q-GMRES |
| `test_qgmres_simple.py` | **Q-GMRES Simple** | Simple Q-GMRES test cases |
| `test_qgmres_large.py` | **Q-GMRES Large Scale** | Performance on large matrices |

### **📊 Matrix Decomposition Tests**
| File | Purpose | What It Tests |
|------|---------|---------------|
| `test_rand_qsvd.py` | **Randomized Q-SVD** | Randomized singular value decomposition |
| `test_pass_eff_qsvd.py` | **Pass-Efficient Q-SVD** | Memory-efficient Q-SVD implementation |
| `test_real_image.py` | **Real Image Processing** | Matrix operations on real image data |

### **🔬 Advanced Algorithms**
| File | Purpose | What It Tests |
|------|---------|---------------|
| `test_power_iteration_simple.py` | **Power Iteration** | Dominant eigenvector computation using power iteration |
| `test_rank.py` | **Matrix Rank** | Rank computation for quaternion matrices |
| `test_rand_unitary.py` | **Random Unitary Matrices** | Generation and validation of unitary matrices |

### **📈 Determinant and Special Functions**
| File | Purpose | What It Tests |
|------|---------|---------------|
| `test_det_demonstration.py` | **Determinant Demo** | Demonstration of Dieudonné determinant computation |
| `test_det_dieudonne_svd_decomposition.py` | **Dieudonné Determinant** | Determinant computation using SVD decomposition |

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
- **Q-GMRES Solver**: All variants tested (basic, accuracy, debug, large-scale)
- **Q-SVD Methods**: Classical, randomized, and pass-efficient implementations
- **Matrix Operations**: Basic algebra, norms, rank, determinant
- **Power Iteration**: Dominant eigenvector computation
- **Unitary Matrices**: Generation and validation

### **🔧 Test Quality**
- **Unit Tests**: Individual function testing
- **Integration Tests**: Component interaction testing
- **Performance Tests**: Scalability and efficiency validation
- **Accuracy Tests**: Numerical precision verification
- **Edge Cases**: Boundary condition handling

---

## 📈 Test Results

### **Expected Outcomes**
- **All tests should pass** with proper environment setup
- **Performance benchmarks** provide timing comparisons
- **Accuracy tests** verify numerical precision
- **Visualization tests** generate plots for analysis

### **Troubleshooting**
- **Import errors**: Ensure virtual environment is activated
- **Performance issues**: Check system resources and dependencies
- **Accuracy failures**: Verify numerical precision settings

---

## 🎯 Key Test Files for Different Use Cases

### **🚀 For Beginners**
- `tutorial_quaternion_basics.py` - Start here to learn the framework

### **🔧 For Developers**
- `test_basic_algebra.py` - Core matrix operations
- `test_qgmres_basics.py` - Linear system solving
- `test_rand_qsvd.py` - Matrix decomposition

### **📊 For Performance Analysis**
- `test_qgmres_large.py` - Large-scale performance
- `test_rand_qsvd.py` - Randomized algorithm performance
- `test_pass_eff_qsvd.py` - Memory-efficient algorithms

### **🔬 For Algorithm Validation**
- `test_power_iteration_simple.py` - Power iteration method
- `test_rank.py` - Rank computation
- `test_rand_unitary.py` - Unitary matrix generation

---

*This directory provides comprehensive testing coverage for all QuatIca functionality. Each test file is designed to be run independently and provides detailed output for debugging and validation.* 