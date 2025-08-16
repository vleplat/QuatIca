# 📊 QuatIca Matrix Decompositions Summary

**A comprehensive guide to all matrix decomposition methods available in QuatIca**

---

## 🎯 Overview

QuatIca provides a complete suite of matrix decomposition algorithms for quaternion matrices, ranging from exact methods for small matrices to efficient approximations for large-scale problems. This document provides a comprehensive overview of all available methods, their requirements, algorithms, and use cases.

---

## 📋 Available Decomposition Methods

### **1. QR Decomposition**

- **Function**: `qr_qua(X_quat)`
- **Input Matrix**: **General quaternion matrix** (any m×n)
- **Algorithm**: Real-block embedding + SciPy QR + contraction
- **Output**: `(Q, R)` where Q has orthonormal columns, R is upper triangular
- **Status**: ✅ **FULLY IMPLEMENTED AND TESTED**

### **2. LU Decomposition (LU Factorization with Partial Pivoting)**

- **Function**: `quaternion_lu(A_quat, return_p=False)`
- **Input Matrix**: **General quaternion matrix** (any m×n)
- **Algorithm**: Gaussian elimination with partial pivoting (MATLAB QTFM implementation)
- **Output**: `(L, U)` or `(L, U, P)` where L is lower triangular with unit diagonal, U is upper triangular, P is permutation matrix
- **Status**: ✅ **FULLY IMPLEMENTED AND TESTED**

### **3. Quaternion SVD (Q-SVD) - Classical Method**

- **Function**: `classical_qsvd(X_quat, R)` (truncated) / `classical_qsvd_full(X_quat)` (full)
- **Input Matrix**: **General quaternion matrix** (any m×n)
- **Algorithm**: Real-block embedding + LAPACK SVD + contraction
- **Output**: `(U, s, V)` where U, V have orthonormal columns, s contains singular values
- **Status**: ✅ **FULLY IMPLEMENTED AND TESTED**

### **4. Eigenvalue Decomposition**

- **Function**: `quaternion_eigendecomposition(A_quat)`
- **Input Matrix**: **Hermitian quaternion matrix only** (square, A = A^H)
- **Algorithm**: Tridiagonalization + numpy.linalg.eig + back transformation
- **Output**: `(eigenvalues, eigenvectors)` where eigenvalues are real
- **Status**: ✅ **FULLY IMPLEMENTED AND TESTED**

### **5. Tridiagonalization**

- **Function**: `tridiagonalize(A_quat)`
- **Input Matrix**: **Hermitian quaternion matrix only** (square, A = A^H)
- **Algorithm**: Householder transformations
- **Output**: `(P, B)` where P*A*P^H = B and B is tridiagonal
- **Status**: ✅ **FULLY IMPLEMENTED AND TESTED**

### **6. Randomized Q-SVD**

- **Function**: `rand_qsvd(X_quat, R, oversample=10, n_iter=2)`
- **Input Matrix**: **General quaternion matrix** (any m×n)
- **Algorithm**: Gaussian sketching + power iterations + QR
- **Output**: `(U, s, V)` (approximate, rank-R)
- **Status**: ✅ **FULLY IMPLEMENTED AND TESTED**

### **6. Pass-Efficient Q-SVD**

- **Function**: `pass_eff_qsvd(X_quat, R, oversample=10, n_passes=2)`
- **Input Matrix**: **General quaternion matrix** (any m×n)
- **Algorithm**: Alternating QR passes for memory efficiency (MATLAB validated)
- **Output**: `(U, s, V)` (approximate, rank-R)
- **Status**: ✅ **FULLY IMPLEMENTED AND TESTED**

### **7. Hessenberg Reduction (Upper Hessenberg Form)**

- **Function**: `hessenbergize(A_quat)`
- **Input Matrix**: **General quaternion matrix** (square n×n)
- **Algorithm**: Householder similarity transformations (zero below first subdiagonal)
- **Output**: `(P, H)` where `H = P * A * P^H` is upper Hessenberg and `P` is unitary
- **Status**: ✅ **FULLY IMPLEMENTED AND TESTED**

### Schur Decomposition (Under Development)

- Note: A prototype quaternion Schur routine exists (`quatica/decomp/schur.py`), but it is under active development. It is not yet production-ready and should not be relied upon for general inputs.
- Hermitian validation: For Hermitian quaternion matrices (A = A^H), the Schur pipeline empirically converges to a diagonal T (as predicted by theory), providing a strong validation signal.
- Synthetic-unitary similarity validation: For matrices with a known Schur form constructed as `A = P S P^H` where `P` is unitary (complex subfield embedding) and `S` is diagonal or upper-triangular in the x-axis complex subfield, our Schur QR variants (rayleigh, implicit) successfully recover an (almost) upper-triangular `T` with low similarity and unitarity residuals.

  - See unit test `tests/unit/test_schur_synthetic.py` (saves |T| heatmaps to `validation_output/`).
  - Demo notebook cells “12d” (Schur synthetic) show |T| and metrics for a reproducible case.

- Reproducible validation (visual and numeric):
  - Run the comparison/visualization script to compare stable variants and view Schur form plots (after activating the venv):
    - Random matrices:
      ```bash
      python tests/validation/compare_schur_variants.py --sizes 50 --iters 1500 --tol 1e-10 --tag rand
      ```
    - Hermitian matrices (A = B^H @ B):
      ```bash
      python tests/validation/compare_schur_variants.py --sizes 50 --iters 1000 --tol 1e-10 --hermitian --tag herm50
      ```
  - The script saves convergence plots and Schur T visualizations (real component) into `validation_output/`:
    - `schur_lead2_variants_<tag>_n<size>.png`
    - `schur_T_<tag>_n<size>_<variant>_real.png`
  - Observed behavior:
    - Hermitian: both Rayleigh-shift and implicit+AED variants converge to diagonal T quickly; implicit+AED is markedly faster, Rayleigh is slightly more accurate (lower similarity residual).
    - Random: Rayleigh achieves stronger deflation with sufficient iterations but is slower; implicit+AED offers speed with higher residual.

---

## 📊 Matrix Type Requirements

| **Decomposition**        | **Matrix Type** | **Shape** | **Conditions** |
| ------------------------ | --------------- | --------- | -------------- |
| **QR**                   | General         | m×n       | None           |
| **LU**                   | General         | m×n       | None           |
| **Q-SVD (Classical)**    | General         | m×n       | None           |
| **Eigenvalue**           | Hermitian       | n×n       | A = A^H        |
| **Tridiagonalization**   | Hermitian       | n×n       | A = A^H        |
| **Randomized Q-SVD**     | General         | m×n       | None           |
| **Pass-Efficient Q-SVD** | General         | m×n       | None           |
| **Hessenberg Reduction** | General         | n×n       | None           |

---

## 🔧 Algorithm Details

### **Real-Block Embedding Method** (QR, Q-SVD)

- **Principle**: Converts quaternion matrix to 4× larger real matrix
- **Process**:
  1. Embed quaternion matrix in real space
  2. Use optimized LAPACK routines
  3. Contract results back to quaternion form
- **Complexity**: O((4m)(4n)min(4m,4n))
- **Advantages**: Leverages highly optimized real matrix libraries
- **Disadvantages**: Memory overhead due to 4× expansion

### **Gaussian Elimination with Partial Pivoting** (LU Decomposition)

- **Principle**: Factorizes matrix into lower and upper triangular factors
- **Process**:
  1. Apply partial pivoting to ensure numerical stability
  2. Perform Gaussian elimination in-place
  3. Extract L (lower triangular with unit diagonal) and U (upper triangular)
  4. Handle permutation matrix for row exchanges
- **Complexity**: O(mn²) for m×n matrix
- **Advantages**: Numerically stable with pivoting, handles rectangular matrices
- **Disadvantages**: Requires non-singular leading principal minors

### **Householder Transformations** (Tridiagonalization)

- **Principle**: Uses Householder reflections to eliminate subdiagonal elements
- **Process**:
  1. Apply Householder transformations iteratively
  2. Preserve Hermitian structure throughout
  3. Achieve tridiagonal form
- **Complexity**: O(n³)
- **Advantages**: Numerically stable, preserves structure
- **Disadvantages**: Requires Hermitian input

### **Householder Similarity (Hessenberg Reduction)**

- **Principle**: Use Householder reflectors to introduce zeros below first subdiagonal
- **Process**:
  1. For each column k = 0..n−3, target the subvector of column k below the first subdiagonal (rows k+2..n−1)
  2. Construct a quaternion Householder reflector `Hk_sub` that maps this subvector to a multiple of `e1`, leaving the (k+1)-th element as the only nonzero entry
  3. Embed `Hk_sub` into an identity matrix to act on the trailing submatrix (rows/cols k+1..n−1)
  4. Apply the similarity transform `H ← Hk * H * Hk^H` to zero entries strictly below the first subdiagonal in column k
  5. Accumulate the overall unitary: `P ← Hk * P`
- **Complexity**: O(n³)
- **Advantages**: Numerically stable, prepares matrix for QR algorithm / Schur form
- **Disadvantages**: General (non-Hermitian) reduction; not tridiagonal

### **Tridiagonalization + Eigendecomposition**

- **Principle**: Two-step process for Hermitian matrices
- **Process**:
  1. Tridiagonalize Hermitian matrix using Householder transformations
  2. Use standard eigendecomposition on tridiagonal form
  3. Transform eigenvectors back to original space
- **Complexity**: O(n³)
- **Advantages**: Efficient for Hermitian matrices, numerically stable
- **Disadvantages**: Only works for Hermitian matrices

### **Randomized Methods** (Randomized Q-SVD, Pass-Efficient Q-SVD)

- **Principle**: Use random sampling to approximate low-rank structure
- **Process**:
  1. Generate random sketching matrices
  2. Apply power iterations for accuracy
  3. Compute SVD on smaller projected matrix
- **Complexity**: O(mn(R+P)) + O((R+P)²n) where P = oversample
- **Advantages**: Fast for large matrices, memory efficient
- **Disadvantages**: Approximate results, requires rank parameter

---

## ⚡ Performance Characteristics

| **Method**               | **Accuracy** | **Speed** | **Memory** | **Use Case**                                       |
| ------------------------ | ------------ | --------- | ---------- | -------------------------------------------------- |
| **QR**                   | Exact        | Fast      | Medium     | Matrix factorization                               |
| **Q-SVD (Classical)**    | Exact        | Medium    | High       | Full SVD, small matrices                           |
| **Eigenvalue**           | Exact        | Fast      | Medium     | Hermitian matrices only                            |
| **Tridiagonalization**   | Exact        | Fast      | Medium     | Preprocessing for eigendecomposition               |
| **Randomized Q-SVD**     | Approximate  | Very Fast | Low        | Large matrices, rank-R approximation               |
| **Pass-Efficient Q-SVD** | Approximate  | Very Fast | Very Low   | Memory-constrained environments, low-rank matrices |

---

## 🎯 Usage Recommendations

### **For General Matrices:**

#### **QR Decomposition**

- **When to use**: Matrix factorization, linear system solving, orthogonalization
- **Example**: `Q, R = qr_qua(X_quat)`
- **Best for**: Small to medium matrices where exact factorization is needed

#### **LU Decomposition**

- **When to use**: Linear system solving, matrix inversion, determinant computation
- **Example**: `L, U, P = quaternion_lu(A_quat, return_p=True)` (with permutation)
- **Example**: `L, U = quaternion_lu(A_quat, return_p=False)` (without permutation)
- **Best for**: Small to medium matrices where exact triangular factorization is needed, especially for solving linear systems

#### **Q-SVD (Classical)**

- **When to use**: Exact SVD, spectral analysis, matrix approximation
- **Example**: `U, s, V = classical_qsvd(X_quat, R)` (truncated)
- **Best for**: Small to medium matrices where exact SVD is required

#### **Randomized Q-SVD**

- **When to use**: Large matrices, rank-R approximation, when speed is priority
- **Example**: `U, s, V = rand_qsvd(X_quat, R, oversample=10, n_iter=2)`
- **Best for**: Large matrices where approximate low-rank structure is sufficient

#### **Pass-Efficient Q-SVD**

- **When to use**: Memory-constrained environments, low-rank matrices, when speed is priority
- **Example**: `U, s, V = pass_eff_qsvd(X_quat, R, oversample=10, n_passes=2)`
- **Best for**: Systems with limited memory, low-rank matrices, 2.8x faster than rand_qsvd

#### **Hessenberg Reduction**

- **When to use**: Preprocessing for QR/Schur algorithms on general (non-Hermitian) matrices
- **Example**: `P, H = hessenbergize(A_quat)`
- **Best for**: Eigenvalue computations and Schur decomposition pipelines

### **For Hermitian Matrices:**

#### **Eigenvalue Decomposition**

- **When to use**: Spectral analysis, diagonalization, principal component analysis
- **Example**: `eigenvalues, eigenvectors = quaternion_eigendecomposition(A_quat)`
- **Best for**: Hermitian matrices where spectral properties are needed

#### **Tridiagonalization**

- **When to use**: Preprocessing step for eigendecomposition, structure analysis
- **Example**: `P, B = tridiagonalize(A_quat)`
- **Best for**: Hermitian matrices where tridiagonal form is useful

---

## 🔍 Implementation Status

### **✅ Fully Implemented and Tested**

- QR Decomposition (`qr_qua`)
- Classical Q-SVD (`classical_qsvd`, `classical_qsvd_full`)
- Eigenvalue Decomposition (`quaternion_eigendecomposition`)
- Tridiagonalization (`tridiagonalize`)
- Randomized Q-SVD (`rand_qsvd`)
- Pass-Efficient Q-SVD (`pass_eff_qsvd`) - **NEW: MATLAB validated, unit tested, performance benchmarked**
- Hessenberg Reduction (`hessenbergize`) - **NEW**

**Note**: All methods have been thoroughly tested and validated for production use.

---

## 📚 Mathematical Background

### **Quaternion Matrices**

- **Structure**: Matrices with quaternion entries (4D numbers: w + xi + yj + zk)
- **Hermitian**: A = A^H where A^H is the conjugate transpose
- **Unitary**: U^H \* U = I where I is the identity matrix

### **Real-Block Embedding**

- **Principle**: Every quaternion matrix can be represented as a 4× larger real matrix
- **Mapping**: Q → [Q_real, Q_i, Q_j, Q_k] where each component is real
- **Advantage**: Enables use of highly optimized real matrix libraries

### **Householder Transformations**

- **Principle**: Use reflections to introduce zeros in specific positions
- **Stability**: Numerically stable and structure-preserving
- **Application**: Tridiagonalization of Hermitian matrices

---

## 🚀 Recent Improvements and Testing

### **✅ Pass-Efficient Q-SVD Implementation (Latest)**

- **MATLAB Validation**: Successfully compared with MATLAB reference implementation
- **Unit Testing**: Comprehensive test suite in `tests/unit/test_pass_eff_qsvd.py`
- **Performance Benchmarking**: Detailed comparison with `rand_qsvd` on 500×300 matrices
- **Key Results**: 2.8x faster than `rand_qsvd`, 4.5x speedup vs full Q-SVD with 2 passes
- **Visualization**: Performance plots available in `output_figures/qsvd_performance_comparison.png`

### **🔧 Implementation Details**

- **Algorithm**: Alternating QR passes matching MATLAB `rand_pass_eff_quater` function
- **Convergence**: Perfect accuracy achieved with 2+ passes
- **Memory Efficiency**: Minimal matrix passes for optimal cache behavior
- **Production Ready**: Fully tested and validated for real-world applications

## 🚀 Future Developments

### **Planned Enhancements**

1. **Advanced Q-SVD**: Implementation of Ma & Bai (2018) structure-preserving one-sided Jacobi method
2. **Parallel Computing**: Multi-core support for large-scale decompositions
3. **GPU Acceleration**: CUDA/OpenCL support for high-performance computing

### **Research Integration**

- **Pass-Efficient Randomized Algorithms**: Based on latest research for communication-efficient matrix approximations
- **Structure-Preserving Methods**: Advanced algorithms that maintain quaternion structure throughout computation
- **Adaptive Methods**: Algorithms that automatically choose optimal parameters based on matrix properties

---

## 📖 References

1. **Quaternion Linear Algebra**: Fundamental theory and applications
2. **Ma & Bai (2018)**: Structure-preserving one-sided Jacobi method for Q-SVD
3. **Ahmadi-Asl et al. (2025)**: Pass-efficient randomized algorithms for quaternion matrices
4. **Householder Transformations**: Classical numerical linear algebra technique
5. **Randomized Matrix Algorithms**: Modern approaches for large-scale matrix computations

---

## 🎯 Conclusion

QuatIca provides a comprehensive suite of matrix decomposition methods that cover the full spectrum of quaternion linear algebra needs. From exact methods for small matrices to efficient approximations for large-scale problems, the library offers solutions for various computational requirements and constraints.

The combination of classical methods (QR, Q-SVD, eigendecomposition) with modern randomized approaches provides users with both accuracy and efficiency options, making QuatIca suitable for a wide range of applications in signal processing, image analysis, and scientific computing.

---

_This document serves as a comprehensive reference for all matrix decomposition capabilities in QuatIca. For detailed implementation examples and tutorials, refer to the main README and demo files._
