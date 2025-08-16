# QuatIca Core Functionality Demo

This directory contains a comprehensive demo script that tests all the core functionality examples from the README.

## Files

- `QuatIca_Core_Functionality_Demo.py` - Python script demonstrating all core functionality
- `README_Demo.md` - This file

## How to Use

### Option 1: Run as Python Script
```bash
python QuatIca_Core_Functionality_Demo.py
```

### Option 2: Convert to Jupyter Notebook
If you have `jupytext` installed:
```bash
pip install jupytext
jupytext --to notebook QuatIca_Core_Functionality_Demo.py
jupyter notebook QuatIca_Core_Functionality_Demo.ipynb
```

### Option 3: Use the Jupyter Notebook
Or simply open and play with the Jupyter notebook `QuatIca_Core_Functionality_Demo.ipynb`

## What the Demo Tests

The demo script tests all 16 core functionality areas with emoji headers:

1. **🧮 Basic Matrix Operations** - Matrix creation, multiplication, and norms
2. **📐 QR Decomposition** - QR factorization of quaternion matrices
3. **🔍 Quaternion SVD (Q-SVD)** - Singular value decomposition for quaternion matrices
4. **🎲 Randomized Q-SVD** - Fast randomized SVD for large matrices
5. **🔢 Eigenvalue Decomposition** - Eigendecomposition of Hermitian quaternion matrices
6. **🔧 LU Decomposition** - LU factorization with partial pivoting
7. **📏 Tridiagonalization** - Householder transformations for Hermitian matrices
8. **⤴️ Pseudoinverse Computation** - Moore-Penrose pseudoinverse using Newton-Schulz
9. **⚙️ Linear System Solving** - Q-GMRES solver for quaternion linear systems
10. **📊 Visualization** - Matrix component visualization
11. **🎯 Determinant Computation** - Dieudonné determinant with unitary matrix validation
12. **📏 Rank Computation** - Matrix rank computation with matrix product validation
13. **🚀 Power Iteration** - Dominant eigenvector computation with eigendecomposition comparison
14. **🔧 Hessenberg Form** - Upper Hessenberg reduction using Householder similarity
15. **🔬 Advanced Eigenvalue Methods** - Hermitian and synthetic unitary similarity cases with validation
16. **🧮 Schur Decomposition** and **📊 Tensor Operations** - Synthetic validation and quaternion tensor algebra

## Expected Output

When run successfully, you should see:
- ✅ All 16 tests passing
- Detailed information about matrix shapes, norms, and errors
- Convergence analysis for power iteration and Q-GMRES
- Performance comparisons between different algorithms
- Advanced eigenvalue validation (Hermitian and synthetic cases)
- Schur decomposition validation with synthetic matrices
- Tensor operations demonstrating quaternion tensor algebra
- Visualization plots (if running in Jupyter)
- Comprehensive summary confirming all functionality works correctly

## Troubleshooting

If you encounter import errors:
1. Make sure you're in the project root directory
2. Ensure all dependencies are installed: `pip install -r requirements.txt`
3. Check that the `core` directory is in your Python path

## Notes

- The script uses small test matrices (3×4, 4×3, etc.) for quick execution
- All numerical errors should be very small (< 1e-10) indicating high precision
- The demo confirms that all code examples in the README work correctly
