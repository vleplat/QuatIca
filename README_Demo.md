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

The demo script tests all 11 core functionality areas:

1. **Basic Matrix Operations** - Matrix creation, multiplication, and norms
2. **QR Decomposition** - QR factorization of quaternion matrices
3. **Quaternion SVD (Q-SVD)** - Singular value decomposition for quaternion matrices
4. **Randomized Q-SVD** - Fast randomized SVD for large matrices
5. **Eigenvalue Decomposition** - Eigendecomposition of Hermitian quaternion matrices
6. **Tridiagonalization** - Householder transformations for Hermitian matrices
7. **Pseudoinverse Computation** - Moore-Penrose pseudoinverse using Newton-Schulz
8. **Linear System Solving** - Q-GMRES solver for quaternion linear systems
9. **Visualization** - Matrix component visualization
10. **Determinant Computation** - Dieudonné determinant with unitary matrix validation
11. **Rank Computation** - Matrix rank computation with matrix product validation
12. **Power Iteration** - Dominant eigenvector computation with eigendecomposition comparison

## Expected Output

When run successfully, you should see:
- ✅ All 12 tests passing
- Detailed information about matrix shapes, norms, and errors
- Convergence analysis for power iteration and Q-GMRES
- Performance comparisons between different algorithms
- Visualization plots (if running in Jupyter)
- Summary confirming all functionality works correctly

## Troubleshooting

If you encounter import errors:
1. Make sure you're in the project root directory
2. Ensure all dependencies are installed: `pip install -r requirements.txt`
3. Check that the `core` directory is in your Python path

## Notes

- The script uses small test matrices (3×4, 4×3, etc.) for quick execution
- All numerical errors should be very small (< 1e-10) indicating high precision
- The demo confirms that all code examples in the README work correctly 