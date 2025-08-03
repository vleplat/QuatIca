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

### Option 3: Manual Jupyter Notebook Creation
1. Open Jupyter Notebook or JupyterLab
2. Create a new Python notebook
3. Copy and paste sections from `QuatIca_Core_Functionality_Demo.py` into separate cells
4. Run each cell to see the functionality in action

## What the Demo Tests

The demo script tests all 8 core functionality areas:

1. **Basic Matrix Operations** - Matrix creation, multiplication, and norms
2. **QR Decomposition** - QR factorization of quaternion matrices
3. **Quaternion SVD (Q-SVD)** - Singular value decomposition for quaternion matrices
4. **Eigenvalue Decomposition** - Eigendecomposition of Hermitian quaternion matrices
5. **Tridiagonalization** - Householder transformations for Hermitian matrices
6. **Pseudoinverse Computation** - Moore-Penrose pseudoinverse using Newton-Schulz
7. **Linear System Solving** - Q-GMRES solver for quaternion linear systems
8. **Visualization** - Matrix component visualization

## Expected Output

When run successfully, you should see:
- ✅ All 8 tests passing
- Detailed information about matrix shapes, norms, and errors
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