# Getting Started

Complete setup guide for QuatIca - from installation to first successful run.

## 🚀 Quick Installation (2 minutes)

### Step 1: Create Virtual Environment

```bash
# Create virtual environment named 'quatica'
python3 -m venv quatica

# Activate it
source quatica/bin/activate   # Mac/Linux
# Windows: quatica\Scripts\activate
```

### Step 2: Install Dependencies

```bash
# Upgrade pip and install requirements
pip install -U pip wheel
pip install -r requirements.txt
```

!!! warning "Critical Performance Note"
**numpy>=2.3.2 is REQUIRED** for optimal performance. numpy 2.3.2 provides **10-15x speedup** for quaternion matrix operations compared to older versions.

### Step 3: Verify Installation

```bash
# Check numpy version (should be ≥2.3.2)
python -c "import numpy; print(f'numpy: {numpy.__version__}')"

# Test QuatIca
python run_analysis.py
```

## 🎯 First Examples

### Learn the Framework

```bash
# Start here - interactive tutorial with visualizations
python run_analysis.py tutorial
```

This command runs the complete QuatIca tutorial covering:

- Quaternion matrix basics
- Matrix operations and norms
- Pseudoinverse computation
- Linear system solving
- Performance analysis

### Test Core Functionality

```bash
# Test Q-GMRES solver
python run_analysis.py qgmres

# Test image processing
python run_analysis.py image_completion
```

## 📋 System Requirements

### Minimum Requirements

- **Python**: 3.9 or higher
- **OS**: macOS, Linux, or Windows
- **RAM**: 4 GB (8+ GB recommended)
- **Storage**: 1 GB free space

### Recommended Setup

- **Python**: 3.10-3.12 (best wheel availability)
- **RAM**: 16 GB for large matrix operations
- **CPU**: Multi-core for parallel operations

## 🔧 Platform-Specific Notes

### Windows Users

1. **Enable Long Path Support**: Required for deep directory structures
2. **Keep repo path short**: Use `C:\src\QuatIca` instead of deep paths
3. **Python version**: Prefer 3.10-3.12 for best wheel availability

### Optional PyTorch Installation

If you need PyTorch for advanced features:

```bash
# CPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.1 version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 🧪 Verification Steps

Run these commands to ensure everything works correctly:

### 1. Check Dependencies

```bash
python -c "import numpy, quaternion, scipy, matplotlib; print('All core dependencies loaded successfully')"
```

### 2. Test Basic Operations

```bash
python -c "
import numpy as np
import quaternion
from quatica.utils import quat_matmat, quat_frobenius_norm
A = quaternion.as_quat_array(np.random.randn(3, 3, 4))
B = quaternion.as_quat_array(np.random.randn(3, 3, 4))
C = quat_matmat(A, B)
norm = quat_frobenius_norm(C)
print(f'Matrix multiplication successful, norm: {norm:.6f}')
"
```

### 3. Run Complete Tutorial

```bash
python run_analysis.py tutorial
```

Expected: Generates 7+ visualization files in `output_figures/` directory.

## 📁 Project Layout Understanding

Once installed, familiarize yourself with the structure:

```
QuatIca/
├── quatica/                   # Core library functions
│   ├── utils.py           # Quaternion matrix operations
│   ├── solver.py          # Pseudoinverse, Q-GMRES solvers
│   ├── decomp/            # Matrix decompositions (QR, SVD, LU, etc.)
│   └── ...
├── tests/                 # Test suites and validation
├── applications/          # Real-world applications
├── output_figures/        # Generated plots (auto-created)
├── validation_output/     # Test validation figures (auto-created)
└── run_analysis.py       # Main script runner
```

## 🎯 Next Steps

### For Beginners

1. **Run the tutorial**: `python run_analysis.py tutorial`
2. **Explore examples**: See [Examples](examples.md) page
3. **Try image processing**: `python run_analysis.py image_completion`

### For Developers

1. **Examine the API**: Browse [API Documentation](api/utils.md)
2. **Study decompositions**: Check [Matrix Decompositions](api/decomp/qsvd.md)
3. **Run unit tests**: `python -m pytest tests/unit/ -v`

### For Researchers

1. **Benchmark performance**: `python run_analysis.py lorenz_benchmark`
2. **Test Schur decomposition**: `python run_analysis.py schur_demo`
3. **Explore signal processing**: `python run_analysis.py lorenz_signal`

## 🔍 Troubleshooting

### Common Issues

#### "Command not found: python"

```bash
# Try python3 instead
python3 --version

# Or install Python from python.org
```

#### "numpy version too old"

```bash
# Upgrade numpy
pip install --upgrade "numpy>=2.3.2"

# Verify version
python -c "import numpy; print(numpy.__version__)"
```

#### "Import error"

```bash
# Make sure virtual environment is activated
source quatica/bin/activate  # Mac/Linux
# Windows: quatica\Scripts\activate

# Should see (quatica) in your prompt
```

#### "Permission denied"

```bash
# Use virtual environment (recommended)
python3 -m venv quatica
source quatica/bin/activate

# Or add --user flag
pip install --user -r requirements.txt
```

#### "Slow performance"

- **Check numpy version**: Must be ≥2.3.2
- **Avoid problematic packages**: opencv-python and tqdm cause 3x slowdown
- **Use recommended hardware**: 16GB RAM, multi-core CPU

### Getting Help

1. **Check examples**: Comprehensive examples in [Examples](examples.md)
2. **Review API docs**: Complete function reference in API section
3. **File issues**: [GitHub Issues](https://github.com/vleplat/QuatIca/issues)
4. **Contact**: v dot leplat [at] innopolis dot ru

## ✅ Success Indicators

You're ready to use QuatIca when:

- ✅ Virtual environment is activated (see `(quatica)` in prompt)
- ✅ numpy version ≥2.3.2 installed
- ✅ `python run_analysis.py tutorial` completes successfully
- ✅ Visualization files appear in `output_figures/` directory
- ✅ No import errors when running examples

**🎉 Congratulations!** You're now ready to explore quaternion linear algebra with QuatIca.
