# Troubleshooting

Common issues and solutions for QuatIca setup and usage.

## 🔧 Installation Issues

### "Command not found: python"

**Problem**: Python is not installed or not in PATH.

**Solutions**:
```bash
# Try python3 instead
python3 --version

# Install Python from python.org
# Download Python 3.9+ from https://python.org

# On macOS with Homebrew
brew install python

# On Ubuntu/Debian
sudo apt update && sudo apt install python3 python3-pip

# On Windows: Download from python.org and check "Add to PATH"
```

### "pip: command not found"

**Problem**: pip is not installed or not accessible.

**Solutions**:
```bash
# Use python -m pip instead
python -m pip --version

# Install pip manually
python -m ensurepip --upgrade

# On Ubuntu/Debian
sudo apt install python3-pip
```

### "Permission denied" when installing packages

**Problem**: Trying to install to system Python without permissions.

**Solutions**:
```bash
# Use virtual environment (RECOMMENDED)
python3 -m venv quatica
source quatica/bin/activate  # Mac/Linux
# Windows: quatica\Scripts\activate

# Then install normally
pip install -r requirements.txt

# Alternative: Install for user only
pip install --user -r requirements.txt
```

## ⚡ Performance Issues

### "Slow performance" - numpy version

**Problem**: Using old numpy version (significant performance impact).

**Diagnosis**:
```bash
# Check numpy version
python -c "import numpy; print(f'numpy version: {numpy.__version__}')"
```

**Solution**:
```bash
# Upgrade to numpy>=2.3.2 (CRITICAL for performance)
pip install --upgrade "numpy>=2.3.2"

# Verify upgrade
python -c "import numpy; print(f'numpy version: {numpy.__version__}')"
```

!!! warning "Performance Impact"
    numpy 2.3.2 provides **10-15x speedup** for quaternion operations compared to version 2.2.6. Using older versions will result in dramatically slower performance.

### "Memory error" during computation

**Problem**: Insufficient RAM for large matrix operations.

**Solutions**:
```bash
# Use smaller problem sizes
python run_analysis.py lorenz_signal --num_points 100  # Instead of 500

# For image processing, use smaller images
python run_analysis.py image_deblurring --size 32  # Instead of 128

# Close other applications to free RAM
# Consider upgrading to 16GB+ RAM for large problems
```

### Slow performance with certain packages

**Problem**: opencv-python and tqdm cause 3x performance degradation.

**Diagnosis**:
```bash
# Check if problematic packages are installed
pip list | grep opencv
pip list | grep tqdm
```

**Solution**:
```bash
# Remove if not needed
pip uninstall opencv-python tqdm

# Or create clean environment
python3 -m venv quatica_clean
source quatica_clean/bin/activate
pip install -r requirements.txt  # Only core dependencies
```

## 🔍 Import and Path Issues

### "ModuleNotFoundError" for core modules

**Problem**: Python can't find QuatIca core modules.

**Diagnosis**:
```bash
# Check if you're in the right directory
pwd
ls  # Should see run_analysis.py, core/, tests/, etc.

# Check if virtual environment is activated
echo $VIRTUAL_ENV  # Should show path to quatica environment
```

**Solutions**:
```bash
# Make sure you're in the QuatIca directory
cd /path/to/QuatIca

# Activate virtual environment
source quatica/bin/activate  # Mac/Linux
# Windows: quatica\Scripts\activate

# Should see (quatica) in your prompt

# Test import
python -c "from core.utils import quat_matmat; print('Import successful')"
```

### "ImportError" for quaternion

**Problem**: numpy-quaternion not installed properly.

**Solutions**:
```bash
# Install quaternion library
pip install numpy-quaternion

# Or upgrade if installed
pip install --upgrade numpy-quaternion

# Test
python -c "import quaternion; print('Quaternion library working')"
```

## 📊 Script Execution Issues

### "Script not found" errors

**Problem**: Scripts can't be found or executed.

**Solutions**:
```bash
# Always run from QuatIca root directory
cd /path/to/QuatIca

# Use the runner script (RECOMMENDED)
python run_analysis.py tutorial

# If running scripts directly, use full paths
python tests/tutorial_quaternion_basics.py
python applications/image_completion/script_real_image_completion.py
```

### No output figures generated

**Problem**: Figures not being saved or displayed.

**Diagnosis**:
```bash
# Check if output_figures directory exists
ls -la | grep output_figures

# Check matplotlib backend
python -c "import matplotlib; print(f'Backend: {matplotlib.get_backend()}')"
```

**Solutions**:
```bash
# Create output directory if missing
mkdir -p output_figures validation_output

# Test matplotlib
python -c "
import matplotlib.pyplot as plt
import numpy as np
plt.plot([1,2,3])
plt.savefig('test_plot.png')
print('Matplotlib working, test_plot.png created')
"

# For headless systems, set backend
export MPLBACKEND=Agg
python run_analysis.py tutorial
```

### Quaternion visualization issues

**Problem**: Quaternion components not displaying correctly.

**Solutions**:
```bash
# Test basic quaternion operations
python -c "
import numpy as np
import quaternion
from core.utils import quat_frobenius_norm
A = quaternion.as_quat_array(np.random.randn(3, 3, 4))
print(f'Matrix shape: {A.shape}')
print(f'Norm: {quat_frobenius_norm(A):.6f}')
print('Quaternion operations working')
"
```

## 🧪 Testing Issues

### Unit tests failing

**Problem**: Tests not passing when run manually.

**Diagnosis**:
```bash
# Run specific test with verbose output
python -m pytest tests/unit/test_basic_algebra.py -v

# Check test environment
python -c "
import sys
print(f'Python version: {sys.version}')
print(f'Python path: {sys.path[:3]}...')
"
```

**Solutions**:
```bash
# Make sure you're in the right environment and directory
source quatica/bin/activate
cd /path/to/QuatIca

# Run tests with proper path
python -m pytest tests/decomp/test_qsvd.py -v

# For single test files
python tests/unit/test_basic_algebra.py
```

### Q-GMRES convergence issues

**Problem**: Q-GMRES solver not converging or giving poor results.

**Solutions**:
```bash
# Try with LU preconditioning
python -c "
from core.solver import QGMRESSolver
solver = QGMRESSolver(preconditioner='left_lu', verbose=True)
# Use solver...
"

# Increase iteration limit
python -c "
solver = QGMRESSolver(max_iter=200, tol=1e-8)
# Use solver...
"

# Check matrix conditioning
python -c "
import numpy as np
from core.utils import matrix_norm
# Check condition number of your matrix
print(f'Matrix 1-norm: {matrix_norm(A, 1)}')
print(f'Matrix inf-norm: {matrix_norm(A, np.inf)}')
"
```

## 🐛 Common Errors and Fixes

### "Segmentation fault" or crashes

**Problem**: Usually related to numpy/BLAS configuration.

**Solutions**:
```bash
# Update numpy and dependencies
pip install --upgrade numpy scipy

# Try different BLAS library
pip uninstall numpy
pip install numpy --no-binary numpy  # Compile from source

# Check for conflicting packages
pip list | grep -E "(mkl|openblas|atlas)"
```

### "RuntimeWarning" about matrix inversion

**Problem**: Ill-conditioned matrices causing numerical warnings.

**Solutions**:
```bash
# Use regularization for better conditioning
python -c "
from core.solver import NewtonSchulzPseudoinverse
solver = NewtonSchulzPseudoinverse(gamma=0.5)  # Damping
# This improves stability for ill-conditioned matrices
"

# For Q-GMRES, use preconditioning
python -c "
from core.solver import QGMRESSolver
solver = QGMRESSolver(preconditioner='left_lu')
"
```

### "DeprecationWarning" messages

**Problem**: Harmless warnings from seaborn or other libraries.

**Solution**:
```bash
# These warnings are normal and don't affect functionality
# To suppress warnings (optional):
python -W ignore run_analysis.py tutorial

# Or in Python:
import warnings
warnings.filterwarnings('ignore')
```

## 🔧 Environment Issues

### Virtual environment problems

**Problem**: Environment not working correctly.

**Solutions**:
```bash
# Recreate environment from scratch
rm -rf quatica  # Remove old environment
python3 -m venv quatica
source quatica/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt

# Verify environment
which python  # Should point to quatica/bin/python
pip list | head -10
```

### Windows-specific issues

**Common Windows problems and solutions**:

```bash
# Long path support
# Enable in Group Policy: Computer Configuration > Administrative Templates > System > Filesystem > Enable Win32 long paths

# PowerShell execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Use forward slashes in paths
cd C:/src/QuatIca  # Instead of C:\src\QuatIca

# Activate environment on Windows
quatica\Scripts\activate.bat  # CMD
quatica\Scripts\Activate.ps1  # PowerShell
```

## 📞 Getting Help

### Before asking for help

1. **Check this troubleshooting guide** first
2. **Verify your setup**:
   ```bash
   python --version  # Should be 3.9+
   python -c "import numpy; print(numpy.__version__)"  # Should be ≥2.3.2
   source quatica/bin/activate  # Environment active?
   ```
3. **Run the tutorial**: `python run_analysis.py tutorial`
4. **Check for error messages** in the console output

### Provide this information when asking for help

```bash
# System information
echo "OS: $(uname -a)"
python --version
python -c "import numpy, quaternion, scipy; print(f'numpy: {numpy.__version__}, quaternion: {quaternion.__version__}, scipy: {scipy.__version__}')"
pip list | grep -E "(numpy|quaternion|scipy|matplotlib)"

# Error reproduction
python run_analysis.py tutorial 2>&1 | tee error_log.txt
```

### Contact and Support

- **GitHub Issues**: [https://github.com/vleplat/QuatIca/issues](https://github.com/vleplat/QuatIca/issues)
- **Email**: v dot leplat [at] innopolis dot ru
- **Documentation**: This site and inline code documentation

### Useful debugging commands

```bash
# Test core functionality
python -c "
import numpy as np
import quaternion
from core.utils import quat_matmat, quat_frobenius_norm
from core.solver import NewtonSchulzPseudoinverse

print('Testing core functionality...')
A = quaternion.as_quat_array(np.random.randn(4, 4, 4))
norm = quat_frobenius_norm(A)
print(f'✓ Matrix norm: {norm:.6f}')

solver = NewtonSchulzPseudoinverse()
A_pinv, _, _ = solver.compute(A)
print(f'✓ Pseudoinverse computed, shape: {A_pinv.shape}')
print('All core functions working!')
"

# Test plotting
python -c "
import matplotlib.pyplot as plt
import numpy as np
plt.figure()
plt.plot([1,2,3,4], [1,4,2,3])
plt.savefig('test_plot.png')
print('✓ Plotting works, test_plot.png created')
"
```

Remember: Most issues are related to environment setup, numpy version, or running from the wrong directory. Double-check these basics first!
