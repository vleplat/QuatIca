# 🐍 Jupyter Notebook Setup Guide for QuatIca

## 🚨 Common Issue: "ModuleNotFoundError: No module named 'quaternion'"

This guide addresses the most common issue users face when trying to use Jupyter notebooks with QuatIca.

## 🔍 Root Cause Analysis

### **Why This Happens:**
1. **Wrong Kernel Selection**: Jupyter is using the default system/Anaconda kernel instead of the QuatIca virtual environment kernel
2. **Missing Kernel Registration**: The virtual environment kernel hasn't been registered with Jupyter
3. **Package Installation Mismatch**: Packages are installed in the `quatica` venv but Jupyter is using a different Python environment
4. **Missing ipykernel**: The `ipykernel` package is required for kernel registration but often forgotten

### **Why Some Users Don't Have Issues:**
- They have properly registered kernels for their virtual environments
- They're selecting the correct kernel when opening notebooks
- Their Jupyter installation is properly configured
- They installed `ipykernel` along with Jupyter packages

## 🛠️ Complete Solution

### **Step 1: Verify Your Current Setup**
```bash
# Activate your virtual environment
source quatica/bin/activate   # Windows: quatica\Scripts\activate

# Run the verification script
python tests/test_jupyter_setup.py
```

### **Step 2: Install Jupyter in Your Virtual Environment**
```bash
# Make sure your virtual environment is activated
source quatica/bin/activate   # Windows: quatica\Scripts\activate

# Install Jupyter packages (ipykernel is CRITICAL for kernel registration)
pip install jupyter notebook jupyterlab ipykernel
```

**Note for Anaconda Users:**
```bash
# If you're using Anaconda, activate your QuatIca environment first
conda activate quatica  # If you created conda env instead of venv
# OR
source quatica/bin/activate  # If you created venv despite having Anaconda

# Then install Jupyter packages
pip install jupyter notebook jupyterlab ipykernel
```

### **Step 3: Register the Kernel**
```bash
# Register a kernel for your virtual environment
python -m ipykernel install --user --name=quatica-venv --display-name="QuatIca (quatica-venv)"
```

**Windows Users - Additional Steps:**
```bash
# On Windows, you might need to specify the full path
python -m ipykernel install --user --name=quatica-venv --display-name="QuatIca (quatica-venv)" --prefix=%USERPROFILE%

# Or if that fails, try without --user flag (requires admin)
python -m ipykernel install --name=quatica-venv --display-name="QuatIca (quatica-venv)"
```

### **Step 4: Launch Jupyter and Select the Correct Kernel**
```bash
# Launch Jupyter (make sure venv is activated)
jupyter notebook QuatIca_Core_Functionality_Demo.ipynb

# Or launch JupyterLab
jupyter lab QuatIca_Core_Functionality_Demo.ipynb
```

### **Step 5: Select the Correct Kernel in Your Notebook**
1. Open the notebook
2. **CRITICAL**: Click on "Kernel" → "Change kernel" → Select **"QuatIca (quatica-venv)"**
3. If you don't see the kernel, restart Jupyter and try again

## 🔧 Troubleshooting

### **Problem 1: `ModuleNotFoundError: No module named 'quaternion'`**
**Solution:**
1. **Check which kernel you're using**: Look at the top-right corner of your notebook
2. If it shows "Python 3" or similar, you're using the wrong kernel
3. **Change to "QuatIca (quatica-venv)" kernel**: Kernel → Change kernel → Select "QuatIca (quatica-venv)"
4. **Restart the kernel**: Kernel → Restart

### **Problem 2: Kernel Not Available in Dropdown**
**Solutions:**
1. **Check registration**: `jupyter kernelspec list` (should show `quatica-venv`)
2. **Re-register if missing**: 
   ```bash
   source quatica/bin/activate
   python -m ipykernel install --user --name=quatica-venv --display-name="QuatIca (quatica-venv)"
   ```
3. **Restart Jupyter completely** and refresh browser

### **Problem 3: Jupyter Opens but Can't Find Packages**
**Step-by-step diagnosis:**
1. **Verify environment activation**: 
   ```bash
   source quatica/bin/activate
   which python  # Should point to quatica/bin/python
   ```
2. **Check package installation**: `pip list | grep quaternion`
3. **Verify kernel configuration**: `jupyter kernelspec show quatica-venv`
4. **Check Python path in kernel**: Look for the `argv` field pointing to correct Python

### **Problem 4: Multiple Kernels Available**
**Best practices:**
- ✅ **Use**: "QuatIca (quatica-venv)" or similar custom kernel
- ❌ **Avoid**: Default "Python 3" kernel (system/Anaconda Python)
- ❌ **Avoid**: "base" or "root" environment kernels

### **Problem 5: Anaconda/Conda Conflicts**
**Solutions:**
1. **If using conda envs**: Make sure you created `quatica` as a conda environment
2. **If mixing conda/pip**: Stick to one package manager within the environment
3. **Conda kernel registration**:
   ```bash
   conda activate quatica
   conda install ipykernel
   python -m ipykernel install --user --name=quatica-venv --display-name="QuatIca (quatica-venv)"
   ```

### **Problem 6: Windows-Specific Issues**
**Common Windows solutions:**
1. **Use Command Prompt or PowerShell**, not Git Bash for kernel registration
2. **Run as Administrator** if kernel registration fails
3. **Check Windows PATH** if `jupyter` command not found
4. **Use forward slashes** in paths even on Windows when possible

## ✅ Verification

After setup, run this in a notebook cell to verify everything works:
```python
import quaternion
import numpy as np
print("✅ QuatIca environment working correctly!")
print(f"numpy version: {np.__version__}")
print(f"quaternion version: {quaternion.__version__}")
```

Or run the verification script:
```bash
python tests/test_jupyter_setup.py
```

## 📋 Quick Reference Commands

### **Complete Setup (Copy-Paste Ready)**
```bash
# 1. Activate virtual environment
source quatica/bin/activate   # Windows: quatica\Scripts\activate

# 2. Install Jupyter packages (all at once)
pip install jupyter notebook jupyterlab ipykernel

# 3. Register kernel
python -m ipykernel install --user --name=quatica-venv --display-name="QuatIca (quatica-venv)"

# 4. Verify setup
python tests/test_jupyter_setup.py

# 5. Launch notebook
jupyter notebook QuatIca_Core_Functionality_Demo.ipynb
# Remember: Select "QuatIca (quatica-venv)" kernel in the notebook!
```

### **Diagnosis Commands**
```bash
# Check if kernel is registered
jupyter kernelspec list

# Check kernel details
jupyter kernelspec show quatica-venv

# Verify packages in environment
source quatica/bin/activate
pip list | grep -E "(quaternion|numpy|jupyter)"

# Check Python path
which python
```

## 🎯 Key Points to Remember

1. **Always activate your virtual environment** before launching Jupyter
2. **Select the correct kernel** in your notebook (not the default Python 3)
3. **Restart the kernel** if you change kernels
4. **Use the verification script** to check your setup

## 📞 Getting Help

If you're still having issues:
1. Run `python tests/test_jupyter_setup.py` and share the output
2. Check which kernel your notebook is using
3. Verify that `numpy-quaternion` is installed in your virtual environment
4. Make sure you're launching Jupyter from the activated virtual environment

---

**Note**: This guide assumes you're using the `quatica` virtual environment. If you named your environment differently, replace `quatica` with your environment name in the commands above.
