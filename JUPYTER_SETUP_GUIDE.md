# 🐍 Jupyter Notebook Setup Guide for QuatIca

## 🚨 Common Issue: "ModuleNotFoundError: No module named 'quaternion'"

This guide addresses the most common issue users face when trying to use Jupyter notebooks with QuatIca.

## 🔍 Root Cause Analysis

### **Why This Happens:**
1. **Wrong Kernel Selection**: Jupyter is using the default system/Anaconda kernel instead of the QuatIca virtual environment kernel
2. **Missing Kernel Registration**: The virtual environment kernel hasn't been registered with Jupyter
3. **Package Installation Mismatch**: Packages are installed in the venv but Jupyter is using a different Python environment

### **Why Some Users Don't Have Issues:**
- They have properly registered kernels for their virtual environments
- They're selecting the correct kernel when opening notebooks
- Their Jupyter installation is properly configured

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

# Install Jupyter packages
pip install jupyter notebook jupyterlab ipykernel
```

### **Step 3: Register the Kernel**
```bash
# Register a kernel for your virtual environment
python -m ipykernel install --user --name=quatica-venv --display-name="QuatIca (quatica-venv)"
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
1. Check which kernel you're using: Look at the top-right corner of your notebook
2. If it shows "Python 3" or similar, you're using the wrong kernel
3. Change to "QuatIca (quatica-venv)" kernel
4. Restart the kernel: "Kernel" → "Restart"

### **Problem 2: Kernel Not Available in Dropdown**
**Solution:**
1. Verify kernel registration: `jupyter kernelspec list`
2. Re-register the kernel if needed
3. Restart Jupyter completely

### **Problem 3: Jupyter Opens but Can't Find Packages**
**Solution:**
1. Make sure your virtual environment is activated when you launch Jupyter
2. Check that packages are installed: `pip list | grep quaternion`
3. Verify the kernel points to the correct Python: `jupyter kernelspec show quatica-venv`

### **Problem 4: Multiple Kernels Available**
**Solution:**
- Use the kernel named **"QuatIca (quatica-venv)"** or similar
- Avoid the default `python3` kernel (this is usually the system/Anaconda kernel)

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

```bash
# Complete setup sequence
source quatica/bin/activate
pip install jupyter notebook jupyterlab ipykernel
python -m ipykernel install --user --name=quatica-venv --display-name="QuatIca (quatica-venv)"
jupyter notebook QuatIca_Core_Functionality_Demo.ipynb
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
