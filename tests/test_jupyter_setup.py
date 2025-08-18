#!/usr/bin/env python3
"""
Jupyter Setup Verification Script for QuatIca

This script helps verify that your Jupyter notebook environment is properly configured
to work with QuatIca. Run this script after following the Jupyter setup instructions
in the README.md file.

Usage:
    python tests/test_jupyter_setup.py
"""

import subprocess
import sys


def check_package(package_name, import_name=None):
    """Check if a package is available and print its version."""
    if import_name is None:
        import_name = package_name

    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "unknown")
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError as e:
        print(f"❌ {package_name}: Not found ({e})")
        return False


def check_jupyter_kernels():
    """Check available Jupyter kernels."""
    try:
        result = subprocess.run(
            ["jupyter", "kernelspec", "list"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print("\n📋 Available Jupyter kernels:")
            print(result.stdout)
            return True
        else:
            print(f"❌ Failed to list kernels: {result.stderr}")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"❌ Jupyter not available: {e}")
        return False


def main():
    print("🔍 QuatIca Jupyter Setup Verification")
    print("=" * 50)

    # Check Python environment
    print(f"🐍 Python: {sys.version}")
    print(f"📍 Python executable: {sys.executable}")

    # Check if we're in a virtual environment
    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        print("✅ Running in a virtual environment")
    else:
        print("⚠️  Not running in a virtual environment (this might be OK)")

    print("\n📦 Checking required packages:")

    # Check core packages
    packages = [
        ("numpy", "numpy"),
        ("numpy-quaternion", "quaternion"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("scikit-learn", "sklearn"),
        ("seaborn", "seaborn"),
    ]

    all_good = True
    for package, import_name in packages:
        if not check_package(package, import_name):
            all_good = False

    # Check Jupyter packages
    print("\n📓 Checking Jupyter packages:")
    jupyter_packages = [
        ("jupyter", "jupyter"),
        ("jupyterlab", "jupyterlab"),
        ("ipykernel", "ipykernel"),
    ]

    jupyter_available = True
    for package, import_name in jupyter_packages:
        if not check_package(package, import_name):
            jupyter_available = False

    # Check Jupyter kernels
    if jupyter_available:
        check_jupyter_kernels()

    # Summary and recommendations
    print("\n" + "=" * 50)
    print("📋 SUMMARY AND RECOMMENDATIONS:")

    if all_good:
        print("✅ All core packages are available!")
    else:
        print("❌ Some core packages are missing.")
        print("   Run: pip install -r requirements.txt")

    if jupyter_available:
        print("✅ Jupyter packages are available!")
        print("\n📝 Next steps for Jupyter:")
        print(
            "1. Register a kernel: python -m ipykernel install --user --name=quat-venv --display-name='QuatIca (quat-venv)'"
        )
        print("2. Launch Jupyter: jupyter notebook QuatIca_Core_Functionality_Demo.ipynb")
        print("3. Select the 'QuatIca (quat-venv)' kernel in your notebook")
    else:
        print("❌ Jupyter packages are not available.")
        print("   Run: pip install jupyter notebook jupyterlab ipykernel")

    print(
        "\n📖 For complete setup instructions, see the 'Jupyter Notebook Setup' section in README.md"
    )


if __name__ == "__main__":
    main()
