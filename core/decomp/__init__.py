"""
Quaternion Matrix Decomposition Module

This module provides implementations of various quaternion matrix decomposition algorithms.
Currently contains placeholder implementations that need to be properly implemented.

All routines operate on quaternion arrays (numpy.quaternion) and leverage existing
utilities from core.utils for quaternion matrix operations.
"""

from .qsvd import classical_qsvd, classical_qsvd_full, rand_qsvd, pass_eff_qsvd
from .eigen import quaternion_eigendecomposition, quaternion_eigenvalues, quaternion_eigenvectors
from .tridiagonalize import tridiagonalize

__all__ = [
    'classical_qsvd', 'classical_qsvd_full', 'rand_qsvd', 'pass_eff_qsvd',
    'quaternion_eigendecomposition', 'quaternion_eigenvalues', 'quaternion_eigenvectors',
    'tridiagonalize'
] 