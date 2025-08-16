"""
Quaternion Matrix Decomposition Module

This module provides implementations of various quaternion matrix decomposition algorithms.
Currently contains placeholder implementations that need to be properly implemented.

All routines operate on quaternion arrays (numpy.quaternion) and leverage existing
utilities from quatica.utils for quaternion matrix operations.
"""

from .eigen import (
    quaternion_eigendecomposition,
    quaternion_eigenvalues,
    quaternion_eigenvectors,
)
from .LU import (
    quaternion_lu,
    quaternion_modulus,
    quaternion_tril,
    quaternion_triu,
    verify_lu_decomposition,
)
from .qsvd import classical_qsvd, classical_qsvd_full, pass_eff_qsvd, rand_qsvd
from .schur import (
    quaternion_schur,
    quaternion_schur_pure,
    quaternion_schur_pure_implicit,
    quaternion_schur_unified,
)
from .tridiagonalize import tridiagonalize

__all__ = [
    "classical_qsvd",
    "classical_qsvd_full",
    "rand_qsvd",
    "pass_eff_qsvd",
    "quaternion_eigendecomposition",
    "quaternion_eigenvalues",
    "quaternion_eigenvectors",
    "tridiagonalize",
    "quaternion_lu",
    "verify_lu_decomposition",
    "quaternion_triu",
    "quaternion_tril",
    "quaternion_modulus",
    "quaternion_schur",
    "quaternion_schur_pure",
    "quaternion_schur_pure_implicit",
    "quaternion_schur_unified",
]
