"""
Quaternion Hermitian eigendecomposition utilities.

This module provides eigenvalue decomposition algorithms for *Hermitian* quaternion matrices.

Key functions
-------------
- quaternion_eigendecomposition(A): returns (eigenvalues, eigenvectors) for Hermitian quaternion A
- quaternion_eigenvalues(A):       returns eigenvalues only (fast path; no eigenvectors)
- quaternion_eigenvectors(A):      returns eigenvectors only

Implementation notes
--------------------
We follow the QTFM-style approach:

1) Tridiagonalize Hermitian quaternion matrix A:
      P * A * P^H = B
   where B is quaternion tridiagonal (and, in the Hermitian case, can be represented
   in a complex subalgebra spanned by {1, i}).

2) Convert B to a complex Hermitian matrix Bc using the (w + i*x) components.

3) Use LAPACK-backed numpy routines:
   - np.linalg.eigvalsh(Bc) for eigenvalues only (fast, stable)
   - np.linalg.eigh(Bc)     for eigenpairs (fast, stable)

4) Map eigenvectors back: V = P^H * V_B (in quaternion arithmetic).

This module is primarily imported as part of the QuatIca package:
    from quatica.decomp.eigen import quaternion_eigendecomposition
but it also supports legacy flat-import contexts used by some tests.

References
----------
- QTFM (Quaternion Toolbox for MATLAB): http://qtfm.sourceforge.net/
  Stephen J. Sangwine & Nicolas Le Bihan
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
import quaternion as nq  # numpy-quaternion

try:
    # Package mode (normal import path)
    from ..utils import quat_frobenius_norm, quat_hermitian, quat_matmat
    from . import tridiagonalize
except Exception:  # pragma: no cover - compatibility path for legacy flat imports
    from utils import quat_frobenius_norm, quat_hermitian, quat_matmat
    try:
        from decomp import tridiagonalize
    except Exception:
        import tridiagonalize


def _is_hermitian(A: np.ndarray, *, atol: float = 1e-10) -> bool:
    """Cheap-ish Hermitian check for quaternion matrices."""
    AH = quat_hermitian(A)
    # Use Frobenius norm rather than np.allclose (less Python overhead on quaternion dtype)
    diff = A - AH
    nA = quat_frobenius_norm(AH)
    nd = quat_frobenius_norm(diff)
    return nd <= atol * max(1.0, nA)


def _quat_to_complex_wi(B: np.ndarray, *, tol_jk: float = 1e-10) -> np.ndarray:
    """Convert quaternion matrix to complex matrix using w + i*x components.

    Parameters
    ----------
    B : np.ndarray (dtype=quaternion)
        Quaternion matrix expected to lie (approximately) in the {1, i} subalgebra.
    tol_jk : float
        Tolerance for j/k components. If exceeded, we still proceed but issue a RuntimeWarning
        (because the reduction to w+ix is only exact when j,k components are zero).

    Returns
    -------
    Bc : np.ndarray (dtype=complex)
        Complex matrix Bc = w + i*x.
    """
    F = nq.as_float_array(B)  # shape (n,n,4) with components (w,x,y,z)
    y = F[..., 2]
    z = F[..., 3]
    if tol_jk is not None:
        mx = float(max(np.max(np.abs(y)), np.max(np.abs(z))))
        if mx > tol_jk:
            import warnings
            warnings.warn(
                f"Quaternion->complex reduction uses only (w,x) but max |j/k|={mx:.2e} > {tol_jk:.2e}. "
                "Proceeding anyway; results may be less accurate.",
                RuntimeWarning,
            )
    return F[..., 0] + 1j * F[..., 1]


def _complex_to_quat_wi(Z: np.ndarray) -> np.ndarray:
    """Embed a complex matrix Z into quaternions using w + i*x (j=k=0)."""
    Z = np.asarray(Z, dtype=np.complex128)
    F = np.zeros(Z.shape + (4,), dtype=float)
    F[..., 0] = Z.real
    F[..., 1] = Z.imag
    return nq.as_quat_array(F)


def quaternion_eigendecomposition(
    A_quat: np.ndarray,
    *,
    check_hermitian: bool = True,
    atol_herm: float = 1e-10,
    tol_jk: float = 1e-10,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute eigendecomposition of a Hermitian quaternion matrix.

    Parameters
    ----------
    A_quat : np.ndarray (dtype=quaternion)
        Square Hermitian quaternion matrix.
    check_hermitian : bool
        If True, validates A is Hermitian (within atol_herm).
    atol_herm : float
        Tolerance for Hermitian check.
    tol_jk : float
        Tolerance for j/k parts in the tridiagonal matrix reduction.
    verbose : bool
        Print diagnostic info.

    Returns
    -------
    w : np.ndarray (float), shape (n,)
        Real eigenvalues (Hermitian quaternion eigenvalues are real).
    V : np.ndarray (dtype=quaternion), shape (n,n)
        Quaternion eigenvectors (columns), unitary up to numerical error.
    """
    A_quat = np.asarray(A_quat)
    m, n = A_quat.shape
    if m != n:
        raise ValueError("Matrix must be square for eigendecomposition")

    if check_hermitian and not _is_hermitian(A_quat, atol=atol_herm):
        raise ValueError("Matrix must be Hermitian for this eigendecomposition method")

    if m == 1:
        # 1x1 Hermitian quaternion: eigenvalue is real part
        lam = float(A_quat[0, 0].w)
        V = np.array([[nq.quaternion(1.0, 0.0, 0.0, 0.0)]], dtype=np.quaternion)
        return np.array([lam], dtype=float), V

    if verbose:
        print(f"[eigen] tridiagonalizing {m}x{m} Hermitian quaternion matrix")

    # Step 1: tridiagonalize A: P*A*P^H = B
    P, B = tridiagonalize.tridiagonalize(A_quat)

    # Step 2: convert tridiagonal B to complex Hermitian form
    Bc = _quat_to_complex_wi(B, tol_jk=tol_jk)

    # Step 3: Hermitian complex eigendecomposition (fast & stable)
    w, Z = np.linalg.eigh(Bc)  # w real (float), Z complex unitary

    # Step 4: embed Z into quaternion vectors in {1,i}
    Zq = _complex_to_quat_wi(Z)

    # Step 5: map back to eigenvectors of A: V = P^H Zq
    V = quat_matmat(quat_hermitian(P), Zq)

    return np.asarray(w, dtype=float), V


def quaternion_eigenvalues(
    A_quat: np.ndarray,
    *,
    check_hermitian: bool = True,
    atol_herm: float = 1e-10,
    tol_jk: float = 1e-10,
) -> np.ndarray:
    """Compute eigenvalues only (fast path) for a Hermitian quaternion matrix.

    This function avoids computing quaternion eigenvectors altogether.
    """
    A_quat = np.asarray(A_quat)
    m, n = A_quat.shape
    if m != n:
        raise ValueError("Matrix must be square for eigenvalues")

    if check_hermitian and not _is_hermitian(A_quat, atol=atol_herm):
        raise ValueError("Matrix must be Hermitian for this eigenvalue method")

    if m == 1:
        return np.array([float(A_quat[0, 0].w)], dtype=float)

    # Tridiagonalize and convert to complex Hermitian
    _, B = tridiagonalize.tridiagonalize(A_quat)
    Bc = _quat_to_complex_wi(B, tol_jk=tol_jk)

    # Hermitian eigenvalues only
    w = np.linalg.eigvalsh(Bc)
    return np.asarray(w, dtype=float)


def quaternion_eigenvectors(
    A_quat: np.ndarray,
    *,
    check_hermitian: bool = True,
    atol_herm: float = 1e-10,
    tol_jk: float = 1e-10,
    verbose: bool = False,
) -> np.ndarray:
    """Compute eigenvectors only for a Hermitian quaternion matrix."""
    _, V = quaternion_eigendecomposition(
        A_quat,
        check_hermitian=check_hermitian,
        atol_herm=atol_herm,
        tol_jk=tol_jk,
        verbose=verbose,
    )
    return V


def verify_eigendecomposition(
    A_quat: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    *,
    tol: float = 1e-6,
) -> dict:
    """Verify A v ≈ v λ for each eigenpair (Hermitian quaternion).

    Returns a dict with max/mean errors and a boolean success flag.
    """
    A_quat = np.asarray(A_quat)
    w = np.asarray(eigenvalues, dtype=float).reshape(-1)
    V = np.asarray(eigenvectors)

    m, n = A_quat.shape
    errs = []
    max_err = 0.0

    for i in range(min(w.size, V.shape[1])):
        v = V[:, i : i + 1]  # column
        Av = quat_matmat(A_quat, v)

        lam = nq.quaternion(float(w[i]), 0.0, 0.0, 0.0)
        lv = lam * v

        err = quat_frobenius_norm(Av - lv)
        errs.append(err)
        max_err = max(max_err, err)

    errs = np.asarray(errs, dtype=float)
    return {
        "max_error": float(max_err),
        "mean_error": float(np.mean(errs) if errs.size else 0.0),
        "errors": errs.tolist(),
        "success": bool(max_err < tol),
    }


__all__ = [
    "quaternion_eigendecomposition",
    "quaternion_eigenvalues",
    "quaternion_eigenvectors",
    "verify_eigendecomposition",
]
