"""
Quaternion Cholesky decompositions.

This module adds:
  - Native dense quaternion Cholesky for Hermitian PD matrices (no embedding)
  - Sparse Cholesky via complex embedding (CHOLMOD via scikit-sparse, optional)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import quaternion  # type: ignore
from scipy import sparse

try:
    # Package mode (normal import path)
    from ..utils import (
        SparseQuaternionMatrix,
        complex_expand_sparse,
        quat_hermitian,
        quat_matmat,
        quat_frobenius_norm,
    )
except Exception:  # pragma: no cover - compatibility path for legacy flat imports
    from utils import (  # type: ignore
        SparseQuaternionMatrix,
        complex_expand_sparse,
        quat_hermitian,
        quat_matmat,
        quat_frobenius_norm,
    )


def _quat_conj(w: float, x: float, y: float, z: float) -> tuple[float, float, float, float]:
    return (w, -x, -y, -z)


def _quat_mul(
    aw: float, ax: float, ay: float, az: float, bw: float, bx: float, by: float, bz: float
) -> tuple[float, float, float, float]:
    # (a)(b) with Hamilton product.
    cw = aw * bw - ax * bx - ay * by - az * bz
    cx = aw * bx + ax * bw + ay * bz - az * by
    cy = aw * by - ax * bz + ay * bw + az * bx
    cz = aw * bz + ax * by - ay * bx + az * bw
    return cw, cx, cy, cz


def chol_quat_dense(
    A: np.ndarray,
    *,
    tol: float = 1e-12,
    hermitianize: bool = False,
    jitter: float = 0.0,
) -> np.ndarray:
    r"""Compute a dense quaternion Cholesky factorization \(A = L L^*\).

    This is a *native* quaternion implementation for Hermitian positive definite
    (HPD) matrices, without any real/complex embedding.

    The algorithm is the quaternion analogue of classical left-looking Cholesky,
    with the key property that each pivot is real:

    - Pivot (real):
      \(s_k = a_{kk} - \\sum_{j<k} L_{kj}\\overline{L_{kj}} = a_{kk} - \\sum_{j<k}|L_{kj}|^2 \\in \\mathbb{R}\)
      and \(L_{kk} = \\sqrt{s_k} \\in \\mathbb{R}_{>0}\).
    - Column update:
      \(t_{ik} = a_{ik} - \\sum_{j<k} L_{ij}\\overline{L_{kj}}\), then
      \(L_{ik} = t_{ik}/L_{kk}\) (safe since \(L_{kk}\\) is real).

    Complexity is \(O(n^3)\). The inner accumulation for the column update is
    implemented with NumPy vectorization over the row index \(i\) (and reduction
    over \(j\)), avoiding Python loops over \(i,j\).

    Args:
        A: Dense quaternion Hermitian matrix of shape (n, n) with dtype
            `np.quaternion`.
        tol: Pivot tolerance. If any pivot \(s_k \\le tol\), the matrix is treated
            as not HPD and a `LinAlgError` is raised.
        hermitianize: If True, symmetrize input as `0.5*(A + A^*)` before
            factorization (useful when small numerical asymmetry is present).
        jitter: Optional diagonal shift added to the *real* diagonal of `A`
            before factorization (common stabilization trick).

    Returns:
        L: Dense quaternion lower-triangular matrix (n, n) such that
        \(A \\approx L L^*\), with real positive diagonal.

    Raises:
        ValueError: If `A` is not a square 2D quaternion array, or if diagonal
            entries are not approximately real.
        numpy.linalg.LinAlgError: If a non-positive pivot is encountered.
    """
    if not (isinstance(A, np.ndarray) and A.dtype == np.quaternion and A.ndim == 2):
        raise ValueError("A must be a 2D dense quaternion ndarray")
    n, n2 = A.shape
    if n != n2:
        raise ValueError("A must be square")

    if hermitianize:
        A = 0.5 * (A + quat_hermitian(A))

    Af = quaternion.as_float_array(A).astype(float, copy=False)  # (n,n,4)
    if jitter != 0.0:
        Af[np.arange(n), np.arange(n), 0] += float(jitter)

    Lw = np.zeros((n, n), dtype=float)
    Lx = np.zeros((n, n), dtype=float)
    Ly = np.zeros((n, n), dtype=float)
    Lz = np.zeros((n, n), dtype=float)

    for k in range(n):
        # Pivot: s_k = a_kk - sum_{j<k} |L_kj|^2 must be real positive.
        a_kk_w, a_kk_x, a_kk_y, a_kk_z = Af[k, k]
        if max(abs(a_kk_x), abs(a_kk_y), abs(a_kk_z)) > 10 * tol:
            raise ValueError(
                f"A[{k},{k}] is not (approximately) real; got imag-norm "
                f"{max(abs(a_kk_x), abs(a_kk_y), abs(a_kk_z)):.3e}."
            )

        if k == 0:
            sum_norm2 = 0.0
        else:
            rw = Lw[k, :k]
            rx = Lx[k, :k]
            ry = Ly[k, :k]
            rz = Lz[k, :k]
            sum_norm2 = float(np.dot(rw, rw) + np.dot(rx, rx) + np.dot(ry, ry) + np.dot(rz, rz))

        s_k = a_kk_w - sum_norm2
        if not np.isfinite(s_k) or s_k <= tol:
            raise np.linalg.LinAlgError(f"Matrix is not HPD at pivot k={k} (s_k={s_k}).")

        Lkk = float(np.sqrt(s_k))
        Lw[k, k] = Lkk  # diagonal is strictly real

        # Column entries (vectorized over i): L_ik = (a_ik - sum_{j<k} L_ij * conj(L_kj)) / Lkk
        if k + 1 < n:
            tw = Af[k + 1 :, k, 0].copy()
            tx = Af[k + 1 :, k, 1].copy()
            ty = Af[k + 1 :, k, 2].copy()
            tz = Af[k + 1 :, k, 3].copy()

            if k > 0:
                # Li? shape (p, k), bk shape (k,)
                aw = Lw[k + 1 :, :k]
                ax = Lx[k + 1 :, :k]
                ay = Ly[k + 1 :, :k]
                az = Lz[k + 1 :, :k]

                bw = Lw[k, :k]
                bx = -Lx[k, :k]
                by = -Ly[k, :k]
                bz = -Lz[k, :k]

                pw = aw * bw - ax * bx - ay * by - az * bz
                px = aw * bx + ax * bw + ay * bz - az * by
                py = aw * by - ax * bz + ay * bw + az * bx
                pz = aw * bz + ax * by - ay * bx + az * bw

                tw -= np.sum(pw, axis=1)
                tx -= np.sum(px, axis=1)
                ty -= np.sum(py, axis=1)
                tz -= np.sum(pz, axis=1)

            inv = 1.0 / Lkk
            Lw[k + 1 :, k] = tw * inv
            Lx[k + 1 :, k] = tx * inv
            Ly[k + 1 :, k] = ty * inv
            Lz[k + 1 :, k] = tz * inv

    Lf = np.zeros((n, n, 4), dtype=float)
    Lf[..., 0] = Lw
    Lf[..., 1] = Lx
    Lf[..., 2] = Ly
    Lf[..., 3] = Lz
    return quaternion.as_quat_array(Lf)


def solve_chol_quat_dense(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    r"""Solve a quaternion linear system using a Cholesky factor.

    Solves \(Ax=b\) given \(A = L L^*\) where `L` is the output of
    `chol_quat_dense`.

    This uses forward/back substitution:
    - Forward: solve \(Ly=b\)
    - Backward: solve \(L^*x=y\) where \((L^*)_{ij} = \\overline{L_{ji}}\)

    Args:
        L: Lower-triangular quaternion matrix of shape (n, n), with real positive
            diagonal, typically returned by `chol_quat_dense`.
        b: Quaternion right-hand side vector (n,) or matrix (n, nrhs).

    Returns:
        x: Quaternion solution with same shape as `b`.

    Raises:
        ValueError: If inputs have incompatible shapes/dtypes.
        numpy.linalg.LinAlgError: If `L` has a non-positive diagonal.
    """
    if not (isinstance(L, np.ndarray) and L.dtype == np.quaternion and L.ndim == 2):
        raise ValueError("L must be a 2D dense quaternion ndarray")
    n, n2 = L.shape
    if n != n2:
        raise ValueError("L must be square")

    if not (isinstance(b, np.ndarray) and b.dtype == np.quaternion):
        raise ValueError("b must be a quaternion ndarray")

    if b.ndim == 1:
        Bf = quaternion.as_float_array(b.reshape(n, 1)).astype(float, copy=False)  # (n,1,4)
        nrhs = 1
    elif b.ndim == 2:
        if b.shape[0] != n:
            raise ValueError("b has incompatible shape")
        Bf = quaternion.as_float_array(b).astype(float, copy=False)  # (n,r,4)
        nrhs = b.shape[1]
    else:
        raise ValueError("b must be 1D or 2D quaternion array")

    Lf = quaternion.as_float_array(L).astype(float, copy=False)  # (n,n,4)
    Lw, Lx, Ly, Lz = Lf[..., 0], Lf[..., 1], Lf[..., 2], Lf[..., 3]

    # Forward solve: L y = b
    Y = np.zeros((n, nrhs, 4), dtype=float)
    for i in range(n):
        diag = Lw[i, i]
        if diag <= 0.0:
            raise np.linalg.LinAlgError("Invalid Cholesky factor: nonpositive diagonal.")
        inv = 1.0 / diag
        for r in range(nrhs):
            tw, tx, ty, tz = Bf[i, r]
            for j in range(i):
                aw, ax, ay, az = Lw[i, j], Lx[i, j], Ly[i, j], Lz[i, j]
                bw, bx, by, bz = Y[j, r]
                pw, px, py, pz = _quat_mul(aw, ax, ay, az, bw, bx, by, bz)
                tw -= pw
                tx -= px
                ty -= py
                tz -= pz
            Y[i, r, 0] = tw * inv
            Y[i, r, 1] = tx * inv
            Y[i, r, 2] = ty * inv
            Y[i, r, 3] = tz * inv

    # Back solve: L^* x = y, where (L^*)_{ij} = conj(L_{ji})
    X = np.zeros((n, nrhs, 4), dtype=float)
    for i in range(n - 1, -1, -1):
        diag = Lw[i, i]
        inv = 1.0 / diag
        for r in range(nrhs):
            tw, tx, ty, tz = Y[i, r]
            for j in range(i + 1, n):
                # conj(L_{j,i}) * x_j
                aw, ax, ay, az = _quat_conj(Lw[j, i], Lx[j, i], Ly[j, i], Lz[j, i])
                bw, bx, by, bz = X[j, r]
                pw, px, py, pz = _quat_mul(aw, ax, ay, az, bw, bx, by, bz)
                tw -= pw
                tx -= px
                ty -= py
                tz -= pz
            X[i, r, 0] = tw * inv
            X[i, r, 1] = tx * inv
            X[i, r, 2] = ty * inv
            X[i, r, 3] = tz * inv

    xq = quaternion.as_quat_array(X)
    if b.ndim == 1:
        return xq.reshape(n)
    return xq


def _pack_quat_rhs_to_complex(b: np.ndarray) -> np.ndarray:
    if not (isinstance(b, np.ndarray) and b.dtype == np.quaternion):
        raise ValueError("b must be a quaternion ndarray")
    if b.ndim == 1:
        bf = quaternion.as_float_array(b).astype(float, copy=False)  # (n,4)
        u = bf[:, 0] + 1j * bf[:, 1]
        v = bf[:, 2] + 1j * bf[:, 3]
        return np.concatenate([u, v], axis=0)
    if b.ndim == 2:
        bf = quaternion.as_float_array(b).astype(float, copy=False)  # (n,r,4)
        u = bf[..., 0] + 1j * bf[..., 1]  # (n,r)
        v = bf[..., 2] + 1j * bf[..., 3]  # (n,r)
        return np.vstack([u, v])
    raise ValueError("b must be 1D or 2D quaternion array")


def _unpack_complex_to_quat_vec(z: np.ndarray, n: int) -> np.ndarray:
    z = np.asarray(z)
    if z.ndim == 1:
        if z.shape[0] != 2 * n:
            raise ValueError("z has incompatible length")
        u = z[:n]
        v = z[n:]
        xf = np.empty((n, 4), dtype=float)
        xf[:, 0] = np.real(u)
        xf[:, 1] = np.imag(u)
        xf[:, 2] = np.real(v)
        xf[:, 3] = np.imag(v)
        return quaternion.as_quat_array(xf)
    if z.ndim == 2:
        if z.shape[0] != 2 * n:
            raise ValueError("z has incompatible shape")
        u = z[:n, :]
        v = z[n:, :]
        xf = np.empty((n, z.shape[1], 4), dtype=float)
        xf[..., 0] = np.real(u)
        xf[..., 1] = np.imag(u)
        xf[..., 2] = np.real(v)
        xf[..., 3] = np.imag(v)
        return quaternion.as_quat_array(xf)
    raise ValueError("z must be 1D or 2D complex array")


@dataclass(frozen=True)
class QuatSparseCholeskyFactor:
    factor: Any
    n: int
    backend: str = "cholmod"

    def solve(self, b: np.ndarray) -> np.ndarray:
        r"""Solve \(Ax=b\) for quaternion `b` using the embedded complex factorization."""
        rhs = _pack_quat_rhs_to_complex(b)
        z = self.factor.solve(rhs)
        return _unpack_complex_to_quat_vec(z, self.n)

    def logdet(self) -> float:
        r"""Return \(\log\det(A)\) for quaternion Hermitian PD \(A\) (when supported).

        For Hermitian quaternion PD matrices, the complex embedding satisfies
        \(\det(\\chi(A)) = \det(A)^2\) (Moore determinant relationship), hence:
        \(\log\\det(A) = \\tfrac12 \log\\det(\\chi(A))\).
        """
        # For quaternion Hermitian PD, det(χ(A)) = det(A)^2 (Moore determinant),
        # so logdet(A) = 0.5 * logdet(χ(A)).
        if not hasattr(self.factor, "logdet"):
            raise NotImplementedError("Backend factor does not provide logdet().")
        return 0.5 * float(self.factor.logdet())


def chol_quat_sparse(
    Aq: SparseQuaternionMatrix,
    *,
    tol: float = 1e-12,
    jitter: float = 0.0,
    ordering: Literal["cholmod", "paired"] = "cholmod",
) -> QuatSparseCholeskyFactor:
    r"""Sparse quaternion Cholesky via complex embedding (CHOLMOD backend).

    This factors a sparse quaternion Hermitian PD matrix \(A\\) by embedding it
    into its \(2n\\times 2n\) complex *adjoint/symplectic* representation:

    \[
        \\chi(A)=\\begin{bmatrix}X & Y\\\\-\\overline{Y} & \\overline{X}\\end{bmatrix},
        \\quad X = A_w + i A_x,\\; Y = A_y + i A_z.
    \]

    Then it calls CHOLMOD on the complex sparse matrix \(\\chi(A)\).
    Solves are performed by packing quaternion right-hand sides \(b\) into a
    complex vector \([u; v]\) (with \(u=b_w + i b_x\), \(v=b_y + i b_z\)).

    Notes:
        - This requires the optional dependency `scikit-sparse` (CHOLMOD).
        - The `ordering="paired"` option is currently a placeholder hook; it
          still uses CHOLMOD's internal ordering.

    Args:
        Aq: Sparse quaternion matrix in component form (`SparseQuaternionMatrix`)
            with shape (n, n).
        tol: Reserved for future numerical checks (kept for API stability).
        jitter: Optional diagonal shift added to \(\\chi(A)\) before factoring.
        ordering: `"cholmod"` (default) or `"paired"` (hook for lifted ordering).

    Returns:
        A `QuatSparseCholeskyFactor` object exposing `.solve(b)` and (optionally)
        `.logdet()`.

    Raises:
        ValueError: If `Aq` is not square or not a `SparseQuaternionMatrix`.
        ImportError: If `scikit-sparse` / CHOLMOD is not available.
    """
    if not isinstance(Aq, SparseQuaternionMatrix):
        raise ValueError("Aq must be a SparseQuaternionMatrix")
    n, n2 = Aq.shape
    if n != n2:
        raise ValueError("Aq must be square")

    # CHOLMOD works best with CSC.
    chiA = complex_expand_sparse(Aq).tocsc()
    if jitter != 0.0:
        chiA = chiA + sparse.eye(2 * n, format="csc", dtype=np.complex128) * complex(jitter)

    try:
        from sksparse.cholmod import cholesky as cholmod_cholesky  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "chol_quat_sparse requires scikit-sparse (sksparse) with CHOLMOD. "
            "Install via `pip install scikit-sparse` (system libs may be required)."
        ) from e

    # Ordering hooks: keep API, but default to CHOLMOD's own ordering.
    if ordering == "cholmod":
        F = cholmod_cholesky(chiA)
        return QuatSparseCholeskyFactor(factor=F, n=n, backend="cholmod")

    if ordering == "paired":
        # Simple "paired" lift: compute permutation on quaternion graph then lift.
        # Here we rely on CHOLMOD ordering for now, but keep parameter for future work.
        F = cholmod_cholesky(chiA)
        return QuatSparseCholeskyFactor(factor=F, n=n, backend="cholmod/paired")

    raise ValueError(f"Unknown ordering: {ordering}")


def _dense_reconstruct_err(A: np.ndarray, L: np.ndarray) -> float:
    LLh = quat_matmat(L, quat_hermitian(L))
    num = quat_frobenius_norm(A - LLh)
    den = quat_frobenius_norm(A) + 1e-30
    return float(num / den)

