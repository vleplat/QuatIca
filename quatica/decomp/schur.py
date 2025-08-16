"""
Quaternion Schur Decomposition via Hessenberg reduction and implicit shifted QR.

Strategy:
- Reduce once to Hessenberg: H0 = P0 * A * P0^H
- Iterate implicit shifted QR steps in Realp (4n x 4n) using quaternion-structured
  Givens rotations and optional Francis double-shift for improved convergence:
    HR_{k+1} = R_k @ Q_k + sigma_k I, where Q_k R_k ≈ HR_k - sigma_k I
- Accumulate total Q in real block, then contract to quaternion and combine with P0

Notes:
- Shift: Rayleigh, Wilkinson (1x1/2x2 trailing block), or double (Francis-style two shifts)
- Deflation: aggressively zero H[i, i-1] when sufficiently small within the active window
- All operations preserve quaternion structure through Realp mapping + fixed 8x8 permutation
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np
import quaternion  # type: ignore

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    ggivens,
    quat_hermitian,
    quat_matmat,
    real_contract,
    real_expand,
)

from .hessenberg import check_hessenberg, hessenbergize
from .tridiagonalize import householder_matrix


def _quat_scalar_abs(q: quaternion.quaternion) -> float:
    return float(np.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z))


# (Legacy helper removed: deflation handled inline in algorithms)


def _estimate_shifts_power_deflate(H: np.ndarray, steps: int = 5) -> list[float]:
    """Estimate a schedule of real scalar shifts via power iteration with simple deflation.

    Strategy: for k = n, n-1, ..., 1 run 'steps' power iterations on the
    leading k x k block H[:k,:k], compute the Rayleigh quotient (real scalar part),
    and record it as a shift. This provides a bottom-up schedule suitable for
    deflating trailing positions during QR.
    """
    n = H.shape[0]
    shifts: list[float] = []
    rng = np.random.default_rng(0)
    for k in range(n, 0, -1):
        Hk = H[:k, :k]
        # initialize random quaternion vector length k
        xr = rng.standard_normal((k,))
        xi = rng.standard_normal((k,))
        xj = rng.standard_normal((k,))
        xk = rng.standard_normal((k,))
        x = quaternion.as_quat_array(np.stack([xr, xi, xj, xk], axis=-1))  # (k,)
        x = x.reshape(k, 1)

        # normalize
        def _vec_norm(v: np.ndarray) -> float:
            vf = quaternion.as_float_array(v.reshape(-1))  # (k,4)
            return float(np.sqrt(np.sum(vf * vf)))

        nrm = _vec_norm(x)
        if nrm > 0:
            x = x / nrm
        for _ in range(steps):
            x = quat_matmat(Hk, x)
            nrm = _vec_norm(x)
            if not np.isfinite(nrm) or nrm == 0.0:
                break
            x = x / nrm
        xH = quat_hermitian(x)
        num = quat_matmat(xH, quat_matmat(Hk, x))[0, 0]
        den = quat_matmat(xH, x)[0, 0]
        mu = float(num.w) / (float(den.w) + 1e-30)
        shifts.append(mu)
    return shifts


def quaternion_schur(
    A: np.ndarray,
    max_iter: int = 5000,
    tol: float = 1e-12,
    shift: str = "wilkinson",
    verbose: bool = False,
    return_diagnostics: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a (quaternion) Schur-like decomposition A = Q T Q^H.

    Parameters:
    - A: square quaternion matrix (n x n)
    - max_iter: maximum number of QR iterations
    - tol: deflation tolerance for subdiagonal entries
    - shift: 'rayleigh' | 'wilkinson' | 'double' (Francis two-shift surrogate)
    - verbose: print progress

    Returns:
    - Q: unitary quaternion matrix
    - T: upper (quasi-)triangular quaternion matrix (Schur form)
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("quaternion_schur requires a square matrix")

    n = A.shape[0]
    if n == 0:
        return A.copy(), A.copy()

    # Step 1: Hessenberg reduction
    P0, H = hessenbergize(A)
    H = check_hessenberg(H)

    # Work entirely in real-block form for QR iterations
    HR = real_expand(H)  # shape (4n, 4n)
    Q_real = np.eye(4 * n)

    eye4n = np.eye(4 * n)

    # (Legacy shift helper removed; shift selection performed inline)

    # Fixed 8x8 permutation mapping contiguous [w_i,x_i,y_i,z_i,w_{i+1},x_{i+1},y_{i+1},z_{i+1}]
    # to stacked [w_i,w_{i+1},x_i,x_{i+1},y_i,y_{i+1},z_i,z_{i+1}]
    P8 = np.zeros((8, 8))
    # mapping indices: 0->0, 4->1, 1->2, 5->3, 2->4, 6->5, 3->6, 7->7
    for src, dst in [(0, 0), (4, 1), (1, 2), (5, 3), (2, 4), (6, 5), (3, 6), (7, 7)]:
        P8[dst, src] = 1.0

    # Diagnostics container
    diag = {
        "iterations": [],  # list of dicts per iteration
        "converged": False,
        "iterations_run": 0,
    }

    # Helper: apply a single implicit shift sweep (bulge chase) with given sigma
    def _apply_single_shift(
        HR_in: np.ndarray, sigma_val: float, m_sz: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply one implicit single-shift sweep (bulge init + full-window chase).

        We construct rotations from the shifted matrix HR - sigma I to properly
        initialize the bulge and then chase it across the active window.
        """
        HRs = HR_in - sigma_val * eye4n
        Qk = np.eye(4 * n)
        # Read components from the shifted matrix (avoid cancelling the shift)
        H_curr = real_contract(HRs, n, n)
        for s in range(0, m_sz - 1):
            h11 = H_curr[s, s]
            h21 = H_curr[s + 1, s]
            x1 = np.array([h11.w, h11.x, h11.y, h11.z])
            x2 = np.array([h21.w, h21.x, h21.y, h21.z])
            G = ggivens(x1, x2)
            r0 = 4 * s
            r1 = 4 * (s + 1)
            row_idx = list(range(r0, r0 + 4)) + list(range(r1, r1 + 4))
            col_idx = row_idx
            Gc_left = P8.T @ G.T @ P8
            Gc_right = P8.T @ G @ P8
            HRs[row_idx, :] = Gc_left @ HRs[row_idx, :]
            HRs[:, col_idx] = HRs[:, col_idx] @ Gc_right
            Qk[:, col_idx] = Qk[:, col_idx] @ Gc_right
        # Add back the shift
        HR_out = HRs + sigma_val * eye4n
        return HR_out, Qk

    # Active trailing block size and stagnation tracking
    m_active = n
    prev_max_sub = float("inf")
    stagnation_count = 0
    k = 0
    while k < max_iter and m_active > 1:
        # Map back to quaternion to choose shift and check deflation
        H = real_contract(HR, n, n)

        # Strong deflation: zero tiny subdiagonals within active window
        deflated_idx = []
        for i in range(1, m_active):
            h_sub = H[i, i - 1]
            denom = (
                _quat_scalar_abs(H[i - 1, i - 1])
                + _quat_scalar_abs(H[i, i])
                + _quat_scalar_abs(h_sub)
            )
            if _quat_scalar_abs(h_sub) <= tol * max(1.0, denom):
                H[i, i - 1] = quaternion.quaternion(0.0, 0.0, 0.0, 0.0)
                deflated_idx.append(int(i))

        # Shrink active window from the bottom if deflated
        while m_active > 1 and _quat_scalar_abs(H[m_active - 1, m_active - 2]) <= tol:
            m_active -= 1

        # Re-expand after deflation
        HR = real_expand(H)

        # Check convergence in active window
        subdiag_norm = 0.0
        for i in range(1, m_active):
            subdiag_norm = max(subdiag_norm, _quat_scalar_abs(H[i, i - 1]))
        if subdiag_norm <= tol:
            if verbose:
                print(
                    f"Converged at iter {k}: active={m_active}, subdiag {subdiag_norm:.2e}"
                )
            # Record final iteration diagnostics
            diag_entry = {
                "iter": k,
                "m_active": int(m_active),
                "shift_mode": None,
                "sigma": None,
                "subdiag_max": float(subdiag_norm),
                "subdiag_vector": [
                    float(
                        np.linalg.norm(
                            [H[i, i - 1].w, H[i, i - 1].x, H[i, i - 1].y, H[i, i - 1].z]
                        )
                    )
                    for i in range(1, m_active)
                ],
                "deflated_indices": deflated_idx,
                "stagnation_count": int(stagnation_count),
            }
            diag["iterations"].append(diag_entry)
            diag["converged"] = True
            diag["iterations_run"] = k + 1
            break

        # Adaptive restart: detect stagnation and swap shift strategy
        if subdiag_norm >= 0.95 * prev_max_sub:
            stagnation_count += 1
        else:
            stagnation_count = 0
        prev_max_sub = subdiag_norm

        current_shift_mode = shift
        # If stagnating, switch strategy (toggle between wilkinson and rayleigh)
        if stagnation_count >= 50:
            current_shift_mode = "rayleigh" if shift == "wilkinson" else "wilkinson"
            stagnation_count = 0
        # If still stagnating for long, try a double-shift step
        if stagnation_count >= 20 and m_active >= 2:
            current_shift_mode = "double"
            stagnation_count = 0

        # Choose shift(s) from trailing block of active window
        if current_shift_mode in ("wilkinson", "double") and m_active >= 2:
            w11 = float(H[m_active - 2, m_active - 2].w)
            w12 = float(H[m_active - 2, m_active - 1].w)
            w21 = float(H[m_active - 1, m_active - 2].w)
            w22 = float(H[m_active - 1, m_active - 1].w)
            B = np.array([[w11, w12], [w21, w22]], dtype=float)
            evals = np.linalg.eigvals(B)
            evals_real = np.real(evals)
            # Prefer a true double-shift when a 2x2 is available
            if evals_real.shape[0] >= 2:
                mu, nu = float(evals_real[0]), float(evals_real[1])
                sigma_list = [mu, nu]
            else:
                idx = np.argmin(np.abs(evals_real - w22))
                sigma = float(evals_real[idx])
                sigma_list = [sigma]
        else:
            sigma = float(H[m_active - 1, m_active - 1].w)
            sigma_list = [sigma]

        # Record iteration diagnostics (pre-sweep)
        diag_entry = {
            "iter": k,
            "m_active": int(m_active),
            "shift_mode": str(current_shift_mode),
            "sigma": float(sigma_list[0]) if sigma_list else None,
            "sigma_list": [float(s) for s in sigma_list],
            "subdiag_max": float(subdiag_norm),
            "subdiag_vector": [
                float(
                    np.linalg.norm(
                        [H[i, i - 1].w, H[i, i - 1].x, H[i, i - 1].y, H[i, i - 1].z]
                    )
                )
                for i in range(1, m_active)
            ],
            "deflated_indices": deflated_idx,
            "stagnation_count": int(stagnation_count),
        }
        diag["iterations"].append(diag_entry)

        # Apply one or two shifts (Francis double-shift surrogate when len>1)
        Qk_real_total = np.eye(4 * n)
        for sigma in sigma_list:
            HR, Qk_part = _apply_single_shift(HR, sigma, m_active)
            Qk_real_total = Qk_real_total @ Qk_part
        Q_real = Q_real @ Qk_real_total

        # Re-enforce Hessenberg structure and apply strong deflation sweep
        H_tmp = real_contract(HR, n, n)
        H_tmp = check_hessenberg(H_tmp)
        # Zero very small subdiagonals relative to local scale
        for i in range(1, m_active):
            h_sub = H_tmp[i, i - 1]
            denom = (
                _quat_scalar_abs(H_tmp[i - 1, i - 1])
                + _quat_scalar_abs(H_tmp[i, i])
                + 1e-30
            )
            if _quat_scalar_abs(h_sub) <= 1e-2 * tol * max(1.0, denom):
                H_tmp[i, i - 1] = quaternion.quaternion(0.0, 0.0, 0.0, 0.0)
        HR = real_expand(H_tmp)

        if verbose and (k % 50 == 0):
            H_tmp = real_contract(HR, n, n)
            resid = 0.0
            for i in range(1, m_active):
                resid = max(resid, _quat_scalar_abs(H_tmp[i, i - 1]))
            print(
                f"Iter {k}: active={m_active}, subdiag max {resid:.2e}, shift {sigma:.3e}"
            )
        k += 1

    # Final contraction
    H_final = real_contract(HR, n, n)
    Q_accum = real_contract(Q_real, n, n)
    # Correct similarity chaining:
    # H0 = P0 A P0^H, H_final = Q_accum^H H0 Q_accum = (Q_accum^H P0) A (P0^H Q_accum)
    # Therefore A = (P0^H Q_accum) H_final (P0^H Q_accum)^H
    Q_total = quat_matmat(quat_hermitian(P0), Q_accum)

    # Optional cleanup small entries below diagonal
    for i in range(n):
        for j in range(0, i):
            if _quat_scalar_abs(H_final[i, j]) <= tol:
                H_final[i, j] = quaternion.quaternion(0.0, 0.0, 0.0, 0.0)

    # Ensure Q is unitary within tolerance (orthonormalization not enforced here)
    if return_diagnostics:
        return Q_total, H_final, diag
    return Q_total, H_final


__all__ = [
    "quaternion_schur",
    "quaternion_schur_pure",
    "quaternion_schur_pure_implicit",
    "quaternion_schur_unified",
    "quaternion_schur_experimental",
]


def quaternion_schur_pure(
    A: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-10,
    verbose: bool = False,
    return_diagnostics: bool = False,
    shift_mode: str = "none",
):
    """Pure quaternion QR iteration (unshifted), no real expansion.

    Algorithm:
      1) Reduce A to Hessenberg H via quaternion Householder similarity.
      2) For k=1..max_iter: compute QR of H using quaternion Householders (left)
         then set H <- R @ Q (QR iteration). Accumulate Q_total.

    Note: Convergence is generally slower than shifted QR; intended as a
    structure-preserving baseline for Lead 2 experiments.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("quaternion_schur_pure requires a square matrix")

    n = A.shape[0]
    if n == 0:
        I0 = np.eye(0, dtype=np.quaternion)
        return I0, I0

    # Step 1: Hessenberg reduction (quaternion Householders)
    P0, H = hessenbergize(A)
    H = check_hessenberg(H)

    Q_accum = np.eye(n, dtype=np.quaternion)
    diag = {"iterations": [], "converged": False, "iterations_run": 0}

    for k in range(max_iter):
        # Optional simple Rayleigh shift using bottom-right scalar part
        sigma = 0.0
        if shift_mode == "rayleigh":
            sigma = float(H[n - 1, n - 1].w)
        qsigma = quaternion.quaternion(float(sigma), 0.0, 0.0, 0.0)

        # Build Q_iter (product of Householders) such that R = Q_iter @ H is upper triangular
        Q_iter = np.eye(n, dtype=np.quaternion)
        R_work = H.copy()
        if sigma != 0.0:
            for i in range(n):
                R_work[i, i] = R_work[i, i] - qsigma
        for j in range(n - 1):
            # Form Householder on subvector R_work[j:, j] to zero entries below j
            col = R_work[j:, j].copy()
            # If already zero below diagonal, skip
            if all(
                (col[t].w == 0 and col[t].x == 0 and col[t].y == 0 and col[t].z == 0)
                for t in range(1, col.shape[0])
            ):
                continue
            # Target vector e1 (real), mapping col -> * e1
            e1 = np.zeros(col.shape[0])
            e1[0] = 1.0
            Hj_sub = householder_matrix(col, e1)
            Hj = np.eye(n, dtype=np.quaternion)
            Hj[j:, j:] = Hj_sub
            # Left-apply to R_work; accumulate Q_iter = Hj @ Q_iter
            R_work = quat_matmat(Hj, R_work)
            Q_iter = quat_matmat(Hj, Q_iter)

        # Now R_work ≈ Q_left * (H - sigma I). In QR iteration, set H <- R * Q_k + sigma I with Q_k = Q_left^H
        Qk = quat_hermitian(Q_iter)
        H = quat_matmat(R_work, Qk)
        if sigma != 0.0:
            for i in range(n):
                H[i, i] = H[i, i] + qsigma
        Q_accum = quat_matmat(Q_accum, Qk)

        # Clean tiny subdiagonals and check convergence
        max_sub = 0.0
        for i in range(1, n):
            h = H[i, i - 1]
            sv = (h.w * h.w + h.x * h.x + h.y * h.y + h.z * h.z) ** 0.5
            max_sub = max(max_sub, sv)
            dscale = (
                (
                    H[i - 1, i - 1].w ** 2
                    + H[i - 1, i - 1].x ** 2
                    + H[i - 1, i - 1].y ** 2
                    + H[i - 1, i - 1].z ** 2
                )
                ** 0.5
                + (H[i, i].w ** 2 + H[i, i].x ** 2 + H[i, i].y ** 2 + H[i, i].z ** 2)
                ** 0.5
                + 1e-30
            )
            if sv <= tol * max(1.0, dscale):
                H[i, i - 1] = quaternion.quaternion(0.0, 0.0, 0.0, 0.0)

        if return_diagnostics:
            diag["iterations"].append({"iter": k, "max_subdiag": float(max_sub)})
        if verbose and (k % 25 == 0 or max_sub <= tol):
            print(f"pure-QR iter {k}: max subdiag {max_sub:.2e}")
        if max_sub <= tol:
            if return_diagnostics:
                diag["converged"] = True
                diag["iterations_run"] = k + 1
            break

    # Compose final Q: A = Q_total T Q_total^H with Q_total = P0^H Q_accum
    Q_total = quat_matmat(quat_hermitian(P0), Q_accum)
    T = H
    if return_diagnostics:
        return Q_total, T, diag
    return Q_total, T


def quaternion_schur_pure_implicit(
    A: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-10,
    verbose: bool = False,
    return_diagnostics: bool = False,
    shift_mode: str = "rayleigh",
):
    """Pure quaternion implicit QR via 2x2 Householder (bulge-chasing style).

    - Keeps computation in quaternion domain
    - Uses optional simple shift (Rayleigh) to accelerate convergence
    - Left-apply 2x2 Householder to zero subdiagonal, right-apply its Hermitian to preserve similarity
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("quaternion_schur_pure_implicit requires a square matrix")

    n = A.shape[0]
    if n == 0:
        I0 = np.eye(0, dtype=np.quaternion)
        return I0, I0

    # Step 1: Hessenberg reduction
    P0, H = hessenbergize(A)
    H = check_hessenberg(H)

    Q_accum = np.eye(n, dtype=np.quaternion)
    diag = {"iterations": [], "converged": False, "iterations_run": 0}

    for k in range(max_iter):
        # Choose shift (real scalar) from trailing element by default
        sigma = 0.0
        if shift_mode == "rayleigh" and n >= 1:
            sigma = float(H[n - 1, n - 1].w)
        qsigma = quaternion.quaternion(float(sigma), 0.0, 0.0, 0.0)

        # Bulge init + chase across the full window
        for s in range(0, n - 1):
            # Form local vector [H[s,s]-sigma; H[s+1,s]] and reflect to e1
            v0 = H[s, s] - qsigma
            v1 = H[s + 1, s]
            local = np.array([v0, v1], dtype=np.quaternion)
            # If already small, skip
            sv = (v1.w * v1.w + v1.x * v1.x + v1.y * v1.y + v1.z * v1.z) ** 0.5
            if sv <= tol:
                continue
            e1 = np.zeros(2)
            e1[0] = 1.0
            Hj_sub = householder_matrix(local, e1)  # 2x2 quaternion
            Hj = np.eye(n, dtype=np.quaternion)
            Hj[s : s + 2, s : s + 2] = Hj_sub
            HjH = quat_hermitian(Hj)

            # Similarity update: H <- Hj H Hj^H
            H = quat_matmat(Hj, H)
            H = quat_matmat(H, HjH)
            # Accumulate Q: Q_accum <- Q_accum Hj^H
            Q_accum = quat_matmat(Q_accum, HjH)

        # Clean tiny subdiagonals and check convergence
        max_sub = 0.0
        for i in range(1, n):
            h = H[i, i - 1]
            sv = (h.w * h.w + h.x * h.x + h.y * h.y + h.z * h.z) ** 0.5
            # Local scale
            dscale = (
                (
                    H[i - 1, i - 1].w ** 2
                    + H[i - 1, i - 1].x ** 2
                    + H[i - 1, i - 1].y ** 2
                    + H[i - 1, i - 1].z ** 2
                )
                ** 0.5
                + (H[i, i].w ** 2 + H[i, i].x ** 2 + H[i, i].y ** 2 + H[i, i].z ** 2)
                ** 0.5
                + 1e-30
            )
            if sv <= tol * max(1.0, dscale):
                H[i, i - 1] = quaternion.quaternion(0.0, 0.0, 0.0, 0.0)
            max_sub = max(max_sub, sv)

        if return_diagnostics:
            diag["iterations"].append({"iter": k, "max_subdiag": float(max_sub)})
        if verbose and (k % 25 == 0 or max_sub <= tol):
            print(f"implicit-QR iter {k}: max subdiag {max_sub:.2e}")
        if max_sub <= tol:
            if return_diagnostics:
                diag["converged"] = True
                diag["iterations_run"] = k + 1
            break

    Q_total = quat_matmat(quat_hermitian(P0), Q_accum)
    T = H
    if return_diagnostics:
        return Q_total, T, diag
    return Q_total, T


def quaternion_schur_unified(
    A: np.ndarray,
    variant: str = "rayleigh",
    max_iter: int = 1000,
    tol: float = 1e-10,
    aed_factor: float | None = None,
    precompute_shifts: bool = True,
    power_shift_steps: int = 5,
    aed_window: int | None = None,
    verbose: bool = False,
    return_diagnostics: bool = False,
):
    """Unified Schur API exposing common variants.

    Variants:
      - 'none'       : pure QR iteration, no shift (Lead 2 baseline)
      - 'rayleigh'   : pure QR iteration with Rayleigh shift (Lead 2 simple shift)
      - 'implicit'   : pure quaternion implicit QR (bulge-chase) with Rayleigh shift
      - 'aed'        : implicit + aggressive early deflation (AED) (quaternion-only)
      - 'ds'         : implicit + double-shift surrogate from trailing 2x2 (quaternion-only)

    Notes:
      - For 'aed' and 'ds', operations stay in the quaternion domain using 2x2 Householder similarities.
      - aed_window (optional): when set, restrict AED checks to the trailing window of size aed_window
        to improve efficiency on larger matrices. Defaults to full sweep when None.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("quaternion_schur_unified requires a square matrix")

    # Map simple variants to existing functions
    if variant == "none":
        return quaternion_schur_pure(
            A,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            return_diagnostics=return_diagnostics,
            shift_mode="none",
        )
    if variant == "rayleigh":
        return quaternion_schur_pure(
            A,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            return_diagnostics=return_diagnostics,
            shift_mode="rayleigh",
        )
    if variant == "implicit":
        return quaternion_schur_pure_implicit(
            A,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            return_diagnostics=return_diagnostics,
            shift_mode="rayleigh",
        )

    # Advanced quaternion-only implicit variants ('aed' and 'ds')
    n = A.shape[0]
    P0, H = hessenbergize(A)
    H = check_hessenberg(H)
    Q_accum = np.eye(n, dtype=np.quaternion)
    diag = {"iterations": [], "converged": False, "iterations_run": 0}

    # Optional: precompute a schedule of scalar shifts using power iteration
    shift_schedule: list[float] | None = None
    shift_idx = 0
    if precompute_shifts:
        shift_schedule = _estimate_shifts_power_deflate(H, steps=power_shift_steps)

    def apply_left_rows(M: np.ndarray, s_idx: int, B: np.ndarray) -> None:
        a, b = B[0, 0], B[0, 1]
        c, d = B[1, 0], B[1, 1]
        r0 = M[s_idx, :].copy()
        r1 = M[s_idx + 1, :].copy()
        M[s_idx, :] = a * r0 + b * r1
        M[s_idx + 1, :] = c * r0 + d * r1

    def apply_right_cols(M: np.ndarray, s_idx: int, B: np.ndarray) -> None:
        BH = quat_hermitian(B)
        a, b = BH[0, 0], BH[0, 1]
        c, d = BH[1, 0], BH[1, 1]
        c0 = M[:, s_idx].copy()
        c1 = M[:, s_idx + 1].copy()
        M[:, s_idx] = c0 * a + c1 * c
        M[:, s_idx + 1] = c0 * b + c1 * d

    # Choose AED factor
    if aed_factor is None:
        # Simple heuristic
        if n <= 20:
            aed_factor = 3.0
        elif n <= 50:
            aed_factor = 5.0
        elif n <= 75:
            aed_factor = 6.0
        else:
            aed_factor = 8.0

    for k in range(max_iter):
        # Select shifts
        if variant == "ds" and n >= 2:
            # Double-shift surrogate: two scalar shifts from trailing 2x2 (real parts)
            if (
                precompute_shifts
                and shift_schedule is not None
                and shift_idx + 1 < len(shift_schedule)
            ):
                sigmas = [
                    float(shift_schedule[shift_idx]),
                    float(shift_schedule[shift_idx + 1]),
                ]
                shift_idx += 2
            else:
                w11 = float(H[n - 2, n - 2].w)
                w12 = float(H[n - 2, n - 1].w)
                w21 = float(H[n - 1, n - 2].w)
                w22 = float(H[n - 1, n - 1].w)
                B2 = np.array([[w11, w12], [w21, w22]], dtype=float)
                evals = np.linalg.eigvals(B2)
                sigmas = [float(np.real(evals[0])), float(np.real(evals[1]))]
        else:
            if (
                precompute_shifts
                and shift_schedule is not None
                and shift_idx < len(shift_schedule)
            ):
                sigmas = [float(shift_schedule[shift_idx])]
                shift_idx += 1
            else:
                sigmas = [float(H[n - 1, n - 1].w)]

        for sigma in sigmas:
            qs = quaternion.quaternion(float(sigma), 0.0, 0.0, 0.0)
            # Bulge init + chase via 2x2 Householders with per-step left/right updates (safer)
            for s in range(0, n - 1):
                v0 = H[s, s] - qs
                v1 = H[s + 1, s]
                sv = (v1.w * v1.w + v1.x * v1.x + v1.y * v1.y + v1.z * v1.z) ** 0.5
                if sv <= tol:
                    continue
                e1 = np.zeros(2)
                e1[0] = 1.0
                Hj_sub = householder_matrix(np.array([v0, v1], dtype=np.quaternion), e1)
                apply_left_rows(H, s, Hj_sub)
                apply_right_cols(H, s, Hj_sub)
                apply_right_cols(Q_accum, s, Hj_sub)

        # AED sweep (optionally restricted to a trailing window)
        max_sub = 0.0
        i_start = 1
        if aed_window is not None and aed_window > 1:
            i_start = max(1, n - aed_window + 1)
        for i in range(i_start, n):
            h = H[i, i - 1]
            # Use squared norms to avoid sqrt in inner loop
            sv_sq = h.w * h.w + h.x * h.x + h.y * h.y + h.z * h.z
            d0 = H[i - 1, i - 1]
            d1 = H[i, i]
            dscale_sq = (
                d0.w * d0.w
                + d0.x * d0.x
                + d0.y * d0.y
                + d0.z * d0.z
                + d1.w * d1.w
                + d1.x * d1.x
                + d1.y * d1.y
                + d1.z * d1.z
            )
            bound_sq = (aed_factor * tol) * (aed_factor * tol) * max(1.0, dscale_sq)
            if variant in ("aed", "ds") and sv_sq <= bound_sq:
                H[i, i - 1] = quaternion.quaternion(0.0, 0.0, 0.0, 0.0)
            # Track residual using actual norm (outside of comparisons)
            sv = (sv_sq) ** 0.5
            max_sub = max(max_sub, sv)

        # Note: no projection; rely on similarity updates to maintain structure

        if return_diagnostics:
            diag["iterations"].append({"iter": k, "max_subdiag": float(max_sub)})
        if verbose and (k % 50 == 0 or max_sub <= tol):
            print(f"unified[{variant}] iter {k}: max subdiag {max_sub:.2e}")
        if max_sub <= tol:
            if return_diagnostics:
                diag["converged"] = True
                diag["iterations_run"] = k + 1
            break

    Q_total = quat_matmat(quat_hermitian(P0), Q_accum)
    T = H
    if return_diagnostics:
        return Q_total, T, diag
    return Q_total, T


def quaternion_schur_experimental(
    A: np.ndarray,
    variant: str = "aed_windowed",
    max_iter: int = 1000,
    tol: float = 1e-10,
    window: int = 12,
    verbose: bool = False,
    return_diagnostics: bool = False,
):
    """Experimental Schur variants with windowed AED and a Francis-like two-shift chase.

    WARNING: Experimental. API and behavior may change. Does not affect stable variants.

    Variants:
      - 'aed_windowed' : restrict bulge-chase and deflation to a trailing window (approx. AED)
      - 'francis_ds'   : perform a two-shift bulge chase per outer iteration (surrogate DS)

    Notes:
      - Operates purely in quaternion domain using 2x2 Householder similarities.
      - Uses real scalar shifts derived from trailing block; keeps quaternion structure.
      - Maintains similarity via left-row and right-column updates.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("quaternion_schur_experimental requires a square matrix")

    n = A.shape[0]
    if n == 0:
        I0 = np.eye(0, dtype=np.quaternion)
        return I0, I0

    # Initial Hessenberg reduction
    P0, H = hessenbergize(A)
    H = check_hessenberg(H)
    Q_accum = np.eye(n, dtype=np.quaternion)

    # Helpers for similarity updates with 2x2 quaternion block B at index s
    def apply_left_rows(M: np.ndarray, s_idx: int, B: np.ndarray) -> None:
        a, b = B[0, 0], B[0, 1]
        c, d = B[1, 0], B[1, 1]
        r0 = M[s_idx, :].copy()
        r1 = M[s_idx + 1, :].copy()
        M[s_idx, :] = a * r0 + b * r1
        M[s_idx + 1, :] = c * r0 + d * r1

    def apply_right_cols(M: np.ndarray, s_idx: int, B: np.ndarray) -> None:
        BH = quat_hermitian(B)
        a, b = BH[0, 0], BH[0, 1]
        c, d = BH[1, 0], BH[1, 1]
        c0 = M[:, s_idx].copy()
        c1 = M[:, s_idx + 1].copy()
        M[:, s_idx] = c0 * a + c1 * c
        M[:, s_idx + 1] = c0 * b + c1 * d

    # Active window [lo, hi]
    lo, hi = 0, n - 1
    diag = {"iterations": [], "converged": False, "iterations_run": 0}

    for k in range(max_iter):
        if hi <= lo:
            break

        # Deflation scan (bottom-up) in trailing region
        i = hi
        while i > lo:
            h = H[i, i - 1]
            sv_sq = h.w * h.w + h.x * h.x + h.y * h.y + h.z * h.z
            d0, d1 = H[i - 1, i - 1], H[i, i]
            dscale_sq = (
                d0.w * d0.w
                + d0.x * d0.x
                + d0.y * d0.y
                + d0.z * d0.z
                + d1.w * d1.w
                + d1.x * d1.x
                + d1.y * d1.y
                + d1.z * d1.z
            )
            if sv_sq <= (tol * tol) * max(1.0, dscale_sq):
                H[i, i - 1] = quaternion.quaternion(0.0, 0.0, 0.0, 0.0)
                hi = i - 1
                break
            i -= 1
        else:
            # No deflation: perform a windowed bulge chase
            win = max(2, min(window, hi - lo + 1))
            start = max(lo, hi - win + 1)

            # Choose shifts
            sigmas: list[float]
            if variant == "francis_ds" and hi - start + 1 >= 2:
                w11 = float(H[hi - 1, hi - 1].w)
                w12 = float(H[hi - 1, hi].w)
                w21 = float(H[hi, hi - 1].w)
                w22 = float(H[hi, hi].w)
                B2 = np.array([[w11, w12], [w21, w22]], dtype=float)
                ev = np.linalg.eigvals(B2)
                sigmas = [float(np.real(ev[0])), float(np.real(ev[1]))]
            else:
                sigmas = [float(H[hi, hi].w)]

            # Apply one or two shift sweeps within the trailing window
            for sigma in sigmas:
                qs = quaternion.quaternion(float(sigma), 0.0, 0.0, 0.0)
                for s in range(start, hi):
                    v0 = H[s, s] - qs
                    v1 = H[s + 1, s]
                    # Skip if already tiny
                    sv_sq = v1.w * v1.w + v1.x * v1.x + v1.y * v1.y + v1.z * v1.z
                    if sv_sq <= (tol * tol):
                        continue
                    e1 = np.zeros(2)
                    e1[0] = 1.0
                    Hj_sub = householder_matrix(
                        np.array([v0, v1], dtype=np.quaternion), e1
                    )
                    apply_left_rows(H, s, Hj_sub)
                    apply_right_cols(H, s, Hj_sub)
                    apply_right_cols(Q_accum, s, Hj_sub)

        # Track residual in active window
        max_sub = 0.0
        for j in range(lo + 1, hi + 1):
            h = H[j, j - 1]
            sv = (h.w * h.w + h.x * h.x + h.y * h.y + h.z * h.z) ** 0.5
            max_sub = max(max_sub, sv)

        if return_diagnostics:
            diag["iterations"].append(
                {
                    "iter": k,
                    "lo": int(lo),
                    "hi": int(hi),
                    "variant": variant,
                    "max_subdiag": float(max_sub),
                }
            )
        if verbose and (k % 50 == 0 or max_sub <= tol):
            print(
                f"experimental[{variant}] iter {k}: window=[{start}:{hi}], max subdiag {max_sub:.2e}"
            )
        if hi <= lo or max_sub <= tol:
            if return_diagnostics:
                diag["converged"] = True
                diag["iterations_run"] = k + 1
            break

    Q_total = quat_matmat(quat_hermitian(P0), Q_accum)
    T = H
    if return_diagnostics:
        return Q_total, T, diag
    return Q_total, T
