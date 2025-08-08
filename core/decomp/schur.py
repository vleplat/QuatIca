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

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import (
    quat_matmat,
    quat_hermitian,
    quat_frobenius_norm,
    real_expand,
    real_contract,
    quat_eye,
    ggivens,
    GRSGivens,
)
from .hessenberg import hessenbergize, check_hessenberg
from .tridiagonalize import householder_matrix


def _quat_scalar_abs(q: quaternion.quaternion) -> float:
    return float(np.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z))


def _deflate_in_place(H: np.ndarray, tol: float) -> bool:
    """Perform simple 1x1 deflation: set tiny subdiagonal entries to zero.

    Returns True if any deflation occurred.
    """
    n = H.shape[0]
    changed = False
    for i in range(1, n):
        h_sub = H[i, i - 1]
        denom = _quat_scalar_abs(H[i - 1, i - 1]) + _quat_scalar_abs(H[i, i]) + 1e-30
        if _quat_scalar_abs(h_sub) <= tol * denom:
            if (h_sub.w != 0) or (h_sub.x != 0) or (h_sub.y != 0) or (h_sub.z != 0):
                H[i, i - 1] = quaternion.quaternion(0.0, 0.0, 0.0, 0.0)
                changed = True
    return changed


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

    def _current_shift(hmat_quat: np.ndarray) -> float:
        # Returns a real scalar shift
        if shift == "rayleigh":
            return float(hmat_quat[-1, -1].w)
        elif shift == "wilkinson":
            # Build 2x2 real proxy matrix from real parts of trailing 2x2 quaternion block
            nloc = hmat_quat.shape[0]
            w11 = float(hmat_quat[nloc - 2, nloc - 2].w)
            w12 = float(hmat_quat[nloc - 2, nloc - 1].w)
            w21 = float(hmat_quat[nloc - 1, nloc - 2].w)
            w22 = float(hmat_quat[nloc - 1, nloc - 1].w)
            B = np.array([[w11, w12], [w21, w22]], dtype=float)
            evals = np.linalg.eigvals(B)
            # Choose eigenvalue closest to bottom-right entry
            if evals.shape[0] == 0:
                return w22
            # ensure real part only if complex
            evals_real = np.real(evals)
            idx = np.argmin(np.abs(evals_real - w22))
            return float(evals_real[idx])
        else:
            return float(hmat_quat[-1, -1].w)

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
    def _apply_single_shift(HR_in: np.ndarray, sigma_val: float, m_sz: int) -> Tuple[np.ndarray, np.ndarray]:
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
    prev_max_sub = float('inf')
    stagnation_count = 0
    k = 0
    while k < max_iter and m_active > 1:
        # Map back to quaternion to choose shift and check deflation
        H = real_contract(HR, n, n)

        # Strong deflation: zero tiny subdiagonals within active window
        deflated_idx = []
        for i in range(1, m_active):
            h_sub = H[i, i - 1]
            denom = _quat_scalar_abs(H[i - 1, i - 1]) + _quat_scalar_abs(H[i, i]) + _quat_scalar_abs(h_sub)
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
                print(f"Converged at iter {k}: active={m_active}, subdiag {subdiag_norm:.2e}")
            # Record final iteration diagnostics
            diag_entry = {
                "iter": k,
                "m_active": int(m_active),
                "shift_mode": None,
                "sigma": None,
                "subdiag_max": float(subdiag_norm),
                "subdiag_vector": [
                    float(np.linalg.norm([H[i, i-1].w, H[i, i-1].x, H[i, i-1].y, H[i, i-1].z]))
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
                float(np.linalg.norm([H[i, i-1].w, H[i, i-1].x, H[i, i-1].y, H[i, i-1].z]))
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
            denom = _quat_scalar_abs(H_tmp[i - 1, i - 1]) + _quat_scalar_abs(H_tmp[i, i]) + 1e-30
            if _quat_scalar_abs(h_sub) <= 1e-2 * tol * max(1.0, denom):
                H_tmp[i, i - 1] = quaternion.quaternion(0.0, 0.0, 0.0, 0.0)
        HR = real_expand(H_tmp)

        if verbose and (k % 50 == 0):
            H_tmp = real_contract(HR, n, n)
            resid = 0.0
            for i in range(1, m_active):
                resid = max(resid, _quat_scalar_abs(H_tmp[i, i - 1]))
            print(f"Iter {k}: active={m_active}, subdiag max {resid:.2e}, shift {sigma:.3e}")
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


__all__ = ["quaternion_schur", "quaternion_schur_pure"]




def quaternion_schur_pure(
    A: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-10,
    verbose: bool = False,
    return_diagnostics: bool = False,
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
        # Build Q_iter (product of Householders) such that R = Q_iter @ H is upper triangular
        Q_iter = np.eye(n, dtype=np.quaternion)
        R_work = H.copy()
        for j in range(n - 1):
            # Form Householder on subvector R_work[j:, j] to zero entries below j
            col = R_work[j:, j].copy()
            # If already zero below diagonal, skip
            if all((col[t].w == 0 and col[t].x == 0 and col[t].y == 0 and col[t].z == 0) for t in range(1, col.shape[0])):
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

        # Now R_work ≈ Q_left * H. In QR iteration, set H <- R * Q_k with Q_k = Q_left^H
        Qk = quat_hermitian(Q_iter)
        H = quat_matmat(R_work, Qk)
        Q_accum = quat_matmat(Q_accum, Qk)

        # Clean tiny subdiagonals and check convergence
        max_sub = 0.0
        for i in range(1, n):
            h = H[i, i - 1]
            sv = (h.w * h.w + h.x * h.x + h.y * h.y + h.z * h.z) ** 0.5
            max_sub = max(max_sub, sv)
            dscale = (
                (H[i - 1, i - 1].w ** 2 + H[i - 1, i - 1].x ** 2 + H[i - 1, i - 1].y ** 2 + H[i - 1, i - 1].z ** 2) ** 0.5
                + (H[i, i].w ** 2 + H[i, i].x ** 2 + H[i, i].y ** 2 + H[i, i].z ** 2) ** 0.5
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
