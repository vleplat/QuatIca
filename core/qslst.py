
"""
QSLST: Quaternion Special Least Squares with Tikhonov Regularization
====================================================================

This module provides a *practical* implementation of Algorithm 2 (QSLST) for
quaternion-valued image restoration. It follows the formulation

    (A^T A + λ I) X = A^T B,

where A is a real-valued blurring operator, X and B are quaternion images.
For convolutional A (Gaussian/motion blur with periodic boundary), we exploit
the diagonalization in the Fourier domain to obtain the closed-form Tikhonov
solution. For a generic real matrix A, we provide a faithful pinv-based path
that implements Algorithm 2 exactly *without explicitly forming* A(T) = A(A^T A + λ I).

Quaternion representation
-------------------------
We represent a quaternion image Q as a float array of shape (H, W, 4) whose
last dimension stores components [q0, q1, q2, q3]. For RGB color, the
common choice is q0 = 0, q1 = R, q2 = G, q3 = B. Use `rgb_to_quat` /
`quat_to_rgb` helpers to convert.

Core API
--------
- qslst_restore_fft(Bq, psf, lam, boundary="periodic")  -> Xq
    Efficient path for convolutional blur with periodic boundary (BCCB).

- qslst_restore_matrix(Bq, A_mat, lam) -> Xq
    Faithful Algorithm 2 using T = A^T A + lam I and pseudo-inverse of T.

- build_psf_gaussian(radius, sigma) -> psf
- build_psf_motion(length, angle_deg) -> psf
- apply_blur_fft(Q, psf, boundary="periodic") -> blurred quaternion
- add_awgn_snr(Q, snr_db, rng=None) -> noisy quaternion
- psnr(x, x_ref), relative_error(x, x_ref)

All functions are fully numpy-based.
"""

from __future__ import annotations
import numpy as np
from numpy.fft import rfft2, irfft2, fft2, ifft2, fftshift
from typing import Tuple, Optional

# -----------------------------
# Quaternion helpers
# -----------------------------

def rgb_to_quat(rgb: np.ndarray, real_part: float = 0.0) -> np.ndarray:
    """
    Convert an RGB image to quaternion form (H, W, 4).

    Parameters
    ----------
    rgb : (H, W, 3) float array in [0, 1] or [0, 255]
    real_part : float, optional
        Value for q0 (often 0.0)

    Returns
    -------
    q : (H, W, 4) float array with [q0, q1, q2, q3] = [real_part, R, G, B]
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, "rgb must be (H, W, 3)"
    H, W, _ = rgb.shape
    q = np.empty((H, W, 4), dtype=np.float64)
    q[..., 0] = real_part
    q[..., 1:] = rgb.astype(np.float64)
    return q


def quat_to_rgb(q: np.ndarray, clip: bool = True) -> np.ndarray:
    """
    Convert quaternion image to RGB by taking imaginary parts [q1, q2, q3].

    Parameters
    ----------
    q : (H, W, 4)
    clip : bool
        Whether to clip RGB to [0, 1] if input appears normalized.

    Returns
    -------
    rgb : (H, W, 3)
    """
    assert q.ndim == 3 and q.shape[2] == 4, "q must be (H, W, 4)"
    rgb = q[..., 1:].copy()
    if clip:
        # Attempt intelligent clipping if values look normalized
        if rgb.max() <= 1.5 and rgb.min() >= -0.5:
            rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def split_quat_channels(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (q0, q1, q2, q3), each (H, W)."""
    return q[..., 0], q[..., 1], q[..., 2], q[..., 3]


def stack_quat_channels(q0: np.ndarray, q1: np.ndarray, q2: np.ndarray, q3: np.ndarray) -> np.ndarray:
    """Stack scalar channels into quaternion image (H, W, 4)."""
    return np.stack([q0, q1, q2, q3], axis=-1)


# -----------------------------
# PSFs and blur application
# -----------------------------

def build_psf_gaussian(radius: int, sigma: float) -> np.ndarray:
    """
    Isotropic Gaussian PSF, truncated to a (2*radius+1) window and normalized.

    Parameters
    ----------
    radius : int
        Truncation radius r
    sigma : float
        Standard deviation

    Returns
    -------
    psf : (K, K) float array, K = 2*radius + 1, sum(psf)=1
    """
    K = 2 * radius + 1
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax, indexing="xy")
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2)) / (2.0 * np.pi * sigma**2)
    psf /= psf.sum()
    return psf


def build_psf_motion(length: int, angle_deg: float) -> np.ndarray:
    """
    Simple linear motion blur PSF of given length and angle, normalized.

    Parameters
    ----------
    length : int
        Number of pixels in the motion kernel (>=1)
    angle_deg : float
        Angle in degrees measured counter-clockwise from +x axis

    Returns
    -------
    psf : (K, K) float array with sum=1. K chosen to tightly contain the line.
    """
    L = max(1, int(length))
    # Determine kernel footprint (square) that can contain the line
    K = L if L % 2 == 1 else L + 1
    psf = np.zeros((K, K), dtype=np.float64)
    c = (K - 1) / 2.0
    theta = np.deg2rad(angle_deg)
    dx, dy = np.cos(theta), np.sin(theta)
    # Sample L points along the line centered at (c, c)
    for t in np.linspace(-(L-1)/2.0, (L-1)/2.0, L):
        x = c + t * dx
        y = c + t * dy
        ix = int(round(x))
        iy = int(round(y))
        if 0 <= ix < K and 0 <= iy < K:
            psf[iy, ix] += 1.0
    s = psf.sum()
    if s > 0:
        psf /= s
    else:
        psf[c.astype(int), c.astype(int)] = 1.0
    return psf


def _pad_psf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Pad/crop PSF to the given image shape with the peak at (0,0) (for FFT)."""
    H, W = shape
    kH, kW = psf.shape
    pad = np.zeros((H, W), dtype=np.float64)
    # Place PSF at top-left corner after centering
    kh2, kw2 = kH // 2, kW // 2
    psf_shifted = np.roll(np.roll(psf, -kh2, axis=0), -kw2, axis=1)
    pad[:kH, :kW] = psf_shifted[:H, :W]
    return pad


def apply_blur_fft(Q: np.ndarray, psf: np.ndarray, boundary: str = "periodic") -> np.ndarray:
    """
    Convolve quaternion image with PSF using FFT (per channel).

    Parameters
    ----------
    Q : (H, W, 4)
    psf : (kH, kW)
    boundary : {"periodic"}
        Only periodic boundary is supported (BCCB).

    Returns
    -------
    B : (H, W, 4) blurred quaternion image
    """
    assert boundary == "periodic", "Only periodic boundary is implemented"
    H, W, _ = Q.shape
    H_pad = _pad_psf(psf, (H, W))
    H_hat = fft2(H_pad)
    B = np.empty_like(Q)
    for c in range(4):
        B[..., c] = np.real(ifft2(fft2(Q[..., c]) * H_hat))
    return B


# -----------------------------
# Noise and metrics
# -----------------------------

def add_awgn_snr(Q: np.ndarray, snr_db: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Add white Gaussian noise to reach a target SNR (in dB) per quaternion image.

    SNR definition used:
        SNR_dB = 10 * log10( ||signal||_F^2 / ||noise||_F^2 )

    Parameters
    ----------
    Q : (H, W, 4) float
    snr_db : float
    rng : np.random.Generator

    Returns
    -------
    noisy : (H, W, 4)
    """
    if rng is None:
        rng = np.random.default_rng()
    power_sig = np.sum(Q**2)
    if power_sig == 0:
        return Q.copy()
    snr = 10.0 ** (snr_db / 10.0)
    power_noise = power_sig / snr
    sigma = math.sqrt(power_noise / Q.size)
    noise = rng.normal(0.0, sigma, size=Q.shape)
    return Q + noise


def psnr(x: np.ndarray, x_ref: np.ndarray, data_range: Optional[float] = None) -> float:
    """
    Peak Signal-to-Noise Ratio for arrays with same shape.

    Parameters
    ----------
    x, x_ref : arrays
    data_range : float, optional
        If None, uses max(x_ref) - min(x_ref)

    Returns
    -------
    PSNR in dB
    """
    x = x.astype(np.float64)
    x_ref = x_ref.astype(np.float64)
    mse = np.mean((x - x_ref) ** 2)
    if mse == 0:
        return float("inf")
    if data_range is None:
        data_range = float(x_ref.max() - x_ref.min() or 1.0)
    return 10.0 * math.log10((data_range ** 2) / mse)


def relative_error(x: np.ndarray, x_ref: np.ndarray) -> float:
    """Relative Frobenius error: ||x - x_ref||_F / ||x_ref||_F"""
    num = np.linalg.norm(x - x_ref)
    den = np.linalg.norm(x_ref)
    return float(num / den) if den != 0 else float("inf")


# -----------------------------
# QSLST restoration
# -----------------------------

def qslst_restore_fft(Bq: np.ndarray, psf: np.ndarray, lam: float, boundary: str = "periodic") -> np.ndarray:
    """
    QSLST (Algorithm 2) specialized to convolutional A with periodic BC.
    Per-channel closed-form Tikhonov solution:

        X_hat = conj(H_hat) * B_hat / (|H_hat|^2 + lam)

    Parameters
    ----------
    Bq : (H, W, 4) observed blurred+noisy quaternion image
    psf : (kH, kW) PSF
    lam : float >= 0
    boundary : {"periodic"}

    Returns
    -------
    Xq : (H, W, 4) restored quaternion
    """
    assert boundary == "periodic", "Only periodic boundary is implemented"
    H, W, _ = Bq.shape
    H_pad = _pad_psf(psf, (H, W))
    H_hat = fft2(H_pad)
    denom = (np.abs(H_hat) ** 2) + lam
    Xq = np.empty_like(Bq)
    H_conj = np.conj(H_hat)
    for c in range(4):
        B_hat = fft2(Bq[..., c])
        X_hat = H_conj * B_hat / denom
        Xq[..., c] = np.real(ifft2(X_hat))
    return Xq


def qslst_restore_matrix(Bq: np.ndarray, A_mat: np.ndarray, lam: float) -> np.ndarray:
    """
    Faithful Algorithm 2 (QSLST) for a generic real matrix A.
    Implements:
        T = A^T A + lam I
        E = A^T B
        X = T^+ E   (Moore-Penrose)

    Since A is real, T is real; then A(T) = I_4 \\otimes T, and
    A(T)^+ = I_4 \\otimes T^+. Therefore we can solve *per quaternion component*.

    Parameters
    ----------
    Bq : (H, W, 4) observed quaternion image
    A_mat : (N, N) real blur matrix (N = H*W)
    lam : float >= 0

    Returns
    -------
    Xq : (H, W, 4)
    """
    H, W, _ = Bq.shape
    N = H * W
    assert A_mat.shape == (N, N), "A_mat must be square (N x N) with N=H*W"
    # Build T and its pseudo-inverse
    T = A_mat.T @ A_mat
    if lam != 0:
        T = T + lam * np.eye(N)
    T_pinv = np.linalg.pinv(T)
    Xq = np.empty_like(Bq)
    for c in range(4):
        b = Bq[..., c].reshape(-1)                  # B component
        e = A_mat.T @ b                             # E = A^T b
        x = T_pinv @ e                              # X = T^+ E
        Xq[..., c] = x.reshape(H, W)
    return Xq


# -----------------------------
# End of module
# -----------------------------
