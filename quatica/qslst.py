"""
QSLST: Quaternion Special Least Squares with Tikhonov Regularization
====================================================================

This module provides a *practical* implementation of Algorithm 2 (QSLST) from [1] for
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

References
----------
[1] Fei, W., Tang, J., & Shan, M.
    Quaternion special least squares with Tikhonov regularization method in image restoration.
    Numerical Algorithms, 1-20. (2025)
    https://doi.org/10.1007/s11075-025-02187-6

"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from numpy.fft import fft2, ifft2

# -----------------------------
# Quaternion helpers
# -----------------------------


def rgb_to_quat(rgb: np.ndarray, real_part: float = 0.0) -> np.ndarray:
    """
    Convert an RGB image to quaternion form (H, W, 4).

    Maps RGB color channels to quaternion imaginary parts with configurable
    real component. Common choice: q0=real_part, q1=R, q2=G, q3=B.

    Parameters:
    -----------
    rgb : np.ndarray
        RGB image array of shape (H, W, 3) with values in [0, 1] or [0, 255]
    real_part : float, optional
        Value for quaternion real component q0 (default: 0.0)

    Returns:
    --------
    np.ndarray
        Quaternion image of shape (H, W, 4) with [q0, q1, q2, q3] = [real_part, R, G, B]

    Notes:
    ------
    This conversion enables processing RGB images as quaternion matrices
    for quaternion-based image restoration algorithms.
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

    Extracts RGB color channels from quaternion imaginary components,
    with optional intelligent clipping for normalized values.

    Parameters:
    -----------
    q : np.ndarray
        Quaternion image array of shape (H, W, 4)
    clip : bool, optional
        Whether to clip RGB to [0, 1] if input appears normalized (default: True)

    Returns:
    --------
    np.ndarray
        RGB image of shape (H, W, 3)

    Notes:
    ------
    Inverse operation of rgb_to_quat. The clipping heuristic checks if values
    are in range [-0.5, 1.5] to determine if normalization is appropriate.
    """
    assert q.ndim == 3 and q.shape[2] == 4, "q must be (H, W, 4)"
    rgb = q[..., 1:].copy()
    if clip:
        # Attempt intelligent clipping if values look normalized
        if rgb.max() <= 1.5 and rgb.min() >= -0.5:
            rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def split_quat_channels(
    q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split quaternion image into individual component channels.

    Separates a quaternion image into its four scalar components for
    independent processing or analysis.

    Parameters:
    -----------
    q : np.ndarray
        Quaternion image array of shape (H, W, 4)

    Returns:
    --------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Four channel arrays (q0, q1, q2, q3), each of shape (H, W)

    Notes:
    ------
    Useful for component-wise operations where quaternion channels
    need to be processed separately.
    """
    return q[..., 0], q[..., 1], q[..., 2], q[..., 3]


def stack_quat_channels(
    q0: np.ndarray, q1: np.ndarray, q2: np.ndarray, q3: np.ndarray
) -> np.ndarray:
    """
    Stack scalar channels into quaternion image (H, W, 4).

    Combines four separate scalar channel arrays into a single quaternion
    image for quaternion matrix operations.

    Parameters:
    -----------
    q0 : np.ndarray
        Real component array of shape (H, W)
    q1 : np.ndarray
        First imaginary component array of shape (H, W)
    q2 : np.ndarray
        Second imaginary component array of shape (H, W)
    q3 : np.ndarray
        Third imaginary component array of shape (H, W)

    Returns:
    --------
    np.ndarray
        Quaternion image of shape (H, W, 4)

    Notes:
    ------
    Inverse operation of split_quat_channels. All input arrays must
    have the same spatial dimensions (H, W).
    """
    return np.stack([q0, q1, q2, q3], axis=-1)


# -----------------------------
# PSFs and blur application
# -----------------------------


def build_psf_gaussian(radius: int, sigma: float) -> np.ndarray:
    """
    Isotropic Gaussian PSF, truncated to a (2*radius+1) window and normalized.

    Creates a 2D Gaussian point spread function for image blurring operations.
    The PSF is truncated to a square window and normalized to unit sum.

    Parameters:
    -----------
    radius : int
        Truncation radius r (PSF size will be 2*radius + 1)
    sigma : float
        Standard deviation of the Gaussian distribution

    Returns:
    --------
    np.ndarray
        Gaussian PSF array of shape (K, K) where K = 2*radius + 1, with sum(psf) = 1

    Notes:
    ------
    Uses the 2D Gaussian formula: exp(-(x² + y²) / (2σ²)) / (2πσ²)
    The PSF is centered and normalized for convolution operations.
    """
    2 * radius + 1
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax, indexing="xy")
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2)) / (2.0 * np.pi * sigma**2)
    psf /= psf.sum()
    return psf


def build_psf_motion(length: int, angle_deg: float) -> np.ndarray:
    """
    Simple linear motion blur PSF of given length and angle, normalized.

    Creates a motion blur kernel representing linear movement during exposure.
    The PSF models uniform motion along a straight line.

    Parameters:
    -----------
    length : int
        Number of pixels in the motion kernel (>= 1)
    angle_deg : float
        Motion angle in degrees measured counter-clockwise from +x axis

    Returns:
    --------
    np.ndarray
        Motion blur PSF array of shape (K, K) with sum = 1, where K is chosen
        to tightly contain the motion line

    Notes:
    ------
    The kernel size K is automatically determined to contain the motion path.
    Points along the line are sampled uniformly and rounded to pixel positions.
    """
    L = max(1, int(length))
    # Determine kernel footprint (square) that can contain the line
    K = L if L % 2 == 1 else L + 1
    psf = np.zeros((K, K), dtype=np.float64)
    c = (K - 1) / 2.0
    theta = np.deg2rad(angle_deg)
    dx, dy = np.cos(theta), np.sin(theta)
    # Sample L points along the line centered at (c, c)
    for t in np.linspace(-(L - 1) / 2.0, (L - 1) / 2.0, L):
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
    """
    Pad/crop PSF to the given image shape with the peak at (0,0) (for FFT).

    Prepares a PSF for FFT-based convolution by padding or cropping to match
    the image dimensions and shifting the center to origin.

    Parameters:
    -----------
    psf : np.ndarray
        Point spread function kernel
    shape : tuple[int, int]
        Target image shape (H, W)

    Returns:
    --------
    np.ndarray
        Padded/cropped PSF of shape (H, W) with center at (0,0)

    Notes:
    ------
    Internal function for FFT-based convolution setup. The PSF center
    is moved to the top-left corner as required by FFT convolution.
    """
    H, W = shape
    kH, kW = psf.shape
    pad = np.zeros((H, W), dtype=np.float64)
    # Place PSF at top-left corner after centering
    kh2, kw2 = kH // 2, kW // 2
    psf_shifted = np.roll(np.roll(psf, -kh2, axis=0), -kw2, axis=1)
    pad[:kH, :kW] = psf_shifted[:H, :W]
    return pad


def apply_blur_fft(
    Q: np.ndarray, psf: np.ndarray, boundary: str = "periodic"
) -> np.ndarray:
    """
    Convolve quaternion image with PSF using FFT (per channel).

    Applies blurring to a quaternion image by convolving each quaternion
    component independently with the given point spread function using FFT.

    Parameters:
    -----------
    Q : np.ndarray
        Quaternion image array of shape (H, W, 4)
    psf : np.ndarray
        Point spread function kernel of shape (kH, kW)
    boundary : str, optional
        Boundary condition, only "periodic" is supported for BCCB (default: "periodic")

    Returns:
    --------
    np.ndarray
        Blurred quaternion image of shape (H, W, 4)

    Notes:
    ------
    Uses FFT-based convolution for computational efficiency. Each quaternion
    component is convolved independently, preserving quaternion structure.
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


def add_awgn_snr(
    Q: np.ndarray, snr_db: float, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Add white Gaussian noise to reach a target SNR (in dB) per quaternion image.

    Adds calibrated noise to achieve a specific signal-to-noise ratio.
    SNR definition: SNR_dB = 10 * log10( ||signal||_F^2 / ||noise||_F^2 )

    Parameters:
    -----------
    Q : np.ndarray
        Clean quaternion image of shape (H, W, 4)
    snr_db : float
        Target signal-to-noise ratio in decibels
    rng : np.random.Generator, optional
        Random number generator (default: None, uses default_rng())

    Returns:
    --------
    np.ndarray
        Noisy quaternion image of shape (H, W, 4)

    Notes:
    ------
    The noise variance is computed to achieve the exact target SNR.
    Returns the original image unchanged if signal power is zero.
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

    Computes PSNR = 10 * log10(data_range² / MSE) where MSE is the mean squared error.
    Higher values indicate better image quality/reconstruction.

    Parameters:
    -----------
    x : np.ndarray
        Estimated/reconstructed array
    x_ref : np.ndarray
        Reference/ground truth array
    data_range : float, optional
        Dynamic range of the data (default: None, uses max(x_ref) - min(x_ref))

    Returns:
    --------
    float
        PSNR value in decibels, or infinity if MSE is zero

    Notes:
    ------
    Standard metric for image quality assessment. Values typically range
    from 20-50 dB for natural images, with higher values indicating better quality.
    """
    x = x.astype(np.float64)
    x_ref = x_ref.astype(np.float64)
    mse = np.mean((x - x_ref) ** 2)
    if mse == 0:
        return float("inf")
    if data_range is None:
        data_range = float(x_ref.max() - x_ref.min() or 1.0)
    return 10.0 * math.log10((data_range**2) / mse)


def relative_error(x: np.ndarray, x_ref: np.ndarray) -> float:
    """
    Relative Frobenius error: ||x - x_ref||_F / ||x_ref||_F.

    Computes the normalized error between two arrays using the Frobenius norm.
    This metric is scale-invariant and commonly used for restoration quality assessment.

    Parameters:
    -----------
    x : np.ndarray
        Estimated/reconstructed array
    x_ref : np.ndarray
        Reference/ground truth array

    Returns:
    --------
    float
        Relative error value, or infinity if reference norm is zero

    Notes:
    ------
    Values closer to 0 indicate better reconstruction quality.
    This metric is particularly useful for comparing restoration algorithms.
    """
    num = np.linalg.norm(x - x_ref)
    den = np.linalg.norm(x_ref)
    return float(num / den) if den != 0 else float("inf")


# -----------------------------
# QSLST restoration
# -----------------------------


def qslst_restore_fft(
    Bq: np.ndarray, psf: np.ndarray, lam: float, boundary: str = "periodic"
) -> np.ndarray:
    """
    QSLST (Algorithm 2) specialized to convolutional A with periodic BC.

    Implements the FFT-based Tikhonov solution for quaternion image restoration
    with convolutional blur operators. Per-channel closed-form solution:
    X_hat = conj(H_hat) * B_hat / (|H_hat|^2 + lam)

    Parameters:
    -----------
    Bq : np.ndarray
        Observed blurred+noisy quaternion image of shape (H, W, 4)
    psf : np.ndarray
        Point spread function of shape (kH, kW)
    lam : float
        Tikhonov regularization parameter (>= 0)
    boundary : str, optional
        Boundary condition, only "periodic" supported (default: "periodic")

    Returns:
    --------
    np.ndarray
        Restored quaternion image of shape (H, W, 4)

    Notes:
    ------
    Efficient FFT-based implementation for BCCB (block-circulant with
    circulant blocks) operators. Each quaternion component is restored
    independently using the same frequency domain filter.
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

    Implements the matrix-based QSLST algorithm for non-convolutional operators:
        T = A^T A + lam I
        E = A^T B
        X = T^+ E   (Moore-Penrose pseudoinverse)

    Since A is real, T is real; then A(T) = I_4 ⊗ T, and A(T)^+ = I_4 ⊗ T^+.
    Therefore we can solve per quaternion component independently.

    Parameters:
    -----------
    Bq : np.ndarray
        Observed quaternion image of shape (H, W, 4)
    A_mat : np.ndarray
        Real blur matrix of shape (N, N) where N = H*W
    lam : float
        Tikhonov regularization parameter (>= 0)

    Returns:
    --------
    np.ndarray
        Restored quaternion image of shape (H, W, 4)

    Notes:
    ------
    General implementation that works with any real matrix A, not just
    convolutional operators. Uses pseudoinverse for robust solution
    even when the system is ill-conditioned.
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
        b = Bq[..., c].reshape(-1)  # B component
        e = A_mat.T @ b  # E = A^T b
        x = T_pinv @ e  # X = T^+ E
        Xq[..., c] = x.reshape(H, W)
    return Xq


# -----------------------------
# End of module
# -----------------------------
