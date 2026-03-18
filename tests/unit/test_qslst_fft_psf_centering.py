import numpy as np


def _manual_pad_psf_center_to_origin(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Pad PSF then roll so its center sits at (0,0)."""
    H, W = shape
    kH, kW = psf.shape
    pad = np.zeros((H, W), dtype=np.float64)
    pad[: min(kH, H), : min(kW, W)] = psf[: min(kH, H), : min(kW, W)]
    pad = np.roll(pad, -(kH // 2), axis=0)
    pad = np.roll(pad, -(kW // 2), axis=1)
    return pad


def _circular_conv2d_from_kernel0(x: np.ndarray, kernel0: np.ndarray) -> np.ndarray:
    """
    Circular convolution using kernel indexed with origin at (0,0):
        y[i,j] = sum_{a,b} kernel0[a,b] * x[i-a, j-b]  (mod H,W)
    Implemented via weighted rolls.
    """
    H, W = x.shape
    assert kernel0.shape == (H, W)
    y = np.zeros_like(x, dtype=np.float64)
    for a in range(H):
        for b in range(W):
            w = kernel0[a, b]
            if w != 0.0:
                y += w * np.roll(np.roll(x, a, axis=0), b, axis=1)
    return y


def test_apply_blur_fft_matches_explicit_circular_convolution_asymmetric_psf() -> None:
    from quatica.qslst import apply_blur_fft

    rng = np.random.default_rng(0)
    H = W = 8

    # Quaternion image (H,W,4); test all channels to match implementation.
    Q = rng.normal(size=(H, W, 4))

    # Asymmetric PSF: centering/phase mistakes show up immediately.
    psf = np.array(
        [
            [0.00, 0.10, 0.00],
            [0.20, 0.30, 0.00],
            [0.00, 0.00, 0.40],
        ],
        dtype=np.float64,
    )
    psf /= psf.sum()

    B_fft = apply_blur_fft(Q, psf, boundary="periodic")

    kernel0 = _manual_pad_psf_center_to_origin(psf, (H, W))
    B_direct = np.empty_like(Q)
    for c in range(4):
        B_direct[..., c] = _circular_conv2d_from_kernel0(Q[..., c], kernel0)

    np.testing.assert_allclose(B_fft, B_direct, rtol=0.0, atol=1e-10)


def test_apply_blur_fft_identity_for_centered_delta_psf() -> None:
    from quatica.qslst import apply_blur_fft

    rng = np.random.default_rng(1)
    H = W = 9
    Q = rng.normal(size=(H, W, 4))

    psf = np.zeros((3, 3), dtype=np.float64)
    psf[1, 1] = 1.0  # delta at the center

    B = apply_blur_fft(Q, psf, boundary="periodic")
    np.testing.assert_allclose(B, Q, rtol=0.0, atol=1e-12)


def test_qslst_fft_noiseless_unregularized_recovers_for_invertible_psf() -> None:
    """
    Regression test: in the noiseless unregularized case (lam=0), QSLST-FFT
    should act like an inverse filter when the PSF is well-conditioned.

    We use a PSF that is a convex combination of a centered delta and a small
    Gaussian blur so that its Fourier response is bounded away from 0.
    """
    from quatica.qslst import apply_blur_fft, build_psf_gaussian, qslst_restore_fft

    rng = np.random.default_rng(0)
    H = W = 16
    Q = rng.normal(size=(H, W, 4))

    delta = np.zeros((5, 5), dtype=np.float64)
    delta[2, 2] = 1.0
    gauss = build_psf_gaussian(radius=2, sigma=1.0)
    assert gauss.shape == delta.shape

    eps = 0.25
    psf = (1.0 - eps) * delta + eps * gauss
    psf /= psf.sum()

    B = apply_blur_fft(Q, psf, boundary="periodic")
    X = qslst_restore_fft(B, psf, lam=0.0, boundary="periodic")

    # Should be essentially exact (no regularization, stable PSF).
    np.testing.assert_allclose(X, Q, rtol=0.0, atol=1e-10)

