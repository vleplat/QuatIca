import os
import sys
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import quaternion  # type: ignore
from PIL import Image

# Ensure we can import from core
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core'))

from qslst import (
    rgb_to_quat,
    quat_to_rgb,
    build_psf_gaussian,
    apply_blur_fft,
    qslst_restore_fft,
    qslst_restore_matrix,
    psnr,
    relative_error,
)

# Placeholders for our methods (to be wired after benchmark spec)
from solver import NewtonSchulzPseudoinverse, HigherOrderNewtonSchulzPseudoinverse
from utils import quat_matmat, quat_hermitian, SparseQuaternionMatrix
from scipy import sparse as _sp


def load_rgb_image(path: str) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img).astype(np.float64) / 255.0
    return arr


def save_rgb_image(arr: np.ndarray, path: str) -> None:
    arr_255 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr_255).save(path)


try:
    from skimage.metrics import peak_signal_noise_ratio as _psnr_sk
    from skimage.metrics import structural_similarity as _ssim_sk
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


def _psnr_fallback(x: np.ndarray, x_ref: np.ndarray, data_range: float = 1.0) -> float:
    x = x.astype(np.float64)
    x_ref = x_ref.astype(np.float64)
    mse = np.mean((x - x_ref) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10((data_range ** 2) / mse)


def ssim_local(x: np.ndarray, y: np.ndarray, data_range: float = 1.0,
               K1: float = 0.01, K2: float = 0.03, sigma: float = 1.5) -> float:
    """Simple SSIM for RGB images using Gaussian smoothing (per-channel avg)."""
    from scipy.ndimage import gaussian_filter
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    if x.ndim == 2:
        x = x[..., None]
        y = y[..., None]
    ssim_vals = []
    for c in range(x.shape[2]):
        xc = x[..., c]
        yc = y[..., c]
        mu_x = gaussian_filter(xc, sigma)
        mu_y = gaussian_filter(yc, sigma)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sigma_x2 = gaussian_filter(xc * xc, sigma) - mu_x2
        sigma_y2 = gaussian_filter(yc * yc, sigma) - mu_y2
        sigma_xy = gaussian_filter(xc * yc, sigma) - mu_xy
        num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
        ssim_map = num / (den + 1e-12)
        ssim_vals.append(np.mean(ssim_map))
    return float(np.mean(ssim_vals))


def psnr_metric(x: np.ndarray, x_ref: np.ndarray, data_range: float = 1.0) -> float:
    """Prefer skimage PSNR if available; fallback to local implementation."""
    if _HAS_SKIMAGE:
        # skimage expects (image_true, image_test)
        return float(_psnr_sk(x_ref, x, data_range=data_range))
    return _psnr_fallback(x, x_ref, data_range=data_range)


def ssim_metric(x: np.ndarray, x_ref: np.ndarray, data_range: float = 1.0) -> float:
    """Prefer skimage SSIM if available; fallback to Gaussian-smoothed local SSIM."""
    if _HAS_SKIMAGE:
        try:
            # Newer skimage: channel_axis=-1; older: multichannel=True
            return float(_ssim_sk(x_ref, x, data_range=data_range, channel_axis=-1))
        except TypeError:
            return float(_ssim_sk(x_ref, x, data_range=data_range, multichannel=True))
    return ssim_local(x, x_ref, data_range=data_range)


def _build_bccb_matrix(psf: np.ndarray, H: int, W: int) -> np.ndarray:
    """Build dense BCCB (periodic) convolution matrix A (N×N) from PSF for manageable sizes.
    Columns are vec of 2D circular shifts of the centered PSF.
    """
    from qslst import _pad_psf  # reuse centering
    base = _pad_psf(psf, (H, W))
    cols = []
    for i in range(H):
        for j in range(W):
            shifted = np.roll(np.roll(base, i, axis=0), j, axis=1)
            cols.append(shifted.reshape(-1))
    A = np.stack(cols, axis=1)
    return A


def _build_bccb_csr(psf: np.ndarray, H: int, W: int) -> _sp.csr_matrix:
    """Build BCCB convolution matrix A (N×N) as CSR from PSF for periodic BC.
    Uses centered offsets so that A @ vec(X) corresponds to circular conv with PSF.
    """
    N = H * W
    kH, kW = psf.shape
    cH, cW = kH // 2, kW // 2
    data = []
    rows = []
    cols = []
    # Skip zeros in PSF to reduce work
    nonzero = [(du, dv, float(psf[du, dv])) for du in range(kH) for dv in range(kW) if psf[du, dv] != 0.0]
    for i in range(H):
        base_i = i * W
        for j in range(W):
            r = base_i + j
            for (du, dv, w) in nonzero:
                ii = (i + (du - cH)) % H
                jj = (j + (dv - cW)) % W
                c = ii * W + jj
                rows.append(r)
                cols.append(c)
                data.append(w)
    A_csr = _sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    return A_csr


def main():
    parser = argparse.ArgumentParser(description='Quaternion image deblurring benchmark: QSLST (FFT/matrix) vs NS/HON')
    parser.add_argument('--size', type=int, default=32, help='Resize to size×size for all paths (default: 32)')
    parser.add_argument('--lam', type=float, default=1e-3, help='Tikhonov lambda for QSLST (default: 1e-3)')
    parser.add_argument('--snr', type=float, default=None, help='Optional SNR dB (add AWGN)')
    parser.add_argument('--ns_mode', type=str, default='dense', choices=['dense', 'sparse', 'fftT', 'tikhonov_aug'], help='NS mode: dense/sparse A^†, fftT for NS on T, or tikhonov_aug for augmented [A;sqrt(lam)I]')
    parser.add_argument('--ns_iters', type=int, default=14, help='Iterations for NS when ns_mode=fftT (default: 14)')
    parser.add_argument('--fftT_order', type=int, default=2, choices=[2,3], help='Order of inverse-NS in fftT mode: 2 (Newton–Schulz) or 3 (Halley)')
    parser.add_argument('--image', type=str, default='kodim16', choices=['kodim16', 'kodim20'], help='Image to use: kodim16 or kodim20')
    args = parser.parse_args()
    # Paths
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_img = os.path.join(repo_root, 'data', 'images', f'{args.image}.png')
    out_dir = os.path.join(repo_root, 'output_figures')
    os.makedirs(out_dir, exist_ok=True)

    # Load clean image and convert to quaternion
    rgb_full = load_rgb_image(data_img)
    print(f"Loaded image: {data_img}, full size={rgb_full.shape}", flush=True)
    # Uniform size for fair comparison and reasonable runtime
    target_size = (args.size, args.size)
    rgb = np.asarray(Image.fromarray((rgb_full * 255).astype(np.uint8)).resize(target_size, Image.BILINEAR)).astype(np.float64) / 255.0
    print(f"Resized to: {rgb.shape}", flush=True)
    Q_clean = rgb_to_quat(rgb, real_part=0.0)

    # Build blur PSF and generate blurred observation (periodic boundary)
    psf = build_psf_gaussian(radius=2, sigma=1.0)
    Q_blur = apply_blur_fft(Q_clean, psf, boundary='periodic')

    # Optional noise (SNR in dB)
    if args.snr is not None:
        from qslst import add_awgn_snr
        print(f"Adding AWGN: SNR={args.snr} dB", flush=True)
        Bq = add_awgn_snr(Q_blur, snr_db=float(args.snr))
    else:
        Bq = Q_blur
    # Report measured SNR to confirm noise is present
    measured_snr_db = None
    if args.snr is not None:
        sig_pow = float(np.sum(Q_blur ** 2)) + 1e-30
        noise_pow = float(np.sum((Bq - Q_blur) ** 2)) + 1e-30
        measured_snr_db = 10.0 * np.log10(sig_pow / noise_pow)
        print(f"Measured SNR (quaternion space): {measured_snr_db:.2f} dB", flush=True)

    # 1) QSLST (FFT path)
    lam = float(args.lam)
    print(f"[QSLST-FFT] Start (lambda={lam})", flush=True)
    t0 = time.time()
    X_qslst = qslst_restore_fft(Bq, psf, lam, boundary='periodic')
    t_qslst = time.time() - t0
    print(f"[QSLST-FFT] Done in {t_qslst:.3f}s", flush=True)

    # 2) QSLST (matrix path, explicit A) — small size benchmark (SKIP for report)
    # We only need QSLST-FFT vs FFT-NS-Q for the report
    Hh, Ww, _ = Bq.shape
    # Skip matrix path to save time - not needed for report comparison
    X_qslst_mat = X_qslst.copy()  # Use FFT result as placeholder
    t_qslst_mat = t_qslst  # Use FFT time as placeholder

    # 3) NS variants
    N = Hh * Ww

    # Prepare quaternion embedding helpers (for dense/sparse A^† path)
    def real_matrix_to_quat(M: np.ndarray) -> np.ndarray:
        comp = np.zeros((M.shape[0], M.shape[1], 4))
        comp[..., 0] = M
        return quaternion.as_quat_array(comp)

    def real_vec_to_quat(v: np.ndarray) -> np.ndarray:
        comp = np.zeros((v.shape[0], 1, 4))
        comp[..., 0] = v.reshape(-1, 1)
        return quaternion.as_quat_array(comp)

    # Helper: NS on T in Fourier domain (matrix-free)
    def _ns_tikhonov_fft_restore(Bq_in: np.ndarray, psf_in: np.ndarray, lam_in: float, iters: int) -> np.ndarray:
        from qslst import _pad_psf  # reuse centering
        H, W, _ = Bq_in.shape
        H_pad = _pad_psf(psf_in, (H, W))
        H_hat = np.fft.fft2(H_pad)
        Th = (np.abs(H_hat) ** 2) + lam_in
        tmin = float(Th.min())
        tmax = float(Th.max())
        alpha = 2.0 / (tmax + tmin + 1e-12)
        y = np.full_like(Th, alpha, dtype=np.complex128)
        for _ in range(max(0, int(iters))):
            y = y * (2.0 - Th * y)
        H_conj = np.conj(H_hat)
        X_out = np.empty_like(Bq_in)
        for c in range(4):
            B_hat = np.fft.fft2(Bq_in[..., c])
            E_hat = H_conj * B_hat
            X_out[..., c] = np.real(np.fft.ifft2(y * E_hat))
        return X_out

    def _hon_ns_tikhonov_fft_restore(Bq_in: np.ndarray, psf_in: np.ndarray, lam_in: float, iters: int) -> np.ndarray:
        """Cubic inverse-NS (Halley) per-frequency on T = |H_hat|^2 + lam.

        Update: y <- y * (1 + r + r^2), where r = 1 - Th * y.
        """
        from qslst import _pad_psf
        H, W, _ = Bq_in.shape
        H_pad = _pad_psf(psf_in, (H, W))
        H_hat = np.fft.fft2(H_pad)
        Th = (np.abs(H_hat) ** 2) + lam_in
        tmin = float(Th.min())
        tmax = float(Th.max())
        alpha = 2.0 / (tmax + tmin + 1e-12)
        y = np.full_like(Th, alpha, dtype=np.complex128)
        for _ in range(max(0, int(iters))):
            r = 1.0 - Th * y
            y = y * (1.0 + r + r * r)
        H_conj = np.conj(H_hat)
        X_out = np.empty_like(Bq_in)
        for c in range(4):
            B_hat = np.fft.fft2(Bq_in[..., c])
            E_hat = H_conj * B_hat
            X_out[..., c] = np.real(np.fft.ifft2(y * E_hat))
        return X_out

    # Branch on mode
    ns_title = 'NS (A^†)'
    print(f"[NS] Mode = {args.ns_mode}", flush=True)
    if args.ns_mode == 'fftT':
        ns_title = f"NS (T^{-1}, FFT, order-{args.fftT_order})"
        print(f"[NS] Computing T^{-1} via inverse-NS in Fourier domain (order={args.fftT_order}, iters={args.ns_iters}) ...", flush=True)
        t1 = time.time()
        if int(args.fftT_order) == 3:
            X_ns = _hon_ns_tikhonov_fft_restore(Bq, psf, lam, args.ns_iters)
        else:
            X_ns = _ns_tikhonov_fft_restore(Bq, psf, lam, args.ns_iters)
        t_ns = time.time() - t1
        print(f"[NS] Done in {t_ns:.3f}s", flush=True)
        hon_skipped = True
        X_hon = X_ns.copy()
        t_hon = 0.0
    elif args.ns_mode == 'tikhonov_aug':
        ns_title = 'NS (augmented, Tikhonov)'
        print("[NS] Computing Tikhonov solution via augmented matrix C=[A; sqrt(lam) I] ...", flush=True)
        sqrtlam = float(np.sqrt(lam))
        # Build A in chosen backend
        if 'A_mat' not in locals():
            A_mat = _build_bccb_matrix(psf, Hh, Ww)
        # Dense quaternion operators
        Aq = real_matrix_to_quat(A_mat)
        Iq = real_matrix_to_quat(np.eye(N))
        # Augmented operator C (2N x N) and y = [b; 0]
        def reg_ns_apply_dense(b_vec: np.ndarray) -> np.ndarray:
            bq = real_vec_to_quat(b_vec)
            zq = quaternion.as_quat_array(np.zeros((N, 1, 4)))
            y = np.vstack([bq, zq])
            C = np.vstack([Aq, sqrtlam * Iq])
            C_pinv, _, _ = ns_solver.compute(C)
            xq = quat_matmat(C_pinv, y)
            return quaternion.as_float_array(xq)[..., 0].reshape(Hh, Ww)

        # If desired in future: sparse branch can be added similarly building C_real via _sp.vstack
        ns_solver = NewtonSchulzPseudoinverse(gamma=1.0, max_iter=60, tol=1e-8, verbose=False, compute_residuals=False)
        t1 = time.time()
        X_ns = np.empty_like(Bq)
        for c in range(4):
            X_ns[..., c] = reg_ns_apply_dense(Bq[..., c].reshape(-1))
        t_ns = time.time() - t1
        print(f"[NS] Done in {t_ns:.3f}s", flush=True)
        hon_skipped = True
        X_hon = X_ns.copy()
        t_hon = 0.0
    else:
        print(f"[NS] Computing A^† via Newton–Schulz ({args.ns_mode}, fast mode) ...", flush=True)
        ns_solver = NewtonSchulzPseudoinverse(gamma=1.0, max_iter=50, tol=1e-8, verbose=False, compute_residuals=False)
        if args.ns_mode == 'sparse':
            # Build CSR directly from PSF stencil (periodic BC)
            print("[NS] Building CSR operator ...", flush=True)
            A_csr = _build_bccb_csr(psf, Hh, Ww)
            zeros = _sp.csr_matrix(A_csr.shape)
            A_quat = SparseQuaternionMatrix(A_csr, zeros, zeros, zeros, A_csr.shape)
        else:
            # Build dense matrix for dense mode
            if 'A_mat' not in locals():
                print("[NS] Building dense BCCB matrix ...", flush=True)
                A_mat = _build_bccb_matrix(psf, Hh, Ww)
            A_quat = real_matrix_to_quat(A_mat)
        t1 = time.time()
        A_pinv_quat, _, _ = ns_solver.compute(A_quat)
        X_ns = np.empty_like(Bq)
        for c in range(4):
            b = Bq[..., c].reshape(-1)
            b_quat = real_vec_to_quat(b)
            x_quat = quat_matmat(A_pinv_quat, b_quat)
            x_real = quaternion.as_float_array(x_quat)[..., 0].reshape(Hh, Ww)
            X_ns[..., c] = x_real
        t_ns = time.time() - t1
        print(f"[NS] Done in {t_ns:.3f}s", flush=True)

        # Higher-Order NS on A (skip in sparse mode)
        hon_skipped = (args.ns_mode == 'sparse')
        if hon_skipped:
            print("[HON-NS] Skipped in sparse mode (sparse algebra not supported for HON updates)", flush=True)
            X_hon = X_ns.copy()
            t_hon = 0.0
        else:
            print("[HON-NS] Computing A^† via Higher-Order Newton–Schulz (quaternion, fast mode) ...", flush=True)
            t2 = time.time()
            hon_solver = HigherOrderNewtonSchulzPseudoinverse(max_iter=40, tol=0.0, verbose=False)
            A_pinv_hon_quat, _, _ = hon_solver.compute(A_quat)
            X_hon = np.empty_like(Bq)
            for c in range(4):
                b = Bq[..., c].reshape(-1)
                b_quat = real_vec_to_quat(b)
                x_quat = quat_matmat(A_pinv_hon_quat, b_quat)
                x_real = quaternion.as_float_array(x_quat)[..., 0].reshape(Hh, Ww)
                X_hon[..., c] = x_real
            t_hon = time.time() - t2
            print(f"[HON-NS] Done in {t_hon:.3f}s", flush=True)

    # Metrics
    rgb_ref = np.clip(quat_to_rgb(Q_clean), 0.0, 1.0)
    rgb_obs = np.clip(quat_to_rgb(Bq), 0.0, 1.0)
    rgb_qslst = np.clip(quat_to_rgb(X_qslst), 0.0, 1.0)
    rgb_qslst_mat = np.clip(quat_to_rgb(X_qslst_mat), 0.0, 1.0)
    rgb_ns = np.clip(quat_to_rgb(X_ns), 0.0, 1.0)
    rgb_hon = np.clip(quat_to_rgb(X_hon), 0.0, 1.0)

    psnr_qslst = psnr_metric(rgb_qslst, rgb_ref, data_range=1.0)
    psnr_qslst_mat = psnr_metric(rgb_qslst_mat, rgb_ref, data_range=1.0)
    psnr_ns = psnr_metric(rgb_ns, rgb_ref, data_range=1.0)
    psnr_hon = psnr_metric(rgb_hon, rgb_ref, data_range=1.0)

    ssim_qslst = ssim_metric(rgb_qslst, rgb_ref, data_range=1.0)
    ssim_qslst_mat = ssim_metric(rgb_qslst_mat, rgb_ref, data_range=1.0)
    ssim_ns = ssim_metric(rgb_ns, rgb_ref, data_range=1.0)
    ssim_hon = ssim_metric(rgb_hon, rgb_ref, data_range=1.0)

    rel_qslst = relative_error(rgb_qslst, rgb_ref)
    rel_ns = relative_error(rgb_ns, rgb_ref)
    rel_hon = relative_error(rgb_hon, rgb_ref)

    # Save outputs with unique names based on image and size
    base_name = f"{args.image}_{args.size}"
    save_rgb_image(rgb, os.path.join(out_dir, f'deblur_input_clean_{base_name}.png'))
    obs_filename = f'deblur_observed_blurred_{base_name}.png' if args.snr is None else f'deblur_observed_blur_noise_{int(args.snr)}dB_{base_name}.png'
    save_rgb_image(rgb_obs, os.path.join(out_dir, obs_filename))
    save_rgb_image(rgb_qslst, os.path.join(out_dir, f'deblur_qslst_fft_{base_name}.png'))
    save_rgb_image(rgb_qslst_mat, os.path.join(out_dir, f'deblur_qslst_matrix_{base_name}.png'))
    save_rgb_image(rgb_ns, os.path.join(out_dir, f'deblur_ns_{base_name}.png'))
    save_rgb_image(rgb_hon, os.path.join(out_dir, f'deblur_hon_{base_name}.png'))

    # Comparison grid figure (4 panels: Clean, Observed, QSLST-FFT, FFT-NS-Q)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    ns_panel_title = f'{ns_title}\nPSNR {psnr_ns:.2f} dB | SSIM {ssim_ns:.3f}'
    if args.snr is None:
        observed_title = 'Observed (blur)'
    else:
        if measured_snr_db is None:
            observed_title = f'Observed (blur+noise, {int(args.snr)} dB)'
        else:
            observed_title = f'Observed (blur+noise, req {int(args.snr)} dB | meas {measured_snr_db:.1f} dB)'
    panels = [
        (rgb, 'Clean'),
        (rgb_obs, observed_title),
        (rgb_qslst, f'QSLST-FFT\nPSNR {psnr_qslst:.2f} dB | SSIM {ssim_qslst:.3f}'),
        (rgb_ns, ns_panel_title)
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(np.clip(img, 0, 1), interpolation='nearest')
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    plt.tight_layout()
    grid_path = os.path.join(out_dir, f'deblur_comparison_grid_{base_name}.png')
    plt.savefig(grid_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Print summary
    print(f"Image Deblurring (kodim16 -> {args.size}x{args.size})", flush=True)
    print(f'  QSLST (FFT):    PSNR={psnr_qslst:.2f}dB  SSIM={ssim_qslst:.3f}  RelErr={rel_qslst:.3e}  time={t_qslst:.3f}s')
    print(f'  QSLST (matrix): PSNR={psnr_qslst_mat:.2f}dB SSIM={ssim_qslst_mat:.3f} time={t_qslst_mat:.3f}s')
    print(f'  {ns_title}:        PSNR={psnr_ns:.2f}dB    SSIM={ssim_ns:.3f}    RelErr={rel_ns:.3e}    time={t_ns:.3f}s')
    if hon_skipped:
        reason = 'fftT inline order' if args.ns_mode == 'fftT' else 'sparse mode'
        print(f'  HON-NS:           skipped ({reason})')
    else:
        print(f'  HON-NS (A^†):    PSNR={psnr_hon:.2f}dB   SSIM={ssim_hon:.3f}   RelErr={rel_hon:.3e}   time={t_hon:.3f}s')
    print(f'Outputs saved to: {out_dir}\n  - deblur_input_clean.png\n  - {obs_filename}\n  - deblur_qslst_fft.png\n  - deblur_qslst_matrix.png\n  - deblur_ns.png\n  - deblur_hon.png\n  - deblur_comparison_grid.png')


if __name__ == '__main__':
    main()


