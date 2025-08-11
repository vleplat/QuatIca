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
from utils import quat_matmat, quat_hermitian


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


def main():
    parser = argparse.ArgumentParser(description='Quaternion image deblurring benchmark: QSLST (FFT/matrix) vs NS/HON')
    parser.add_argument('--size', type=int, default=32, help='Resize to size×size for all paths (default: 32)')
    parser.add_argument('--lam', type=float, default=1e-3, help='Tikhonov lambda for QSLST (default: 1e-3)')
    parser.add_argument('--snr', type=float, default=None, help='Optional SNR dB (add AWGN)')
    args = parser.parse_args()
    # Paths
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_img = os.path.join(repo_root, 'data', 'images', 'kodim16.png')
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

    # 1) QSLST (FFT path)
    lam = float(args.lam)
    print(f"[QSLST-FFT] Start (lambda={lam})", flush=True)
    t0 = time.time()
    X_qslst = qslst_restore_fft(Bq, psf, lam, boundary='periodic')
    t_qslst = time.time() - t0
    print(f"[QSLST-FFT] Done in {t_qslst:.3f}s", flush=True)

    # 2) QSLST (matrix path, explicit A) — small size benchmark
    Hh, Ww, _ = Bq.shape
    print("[QSLST-Matrix] Building explicit BCCB A ...", flush=True)
    A_mat = _build_bccb_matrix(psf, Hh, Ww)  # (N x N)
    print(f"[QSLST-Matrix] A shape={A_mat.shape}", flush=True)
    print(f"[QSLST-Matrix] Start (lambda={lam})", flush=True)
    t_m = time.time()
    X_qslst_mat = qslst_restore_matrix(Bq, A_mat, lam)
    t_qslst_mat = time.time() - t_m
    print(f"[QSLST-Matrix] Done in {t_qslst_mat:.3f}s", flush=True)

    # 3) NS and Higher-Order NS solving directly X = A^† B in quaternion domain (no real embedding mapping)
    N = Hh * Ww

    # Prepare quaternion embedding helpers
    def real_matrix_to_quat(M: np.ndarray) -> np.ndarray:
        comp = np.zeros((M.shape[0], M.shape[1], 4))
        comp[..., 0] = M
        return quaternion.as_quat_array(comp)

    def real_vec_to_quat(v: np.ndarray) -> np.ndarray:
        comp = np.zeros((v.shape[0], 1, 4))
        comp[..., 0] = v.reshape(-1, 1)
        return quaternion.as_quat_array(comp)

    # NS on A
    print("[NS] Computing A^† via Newton–Schulz (quaternion) ...", flush=True)
    t1 = time.time()
    ns_solver = NewtonSchulzPseudoinverse(gamma=1.0, max_iter=50, tol=1e-8, verbose=True)
    A_quat = real_matrix_to_quat(A_mat)
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

    # Higher-Order NS on A
    print("[HON-NS] Computing A^† via Higher-Order Newton–Schulz (quaternion) ...", flush=True)
    t2 = time.time()
    hon_solver = HigherOrderNewtonSchulzPseudoinverse(max_iter=40, tol=0.0, verbose=True)
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

    # Save outputs
    save_rgb_image(rgb, os.path.join(out_dir, 'deblur_input_clean.png'))
    save_rgb_image(rgb_obs, os.path.join(out_dir, 'deblur_observed_blurred.png'))
    save_rgb_image(rgb_qslst, os.path.join(out_dir, 'deblur_qslst_fft.png'))
    save_rgb_image(rgb_qslst_mat, os.path.join(out_dir, 'deblur_qslst_matrix.png'))
    save_rgb_image(rgb_ns, os.path.join(out_dir, 'deblur_ns.png'))
    save_rgb_image(rgb_hon, os.path.join(out_dir, 'deblur_hon.png'))

    # Comparison grid figure
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    panels = [
        (rgb, 'Clean'),
        (rgb_obs, 'Observed'),
        (rgb_qslst, f'QSLST-FFT\nPSNR {psnr_qslst:.2f} dB | SSIM {ssim_qslst:.3f}'),
        (rgb_qslst_mat, f'QSLST-Matrix\nPSNR {psnr_qslst_mat:.2f} dB | SSIM {ssim_qslst_mat:.3f}'),
        (rgb_ns, f'NS (A^†)\nPSNR {psnr_ns:.2f} dB | SSIM {ssim_ns:.3f}'),
        (rgb_hon, f'HON-NS (A^†)\nPSNR {psnr_hon:.2f} dB | SSIM {ssim_hon:.3f}')
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    grid_path = os.path.join(out_dir, 'deblur_comparison_grid.png')
    plt.savefig(grid_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Print summary
    print(f"Image Deblurring (kodim16 -> {args.size}x{args.size})", flush=True)
    print(f'  QSLST (FFT):    PSNR={psnr_qslst:.2f}dB  SSIM={ssim_qslst:.3f}  RelErr={rel_qslst:.3e}  time={t_qslst:.3f}s')
    print(f'  QSLST (matrix): PSNR={psnr_qslst_mat:.2f}dB SSIM={ssim_qslst_mat:.3f} time={t_qslst_mat:.3f}s')
    print(f'  NS (A^†):        PSNR={psnr_ns:.2f}dB    SSIM={ssim_ns:.3f}    RelErr={rel_ns:.3e}    time={t_ns:.3f}s')
    print(f'  HON-NS (A^†):    PSNR={psnr_hon:.2f}dB   SSIM={ssim_hon:.3f}   RelErr={rel_hon:.3e}   time={t_hon:.3f}s')
    print(f'Outputs saved to: {out_dir}\n  - deblur_input_clean.png\n  - deblur_observed_blurred.png\n  - deblur_qslst_fft.png\n  - deblur_qslst_matrix.png\n  - deblur_ns.png\n  - deblur_hon.png\n  - deblur_comparison_grid.png')


if __name__ == '__main__':
    main()


