import argparse
import json
import os
import sys
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import quaternion  # type: ignore
from PIL import Image
from scipy import sparse as sp

# Prefer package imports; fall back to repository-local execution.
try:
    from quatica.qslst import (
        add_awgn_snr,
        apply_blur_fft,
        build_psf_gaussian,
        psnr as qslst_psnr,
        qslst_restore_fft,
        qslst_restore_matrix,
        quat_to_rgb,
        relative_error,
        rgb_to_quat,
    )
    from quatica.solver import (
        HigherOrderNewtonSchulzPseudoinverse,
        NewtonSchulzPseudoinverse,
    )
    from quatica.utils import SparseQuaternionMatrix, quat_matmat
except Exception:
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    QUATICA_DIR = os.path.join(REPO_ROOT, "quatica")
    if QUATICA_DIR not in sys.path:
        sys.path.append(QUATICA_DIR)
    from qslst import (  # type: ignore
        add_awgn_snr,
        apply_blur_fft,
        build_psf_gaussian,
        psnr as qslst_psnr,
        qslst_restore_fft,
        qslst_restore_matrix,
        quat_to_rgb,
        relative_error,
        rgb_to_quat,
    )
    from solver import (  # type: ignore
        HigherOrderNewtonSchulzPseudoinverse,
        NewtonSchulzPseudoinverse,
    )
    from utils import SparseQuaternionMatrix, quat_matmat  # type: ignore


def load_rgb_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float64) / 255.0


def save_rgb_image(arr: np.ndarray, path: str) -> None:
    arr_255 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr_255).save(path)


try:
    from skimage.metrics import structural_similarity as _ssim_sk

    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


def ssim_local(
    x: np.ndarray,
    y: np.ndarray,
    data_range: float = 1.0,
    K1: float = 0.01,
    K2: float = 0.03,
    sigma: float = 1.5,
) -> float:
    """Simple SSIM for RGB images using Gaussian smoothing (per-channel average)."""
    from scipy.ndimage import gaussian_filter

    x = x.astype(np.float64)
    y = y.astype(np.float64)
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    if x.ndim == 2:
        x = x[..., None]
        y = y[..., None]
    vals = []
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
        vals.append(float(np.mean(num / (den + 1e-12))))
    return float(np.mean(vals))


def ssim_metric(x: np.ndarray, x_ref: np.ndarray, data_range: float = 1.0) -> float:
    if _HAS_SKIMAGE:
        try:
            return float(_ssim_sk(x_ref, x, data_range=data_range, channel_axis=-1))
        except TypeError:
            return float(_ssim_sk(x_ref, x, data_range=data_range, multichannel=True))
    return ssim_local(x, x_ref, data_range=data_range)


def _build_bccb_csr(psf: np.ndarray, H: int, W: int) -> sp.csr_matrix:
    """Build periodic BCCB convolution matrix A (N x N) as CSR.

    For output pixel (i,j), the input pixel selected by a PSF entry (du,dv)
    is ((i - (du-cH)) mod H, (j - (dv-cW)) mod W), consistent with
    circular convolution using a centered PSF.
    """
    N = H * W
    kH, kW = psf.shape
    cH, cW = kH // 2, kW // 2
    data = []
    rows = []
    cols = []
    nonzero = [
        (du, dv, float(psf[du, dv]))
        for du in range(kH)
        for dv in range(kW)
        if psf[du, dv] != 0.0
    ]
    for i in range(H):
        base_i = i * W
        for j in range(W):
            r = base_i + j
            for du, dv, w in nonzero:
                ii = (i - (du - cH)) % H
                jj = (j - (dv - cW)) % W
                c = ii * W + jj
                rows.append(r)
                cols.append(c)
                data.append(w)
    return sp.csr_matrix((data, (rows, cols)), shape=(N, N))


def _build_bccb_matrix(psf: np.ndarray, H: int, W: int) -> np.ndarray:
    return _build_bccb_csr(psf, H, W).toarray()


def _validate_bccb_against_fft(psf: np.ndarray, H: int, W: int, rng: np.random.Generator) -> float:
    """Return relative mismatch between explicit A_mat and apply_blur_fft on one random scalar image."""
    A_mat = _build_bccb_matrix(psf, H, W)
    x = rng.standard_normal((H, W))
    Q = np.zeros((H, W, 4), dtype=np.float64)
    Q[..., 0] = x
    y_fft = apply_blur_fft(Q, psf, boundary="periodic")[..., 0].reshape(-1)
    y_mat = A_mat @ x.reshape(-1)
    return float(np.linalg.norm(y_fft - y_mat) / (np.linalg.norm(y_fft) + 1e-30))


def _real_matrix_to_quat(M: np.ndarray) -> np.ndarray:
    comp = np.zeros((M.shape[0], M.shape[1], 4), dtype=np.float64)
    comp[..., 0] = M
    return quaternion.as_quat_array(comp)


def _real_vec_to_quat(v: np.ndarray) -> np.ndarray:
    comp = np.zeros((v.shape[0], 1, 4), dtype=np.float64)
    comp[..., 0] = v.reshape(-1, 1)
    return quaternion.as_quat_array(comp)


def _ns_tikhonov_fft_restore(
    Bq_in: np.ndarray, psf_in: np.ndarray, lam_in: float, iters: int, order: int = 2
) -> np.ndarray:
    """Inverse iteration on T = |H_hat|^2 + lam in the Fourier domain.

    order=2 uses Newton-Schulz: y <- y * (2 - T y)
    order=3 uses cubic hyperpower: y <- y * (1 + r + r^2), r = 1 - T y
    """
    H, W, _ = Bq_in.shape
    # Build centered/padded PSF through public blur path convention.
    # Reconstruct H_hat by blurring a delta image once.
    delta = np.zeros((H, W, 4), dtype=np.float64)
    delta[0, 0, 0] = 1.0
    h_padded = apply_blur_fft(delta, psf_in, boundary="periodic")[..., 0]
    H_hat = np.fft.fft2(h_padded)
    Th = np.abs(H_hat) ** 2 + lam_in
    tmax = float(Th.max())
    # Safer NS initialization: for scalar t>0, Newton–Schulz for 1/t converges if
    # 0 < y0 < 2/t. Using y0≈2/tmax can be borderline when lam=0 and tmin≈0.
    alpha = 0.9 / (tmax + 1e-30)

    # Optional floor for the unregularized case to avoid huge 1/Th at very
    # small frequencies (or exact zeros), which can destabilize finite-iteration
    # inverse iteration. This is only relevant when lam_in=0.
    if lam_in == 0.0:
        floor = (1e-12) * (tmax + 1e-30)
        Th = np.maximum(Th, floor)

    y = np.full_like(Th, alpha, dtype=np.complex128)
    for _ in range(max(0, int(iters))):
        if order == 3:
            r = 1.0 - Th * y
            y = y * (1.0 + r + r * r)
        else:
            y = y * (2.0 - Th * y)
    H_conj = np.conj(H_hat)
    X_out = np.empty_like(Bq_in)
    for c in range(4):
        B_hat = np.fft.fft2(Bq_in[..., c])
        E_hat = H_conj * B_hat
        X_out[..., c] = np.real(np.fft.ifft2(y * E_hat))
    return X_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quaternion image deblurring demo: QSLST vs Newton-Schulz-based restoration"
    )
    parser.add_argument("--size", type=int, default=32, help="Resize to size x size (default: 32)")
    parser.add_argument("--lam", type=float, default=1e-3, help="Tikhonov lambda (default: 1e-3)")
    parser.add_argument("--snr", type=float, default=None, help="Optional AWGN SNR in dB")
    parser.add_argument(
        "--psf_radius",
        type=int,
        default=4,
        help="Gaussian PSF radius r (kernel size = 2r+1). Default: 4 (9x9).",
    )
    parser.add_argument(
        "--psf_sigma",
        type=float,
        default=1.0,
        help="Gaussian PSF sigma. Default: 1.0 (moderate blur).",
    )
    parser.add_argument(
        "--ns_mode",
        type=str,
        default="fftT",
        choices=["fftT", "dense", "sparse", "tikhonov_aug"],
        help=(
            "Comparison mode: fftT = inverse iteration on T=|H|^2+lambda in Fourier space; "
            "dense/sparse = NS pseudoinverse of the blur operator A; "
            "tikhonov_aug = NS pseudoinverse of the augmented Tikhonov operator [A; sqrt(lambda) I]."
        ),
    )
    parser.add_argument("--ns_iters", type=int, default=14, help="Iterations for fftT mode (default: 14)")
    parser.add_argument(
        "--fftT_order",
        type=int,
        default=2,
        choices=[2, 3],
        help="Order for fftT inverse iteration: 2 (Newton-Schulz) or 3 (cubic)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="kodim16",
        choices=["kodim16", "kodim20"],
        help="Image to use",
    )
    parser.add_argument(
        "--run_qslst_matrix",
        action="store_true",
        help="Also compute the explicit matrix-based QSLST reference path (slower).",
    )
    parser.add_argument(
        "--validate_bccb",
        action="store_true",
        help="Validate that the explicit BCCB matrix matches FFT blur on a random test image.",
    )
    parser.add_argument(
        "--metrics_json",
        type=str,
        default=None,
        help=(
            "Optional path to write machine-readable metrics/results as JSON. "
            "This avoids brittle stdout parsing in wrapper scripts."
        ),
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_img = os.path.join(repo_root, "data", "images", f"{args.image}.png")
    out_dir = os.path.join(repo_root, "output_figures")
    os.makedirs(out_dir, exist_ok=True)

    rgb_full = load_rgb_image(data_img)
    print(f"Loaded image: {data_img}, full size={rgb_full.shape}", flush=True)
    target_size = (args.size, args.size)
    rgb = (
        np.asarray(Image.fromarray((rgb_full * 255).astype(np.uint8)).resize(target_size, Image.BILINEAR))
        .astype(np.float64)
        / 255.0
    )
    print(f"Resized to: {rgb.shape}", flush=True)
    Q_clean = rgb_to_quat(rgb, real_part=0.0)

    psf = build_psf_gaussian(radius=int(args.psf_radius), sigma=float(args.psf_sigma))
    Q_blur = apply_blur_fft(Q_clean, psf, boundary="periodic")

    # Save a blur-only output for visual sanity checking (before any noise injection).
    rgb_blur = np.clip(quat_to_rgb(Q_blur), 0.0, 1.0)

    if args.snr is not None:
        print(f"Adding AWGN: requested SNR={args.snr} dB", flush=True)
        Bq = add_awgn_snr(Q_blur, snr_db=float(args.snr))
    else:
        Bq = Q_blur

    measured_snr_db: Optional[float] = None
    if args.snr is not None:
        sig_pow = float(np.sum(Q_blur**2)) + 1e-30
        noise_pow = float(np.sum((Bq - Q_blur) ** 2)) + 1e-30
        measured_snr_db = 10.0 * np.log10(sig_pow / noise_pow)
        print(f"Measured SNR (quaternion space): {measured_snr_db:.2f} dB", flush=True)

    Hh, Ww, _ = Bq.shape
    N = Hh * Ww
    rng = np.random.default_rng(0)

    if args.validate_bccb:
        rel_bccb = _validate_bccb_against_fft(psf, Hh, Ww, rng)
        print(f"BCCB validation rel. mismatch = {rel_bccb:.3e}", flush=True)
    else:
        rel_bccb = None

    print(f"[QSLST-FFT] Start (lambda={args.lam})", flush=True)
    t0 = time.time()
    X_qslst_fft = qslst_restore_fft(Bq, psf, args.lam, boundary="periodic")
    t_qslst_fft = time.time() - t0
    print(f"[QSLST-FFT] Done in {t_qslst_fft:.3f}s", flush=True)

    A_mat: Optional[np.ndarray] = None
    X_qslst_mat: Optional[np.ndarray] = None
    t_qslst_mat: Optional[float] = None
    if args.run_qslst_matrix:
        print("[QSLST-matrix] Building explicit BCCB matrix ...", flush=True)
        A_mat = _build_bccb_matrix(psf, Hh, Ww)
        print("[QSLST-matrix] Solving matrix-based reference path ...", flush=True)
        t1 = time.time()
        X_qslst_mat = qslst_restore_matrix(Bq, A_mat, args.lam)
        t_qslst_mat = time.time() - t1
        print(f"[QSLST-matrix] Done in {t_qslst_mat:.3f}s", flush=True)

    X_ns: np.ndarray
    X_hon: Optional[np.ndarray] = None
    t_ns: float = 0.0
    t_hon: Optional[float] = None
    ns_title = ""

    if args.ns_mode == "fftT":
        ns_title = f"NS on T (FFT, order-{args.fftT_order})"
        print(
            f"[NS] Inverse iteration on T=|H|^2+lambda in Fourier domain (order={args.fftT_order}, iters={args.ns_iters}) ...",
            flush=True,
        )
        t1 = time.time()
        X_ns = _ns_tikhonov_fft_restore(Bq, psf, args.lam, args.ns_iters, order=args.fftT_order)
        t_ns = time.time() - t1
        print(f"[NS] Done in {t_ns:.3f}s", flush=True)
    elif args.ns_mode == "tikhonov_aug":
        ns_title = "NS on augmented Tikhonov system"
        if A_mat is None:
            print("[NS] Building explicit BCCB matrix for augmented Tikhonov solve ...", flush=True)
            A_mat = _build_bccb_matrix(psf, Hh, Ww)
        Aq = _real_matrix_to_quat(A_mat)
        Iq = _real_matrix_to_quat(np.eye(N))
        sqrtlam = float(np.sqrt(args.lam))
        ns_solver = NewtonSchulzPseudoinverse(
            gamma=1.0, max_iter=60, tol=1e-8, verbose=False, compute_residuals=False
        )
        C = np.vstack([Aq, sqrtlam * Iq])
        print("[NS] Computing pseudoinverse of augmented operator C=[A; sqrt(lambda) I] ...", flush=True)
        t1 = time.time()
        C_pinv, _, _ = ns_solver.compute(C)
        X_ns = np.empty_like(Bq)
        zq = quaternion.as_quat_array(np.zeros((N, 1, 4), dtype=np.float64))
        for c in range(4):
            bq = _real_vec_to_quat(Bq[..., c].reshape(-1))
            y = np.vstack([bq, zq])
            xq = quat_matmat(C_pinv, y)
            X_ns[..., c] = quaternion.as_float_array(xq)[..., 0].reshape(Hh, Ww)
        t_ns = time.time() - t1
        print(f"[NS] Done in {t_ns:.3f}s", flush=True)
    else:
        if args.ns_mode == "sparse":
            ns_title = "NS pseudoinverse of sparse blur matrix A"
            print("[NS] Building sparse BCCB operator ...", flush=True)
            A_csr = _build_bccb_csr(psf, Hh, Ww)
            zeros = sp.csr_matrix(A_csr.shape)
            A_quat = SparseQuaternionMatrix(A_csr, zeros, zeros, zeros, A_csr.shape)
        else:
            ns_title = "NS pseudoinverse of dense blur matrix A"
            if A_mat is None:
                print("[NS] Building dense BCCB matrix ...", flush=True)
                A_mat = _build_bccb_matrix(psf, Hh, Ww)
            A_quat = _real_matrix_to_quat(A_mat)

        ns_solver = NewtonSchulzPseudoinverse(
            gamma=1.0, max_iter=50, tol=1e-8, verbose=False, compute_residuals=False
        )
        print("[NS] Computing pseudoinverse of blur operator A ...", flush=True)
        t1 = time.time()
        A_pinv_quat, _, _ = ns_solver.compute(A_quat)
        X_ns = np.empty_like(Bq)
        for c in range(4):
            bq = _real_vec_to_quat(Bq[..., c].reshape(-1))
            xq = quat_matmat(A_pinv_quat, bq)
            X_ns[..., c] = quaternion.as_float_array(xq)[..., 0].reshape(Hh, Ww)
        t_ns = time.time() - t1
        print(f"[NS] Done in {t_ns:.3f}s", flush=True)

        if args.ns_mode != "sparse":
            print("[HON-NS] Computing higher-order pseudoinverse of blur operator A ...", flush=True)
            hon_solver = HigherOrderNewtonSchulzPseudoinverse(max_iter=40, tol=0.0, verbose=False)
            t2 = time.time()
            A_pinv_hon_quat, _, _ = hon_solver.compute(A_quat)
            X_hon = np.empty_like(Bq)
            for c in range(4):
                bq = _real_vec_to_quat(Bq[..., c].reshape(-1))
                xq = quat_matmat(A_pinv_hon_quat, bq)
                X_hon[..., c] = quaternion.as_float_array(xq)[..., 0].reshape(Hh, Ww)
            t_hon = time.time() - t2
            print(f"[HON-NS] Done in {t_hon:.3f}s", flush=True)

    rgb_ref = np.clip(quat_to_rgb(Q_clean), 0.0, 1.0)
    rgb_obs = np.clip(quat_to_rgb(Bq), 0.0, 1.0)
    rgb_qslst_fft = np.clip(quat_to_rgb(X_qslst_fft), 0.0, 1.0)
    rgb_ns = np.clip(quat_to_rgb(X_ns), 0.0, 1.0)

    psnr_qslst_fft = qslst_psnr(rgb_qslst_fft, rgb_ref, data_range=1.0)
    ssim_qslst_fft = ssim_metric(rgb_qslst_fft, rgb_ref, data_range=1.0)
    rel_qslst_fft = relative_error(rgb_qslst_fft, rgb_ref)

    psnr_ns = qslst_psnr(rgb_ns, rgb_ref, data_range=1.0)
    ssim_ns = ssim_metric(rgb_ns, rgb_ref, data_range=1.0)
    rel_ns = relative_error(rgb_ns, rgb_ref)

    rgb_qslst_mat = None
    if X_qslst_mat is not None:
        rgb_qslst_mat = np.clip(quat_to_rgb(X_qslst_mat), 0.0, 1.0)
        psnr_qslst_mat = qslst_psnr(rgb_qslst_mat, rgb_ref, data_range=1.0)
        ssim_qslst_mat = ssim_metric(rgb_qslst_mat, rgb_ref, data_range=1.0)
        rel_qslst_mat = relative_error(rgb_qslst_mat, rgb_ref)
    else:
        psnr_qslst_mat = ssim_qslst_mat = rel_qslst_mat = None

    rgb_hon = None
    if X_hon is not None:
        rgb_hon = np.clip(quat_to_rgb(X_hon), 0.0, 1.0)
        psnr_hon = qslst_psnr(rgb_hon, rgb_ref, data_range=1.0)
        ssim_hon = ssim_metric(rgb_hon, rgb_ref, data_range=1.0)
        rel_hon = relative_error(rgb_hon, rgb_ref)
    else:
        psnr_hon = ssim_hon = rel_hon = None

    base_name = f"{args.image}_{args.size}"
    saved = []
    clean_path = os.path.join(out_dir, f"deblur_input_clean_{base_name}.png")
    save_rgb_image(rgb, clean_path)
    saved.append(os.path.basename(clean_path))

    blur_path = os.path.join(out_dir, f"deblur_blurred_{base_name}.png")
    save_rgb_image(rgb_blur, blur_path)
    saved.append(os.path.basename(blur_path))

    obs_filename = (
        f"deblur_observed_blurred_{base_name}.png"
        if args.snr is None
        else f"deblur_observed_blur_noise_{int(args.snr)}dB_{base_name}.png"
    )
    obs_path = os.path.join(out_dir, obs_filename)
    save_rgb_image(rgb_obs, obs_path)
    saved.append(os.path.basename(obs_path))

    qslst_fft_path = os.path.join(out_dir, f"deblur_qslst_fft_{base_name}.png")
    save_rgb_image(rgb_qslst_fft, qslst_fft_path)
    saved.append(os.path.basename(qslst_fft_path))

    ns_path = os.path.join(out_dir, f"deblur_ns_{args.ns_mode}_{base_name}.png")
    save_rgb_image(rgb_ns, ns_path)
    saved.append(os.path.basename(ns_path))

    if rgb_qslst_mat is not None:
        qslst_mat_path = os.path.join(out_dir, f"deblur_qslst_matrix_{base_name}.png")
        save_rgb_image(rgb_qslst_mat, qslst_mat_path)
        saved.append(os.path.basename(qslst_mat_path))

    if rgb_hon is not None:
        hon_path = os.path.join(out_dir, f"deblur_hon_{base_name}.png")
        save_rgb_image(rgb_hon, hon_path)
        saved.append(os.path.basename(hon_path))

    panels = [(rgb_ref, "Clean")]
    if args.snr is None:
        observed_title = "Observed (blur)"
    else:
        observed_title = (
            f"Observed (blur+noise, req {int(args.snr)} dB | meas {measured_snr_db:.1f} dB)"
            if measured_snr_db is not None
            else f"Observed (blur+noise, {int(args.snr)} dB)"
        )
    panels.append((rgb_obs, observed_title))
    panels.append((rgb_qslst_fft, f"QSLST-FFT\nPSNR {psnr_qslst_fft:.2f} dB | SSIM {ssim_qslst_fft:.3f}"))
    panels.append((rgb_ns, f"{ns_title}\nPSNR {psnr_ns:.2f} dB | SSIM {ssim_ns:.3f}"))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(np.clip(img, 0, 1), interpolation="nearest")
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    plt.tight_layout()
    grid_path = os.path.join(out_dir, f"deblur_comparison_grid_{base_name}.png")
    plt.savefig(grid_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved.append(os.path.basename(grid_path))

    print(f"Image Deblurring ({args.image} -> {args.size}x{args.size})", flush=True)
    print(
        f"  QSLST (FFT):     PSNR={psnr_qslst_fft:.2f}dB  SSIM={ssim_qslst_fft:.3f}  RelErr={rel_qslst_fft:.3e}  time={t_qslst_fft:.3f}s",
        flush=True,
    )
    if rgb_qslst_mat is not None:
        print(
            f"  QSLST (matrix):  PSNR={psnr_qslst_mat:.2f}dB  SSIM={ssim_qslst_mat:.3f}  RelErr={rel_qslst_mat:.3e}  time={t_qslst_mat:.3f}s",
            flush=True,
        )
    else:
        print("  QSLST (matrix):  not run", flush=True)

    print(
        f"  {ns_title}: PSNR={psnr_ns:.2f}dB  SSIM={ssim_ns:.3f}  RelErr={rel_ns:.3e}  time={t_ns:.3f}s",
        flush=True,
    )
    if rgb_hon is not None:
        print(
            f"  HON-NS (A^dagger): PSNR={psnr_hon:.2f}dB  SSIM={ssim_hon:.3f}  RelErr={rel_hon:.3e}  time={t_hon:.3f}s",
            flush=True,
        )
    elif args.ns_mode in {"fftT", "tikhonov_aug", "sparse"}:
        print("  HON-NS: not run in this mode", flush=True)

    print(f"Outputs saved to: {out_dir}", flush=True)
    for name in saved:
        print(f"  - {name}", flush=True)

    if args.metrics_json:
        payload = {
            "image": args.image,
            "size": int(args.size),
            "snr": None if args.snr is None else float(args.snr),
            "snr_measured_db": None if measured_snr_db is None else float(measured_snr_db),
            "lam": float(args.lam),
            "psf_radius": int(args.psf_radius),
            "psf_sigma": float(args.psf_sigma),
            "ns_mode": str(args.ns_mode),
            "ns_iters": int(args.ns_iters),
            "fftT_order": int(args.fftT_order),
            "validate_bccb_rel_mismatch": None if rel_bccb is None else float(rel_bccb),
            "qslst_fft": {
                "psnr": float(psnr_qslst_fft),
                "ssim": float(ssim_qslst_fft),
                "rel_err": float(rel_qslst_fft),
                "time_s": float(t_qslst_fft),
            },
            "qslst_matrix": None
            if rgb_qslst_mat is None
            else {
                "psnr": float(psnr_qslst_mat),
                "ssim": float(ssim_qslst_mat),
                "rel_err": float(rel_qslst_mat),
                "time_s": None if t_qslst_mat is None else float(t_qslst_mat),
            },
            "ns": {
                "title": str(ns_title),
                "psnr": float(psnr_ns),
                "ssim": float(ssim_ns),
                "rel_err": float(rel_ns),
                "time_s": float(t_ns),
            },
            "hon": None
            if rgb_hon is None
            else {
                "psnr": float(psnr_hon),
                "ssim": float(ssim_hon),
                "rel_err": float(rel_hon),
                "time_s": None if t_hon is None else float(t_hon),
            },
            "output_dir": str(out_dir),
            "saved_files": list(saved),
            "base_name": str(base_name),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.metrics_json)), exist_ok=True)
        with open(args.metrics_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
