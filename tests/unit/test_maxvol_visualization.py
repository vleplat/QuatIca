import os
import numpy as np
import quaternion
import matplotlib.pyplot as plt

from quatica import maxvol_submatrix_quat
from quatica.decomp.qsvd import classical_qsvd_full


def quaternion_volume(B: np.ndarray) -> float:
    U, s, V = classical_qsvd_full(B)
    if len(s) == 0:
        return 0.0
    prod = 1.0
    for val in s[: min(B.shape)]:
        prod *= float(val)
    return float(prod)


def generate_quat_matrix(m: int, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    real = rng.standard_normal((m, n))
    i = rng.standard_normal((m, n))
    j = rng.standard_normal((m, n))
    k = rng.standard_normal((m, n))
    return quaternion.as_quat_array(np.stack([real, i, j, k], axis=-1))


def to_magnitude(A: np.ndarray) -> np.ndarray:
    comp = quaternion.as_float_array(A)
    mag = np.sqrt(np.sum(comp ** 2, axis=-1))
    return mag


def test_maxvol_visualization_outputs():
    m, n, k = 20, 16, 5
    A = generate_quat_matrix(m, n, seed=123)

    I, J, B, info = maxvol_submatrix_quat(A, k=k, tol=1e-8, max_sweeps=30, track_history=True)

    # Compute volume history
    vols = []
    for Bi in info.get("B_history", []):
        vols.append(quaternion_volume(Bi))
    # Ensure we have some history
    assert len(vols) >= 1

    # Prepare output dir
    out_dir = os.path.join(os.path.dirname(__file__), "validation_output")
    os.makedirs(out_dir, exist_ok=True)

    # Plot original matrix magnitude and selected submatrix mask
    A_mag = to_magnitude(A)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axes[0].imshow(A_mag, aspect='auto', cmap='viridis')
    axes[0].set_title("|A| (magnitude)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Selection mask
    mask = np.zeros_like(A_mag)
    mask[np.ix_(I, J)] = 1.0
    im1 = axes[1].imshow(mask, aspect='auto', cmap='magma')
    axes[1].set_title("Selected submatrix mask")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Volume evolution
    axes[2].plot(vols, marker='o')
    axes[2].set_title("Volume evolution")
    axes[2].set_xlabel("Accepted swap index")
    axes[2].set_ylabel("Vol_H(B)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(out_dir, "maxvol_visualization.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # Save a text summary with indices and final volume
    summary_path = os.path.join(out_dir, "maxvol_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Rows I = {I}\n")
        f.write(f"Cols J = {J}\n")
        f.write(f"Final volume = {quaternion_volume(B):.6e}\n")
        f.write(f"History length = {len(vols)}\n")
        if len(vols) >= 2:
            diffs = [vols[t+1] - vols[t] for t in range(len(vols)-1)]
            f.write(f"Monotone nondecreasing (allowing tiny eps): {all(d>=-1e-10 for d in diffs)}\n")

    # Basic existence checks
    assert os.path.exists(fig_path)
    assert os.path.exists(summary_path)


