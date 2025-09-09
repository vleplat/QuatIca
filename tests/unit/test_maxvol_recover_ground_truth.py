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


def to_magnitude(A: np.ndarray) -> np.ndarray:
    comp = quaternion.as_float_array(A)
    return np.sqrt(np.sum(comp ** 2, axis=-1))


def generate_quat_matrix(m: int, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    real = rng.standard_normal((m, n))
    i = rng.standard_normal((m, n))
    j = rng.standard_normal((m, n))
    k = rng.standard_normal((m, n))
    return quaternion.as_quat_array(np.stack([real, i, j, k], axis=-1))


def convex_combine(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    # Elementwise quaternion convex combination: (1-alpha)*a + alpha*b
    return (1.0 - alpha) * a + alpha * b


def test_maxvol_recovers_ground_truth_from_convex_aug():
    # Ground-truth small core (k×k)
    k = 4
    B_true = generate_quat_matrix(k, k, seed=1)
    vol_true = quaternion_volume(B_true)

    # Build larger matrix A = P @ B_true @ Q^T where extra rows/cols
    # are convex combinations of the true ones (with sum < 1)
    m, n = 18, 15
    assert k < m and k < n
    rng = np.random.default_rng(3)

    # Real mixing matrices (commute with quaternion entries)
    P = np.zeros((m, k))
    for i in range(k):
        P[i, i] = 1.0
    for i in range(k, m):
        w = rng.random(k)
        w = w / (w.sum() + 1e-12)
        beta = rng.uniform(0.4, 0.9)  # sum < 1 ensures reduced volume
        P[i, :] = beta * w

    Q = np.zeros((n, k))
    for j in range(k):
        Q[j, j] = 1.0
    for j in range(k, n):
        w = rng.random(k)
        w = w / (w.sum() + 1e-12)
        beta = rng.uniform(0.4, 0.9)
        Q[j, :] = beta * w

    # Compute A = P @ B_true @ Q^T with quaternion-native ops
    # First compute S = P @ B_true (m×k), left real scaling of rows
    S = np.zeros((m, k), dtype=np.quaternion)
    for a in range(k):
        S += P[:, a][:, None] * B_true[a : a + 1, :]
    # Then A = S @ Q^T (m×n), right real scaling of columns
    A = np.zeros((m, n), dtype=np.quaternion)
    for b in range(k):
        A += S[:, b : b + 1] * Q[:, b][None, :]

    # Random row/col permutations (so indices are not trivially 0..k-1)
    perm_rows = rng.permutation(m)
    perm_cols = rng.permutation(n)
    A = A[perm_rows, :][:, perm_cols]

    # Ground-truth index sets after permutation
    I0 = [i for i, orig in enumerate(perm_rows) if orig < k]
    J0 = [j for j, orig in enumerate(perm_cols) if orig < k]

    # Run maxvol
    I, J, B_hat, info = maxvol_submatrix_quat(A, k=k, tol=1e-10, max_sweeps=80, track_history=True)

    # Check that ground-truth is recovered (order may differ)
    assert set(I) == set(I0)
    assert set(J) == set(J0)

    # Volumes: final ≥ true (equal ideally), and history monotone non-decreasing
    vol_hat = quaternion_volume(B_hat)
    assert vol_hat >= vol_true - 1e-8
    vols = [quaternion_volume(Bi) for Bi in info.get("B_history", [])]
    for a, b in zip(vols, vols[1:]):
        assert b >= a - 1e-10

    # Save visual validation
    out_dir = os.path.join(os.path.dirname(__file__), "validation_output")
    os.makedirs(out_dir, exist_ok=True)
    A_mag = to_magnitude(A)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(A_mag, aspect='auto', cmap='viridis')
    axes[0].set_title("|A| magnitude")
    mask_gt = np.zeros_like(A_mag)
    mask_gt[np.ix_(I0, J0)] = 1.0
    axes[1].imshow(mask_gt, aspect='auto', cmap='magma')
    axes[1].set_title("Ground-truth mask")
    mask_sel = np.zeros_like(A_mag)
    mask_sel[np.ix_(I, J)] = 1.0
    axes[2].imshow(mask_sel, aspect='auto', cmap='magma')
    axes[2].set_title("Recovered mask")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "maxvol_recover_gt.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    test_maxvol_recovers_ground_truth_from_convex_aug()
    print("Test completed successfully!")
