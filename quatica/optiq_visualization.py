"""OptiQ-specific visualization helpers.

This module is intentionally *additive* and does not modify QuatIca's core
`visualization.py`. It provides a couple of convenience functions used by the
OptiQ regression scripts.

All functions accept dense numpy-quaternion matrices (dtype=quaternion).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import quaternion  # type: ignore


def _as_float_components(A: np.ndarray) -> np.ndarray:
    """Return float components array with shape (n, n, 4) for quaternion matrix A."""
    comp = quaternion.as_float_array(A)
    if comp.ndim != 3 or comp.shape[-1] != 4:
        raise ValueError("Expected a dense quaternion matrix (n×n) convertible via quaternion.as_float_array")
    return comp


def save_quaternion_block_comparison(
    A_left: np.ndarray,
    A_right: np.ndarray,
    block: int = 5,
    labels: Tuple[str, str] = ("X_star", "X(mu)"),
    title: str = "Top-left block comparison (quaternion components)",
    save_path: Optional[str] = None,
    annotate: bool = True,
) -> None:
    """Save a side-by-side comparison of the top-left (block×block) entries.

    Produces a 4×2 grid (components Re/i/j/k × {left,right}). Each subplot is a heatmap
    of the selected component; optionally annotates each entry.

    Parameters
    ----------
    A_left, A_right:
        Dense quaternion matrices.
    block:
        Size of the top-left block to visualize.
    labels:
        Labels for left/right matrices.
    title:
        Figure title.
    save_path:
        If provided, saves to this path (PNG recommended).
    annotate:
        If True, write numeric values in each cell (recommended for block <= 6).
    """
    compL = _as_float_components(A_left)
    compR = _as_float_components(A_right)

    b = int(block)
    b = max(1, b)
    b = min(b, compL.shape[0], compL.shape[1], compR.shape[0], compR.shape[1])

    # Keep annotation readable.
    if b > 7:
        annotate = False

    names = ["Re", "i", "j", "k"]
    fig, axes = plt.subplots(4, 2, figsize=(9, 11))
    fig.suptitle(title, fontsize=12)

    for ci in range(4):
        for col, (comp, lab) in enumerate([(compL, labels[0]), (compR, labels[1])]):
            ax = axes[ci, col]
            data = comp[:b, :b, ci]
            im = ax.imshow(data, aspect="equal", origin="upper")
            ax.set_title(f"{lab} — {names[ci]}", fontsize=10)
            ax.set_xticks(range(b))
            ax.set_yticks(range(b))
            ax.tick_params(axis="both", labelsize=8)

            if annotate:
                for i in range(b):
                    for j in range(b):
                        val = float(data[i, j])
                        ax.text(j, i, f"{val:+.2e}", ha="center", va="center", fontsize=6)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def save_quaternion_abs_comparison(
    A_left: np.ndarray,
    A_right: np.ndarray,
    labels: Tuple[str, str] = ("X_star", "X(mu)"),
    title: str = "Quaternion matrix magnitude comparison",
    save_path: Optional[str] = None,
) -> None:
    """Save a 1×3 figure: |A_left|, |A_right|, and |A_left-A_right|."""
    cL = _as_float_components(A_left)
    cR = _as_float_components(A_right)

    absL = np.sqrt(np.sum(cL**2, axis=-1))
    absR = np.sqrt(np.sum(cR**2, axis=-1))
    absD = np.sqrt(np.sum((cL - cR) ** 2, axis=-1))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(title, fontsize=12)

    ims = []
    ims.append(axes[0].imshow(absL, aspect="auto"))
    axes[0].set_title(f"|{labels[0]}|")
    ims.append(axes[1].imshow(absR, aspect="auto"))
    axes[1].set_title(f"|{labels[1]}|")
    ims.append(axes[2].imshow(absD, aspect="auto"))
    axes[2].set_title(f"|{labels[0]} - {labels[1]}|")

    for ax in axes:
        ax.set_xlabel("col")
        ax.set_ylabel("row")

    for ax, im in zip(axes, ims):
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path:
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()
