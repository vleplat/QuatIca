from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import quaternion


class Visualizer:
    @staticmethod
    def plot_residuals(
        residuals: dict[str, list[float]],
        title: str = "Residual Norms",
        subtitle: str = "",
    ) -> None:
        """
        Plot Moore-Penrose residual norms over iterations.

        Creates a logarithmic plot showing the convergence behavior of different
        residual types during pseudoinverse computation.

        Parameters:
        -----------
        residuals : dict[str, list[float]]
            Dictionary mapping residual names to their values over iterations
        title : str, optional
            Main plot title (default: "Residual Norms")
        subtitle : str, optional
            Additional subtitle text (default: "")

        Notes:
        ------
        Uses semilogy scale to better visualize exponential convergence patterns
        typical in iterative pseudoinverse algorithms.
        """
        plt.figure(figsize=(6, 4))
        for key, vals in residuals.items():
            plt.semilogy(vals, label=key)
        plt.xlabel("Iteration")
        plt.ylabel("Residual norm")
        plt.legend()
        plt.title(title)
        if subtitle:
            plt.suptitle(subtitle, fontsize=10, y=0.98)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout(rect=[0, 0, 1, 0.95] if subtitle else None)
        plt.show()

    @staticmethod
    def plot_covariances(
        covariances: list[float], title: str = "Covariance Deviation", subtitle: str = ""
    ) -> None:
        """
        Plot covariance deviation ||AX - I|| or ||XA - I|| over iterations.

        Visualizes how well the computed pseudoinverse satisfies the covariance
        conditions during Newton-Schulz iterations.

        Parameters:
        -----------
        covariances : list[float]
            List of covariance deviation values over iterations
        title : str, optional
            Main plot title (default: "Covariance Deviation")
        subtitle : str, optional
            Additional subtitle text (default: "")

        Notes:
        ------
        Uses logarithmic scale to track convergence. The covariance deviation
        measures how close XA (or AX) is to the identity matrix.
        """
        plt.figure(figsize=(6, 4))
        plt.semilogy(covariances, marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("Covariance deviation")
        plt.title(title)
        if subtitle:
            plt.suptitle(subtitle, fontsize=10, y=0.98)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout(rect=[0, 0, 1, 0.95] if subtitle else None)
        plt.show()

    @staticmethod
    def visualize_matrix(
        A: np.ndarray,
        component: int = 0,
        cmap: str = "viridis",
        title: str = "Matrix Component Heatmap",
        subtitle: str = "",
    ) -> None:
        """
        Heatmap of a chosen quaternion component (0=w, 1=x, 2=y, 3=z).

        Displays a specific quaternion component as a 2D heatmap to visualize
        the structure and patterns within the matrix.

        Parameters:
        -----------
        A : np.ndarray or SparseQuaternionMatrix
            Input quaternion matrix
        component : int, optional
            Quaternion component to visualize: 0=w, 1=x, 2=y, 3=z (default: 0)
        cmap : str, optional
            Matplotlib colormap name (default: 'viridis')
        title : str, optional
            Main plot title (default: "Matrix Component Heatmap")
        subtitle : str, optional
            Additional subtitle text (default: "")

        Notes:
        ------
        Automatically handles both dense and sparse quaternion matrices by
        converting sparse matrices to dense format for visualization.
        """
        from utils import SparseQuaternionMatrix

        if isinstance(A, SparseQuaternionMatrix):
            A = quaternion.as_quat_array(
                np.stack(
                    [A.real.toarray(), A.i.toarray(), A.j.toarray(), A.k.toarray()],
                    axis=-1,
                )
            )
        comp = quaternion.as_float_array(A)[..., component]
        plt.figure(figsize=(5, 5))
        plt.imshow(comp, cmap=cmap)
        plt.colorbar()
        plt.title(title)
        if subtitle:
            plt.suptitle(subtitle + f" (component {component})", fontsize=10, y=0.98)
        else:
            plt.suptitle(f"Component {component} heatmap", fontsize=10, y=0.98)
        plt.grid(False)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    @staticmethod
    def visualize_matrix_abs(
        A: np.ndarray,
        cmap: str = "viridis",
        title: str = "Matrix Absolute Value",
        subtitle: str = "",
    ) -> None:
        """
        Heatmap of quaternion matrix absolute values |q| = sqrt(w² + x² + y² + z²).

        Displays the magnitude of each quaternion entry as a 2D heatmap,
        providing insight into the overall structure and numerical behavior.

        Parameters:
        -----------
        A : np.ndarray or SparseQuaternionMatrix
            Input quaternion matrix
        cmap : str, optional
            Matplotlib colormap name (default: 'viridis')
        title : str, optional
            Main plot title (default: "Matrix Absolute Value")
        subtitle : str, optional
            Additional subtitle text (default: "")

        Notes:
        ------
        The absolute value (magnitude) is computed as the quaternion norm,
        which is invariant under quaternion rotations and provides a
        scalar measure of quaternion "size".
        """
        from utils import SparseQuaternionMatrix

        if isinstance(A, SparseQuaternionMatrix):
            A = quaternion.as_quat_array(
                np.stack(
                    [A.real.toarray(), A.i.toarray(), A.j.toarray(), A.k.toarray()],
                    axis=-1,
                )
            )

        # Compute absolute values
        comp = quaternion.as_float_array(A)
        abs_vals = np.sqrt(np.sum(comp**2, axis=-1))

        plt.figure(figsize=(6, 5))
        im = plt.imshow(abs_vals, cmap=cmap, aspect="auto")
        plt.colorbar(im, label="|q|")
        plt.title(title)
        if subtitle:
            plt.suptitle(subtitle, fontsize=10, y=0.98)
        plt.xlabel("Column index")
        plt.ylabel("Row index")
        plt.grid(False)
        plt.tight_layout(rect=[0, 0, 1, 0.95] if subtitle else None)
        plt.show()

    @staticmethod
    def visualize_tensor_slice(
        T: np.ndarray,
        mode: int = 0,
        slice_idx: int = 0,
        cmap: str = "viridis",
        title: str = "Tensor Slice",
        show_abs: bool = True,
    ) -> None:
        """Visualize a 2D slice of a quaternion tensor T(I×J×K)."""
        if len(T.shape) != 3:
            raise ValueError("Tensor must be 3D (I×J×K)")

        if mode == 0:  # Fix first index
            slice_data = T[slice_idx, :, :]
            xlabel, ylabel = "K (3rd mode)", "J (2nd mode)"
        elif mode == 1:  # Fix second index
            slice_data = T[:, slice_idx, :]
            xlabel, ylabel = "K (3rd mode)", "I (1st mode)"
        elif mode == 2:  # Fix third index
            slice_data = T[:, :, slice_idx]
            xlabel, ylabel = "J (2nd mode)", "I (1st mode)"
        else:
            raise ValueError("Mode must be 0, 1, or 2")

        if show_abs:
            # Show absolute values of quaternions
            comp = quaternion.as_float_array(slice_data)
            plot_data = np.sqrt(np.sum(comp**2, axis=-1))
            label = "|T|"
        else:
            # Show real component only
            comp = quaternion.as_float_array(slice_data)
            plot_data = comp[..., 0]
            label = "Real(T)"

        plt.figure(figsize=(6, 5))
        im = plt.imshow(plot_data, cmap=cmap, aspect="auto", origin="lower")
        plt.colorbar(im, label=label)
        plt.title(f"{title} - Mode {mode}, Slice {slice_idx}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(False)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_schur_structure(
        T: np.ndarray,
        title: str = "Schur Form Structure",
        subtitle: str = "",
        threshold: float = 1e-12,
    ) -> Tuple[float, float]:
        """
        Visualize the structure of a Schur form matrix T and compute structure metrics.

        Returns:
            Tuple of (below_diagonal_max, subdiagonal_max) for quantitative analysis
        """
        n = T.shape[0]

        # Compute absolute values
        comp = quaternion.as_float_array(T)
        abs_vals = np.sqrt(np.sum(comp**2, axis=-1))

        # Create structure visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Left plot: Absolute value heatmap
        im1 = ax1.imshow(abs_vals, cmap="viridis", aspect="auto")
        ax1.set_title("|T| - Absolute Values")
        ax1.set_xlabel("Column index")
        ax1.set_ylabel("Row index")
        plt.colorbar(im1, ax=ax1, label="|T_ij|")

        # Right plot: Structure analysis with thresholding
        structure = abs_vals.copy()
        structure[abs_vals < threshold] = 0  # Zero out small elements

        # Color-code different regions
        colored_structure = np.zeros_like(structure)
        for i in range(n):
            for j in range(n):
                if i == j:  # Diagonal
                    colored_structure[i, j] = 3 if structure[i, j] > threshold else 0
                elif i == j + 1:  # Subdiagonal
                    colored_structure[i, j] = 2 if structure[i, j] > threshold else 0
                elif i < j:  # Upper triangular
                    colored_structure[i, j] = 1 if structure[i, j] > threshold else 0
                else:  # Strictly below diagonal
                    colored_structure[i, j] = -1 if structure[i, j] > threshold else 0

        im2 = ax2.imshow(
            colored_structure, cmap="RdYlBu_r", aspect="auto", vmin=-1, vmax=3
        )
        ax2.set_title(f"Structure Analysis (threshold={threshold:.1e})")
        ax2.set_xlabel("Column index")
        ax2.set_ylabel("Row index")

        # Add colorbar with labels
        cbar = plt.colorbar(im2, ax=ax2, ticks=[-1, 0, 1, 2, 3])
        cbar.ax.set_yticklabels(
            ["Below diag", "Zero", "Upper tri", "Subdiag", "Diagonal"]
        )

        if subtitle:
            fig.suptitle(f"{title} - {subtitle}", fontsize=12)
        else:
            fig.suptitle(title, fontsize=12)

        plt.tight_layout()
        plt.show()

        # Compute quantitative metrics
        below_diag_max = 0.0
        for i in range(1, n):
            for j in range(i):
                below_diag_max = max(below_diag_max, abs_vals[i, j])

        subdiag_max = 0.0
        for i in range(1, n):
            subdiag_max = max(subdiag_max, abs_vals[i, i - 1])

        return below_diag_max, subdiag_max

    @staticmethod
    def plot_convergence_comparison(
        data_dict: dict,
        title: str = "Convergence Comparison",
        xlabel: str = "Iteration",
        ylabel: str = "Residual",
        logscale: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot convergence comparison for multiple algorithms/variants."""
        plt.figure(figsize=(8, 6))

        colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]
        markers = ["o", "s", "^", "D", "v", "<", ">", "p"]

        for i, (label, values) in enumerate(data_dict.items()):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            if logscale:
                plt.semilogy(
                    values,
                    label=label,
                    color=color,
                    marker=marker,
                    markersize=4,
                    markevery=max(1, len(values) // 20),
                )
            else:
                plt.plot(
                    values,
                    label=label,
                    color=color,
                    marker=marker,
                    markersize=4,
                    markevery=max(1, len(values) // 20),
                )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, which="both", linestyle="--", alpha=0.7)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {save_path}")

        plt.show()
