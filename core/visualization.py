import matplotlib.pyplot as plt
import quaternion
import numpy as np

class Visualizer:
    @staticmethod
    def plot_residuals(residuals: dict[str, list[float]], title: str = "Residual Norms", subtitle: str = "") -> None:
        """Plot MP residual norms over iterations."""
        plt.figure(figsize=(6,4))
        for key, vals in residuals.items():
            plt.semilogy(vals, label=key)
        plt.xlabel('Iteration')
        plt.ylabel('Residual norm')
        plt.legend()
        plt.title(title)
        if subtitle:
            plt.suptitle(subtitle, fontsize=10, y=0.98)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout(rect=[0, 0, 1, 0.95] if subtitle else None)
        plt.show()

    @staticmethod
    def plot_covariances(covariances: list[float], title: str = "Covariance Deviation", subtitle: str = "") -> None:
        """Plot covariance deviation ||AX - I|| or ||XA - I||."""
        plt.figure(figsize=(6,4))
        plt.semilogy(covariances, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Covariance deviation')
        plt.title(title)
        if subtitle:
            plt.suptitle(subtitle, fontsize=10, y=0.98)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout(rect=[0, 0, 1, 0.95] if subtitle else None)
        plt.show()

    @staticmethod
    def visualize_matrix(A: np.ndarray, component: int = 0, cmap: str = 'viridis', title: str = "Matrix Component Heatmap", subtitle: str = "") -> None:
        """Heatmap of a chosen quaternion component (0=w,1=x,2=y,3=z)."""
        from utils import SparseQuaternionMatrix
        if isinstance(A, SparseQuaternionMatrix):
            A = quaternion.as_quat_array(
                np.stack([
                    A.real.toarray(),
                    A.i.toarray(),
                    A.j.toarray(),
                    A.k.toarray()
                ], axis=-1)
            )
        comp = quaternion.as_float_array(A)[..., component]
        plt.figure(figsize=(5,5))
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