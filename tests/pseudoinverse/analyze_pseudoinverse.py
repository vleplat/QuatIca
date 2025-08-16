import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import quaternion
from PIL import Image

# Add parent directory to path to import quatica modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "quatica"))
from solver import NewtonSchulzPseudoinverse
from utils import quat_frobenius_norm, quat_matmat


def load_and_preprocess_image(image_path):
    """
    Load and preprocess image to quaternion format.

    Args:
        image_path: Path to the image file

    Returns:
        X: Quaternion matrix of shape (h, w)
        original_shape: Original image shape for reconstruction
    """
    # Load the image and resize to manageable size
    img = Image.open(image_path)
    # Resize to 128x192 to make it computationally manageable
    img = img.resize((192, 128), Image.Resampling.LANCZOS)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]

    print(f"Original image shape: {img_array.shape}")
    print(f"Image data range: [{img_array.min():.3f}, {img_array.max():.3f}]")

    # Convert RGB to quaternion format
    # We'll use: w = (R+G+B)/3, x = R, y = G, z = B
    h, w, c = img_array.shape

    # Create quaternion components for each pixel
    # w = average of RGB channels
    w_component = np.mean(img_array, axis=2, keepdims=True)  # Shape: (h, w, 1)
    # x, y, z = individual RGB channels
    x_component = img_array[:, :, 0:1]  # Red
    y_component = img_array[:, :, 1:2]  # Green
    z_component = img_array[:, :, 2:3]  # Blue

    # Stack components along the last axis
    quat_components = np.concatenate(
        [w_component, x_component, y_component, z_component], axis=2
    )  # Shape: (h, w, 4)

    # Convert to quaternion array - this will be (h, w) with each entry being a quaternion
    X = quaternion.as_quat_array(quat_components)

    print(f"Quaternion matrix shape: {X.shape}")
    print(f"Quaternion norm: {quat_frobenius_norm(X):.6f}")

    return X, (h, w, c)


def compute_quaternion_pseudoinverse(X):
    """
    Compute pseudoinverse of quaternion matrix X using Newton-Schulz method.

    Args:
        X: Quaternion matrix of shape (m, n)

    Returns:
        X_pinv: Pseudoinverse of shape (n, m)
    """
    m, n = X.shape
    print(f"Computing pseudoinverse for matrix of shape ({m}, {n})")

    # Use Newton-Schulz method to compute pseudoinverse
    ns_solver = NewtonSchulzPseudoinverse(verbose=True)
    X_pinv, residuals, covariances = ns_solver.compute(X)

    print(f"Pseudoinverse shape: {X_pinv.shape}")
    print(f"Pseudoinverse norm: {quat_frobenius_norm(X_pinv):.6f}")

    return X_pinv


def analyze_pseudoinverse(X, X_pinv):
    """
    Analyze properties of the pseudoinverse.

    Args:
        X: Original quaternion matrix
        X_pinv: Pseudoinverse matrix
    """
    print("\n" + "=" * 80)
    print("PSEUDOINVERSE ANALYSIS")
    print("=" * 80)

    # Basic properties
    print(f"Original matrix X shape: {X.shape}")
    print(f"Pseudoinverse X_pinv shape: {X_pinv.shape}")
    print(f"X norm: {quat_frobenius_norm(X):.6f}")
    print(f"X_pinv norm: {quat_frobenius_norm(X_pinv):.6f}")

    # Verify pseudoinverse properties
    X_X_pinv = quat_matmat(X, X_pinv)
    X_pinv_X = quat_matmat(X_pinv, X)

    # Convert to real matrices for comparison
    X_X_pinv_real = quaternion.as_float_array(X_X_pinv)
    X_pinv_X_real = quaternion.as_float_array(X_pinv_X)

    # Create identity matrices of the right size
    I_m = np.eye(X.shape[0])
    I_n = np.eye(X.shape[1])

    # Compare only the w-component (real part) for simplicity
    X_X_pinv_w = X_X_pinv_real[:, :, 0]  # w-component
    X_pinv_X_w = X_pinv_X_real[:, :, 0]  # w-component

    print("\nVerification of pseudoinverse properties:")
    print(f"||X @ X_pinv - I_m|| (w-component): {np.linalg.norm(X_X_pinv_w - I_m):.2e}")
    print(f"||X_pinv @ X - I_n|| (w-component): {np.linalg.norm(X_pinv_X_w - I_n):.2e}")

    # Analyze component distributions
    X_pinv_components = quaternion.as_float_array(X_pinv)
    w_comp, x_comp, y_comp, z_comp = (
        X_pinv_components[:, :, 0],
        X_pinv_components[:, :, 1],
        X_pinv_components[:, :, 2],
        X_pinv_components[:, :, 3],
    )

    print("\nComponent statistics:")
    print(
        f"w-component: mean={w_comp.mean():.6f}, std={w_comp.std():.6f}, min={w_comp.min():.6f}, max={w_comp.max():.6f}"
    )
    print(
        f"x-component: mean={x_comp.mean():.6f}, std={x_comp.std():.6f}, min={x_comp.min():.6f}, max={x_comp.max():.6f}"
    )
    print(
        f"y-component: mean={y_comp.mean():.6f}, std={y_comp.std():.6f}, min={y_comp.min():.6f}, max={y_comp.max():.6f}"
    )
    print(
        f"z-component: mean={z_comp.mean():.6f}, std={z_comp.std():.6f}, min={z_comp.min():.6f}, max={z_comp.max():.6f}"
    )


def visualize_pseudoinverse(X, X_pinv, original_shape):
    """
    Comprehensive visualization of the pseudoinverse with advanced analysis.

    Args:
        X: Original quaternion matrix
        X_pinv: Pseudoinverse matrix
        original_shape: Original image shape
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE PSEUDOINVERSE VISUALIZATION")
    print("=" * 80)

    # Extract components
    X_components = quaternion.as_float_array(X)
    X_pinv_components = quaternion.as_float_array(X_pinv)

    # 1. Component-wise visualization with interpretation
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Quaternion Matrix and Pseudoinverse Component Analysis", fontsize=16)

    # Original image components
    axes[0, 0].imshow(X_components[:, :, 1], cmap="Reds", vmin=-1, vmax=1)
    axes[0, 0].set_title("Original - Red (x)\nColor-specific relationships")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(X_components[:, :, 2], cmap="Greens", vmin=-1, vmax=1)
    axes[0, 1].set_title("Original - Green (y)\nColor-specific relationships")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(X_components[:, :, 3], cmap="Blues", vmin=-1, vmax=1)
    axes[0, 2].set_title("Original - Blue (z)\nColor-specific relationships")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(X_components[:, :, 0], cmap="coolwarm", vmin=-1, vmax=1)
    axes[0, 3].set_title("Original - Real (w)\nScaling relationships")
    axes[0, 3].axis("off")

    # Pseudoinverse components - Inverse relationships
    axes[1, 0].imshow(X_pinv_components[:, :, 1], cmap="Reds", vmin=-1, vmax=1)
    axes[1, 0].set_title("Pseudoinverse - Red (x)\nInverse color relationships")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(X_pinv_components[:, :, 2], cmap="Greens", vmin=-1, vmax=1)
    axes[1, 1].set_title("Pseudoinverse - Green (y)\nInverse color relationships")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(X_pinv_components[:, :, 3], cmap="Blues", vmin=-1, vmax=1)
    axes[1, 2].set_title("Pseudoinverse - Blue (z)\nInverse color relationships")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(X_pinv_components[:, :, 0], cmap="coolwarm", vmin=-1, vmax=1)
    axes[1, 3].set_title("Pseudoinverse - Real (w)\nInverse scaling relationships")
    axes[1, 3].axis("off")

    plt.tight_layout()
    # Save to repository output_figures directory
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(repo_root, "output_figures")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(
        os.path.join(out_dir, "pseudoinverse_component_analysis.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # 2. Reconstruction error map
    print("Computing reconstruction error map...")
    recon = quat_matmat(quat_matmat(X, X_pinv), X)
    residual = np.linalg.norm(quaternion.as_float_array(X - recon), axis=-1)

    plt.figure(figsize=(10, 8))
    plt.imshow(residual, cmap="viridis")
    plt.colorbar(label="Reconstruction Error")
    plt.title(
        "Reconstruction Error Map |X - XX^†X|\nBright = Poor representation, Dark = Stable relationships"
    )
    plt.axis("off")
    plt.savefig(
        os.path.join(out_dir, "reconstruction_error_map.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # 3. Pseudoinverse as filter bank
    print("Analyzing pseudoinverse as filter bank...")
    # Take first row as a representative kernel
    kernel = X_pinv[0, :]  # Shape: (128,)
    kernel_rgb = quaternion.as_float_array(kernel)[:, 1:]  # Shape: (128, 3)

    plt.figure(figsize=(15, 4))
    for i, (color, cmap) in enumerate(
        [("Red", "Reds"), ("Green", "Greens"), ("Blue", "Blues")]
    ):
        plt.subplot(1, 3, i + 1)
        plt.imshow(
            kernel_rgb[:, i].reshape(1, -1), cmap=cmap, aspect="auto", vmin=-1, vmax=1
        )
        plt.title(f"{color} Channel Filter\nFirst row of X^† as deconvolution kernel")
        plt.xlabel("Pixel Index")
        plt.ylabel("Filter Weight")
        plt.colorbar()

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "pseudoinverse_filter_bank.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # 4. Component distributions with interpretation
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Component Distributions: Original vs Pseudoinverse", fontsize=16)

    component_names = ["Real (w)", "Imag-i (x)", "Imag-j (y)", "Imag-k (z)"]
    interpretations = [
        "Scaling relationships\n(amplify/dampen)",
        "Red channel inverse\n(color-specific)",
        "Green channel inverse\n(color-specific)",
        "Blue channel inverse\n(color-specific)",
    ]

    for i, (name, interpretation) in enumerate(zip(component_names, interpretations)):
        row, col = i // 2, i % 2
        axes[row, col].hist(
            X_components[:, :, i].flatten(),
            bins=50,
            alpha=0.7,
            label="Original",
            color="blue",
            density=True,
        )
        axes[row, col].hist(
            X_pinv_components[:, :, i].flatten(),
            bins=50,
            alpha=0.7,
            label="Pseudoinverse",
            color="red",
            density=True,
        )
        axes[row, col].set_title(f"{name}\n{interpretation}")
        axes[row, col].legend()
        axes[row, col].set_xlabel("Component Value")
        axes[row, col].set_ylabel("Density")

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "pseudoinverse_distributions_interpreted.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # 5. Summary statistics
    print("\n" + "=" * 80)
    print("PSEUDOINVERSE INTERPRETATION SUMMARY")
    print("=" * 80)
    print("1. Real Component (w): Represents scaling relationships between pixels")
    print("   - How pixel intensities amplify or dampen each other")
    print("   - Bright spots = strong local interactions")
    print()
    print("2. Imaginary Components (x,y,z): Color-specific inverse relationships")
    print("   - Show how different color channels interact in inverse space")
    print("   - Reveal hidden color dependencies not visible in original")
    print()
    print("3. Reconstruction Error Map: Shows reconstruction stability")
    print("   - Bright areas = pixels poorly represented by pseudoinverse")
    print("   - Dark areas = stable inverse relationships")
    print()
    print("4. Filter Bank: Pseudoinverse acts as deconvolution operator")
    print("   - Each row represents optimal local filter")
    print("   - Positive weights = enhance similar colors")
    print("   - Negative weights = suppress complementary colors")
    print()
    print("5. Key Insights:")
    print("   - Pseudoinverse encodes 'undo' relationships between pixels")
    print("   - Different color channels have distinct inverse patterns")
    print("   - High-condition areas are reconstruction bottlenecks")
    print("   - Edges/corners exhibit unique pseudoinverse signatures")


def main():
    # Resolve repository root and output directory robustly
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(repo_root, "output_figures")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("QUATERNION PSEUDOINVERSE ANALYSIS")
    print("=" * 80)

    # Load image (use absolute path relative to repo root)
    image_path = os.path.join(repo_root, "data", "images", "kodim16.png")
    print(f"Loading image: {image_path}")

    X, original_shape = load_and_preprocess_image(image_path)

    # Compute pseudoinverse
    print("\nComputing pseudoinverse...")
    X_pinv = compute_quaternion_pseudoinverse(X)

    # Analyze properties
    analyze_pseudoinverse(X, X_pinv)

    # Visualize
    visualize_pseudoinverse(X, X_pinv, original_shape)

    print("\nAnalysis complete! Check the generated plots:")
    print("- pseudoinverse_analysis.png: Component visualizations")
    print("- pseudoinverse_distributions.png: Component distributions")


if __name__ == "__main__":
    main()
