import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import quaternion
import os
import sys
import time
from matplotlib.colors import hsv_to_rgb
from sklearn.decomposition import PCA
import seaborn as sns

# Add parent directory to path to import core modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core'))
from utils import quat_frobenius_norm, quat_matmat
from solver import NewtonSchulzPseudoinverse

def load_small_images_from_directory(directory_path, target_size=(32, 32), max_images=20):
    """
    Load multiple small images from a directory and convert to quaternion format.
    
    Args:
        directory_path: Path to directory containing images
        target_size: Target size for all images (h, w)
        max_images: Maximum number of images to load
        
    Returns:
        X: Quaternion matrix where each row is a vectorized image
        image_names: List of image filenames
        original_shapes: List of original image shapes
    """
    print(f"Loading images from: {directory_path}")
    
    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    # Get list of image files
    image_files = []
    for filename in os.listdir(directory_path):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(filename)
    
    # Limit number of images
    image_files = image_files[:max_images]
    print(f"Found {len(image_files)} images: {image_files}")
    
    # Load and process images
    quaternion_rows = []
    image_names = []
    original_shapes = []
    
    for i, filename in enumerate(image_files):
        try:
            image_path = os.path.join(directory_path, filename)
            img = Image.open(image_path)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to target size
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            img_array = np.array(img) / 255.0  # Normalize to [0, 1]
            
            print(f"  {i+1}/{len(image_files)}: {filename} -> {img_array.shape}")
            
            # Convert to quaternion format
            h, w, c = img_array.shape
            
            # Create quaternion components
            # w = average of RGB channels
            w_component = np.mean(img_array, axis=2, keepdims=True)
            # x, y, z = individual RGB channels
            x_component = img_array[:, :, 0:1]  # Red
            y_component = img_array[:, :, 1:2]  # Green  
            z_component = img_array[:, :, 2:3]  # Blue
            
            # Stack components along the last axis
            quat_components = np.concatenate([w_component, x_component, y_component, z_component], axis=2)
            
            # Convert to quaternion array
            X_quat = quaternion.as_quat_array(quat_components)
            
            # Vectorize (flatten) the quaternion matrix
            X_vectorized = X_quat.flatten()  # Shape: (h*w,)
            
            quaternion_rows.append(X_vectorized)
            image_names.append(filename)
            original_shapes.append((h, w, c))
            
        except Exception as e:
            print(f"  Error loading {filename}: {e}")
            continue
    
    if not quaternion_rows:
        raise ValueError("No images were successfully loaded!")
    
    # Stack all vectorized images as rows
    X = np.stack(quaternion_rows, axis=0)  # Shape: (n_images, h*w)
    
    print(f"\nFinal matrix X shape: {X.shape}")
    print(f"Each row represents a {target_size[0]}x{target_size[1]} image vectorized to {target_size[0]*target_size[1]} quaternions")
    
    return X, image_names, original_shapes

def compute_pseudoinverse(X):
    """
    Compute pseudoinverse of quaternion matrix X using Newton-Schulz method.
    
    Args:
        X: Quaternion matrix of shape (n_images, n_features)
        
    Returns:
        X_pinv: Pseudoinverse of shape (n_features, n_images)
    """
    n_images, n_features = X.shape
    print(f"Computing pseudoinverse for matrix of shape ({n_images}, {n_features})")
    
    # Use Newton-Schulz method to compute pseudoinverse
    ns_solver = NewtonSchulzPseudoinverse(verbose=True)
    X_pinv, residuals, covariances = ns_solver.compute(X)
    
    print(f"Pseudoinverse shape: {X_pinv.shape}")
    print(f"Pseudoinverse norm: {quat_frobenius_norm(X_pinv):.6f}")
    
    return X_pinv, residuals, covariances

def visualize_sample_images(X, image_names, image_shape=(32, 32), n_samples=8):
    """Visualize sample images to verify data loading."""
    print("\n" + "="*80)
    print("SAMPLE IMAGES VERIFICATION")
    print("="*80)
    
    n_images = len(image_names)
    n_show = min(n_samples, n_images)
    
    # Calculate grid dimensions
    cols = min(4, n_show)
    rows = (n_show + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    fig.suptitle('Sample Images from Dataset (Verification)', fontsize=16)
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_show):
        row, col = i // cols, i % cols
        ax = axes[row, col]
        
        # Get the quaternion image data
        quat_image = X[i]  # Shape: (n_features,) quaternion array
        
        # Convert to float array and reshape to spatial dimensions
        float_image = quaternion.as_float_array(quat_image)  # Shape: (n_features, 4)
        spatial_image = float_image.reshape(image_shape[0], image_shape[1], 4)  # Shape: (h, w, 4)
        
        # Extract RGB components (x, y, z components)
        rgb_image = spatial_image[:, :, 1:]  # Shape: (h, w, 3)
        
        # Normalize to [0, 1] for display
        rgb_normalized = np.clip(rgb_image, 0, 1)
        
        # Plot
        ax.imshow(rgb_normalized)
        ax.set_title(f'{image_names[i].split(".")[0]}\nImage {i}')
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_show, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"../../output_figures/multiple_images_sample_verification_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Sample verification figure saved to: {filename}")
    plt.show()
    
    # Print some statistics
    print(f"\nData Loading Statistics:")
    print(f"  Total images loaded: {n_images}")
    print(f"  Image shape: {X.shape}")
    print(f"  Quaternion components per image: {X.shape[1]}")
    print(f"  Spatial dimensions: {image_shape[0]}x{image_shape[1]} pixels")
    print(f"  Color channels: RGB (from quaternion x,y,z components)")

def visualize_image_average_filters(X_pinv, image_names, image_shape=(32, 32)):
    """Visualize image-average reconstruction filters from pseudoinverse columns."""
    print("\n" + "="*80)
    print("IMAGE-AVERAGE RECONSTRUCTION FILTERS")
    print("="*80)
    
    n_images = len(image_names)
    n_show = min(8, n_images)  # Show first 8 images
    
    # Calculate grid dimensions
    cols = 4  # Show 4 components per image
    rows = n_show
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    fig.suptitle('Image-Average Reconstruction Filters (Pseudoinverse Columns)', fontsize=16)
    
    # Collect all data for consistent colorbar scaling
    all_data = []
    for i in range(n_show):
        col = X_pinv[:, i]  # Get column for this image
        col_q = quaternion.as_float_array(col)
        col_img = col_q.reshape(image_shape[0], image_shape[1], 4)
        all_data.append(col_img)
    
    # Find global min/max for consistent colorbar
    all_data_array = np.array(all_data)
    global_min = np.min(all_data_array)
    global_max = np.max(all_data_array)
    
    for i in range(n_show):
        print(f"Computing filter for image {image_names[i].split('.')[0]} ({i+1}/{n_show})")
        
        # Get column for this image: shape (n_features,)
        col = X_pinv[:, i]
        
        # Turn quaternion array into float array (n_features×4)
        col_q = quaternion.as_float_array(col)
        
        # Reshape back to spatial form
        col_img = col_q.reshape(image_shape[0], image_shape[1], 4)
        
        comps = ['Real','Red','Green','Blue']
        for c in range(4):
            ax = axes[i, c]
            im = ax.imshow(col_img[:, :, c], cmap='seismic', 
                          vmin=global_min, vmax=global_max)
            ax.set_title(f"{image_names[i].split('.')[0]}\n{comps[c]}")
            ax.axis('off')
    
    # Add single shared colorbar
    cbar = fig.colorbar(axes[0, 0].images[0], ax=axes, orientation='horizontal', 
                        fraction=0.05, pad=0.1)
    cbar.set_label('Filter Weight Value')
    
    plt.tight_layout()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"../../output_figures/multiple_images_average_filters_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Average filters figure saved to: {filename}")
    plt.show()
    
    # Print statistics for each image
    print("\n" + "="*80)
    print("IMAGE-AVERAGE FILTER STATISTICS")
    print("="*80)
    
    for i in range(n_show):
        col = X_pinv[:, i]
        col_q = quaternion.as_float_array(col)
        
        print(f"\n{image_names[i].split('.')[0]} (image {i}):")
        for c, comp_name in enumerate(['Real', 'Red', 'Green', 'Blue']):
            comp_data = col_q[:, c]
            print(f"  {comp_name}: mean={comp_data.mean():.6f}, std={comp_data.std():.6f}, "
                  f"min={comp_data.min():.6f}, max={comp_data.max():.6f}")

def visualize_pca_analysis(X_pinv, image_names):
    """Perform PCA on pseudoinverse columns to reveal clustering."""
    print("\n" + "="*80)
    print("PCA ANALYSIS OF PSEUDOINVERSE COLUMNS")
    print("="*80)
    
    # Convert X_pinv to real array: (F, N, 4) → (F*4, N) → (N, F*4)
    P = quaternion.as_float_array(X_pinv).reshape(-1, X_pinv.shape[1]).T  # shape (N, F*4)
    
    print(f"PCA input shape: {P.shape}")
    print(f"Computing PCA on {P.shape[1]} features for {P.shape[0]} samples...")
    
    # Perform PCA with automatic component selection
    pca = PCA(n_components=0.95)  # Automatically select components for 95% variance
    Z = pca.fit_transform(P)
    
    print(f"Selected {Z.shape[1]} components for 95% variance")
    
    # Plot explained variance
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    plt.plot(range(1, len(explained_var)+1), explained_var, 'o-', label='Individual')
    plt.plot(range(1, len(cumulative_var)+1), cumulative_var, 's-', label='Cumulative')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.legend()
    plt.grid(True)
    
    # Plot first two components
    plt.subplot(1, 2, 2)
    plt.scatter(Z[:, 0], Z[:, 1], c='blue', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add image names as annotations if not too many
    if len(image_names) <= 15:
        for i, name in enumerate(image_names):
            plt.annotate(name.split('.')[0], (Z[i, 0], Z[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
    plt.title('PCA of Pseudoinverse Columns')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"../../output_figures/multiple_images_pca_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"PCA analysis figure saved to: {filename}")
    plt.show()
    
    # Print PCA insights
    print(f"\nPCA Results:")
    print(f"  Components for 95% variance: {Z.shape[1]}")
    print(f"  Total variance explained by first 2 components: {cumulative_var[1]:.1%}")
    if len(cumulative_var) > 4:
        print(f"  Total variance explained by first 5 components: {cumulative_var[4]:.1%}")
    else:
        print(f"  Total variance explained by all {len(cumulative_var)} components: {cumulative_var[-1]:.1%}")
    print(f"  Number of components for 90% variance: {np.argmax(cumulative_var >= 0.90) + 1}")
    print(f"  Number of components for 99% variance: {np.argmax(cumulative_var >= 0.99) + 1}")

def visualize_advanced_insights(X, X_pinv, image_names, image_shape=(32, 32)):
    """Visualize advanced insights from quaternion pseudoinverse"""
    print("\n" + "="*80)
    print("ADVANCED PSEUDOINVERSE INSIGHTS VISUALIZATION")
    print("="*80)
    
    n_images = len(image_names)
    
    # 1. Spectral Analysis (Singular Value Decomposition)
    print("1. Computing spectral analysis...")
    
    X_real = quaternion.as_float_array(X).reshape(-1, 4)
    U, s, Vt = np.linalg.svd(X_real, full_matrices=False)
    
    plt.figure(figsize=(12, 8))
    
    # Plot singular values
    plt.subplot(2, 2, 1)
    plt.plot(s / s.max(), 'o-', markersize=3)
    plt.yscale('log')
    plt.title('Singular Value Spectrum')
    plt.xlabel('Component Index')
    plt.ylabel('Normalized Singular Value (log scale)')
    plt.grid(True)
    
    # Plot cumulative energy
    plt.subplot(2, 2, 2)
    cumulative_energy = np.cumsum(s**2) / np.sum(s**2)
    plt.plot(cumulative_energy, 'o-', markersize=3)
    plt.title('Cumulative Energy')
    plt.xlabel('Component Index')
    plt.ylabel('Cumulative Energy Fraction')
    plt.grid(True)
    
    # Plot first few singular vectors
    plt.subplot(2, 2, 3)
    for i in range(min(5, len(s))):
        plt.plot(Vt[i, :100], label=f'SV {i+1}')
    plt.title('First 5 Singular Vectors (first 100 components)')
    plt.xlabel('Component Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Plot singular value distribution
    plt.subplot(2, 2, 4)
    plt.hist(np.log10(s/s.max()), bins=50, alpha=0.7)
    plt.title('Distribution of Log Singular Values')
    plt.xlabel('Log10(Normalized Singular Value)')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.tight_layout()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"../../output_figures/multiple_images_spectral_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Spectral analysis figure saved to: {filename}")
    plt.show()
    
    # 2. Pseudoinverse as Image Manifold
    print("2. Computing pseudoinverse manifold...")
    
    manifold = quat_matmat(X_pinv, X)  # Should be ≈ identity for perfect reconstruction
    
    # Extract phase and magnitude information
    comp = quaternion.as_float_array(manifold)
    magnitude = np.linalg.norm(comp, axis=-1)
    phase = np.arctan2(comp[..., 1], comp[..., 0])  # Red vs Real
    
    # Create HSV representation: Hue=phase, Saturation=1, Value=magnitude
    h = (phase + np.pi) / (2 * np.pi)  # Normalize to [0,1]
    s = np.ones_like(h)
    v = magnitude / magnitude.max()
    hsv = np.stack([h, s, v], axis=-1)
    rgb = hsv_to_rgb(hsv)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(rgb)
    plt.title('Pseudoinverse Manifold (Phase and Magnitude)\nHue=Phase, Value=Magnitude')
    plt.axis('off')
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"../../output_figures/multiple_images_manifold_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Manifold figure saved to: {filename}")
    plt.show()
    
    # 3. Color Channel Correlations
    print("3. Computing color channel correlations...")
    
    comp_pinv = quaternion.as_float_array(X_pinv)
    corr_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            corr_matrix[i, j] = np.corrcoef(comp_pinv[..., i].flatten(), 
                                            comp_pinv[..., j].flatten())[0, 1]
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.xticks(range(4), ['Real', 'Red', 'Green', 'Blue'])
    plt.yticks(range(4), ['Real', 'Red', 'Green', 'Blue'])
    plt.title('Color Channel Correlations in Pseudoinverse')
    
    # Add correlation values as text
    for i in range(4):
        for j in range(4):
            plt.text(j, i, f'{corr_matrix[i, j]:.3f}', 
                    ha='center', va='center', fontsize=10,
                    color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"../../output_figures/multiple_images_channel_correlations_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Channel correlations figure saved to: {filename}")
    plt.show()
    
    # 4. Image-to-Image Relationships
    print("4. Computing image-to-image relationships...")
    
    X_X_pinv = quat_matmat(X, X_pinv)
    X_X_pinv_real = quaternion.as_float_array(X_X_pinv)[:, :, 0]  # w-component
    
    plt.figure(figsize=(12, 10))
    plt.imshow(X_X_pinv_real, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Relationship Strength')
    plt.title(f'Image-to-Image Relationships (X @ X^†)\nDiagonal ≈ 1, Off-diagonal ≈ 0 for independence')
    plt.xlabel('Image Index')
    plt.ylabel('Image Index')
    
    # Add image names as tick labels if not too many
    if n_images <= 15:
        plt.xticks(range(n_images), [name.split('.')[0] for name in image_names], rotation=45, ha='right')
        plt.yticks(range(n_images), [name.split('.')[0] for name in image_names])
    
    plt.tight_layout()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"../../output_figures/multiple_images_relationships_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Image relationships figure saved to: {filename}")
    plt.show()
    
    # 5. Summary insights
    print("\n" + "="*80)
    print("ADVANCED PSEUDOINVERSE INSIGHTS")
    print("="*80)
    print("1. Spectral Analysis:")
    print(f"   - Effective rank: {np.sum(s > s.max() * 0.01)} components")
    print(f"   - 90% energy in first {np.argmax(cumulative_energy > 0.9)} components")
    print("   - Rapid decay indicates high image redundancy")
    print("   - Singular vectors show principal image patterns")
    print()
    print("2. Pseudoinverse Manifold:")
    print("   - Phase patterns reveal rotational symmetries")
    print("   - Magnitude shows important anchor pixels")
    print("   - Color relationships indicate opponent processing")
    print()
    print("3. Color Channel Correlations:")
    print("   - Strong correlations indicate shared patterns")
    print("   - Negative correlations show opponent processes")
    print("   - Real component captures luminance information")
    print()
    print("4. Image-to-Image Relationships:")
    print("   - Diagonal elements ≈ 1: Each image can reconstruct itself")
    print("   - Off-diagonal elements: How much images influence each other")
    print("   - Low off-diagonal values = independent images")
    print("   - High off-diagonal values = similar images")

def analyze_multiple_images_pseudoinverse(X, X_pinv, image_names, original_shapes):
    """
    Analyze properties of the pseudoinverse for multiple images.
    
    Args:
        X: Original quaternion matrix (n_images, n_features)
        X_pinv: Pseudoinverse matrix (n_features, n_images)
        image_names: List of image filenames
        original_shapes: List of original image shapes
    """
    print("\n" + "="*80)
    print("MULTIPLE IMAGES PSEUDOINVERSE ANALYSIS")
    print("="*80)
    
    n_images, n_features = X.shape
    
    # Basic properties
    print(f"Number of images: {n_images}")
    print(f"Features per image: {n_features}")
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
    I_m = np.eye(n_images)
    I_n = np.eye(n_features)
    
    # Compare only the w-component (real part) for simplicity
    X_X_pinv_w = X_X_pinv_real[:, :, 0]  # w-component
    X_pinv_X_w = X_pinv_X_real[:, :, 0]  # w-component
    
    print(f"\nVerification of pseudoinverse properties:")
    print(f"||X @ X_pinv - I_m|| (w-component): {np.linalg.norm(X_X_pinv_w - I_m):.2e}")
    print(f"||X_pinv @ X - I_n|| (w-component): {np.linalg.norm(X_pinv_X_w - I_n):.2e}")
    
    # Analyze component distributions
    X_pinv_components = quaternion.as_float_array(X_pinv)
    w_comp, x_comp, y_comp, z_comp = X_pinv_components[:, :, 0], X_pinv_components[:, :, 1], X_pinv_components[:, :, 2], X_pinv_components[:, :, 3]
    
    print(f"\nComponent statistics:")
    print(f"w-component: mean={w_comp.mean():.6f}, std={w_comp.std():.6f}, min={w_comp.min():.6f}, max={w_comp.max():.6f}")
    print(f"x-component: mean={x_comp.mean():.6f}, std={x_comp.std():.6f}, min={x_comp.min():.6f}, max={x_comp.max():.6f}")
    print(f"y-component: mean={y_comp.mean():.6f}, std={y_comp.std():.6f}, min={y_comp.min():.6f}, max={y_comp.max():.6f}")
    print(f"z-component: mean={z_comp.mean():.6f}, std={z_comp.std():.6f}, min={z_comp.min():.6f}, max={z_comp.max():.6f}")

def main():
    print("="*80)
    print("MULTIPLE IMAGES QUATERNION PSEUDOINVERSE ANALYSIS")
    print("="*80)
    
    # Load multiple small images
    image_directory = "../../data/images"
    target_size = (32, 32)  # Small size for better analysis
    max_images = 15  # Reasonable number for visualization
    
    try:
        X, image_names, original_shapes = load_small_images_from_directory(
            image_directory, target_size, max_images
        )
    except Exception as e:
        print(f"Error loading images: {e}")
        print("Creating synthetic test data instead...")
        
        # Create synthetic test data if no images found
        n_images = 10
        h, w = target_size
        n_features = h * w
        
        # Create synthetic quaternion images
        X = np.zeros((n_images, n_features), dtype=np.quaternion)
        image_names = [f"synthetic_{i:02d}" for i in range(n_images)]
        original_shapes = [(h, w, 3)] * n_images
        
        for i in range(n_images):
            # Create random RGB data
            rgb_data = np.random.rand(h, w, 3)
            
            # Convert to quaternion
            w_comp = np.mean(rgb_data, axis=2, keepdims=True)
            x_comp = rgb_data[:, :, 0:1]
            y_comp = rgb_data[:, :, 1:2]
            z_comp = rgb_data[:, :, 2:3]
            
            quat_components = np.concatenate([w_comp, x_comp, y_comp, z_comp], axis=2)
            X_quat = quaternion.as_quat_array(quat_components)
            X[i] = X_quat.flatten()
        
        print(f"Created synthetic data: {X.shape}")
    
    # Visualize sample images
    visualize_sample_images(X, image_names, target_size)
    
    # Compute pseudoinverse
    print("\nComputing pseudoinverse...")
    X_pinv, residuals, covariances = compute_pseudoinverse(X)
    
    # Analyze pseudoinverse
    analyze_multiple_images_pseudoinverse(X, X_pinv, image_names, original_shapes)
    
    # Advanced visualizations
    visualize_image_average_filters(X_pinv, image_names, target_size)
    visualize_pca_analysis(X_pinv, image_names)
    visualize_advanced_insights(X, X_pinv, image_names, target_size)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("Generated plots:")
    print("- multiple_images_sample_verification_*.png: Sample images verification")
    print("- multiple_images_average_filters_*.png: Image-average reconstruction filters")
    print("- multiple_images_pca_analysis_*.png: PCA analysis of pseudoinverse columns")
    print("- multiple_images_spectral_analysis_*.png: Spectral analysis with SVD")
    print("- multiple_images_manifold_*.png: Pseudoinverse manifold visualization")
    print("- multiple_images_channel_correlations_*.png: Color channel correlations")
    print("- multiple_images_relationships_*.png: Image-to-image relationships")

if __name__ == "__main__":
    main() 