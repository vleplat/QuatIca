import numpy as np
import matplotlib.pyplot as plt
import quaternion
import sys
import os

# Add parent directory to path to import core modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core'))
from utils import quat_frobenius_norm, quat_matmat
from solver import NewtonSchulzPseudoinverse

from torchvision import datasets, transforms
import torch
from matplotlib.colors import hsv_to_rgb
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

def visualize_sample_images(X, labels, class_names, n_samples_per_class=5):
    """Visualize sample images from each class to verify data loading."""
    print("\n" + "="*80)
    print("SAMPLE IMAGES VERIFICATION")
    print("="*80)
    
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, n_samples_per_class, figsize=(2*n_samples_per_class, 2*n_classes))
    fig.suptitle('Sample Images from Each Class (Verification)', fontsize=16)
    
    for cls in range(n_classes):
        # Get indices for this class
        class_indices = np.where(labels == cls)[0]
        
        # Sample n_samples_per_class images
        sample_indices = class_indices[:n_samples_per_class]
        
        for i, idx in enumerate(sample_indices):
            # Get the quaternion image data
            quat_image = X[idx]  # Shape: (1024,) quaternion array
            
            # Convert to float array and reshape to spatial dimensions
            float_image = quaternion.as_float_array(quat_image)  # Shape: (1024, 4)
            spatial_image = float_image.reshape(32, 32, 4)  # Shape: (32, 32, 4)
            
            # Extract RGB components (x, y, z components)
            rgb_image = spatial_image[:, :, 1:]  # Shape: (32, 32, 3)
            
            # Normalize to [0, 1] for display
            rgb_normalized = np.clip(rgb_image, 0, 1)
            
            # Plot
            if n_classes == 1:
                ax = axes[i]
            else:
                ax = axes[cls, i]
            
            ax.imshow(rgb_normalized)
            ax.set_title(f'{class_names[cls]}\nImage {idx}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../../output_figures/sample_images_verification.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print some statistics
    print(f"\nData Loading Statistics:")
    print(f"  Total images loaded: {len(X)}")
    print(f"  Image shape: {X.shape}")
    print(f"  Quaternion components per image: {X.shape[1]}")
    print(f"  Spatial dimensions: 32x32 pixels")
    print(f"  Color channels: RGB (from quaternion x,y,z components)")
    
    for cls in range(n_classes):
        class_indices = np.where(labels == cls)[0]
        print(f"  {class_names[cls]}: {len(class_indices)} images")

def load_cifar10_subset(n_samples_per_class=30, classes_to_use=None):
    """Load a subset of CIFAR-10 images."""
    print("Loading CIFAR-10 subset...")
    
    # CIFAR-10 class names
    cifar10_classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    if classes_to_use is None:
        classes_to_use = [0, 1, 2, 3, 4]  # First 5 classes
    
    class_names = [cifar10_classes[i] for i in classes_to_use]
    print(f"Using classes: {class_names}")
    
    # Load CIFAR-10
    dataset = datasets.CIFAR10(root='../../data', train=True, download=True, transform=transforms.ToTensor())
    
    # Collect images by class
    images_by_class = {i: [] for i in classes_to_use}
    labels_by_class = {i: [] for i in classes_to_use}
    
    for idx, (image, label) in enumerate(dataset):
        if label in classes_to_use and len(images_by_class[label]) < n_samples_per_class:
            images_by_class[label].append(image.numpy())
            labels_by_class[label].append(label)
    
    # Convert to quaternion format
    quaternion_rows = []
    labels = []
    
    for class_idx in classes_to_use:
        class_images = images_by_class[class_idx]
        print(f"  Class {cifar10_classes[class_idx]}: {len(class_images)} images")
        
        for image in class_images:
            # image is (3, 32, 32), convert to (32, 32, 3)
            image = np.transpose(image, (1, 2, 0))
            
            # Convert to quaternion format
            h, w, c = image.shape
            
            # Create quaternion components
            w_component = np.mean(image, axis=2, keepdims=True)
            x_component = image[:, :, 0:1]  # Red
            y_component = image[:, :, 1:2]  # Green  
            z_component = image[:, :, 2:3]  # Blue
            
            quat_components = np.concatenate([w_component, x_component, y_component, z_component], axis=2)
            X_quat = quaternion.as_quat_array(quat_components)
            X_vectorized = X_quat.flatten()
            
            quaternion_rows.append(X_vectorized)
            labels.append(class_idx)
    
    X = np.stack(quaternion_rows, axis=0)
    print(f"\nFinal matrix X shape: {X.shape}")
    return X, np.array(labels), class_names

def compute_pseudoinverse(X):
    """Compute pseudoinverse using Newton-Schulz method."""
    n_images, n_features = X.shape
    print(f"Computing pseudoinverse for matrix of shape ({n_images}, {n_features})")
    
    ns_solver = NewtonSchulzPseudoinverse(gamma=1,verbose=True)
    X_pinv, residuals, covariances = ns_solver.compute(X)
    
    print(f"Pseudoinverse shape: {X_pinv.shape}")
    print(f"Pseudoinverse norm: {quat_frobenius_norm(X_pinv):.6f}")
    
    return X_pinv, residuals, covariances

def visualize_class_average_filters(X_pinv, labels, class_names, image_shape=(32,32)):
    """Visualize class-average reconstruction filters from pseudoinverse columns."""
    print("\n" + "="*80)
    print("CLASS-AVERAGE RECONSTRUCTION FILTERS")
    print("="*80)
    
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, 4, figsize=(16, 4*n_classes))
    fig.suptitle('Class-Average Reconstruction Filters (Pseudoinverse Columns)', fontsize=16)
    
    # Collect all data for consistent colorbar scaling
    all_data = []
    for cls in range(n_classes):
        idxs = np.where(labels == cls)[0]
        avg_col = np.mean(X_pinv[:, idxs], axis=1)
        avg_q = quaternion.as_float_array(avg_col)
        avg_q_img = avg_q.reshape(image_shape[0], image_shape[1], 4)
        all_data.append(avg_q_img)
    
    # Find global min/max for consistent colorbar
    all_data_array = np.array(all_data)
    global_min = np.min(all_data_array)
    global_max = np.max(all_data_array)
    
    for cls in range(n_classes):
        idxs = np.where(labels == cls)[0]
        print(f"Computing average for class {class_names[cls]} ({len(idxs)} images)")
        
        # Average columns for this class: shape (F,)
        avg_col = np.mean(X_pinv[:, idxs], axis=1)
        
        # Turn quaternion array into float array (F×4)
        avg_q = quaternion.as_float_array(avg_col)  # (4096,4)
        
        # Reshape back to spatial form
        avg_q_img = avg_q.reshape(image_shape[0], image_shape[1], 4)
        
        comps = ['Real','Red','Green','Blue']
        for c in range(4):
            ax = axes[cls, c]
            im = ax.imshow(avg_q_img[:, :, c], cmap='seismic', 
                          vmin=global_min, vmax=global_max)
            ax.set_title(f"{class_names[cls]}\n{comps[c]}")
            ax.axis('off')
    
    # Add single shared colorbar
    cbar = fig.colorbar(axes[0, 0].images[0], ax=axes, orientation='horizontal', 
                        fraction=0.05, pad=0.1)
    cbar.set_label('Filter Weight Value')
    
    plt.tight_layout()
    plt.savefig('../../output_figures/class_average_filters.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics for each class
    print("\n" + "="*80)
    print("CLASS-AVERAGE FILTER STATISTICS")
    print("="*80)
    
    for cls in range(n_classes):
        idxs = np.where(labels == cls)[0]
        avg_col = np.mean(X_pinv[:, idxs], axis=1)
        avg_q = quaternion.as_float_array(avg_col)
        
        print(f"\n{class_names[cls]} (average of {len(idxs)} images):")
        for c, comp_name in enumerate(['Real', 'Red', 'Green', 'Blue']):
            comp_data = avg_q[:, c]
            print(f"  {comp_name}: mean={comp_data.mean():.6f}, std={comp_data.std():.6f}, "
                  f"min={comp_data.min():.6f}, max={comp_data.max():.6f}")

def visualize_pca_analysis(X_pinv, labels=None, class_names=None):
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
    if labels is not None and class_names is not None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
        for i, class_name in enumerate(class_names):
            mask = labels == i
            plt.scatter(Z[mask, 0], Z[mask, 1], c=[colors[i]], label=class_name, 
                       s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        plt.legend()
    else:
        plt.scatter(Z[:, 0], Z[:, 1], c='gray', s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
    plt.xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
    plt.title('PCA of Pseudoinverse Columns')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../output_figures/pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print PCA insights
    print(f"\nPCA Results:")
    print(f"  Components for 95% variance: {Z.shape[1]}")
    print(f"  Total variance explained by first 2 components: {cumulative_var[1]:.1%}")
    print(f"  Total variance explained by first 5 components: {cumulative_var[4]:.1%}")
    print(f"  Number of components for 90% variance: {np.argmax(cumulative_var >= 0.90) + 1}")
    print(f"  Number of components for 99% variance: {np.argmax(cumulative_var >= 0.99) + 1}")

def visualize_pseudoinverse_insights(X, X_pinv, labels, image_shape=(32, 32)):
    """Visualize meaningful insights from quaternion pseudoinverse"""
    print("\n" + "="*80)
    print("ADVANCED PSEUDOINVERSE INSIGHTS VISUALIZATION")
    print("="*80)
    
    # 1. Reconstruction Filters (Per-Pixel Inverse Kernels)
    print("1. Analyzing pixel-level reconstruction filters...")
    
    # Choose several interesting pixel positions (adjust for transposed case)
    print(f"X shape: {X.shape}, X_pinv shape: {X_pinv.shape}")
    
    if X.shape[0] == 1024:  # Transposed case: X is (1024, 250)
        pixel_positions = [
            (50, "Image 50 (middle of first class)"),
            (0, "First image (0)"),
            (249, "Last image (249)"),
            (100, "Image 100 (middle of second class)"),
            (200, "Image 200 (middle of fourth class)")
        ]
    else:  # Original case: X is (250, 1024)
        pixel_positions = [
            (256, "Center pixel (8,8)"),
            (0, "Top-left corner (0,0)"),
            (511, "Bottom-right corner (15,31)"),
            (16, "Top edge (0,16)"),
            (496, "Bottom edge (15,16)")
        ]
    
    fig, axes = plt.subplots(len(pixel_positions), 4, figsize=(16, 4*len(pixel_positions)))
    fig.suptitle('Pixel-Level Reconstruction Filters (150 image weights)', fontsize=16)
    
    for idx, (pixel_idx, description) in enumerate(pixel_positions):
        # Get reconstruction filter for this pixel
        filter = X_pinv[pixel_idx, :]  # Filter to reconstruct pixel (150 components)
        
        # Convert to float array (150, 4)
        filter_components = quaternion.as_float_array(filter)  # Shape: (150, 4)
        
        # For visualization, we'll show the filter as a 1D array with 4 components
        # Reshape to show spatial structure if possible, otherwise show as 1D
        if len(filter_components) >= 32*32:
            # If we have enough components, reshape to spatial
            filter_spatial = filter_components[:32*32].reshape(32, 32, 4)
        else:
            # Otherwise, pad or truncate to fit spatial dimensions
            filter_padded = np.zeros((32*32, 4))
            filter_padded[:len(filter_components), :] = filter_components
            filter_spatial = filter_padded.reshape(32, 32, 4)
        
        # Visualize each component as 1D arrays
        components = ['Real', 'Red', 'Green', 'Blue']
        for comp_idx in range(4):
            ax = axes[idx, comp_idx]
            
            # Plot as 1D array
            ax.plot(filter_components[:, comp_idx], 'o-', markersize=2, alpha=0.7)
            ax.set_title(f'{description}\n{components[comp_idx]} Component')
            ax.set_xlabel('Image Index')
            ax.set_ylabel('Weight Value')
            ax.grid(True, alpha=0.3)
            
            # Add horizontal line at zero
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('../../output_figures/pixel_reconstruction_filters.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Spectral Analysis (Singular Value Decomposition)
    print("2. Computing spectral analysis...")
    
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
    plt.savefig('../../output_figures/spectral_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Pseudoinverse as Image Manifold
    print("3. Computing pseudoinverse manifold...")
    
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
    plt.savefig('../../output_figures/pseudoinverse_manifold.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Color Channel Correlations
    print("4. Computing color channel correlations...")
    
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
    
    plt.savefig('../../output_figures/channel_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Class-specific spectral analysis
    print("5. Computing class-specific spectral analysis...")
    
    # Analyze each class separately
    n_classes = 5
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Class-Specific Spectral Analysis', fontsize=16)
    
    for class_idx in range(n_classes):
        row, col = class_idx // 3, class_idx % 3
        
        # Get images from this class
        class_mask = labels == class_idx
        X_class = X[class_mask]
        
        # Compute SVD for this class
        X_class_real = quaternion.as_float_array(X_class).reshape(-1, 4)
        U_class, s_class, Vt_class = np.linalg.svd(X_class_real, full_matrices=False)
        
        # Plot singular values
        axes[row, col].plot(s_class / s_class.max(), 'o-', markersize=2)
        axes[row, col].set_yscale('log')
        axes[row, col].set_title(f'{class_names[class_idx]}\nSingular Values')
        axes[row, col].set_xlabel('Component Index')
        axes[row, col].set_ylabel('Normalized SV (log scale)')
        axes[row, col].grid(True)
    
    # Remove the last subplot if not needed
    if n_classes < 6:
        axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig('../../output_figures/class_spectral_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Summary insights
    print("\n" + "="*80)
    print("ADVANCED PSEUDOINVERSE INSIGHTS")
    print("="*80)
    print("1. Pixel-Level Reconstruction Filters:")
    print("   - Each pixel has a unique reconstruction kernel")
    print("   - Center pixels show self-reinforcement patterns")
    print("   - Edge pixels show directional preferences")
    print("   - Color components reveal opponent processing")
    print()
    print("2. Spectral Analysis:")
    print(f"   - Effective rank: {np.sum(s > s.max() * 0.01)} components")
    print(f"   - 90% energy in first {np.argmax(cumulative_energy > 0.9)} components")
    print("   - Rapid decay indicates high image redundancy")
    print("   - Singular vectors show principal image patterns")
    print()
    print("3. Pseudoinverse Manifold:")
    print("   - Phase patterns reveal rotational symmetries")
    print("   - Magnitude shows important anchor pixels")
    print("   - Color relationships indicate opponent processing")
    print()
    print("4. Color Channel Correlations:")
    print("   - Strong correlations indicate shared patterns")
    print("   - Negative correlations show opponent processes")
    print("   - Real component captures luminance information")
    print()
    print("5. Class-Specific Patterns:")
    print("   - Different classes have distinct spectral signatures")
    print("   - Reflects class-specific visual features")
    print("   - Shows how pseudoinverse adapts to different object types")

def analyze_cifar10_pseudoinverse(X, X_pinv, labels, class_names):
    """Analyze CIFAR-10 pseudoinverse properties."""
    print("\n" + "="*80)
    print("CIFAR-10 PSEUDOINVERSE ANALYSIS")
    print("="*80)
    
    n_images, n_features = X.shape
    n_classes = len(class_names)
    
    print(f"Number of images: {n_images}")
    print(f"Number of classes: {n_classes}")
    print(f"Features per image: {n_features}")
    print(f"X norm: {quat_frobenius_norm(X):.6f}")
    print(f"X_pinv norm: {quat_frobenius_norm(X_pinv):.6f}")
    
    # Class distribution
    print(f"\nClass distribution:")
    for i, class_name in enumerate(class_names):
        count = np.sum(labels == i)
        print(f"  {class_name}: {count} images")
    
    # Verify pseudoinverse properties
    X_X_pinv = quat_matmat(X, X_pinv)
    X_X_pinv_real = quaternion.as_float_array(X_X_pinv)[:, :, 0]
    I_m = np.eye(n_images)
    print(f"\n||X @ X_pinv - I_m|| (w-component): {np.linalg.norm(X_X_pinv_real - I_m):.2e}")

# Remove all X^T analysis logic and restore main()
def main():
    print("="*80)
    print("CIFAR-10 QUATERNION PSEUDOINVERSE ANALYSIS")
    print("="*80)
    
    # Load CIFAR-10 subset
    n_samples_per_class = 50
    classes_to_use = [0, 1, 2, 3, 4]  # airplane, automobile, bird, cat, deer
    
    try:
        X, labels, class_names = load_cifar10_subset(n_samples_per_class, classes_to_use)
    except Exception as e:
        print(f"Error loading CIFAR-10: {e}")
        return
    
    # Compute pseudoinverse
    print("\nComputing pseudoinverse...")
    X_pinv, residuals, covariances = compute_pseudoinverse(X)
    
    # Analyze pseudoinverse
    analyze_cifar10_pseudoinverse(X, X_pinv, labels, class_names)
    
    # Advanced visualization with insights
    visualize_pseudoinverse_insights(X, X_pinv, labels, image_shape=(32, 32))
    
    # Class-specific column analysis
    visualize_class_average_filters(X_pinv, labels, class_names, image_shape=(32, 32))
    
    # PCA analysis
    visualize_pca_analysis(X_pinv, labels, class_names)
    
    # Sample image verification
    visualize_sample_images(X, labels, class_names)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("ANALYZED: X (images × features)")
    print("Standard perspective:")
    print("- Rows of X = images")
    print("- Columns of X = pixels/features")
    print("- X_pinv = how each image reconstructs from all pixels")
    print("- See all generated plots for results.")
    print("\nGenerated plots:")
    print("- pixel_reconstruction_filters.png: Pixel-level reconstruction kernels")
    print("- spectral_analysis.png: Singular value analysis")
    print("- pseudoinverse_manifold.png: Phase and magnitude visualization")
    print("- channel_correlations.png: Color channel correlations")
    print("- class_spectral_analysis.png: Class-specific spectral patterns")
    print("- class_average_filters.png: Class-average reconstruction filters")
    print("- pca_analysis.png: PCA and t-SNE analysis of pseudoinverse columns")
    print("- sample_images_verification.png: Sample image verification")

if __name__ == "__main__":
    main() 