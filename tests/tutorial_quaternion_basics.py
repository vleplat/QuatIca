#!/usr/bin/env python3
"""
QuatIca Framework Tutorial: Quaternion Matrix Operations Basics
================================================================

This tutorial demonstrates the fundamental operations available in the QuatIca framework.
It covers creating quaternion matrices, basic operations, and advanced computations.

Author: QuatIca Team
Date: 2024
"""

import numpy as np
import quaternion
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Add core directory to path to import our framework functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core'))
from utils import quat_frobenius_norm, quat_matmat, SparseQuaternionMatrix
from data_gen import create_test_matrix, create_sparse_quat_matrix
from solver import NewtonSchulzPseudoinverse

def print_section(title, level=1):
    """Print a formatted section header."""
    if level == 1:
        print("\n" + "="*80)
        print(f" {title}")
        print("="*80)
    elif level == 2:
        print("\n" + "-"*60)
        print(f" {title}")
        print("-"*60)
    else:
        print(f"\n{title}")

def visualize_quaternion_matrix(A, name="Matrix", save_plot=True):
    """Create a beautiful visualization of a quaternion matrix."""
    # Handle sparse matrices
    if isinstance(A, SparseQuaternionMatrix):
        A = quaternion.as_quat_array(
            np.stack([
                A.real.toarray(),
                A.i.toarray(),
                A.j.toarray(),
                A.k.toarray()
            ], axis=-1)
        )
    
    # Convert to float array
    A_float = quaternion.as_float_array(A)
    
    # Create subplot for each component
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Quaternion Matrix: {name} (shape: {A.shape})', fontsize=16, fontweight='bold')
    
    components = ['Real (w)', 'Imaginary i', 'Imaginary j', 'Imaginary k']
    colors = ['viridis', 'Reds', 'Blues', 'Greens']
    
    for idx, (comp, color) in enumerate(zip(components, colors)):
        row, col = idx // 2, idx % 2
        im = axes[row, col].imshow(A_float[..., idx], cmap=color, aspect='auto')
        axes[row, col].set_title(f'{comp} Component', fontweight='bold')
        axes[row, col].set_xlabel('Column')
        axes[row, col].set_ylabel('Row')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[row, col])
        cbar.set_label('Magnitude')
        
        # Add text annotations for small matrices
        if A.shape[0] <= 8 and A.shape[1] <= 8:
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    val = A_float[i, j, idx]
                    color = 'white' if abs(val) > 0.5 else 'black'
                    axes[row, col].text(j, i, f'{val:.2f}', 
                                      ha='center', va='center', color=color, fontsize=8)
    
    plt.tight_layout()
    if save_plot:
        # Ensure output directory exists
        import os
        os.makedirs('../../output_figures', exist_ok=True)
        plt.savefig(f'../../output_figures/{name.replace(" ", "_").lower()}_visualization.png', 
                   dpi=300, bbox_inches='tight')
    plt.show()

def plot_convergence(residuals, covariances, title="Newton-Schulz Convergence", save_plot=True):
    """Plot convergence curves for the Newton-Schulz algorithm."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot residuals
    ax1.semilogy(range(1, len(residuals['AXA-A']) + 1), residuals['AXA-A'], 
                'o-', label='AXA-A', linewidth=2, markersize=6)
    ax1.semilogy(range(1, len(residuals['XAX-X']) + 1), residuals['XAX-X'], 
                's-', label='XAX-X', linewidth=2, markersize=6)
    ax1.semilogy(range(1, len(residuals['AX-herm']) + 1), residuals['AX-herm'], 
                '^-', label='AX-herm', linewidth=2, markersize=6)
    ax1.semilogy(range(1, len(residuals['XA-herm']) + 1), residuals['XA-herm'], 
                'd-', label='XA-herm', linewidth=2, markersize=6)
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Residual Norm', fontsize=12)
    ax1.set_title('Moore-Penrose Residuals', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot covariance deviation
    ax2.semilogy(range(1, len(covariances) + 1), covariances, 
                'o-', color='red', linewidth=2, markersize=6, label='Covariance Deviation')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Covariance Deviation', fontsize=12)
    ax2.set_title('Covariance Deviation ||AX-I|| or ||XA-I||', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_plot:
        # Ensure output directory exists
        import os
        os.makedirs('../../output_figures', exist_ok=True)
        plt.savefig(f'../../output_figures/{title.replace(" ", "_").lower()}_convergence.png', 
                   dpi=300, bbox_inches='tight')
    plt.show()

def plot_performance_comparison(sizes, times, iterations, title="Performance Scaling", save_plot=True):
    """Plot performance comparison across different matrix sizes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract matrix dimensions for x-axis
    matrix_sizes = [f"{m}×{n}" for m, n in sizes]
    
    # Plot computation time
    ax1.plot(matrix_sizes, times, 'o-', color='blue', linewidth=3, markersize=8)
    ax1.set_xlabel('Matrix Size', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Computation Time vs Matrix Size', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on points
    for i, (size, time) in enumerate(zip(matrix_sizes, times)):
        ax1.annotate(f'{time:.3f}s', (i, time), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    # Plot iterations
    ax2.plot(matrix_sizes, iterations, 's-', color='red', linewidth=3, markersize=8)
    ax2.set_xlabel('Matrix Size', fontsize=12)
    ax2.set_ylabel('Iterations to Convergence', fontsize=12)
    ax2.set_title('Convergence Iterations vs Matrix Size', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on points
    for i, (size, iter_count) in enumerate(zip(matrix_sizes, iterations)):
        ax2.annotate(f'{iter_count}', (i, iter_count), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_plot:
        # Ensure output directory exists
        import os
        os.makedirs('../../output_figures', exist_ok=True)
        plt.savefig(f'../../output_figures/{title.replace(" ", "_").lower()}_performance.png', 
                   dpi=300, bbox_inches='tight')
    plt.show()

def display_quaternion_matrix(A, name="Matrix", max_display=4):
    """Display a quaternion matrix in a readable format."""
    print(f"\n{name} (shape: {A.shape}):")
    
    # Handle sparse matrices
    if isinstance(A, SparseQuaternionMatrix):
        A = quaternion.as_quat_array(
            np.stack([
                A.real.toarray(),
                A.i.toarray(),
                A.j.toarray(),
                A.k.toarray()
            ], axis=-1)
        )
    
    # Convert to float array for easier display
    A_float = quaternion.as_float_array(A)
    
    # Display only a subset if matrix is large
    if A.shape[0] > max_display or A.shape[1] > max_display:
        print("  (showing first few entries)")
        rows = min(max_display, A.shape[0])
        cols = min(max_display, A.shape[1])
    else:
        rows, cols = A.shape
    
    for i in range(rows):
        row_str = "  ["
        for j in range(cols):
            w, x, y, z = A_float[i, j]
            # Format quaternion nicely
            if abs(w) < 1e-10: w = 0
            if abs(x) < 1e-10: x = 0
            if abs(y) < 1e-10: y = 0
            if abs(z) < 1e-10: z = 0
            
            comp_str = f"{w:.3f}"
            if x != 0: comp_str += f"{x:+.3f}i"
            if y != 0: comp_str += f"{y:+.3f}j"
            if z != 0: comp_str += f"{z:+.3f}k"
            
            row_str += f" {comp_str:>12}"
        row_str += " ]"
        print(row_str)
    
    if A.shape[0] > max_display or A.shape[1] > max_display:
        print("  ...")

def main():
    """Main tutorial function demonstrating QuatIca capabilities."""
    
    print_section("QUATICA FRAMEWORK TUTORIAL: QUATERNION MATRIX OPERATIONS", 1)
    print("This tutorial demonstrates the key features of the QuatIca framework.")
    print("We'll cover: matrix creation, basic operations, and advanced computations.")
    
    # ========================================================================
    # SECTION 1: CREATING QUATERNION MATRICES
    # ========================================================================
    print_section("1. CREATING QUATERNION MATRICES", 2)
    
    print("1.1 Creating a random dense quaternion matrix:")
    # Create a 3x4 random quaternion matrix
    A_dense = create_test_matrix(3, 4)
    display_quaternion_matrix(A_dense, "A_dense")
    visualize_quaternion_matrix(A_dense, "Dense Matrix A_dense")
    
    print("\n1.2 Creating a sparse quaternion matrix:")
    # Create a 4x3 sparse quaternion matrix with 50% sparsity
    A_sparse = create_sparse_quat_matrix(4, 3, density=0.5)
    display_quaternion_matrix(A_sparse, "A_sparse")
    visualize_quaternion_matrix(A_sparse, "Sparse Matrix A_sparse")
    
    print("\n1.3 Creating a quaternion vector:")
    # Create a random quaternion vector
    v = create_test_matrix(4, 1)  # 4x1 vector
    display_quaternion_matrix(v, "v (vector)")
    visualize_quaternion_matrix(v, "Vector v")
    
    # ========================================================================
    # SECTION 2: BASIC MATRIX OPERATIONS
    # ========================================================================
    print_section("2. BASIC MATRIX OPERATIONS", 2)
    
    print("2.1 Matrix-Vector Multiplication (A * v):")
    result_matvec = quat_matmat(A_dense, v)
    display_quaternion_matrix(result_matvec, "A_dense * v")
    
    print("\n2.2 Matrix-Matrix Multiplication (A * B):")
    # Create another matrix for multiplication
    B = create_test_matrix(4, 2)
    display_quaternion_matrix(B, "B")
    
    result_matmat = quat_matmat(A_dense, B)
    display_quaternion_matrix(result_matmat, "A_dense * B")
    
    print("\n2.3 Computing Frobenius Norm:")
    norm_A = quat_frobenius_norm(A_dense)
    norm_B = quat_frobenius_norm(B)
    norm_v = quat_frobenius_norm(v)
    
    print(f"  ||A_dense||_F = {norm_A:.6f}")
    print(f"  ||B||_F = {norm_B:.6f}")
    print(f"  ||v||_F = {norm_v:.6f}")
    
    # ========================================================================
    # SECTION 3: ADVANCED OPERATIONS
    # ========================================================================
    print_section("3. ADVANCED OPERATIONS", 2)
    
    print("3.1 Computing Pseudoinverse using Newton-Schulz method:")
    print("   This is the core feature of QuatIca!")
    
    # Create a matrix for pseudoinverse computation
    A_pinv = create_test_matrix(3, 5)
    display_quaternion_matrix(A_pinv, "A_pinv (original)")
    
    # Compute pseudoinverse
    ns_solver = NewtonSchulzPseudoinverse(verbose=True)
    A_pinv_result, residuals, covariances = ns_solver.compute(A_pinv)
    
    display_quaternion_matrix(A_pinv_result, "A_pinv^† (pseudoinverse)")
    
    print(f"\n   Convergence: {len(covariances)} iterations")
    print(f"   Final residual: {max(residuals[key][-1] for key in residuals):.2e}")
    print(f"   Final covariance deviation: {covariances[-1]:.2e}")
    
    # Plot convergence curves
    plot_convergence(residuals, covariances, "Pseudoinverse Convergence")
    
    # ========================================================================
    # SECTION 4: VERIFICATION AND PROPERTIES
    # ========================================================================
    print_section("4. VERIFICATION AND PROPERTIES", 2)
    
    print("4.1 Verifying Pseudoinverse Properties:")
    print("   We check: A * A^† * A ≈ A (first property)")
    
    # Compute A * A^† * A
    temp1 = quat_matmat(A_pinv, A_pinv_result)
    verification = quat_matmat(temp1, A_pinv)
    
    # Compute difference: A * A^† * A - A
    diff = verification - A_pinv
    diff_norm = quat_frobenius_norm(diff)
    
    print(f"   ||A * A^† * A - A||_F = {diff_norm:.2e}")
    print(f"   Relative error: {diff_norm / quat_frobenius_norm(A_pinv):.2e}")
    
    print("\n4.2 Checking Matrix Dimensions:")
    print(f"   A_pinv shape: {A_pinv.shape}")
    print(f"   A_pinv^† shape: {A_pinv_result.shape}")
    print(f"   Expected: ({A_pinv.shape[1]}, {A_pinv.shape[0]}) ✓")
    
    # ========================================================================
    # SECTION 5: PRACTICAL EXAMPLES
    # ========================================================================
    print_section("5. PRACTICAL EXAMPLES", 2)
    
    print("5.1 Solving a Linear System (A * x = b):")
    # Create a system A * x = b
    A_system = create_test_matrix(3, 3)
    b = create_test_matrix(3, 1)
    
    display_quaternion_matrix(A_system, "A_system")
    display_quaternion_matrix(b, "b")
    
    # Solve using pseudoinverse: x = A^† * b
    A_system_pinv, _, _ = ns_solver.compute(A_system)
    x_solution = quat_matmat(A_system_pinv, b)
    
    display_quaternion_matrix(x_solution, "x_solution (A^† * b)")
    
    # Verify solution: check that A*x ≈ b
    b_computed = quat_matmat(A_system, x_solution)
    residual_norm = quat_frobenius_norm(b - b_computed)
    relative_error = residual_norm / quat_frobenius_norm(b)
    print(f"   ||A*x - b||_F = {residual_norm:.2e}")
    print(f"   Relative error: {relative_error:.2e}")
    print(f"   ✓ Solution verification: residual is {'small' if residual_norm < 1e-6 else 'large'}")
    
    print("\n5.2 Matrix Factorization Example:")
    # Create a larger matrix for demonstration
    A_large = create_test_matrix(5, 4)
    display_quaternion_matrix(A_large, "A_large")
    
    # Compute pseudoinverse
    A_large_pinv, _, _ = ns_solver.compute(A_large)
    
    # Show that A * A^† is a projection matrix
    projection = quat_matmat(A_large, A_large_pinv)
    display_quaternion_matrix(projection, "A * A^† (projection matrix)")
    
    # Check projection property: (A * A^†)^2 = A * A^†
    projection_squared = quat_matmat(projection, projection)
    projection_error = quat_frobenius_norm(projection - projection_squared)
    print(f"   Projection property error: {projection_error:.2e}")
    
    # ========================================================================
    # SECTION 6: PERFORMANCE AND SCALING
    # ========================================================================
    print_section("6. PERFORMANCE AND SCALING", 2)
    
    print("6.1 Testing with different matrix sizes:")
    sizes = [(10, 8), (20, 15), (30, 25)]
    times = []
    iterations = []
    
    for m, n in sizes:
        print(f"\n   Testing {m}x{n} matrix:")
        A_test = create_test_matrix(m, n)
        
        # Time the pseudoinverse computation
        import time
        start_time = time.time()
        A_test_pinv, residuals, _ = ns_solver.compute(A_test)
        end_time = time.time()
        
        computation_time = end_time - start_time
        iter_count = len(residuals)
        
        times.append(computation_time)
        iterations.append(iter_count)
        
        print(f"     Size: {m}x{n}")
        print(f"     Time: {computation_time:.3f} seconds")
        print(f"     Iterations: {iter_count}")
        print(f"     Final residual: {max(residuals[key][-1] for key in residuals):.2e}")
    
    # Plot performance comparison
    plot_performance_comparison(sizes, times, iterations, "Quaternion Pseudoinverse Performance")
    
    # ========================================================================
    # SECTION 7: SUMMARY AND BEST PRACTICES
    # ========================================================================
    print_section("7. SUMMARY AND BEST PRACTICES", 2)
    
    print("Key Takeaways:")
    print("1. Use create_test_matrix() for dense matrices")
    print("2. Use create_sparse_quat_matrix() for sparse matrices")
    print("3. Use quat_matmat() for matrix-matrix multiplication")
    print("4. Use quat_matmat() for matrix-vector multiplication (vectors are matrices with one column)")
    print("5. Use quat_frobenius_norm() for computing norms")
    print("6. Use NewtonSchulzPseudoinverse() for pseudoinverse computation")
    
    print("\nBest Practices:")
    print("1. Always check matrix dimensions before operations")
    print("2. Verify pseudoinverse properties for validation")
    print("3. Monitor convergence in Newton-Schulz iterations")
    print("4. Use appropriate scaling for numerical stability")
    print("5. Consider sparsity for large matrices")
    
    print("\nCommon Use Cases:")
    print("1. Linear system solving: x = A^† * b")
    print("2. Least squares problems")
    print("3. Matrix factorization and decomposition")
    print("4. Image processing and computer vision")
    print("5. Signal processing applications")
    
    # Create a summary visualization
    def create_summary_visualization():
        """Create a creative summary visualization of the tutorial."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a colorful background
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        
        # Add title
        ax.text(5, 7.5, 'QuatIca Framework Tutorial', fontsize=20, fontweight='bold', 
               ha='center', color='darkblue')
        ax.text(5, 7, 'Quaternion Matrix Operations Complete!', fontsize=14, 
               ha='center', color='darkgreen')
        
        # Create feature boxes
        features = [
            ('Matrix Creation', 2, 6, 'lightblue'),
            ('Basic Operations', 5, 6, 'lightgreen'),
            ('Pseudoinverse', 8, 6, 'lightcoral'),
            ('Verification', 2, 4, 'lightyellow'),
            ('Linear Systems', 5, 4, 'lightpink'),
            ('Performance', 8, 4, 'lightgray'),
            ('Best Practices', 5, 2, 'lightcyan')
        ]
        
        for feature, x, y, color in features:
            rect = Rectangle((x-0.8, y-0.5), 1.6, 1, facecolor=color, 
                           edgecolor='black', linewidth=2, alpha=0.8)
            ax.add_patch(rect)
            ax.text(x, y, feature, fontsize=10, fontweight='bold', 
                   ha='center', va='center')
        
        # Add connecting arrows
        arrows = [
            ((2, 5.5), (5, 5.5)),  # Matrix Creation -> Basic Operations
            ((5, 5.5), (8, 5.5)),  # Basic Operations -> Pseudoinverse
            ((8, 5.5), (8, 4.5)),  # Pseudoinverse -> Performance
            ((2, 4.5), (5, 4.5)),  # Verification -> Linear Systems
            ((5, 4.5), (8, 4.5)),  # Linear Systems -> Performance
            ((5, 3.5), (5, 2.5)),  # Linear Systems -> Best Practices
        ]
        
        for start, end in arrows:
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        
        # Add quaternion symbols
        quat_symbols = ['1', 'i', 'j', 'k']
        for i, symbol in enumerate(quat_symbols):
            x = 1 + i * 2
            ax.text(x, 1, symbol, fontsize=24, fontweight='bold', 
                   ha='center', va='center', color='purple')
        
        ax.text(5, 0.5, 'Quaternion Algebra: 1, i, j, k', fontsize=12, 
               ha='center', style='italic')
        
        ax.set_xlabel('Framework Components', fontsize=12)
        ax.set_ylabel('Learning Progress', fontsize=12)
        ax.set_title('Tutorial Journey Summary', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # Ensure output directory exists
        import os
        os.makedirs('../../output_figures', exist_ok=True)
        plt.savefig('../../output_figures/tutorial_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    create_summary_visualization()
    
    print_section("TUTORIAL COMPLETE!", 1)
    print("You now have a solid understanding of the QuatIca framework!")
    print("Explore the other scripts in the tests/ directory for more advanced examples.")
    print("\n📊 Visualizations have been saved to the output_figures/ directory!")

if __name__ == "__main__":
    main() 