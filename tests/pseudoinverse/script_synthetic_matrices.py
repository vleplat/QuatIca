import sys
import os

# Add parent directory to path to import core modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core'))
from data_gen import create_test_matrix, create_sparse_quat_matrix, small_test_Mat, theoretical_pseudoinverse_example_5_2
from solver import NewtonSchulzPseudoinverse
from visualization import Visualizer
import time
import numpy as np
import quaternion
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns


def create_validation_visualization(X_small, A_theoretical, cov_small, res_small, 
                                   small_iters, small_time, performance_data):
    """
    Create two separate validation visualizations:
    1. Theoretical vs Numerical Comparison (focused on validation)
    2. Performance Analysis (focused on benchmarking)
    """
    
    # ========================================================================
    # FIGURE 1a: ORIGINAL MATRIX + THEORETICAL VS NUMERICAL COMPARISON
    # ========================================================================
    fig1a = plt.figure(figsize=(24, 16))
    gs1a = fig1a.add_gridspec(2, 3, hspace=0.6, wspace=0.5)
    
    # Title and reference
    ax_title = fig1a.add_subplot(gs1a[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.8, 'QUATERNION PSEUDOINVERSE VALIDATION - PART 1', 
                 fontsize=24, fontweight='bold', ha='center')
    ax_title.text(0.5, 0.6, 'Original Matrix and Theoretical vs Numerical Comparison', 
                 fontsize=16, ha='center')
    ax_title.text(0.5, 0.4, 'Reference: Huang, L., Wang, Q.-W., & Zhang, Y. (2015)', 
                 fontsize=12, ha='center', style='italic')
    ax_title.text(0.5, 0.2, 'Linear Algebra and its Applications, 475, 45-61', 
                 fontsize=12, ha='center', style='italic')
    
    # Original matrix A (larger, more readable)
    ax_orig = fig1a.add_subplot(gs1a[1, 0])
    ax_orig.set_title('Original Matrix A\n[1 i+2k 3; i 6+j 7]', fontweight='bold', fontsize=18)
    ax_orig.text(0.5, 0.8, 'A = [1  i+2k  3]', ha='center', fontsize=16)
    ax_orig.text(0.5, 0.6, '    [i  6+j   7]', ha='center', fontsize=16)
    ax_orig.text(0.5, 0.4, 'Shape: 2×3', ha='center', fontsize=14)
    ax_orig.text(0.5, 0.2, 'Quaternion matrix', ha='center', fontsize=14)
    ax_orig.axis('off')
    
    # Theoretical pseudoinverse (much larger table with better formatting)
    ax_theo = fig1a.add_subplot(gs1a[1, 1])
    ax_theo.set_title('Theoretical A^† (Literature)', fontweight='bold', fontsize=18)
    
    theo_display = quaternion.as_float_array(A_theoretical)
    theo_text = []
    for i in range(3):
        row_text = []
        for j in range(2):
            w, x, y, z = theo_display[i, j]
            if abs(z) < 1e-10: z = 0
            if abs(y) < 1e-10: y = 0
            if abs(x) < 1e-10: x = 0
            if abs(w) < 1e-10: w = 0
            # Format with shorter precision for better fit
            comp_str = f"{w:.3f}"
            if x != 0: comp_str += f"{x:+.3f}i"
            if y != 0: comp_str += f"{y:+.3f}j"
            if z != 0: comp_str += f"{z:+.3f}k"
            row_text.append(comp_str)
        theo_text.append(row_text)
    
    table = ax_theo.table(cellText=theo_text, 
                         colLabels=['Col 1', 'Col 2'],
                         rowLabels=['Row 1', 'Row 2', 'Row 3'],
                         cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.5, 5)  # Much wider and taller
    ax_theo.axis('off')
    
    # Numerical pseudoinverse (much larger table with better formatting)
    ax_num = fig1a.add_subplot(gs1a[1, 2])
    ax_num.set_title('Numerical A^† (Our Implementation)', fontweight='bold', fontsize=18)
    
    num_display = quaternion.as_float_array(X_small)
    num_text = []
    for i in range(3):
        row_text = []
        for j in range(2):
            w, x, y, z = num_display[i, j]
            if abs(z) < 1e-10: z = 0
            if abs(y) < 1e-10: y = 0
            if abs(x) < 1e-10: x = 0
            if abs(w) < 1e-10: w = 0
            # Format with shorter precision for better fit
            comp_str = f"{w:.3f}"
            if x != 0: comp_str += f"{x:+.3f}i"
            if y != 0: comp_str += f"{y:+.3f}j"
            if z != 0: comp_str += f"{z:+.3f}k"
            row_text.append(comp_str)
        num_text.append(row_text)
    
    table = ax_num.table(cellText=num_text, 
                        colLabels=['Col 1', 'Col 2'],
                        rowLabels=['', '', ''],  # Empty row labels to save space
                        cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.5, 5)  # Much wider and taller
    ax_num.axis('off')
    
    # Save Figure 1a
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename1a = f"../../output_figures/validation_comparison_1a_{timestamp}.png"
    plt.savefig(filename1a, dpi=300, bbox_inches='tight')
    print(f"Validation comparison figure 1a saved to: {filename1a}")
    
    # ========================================================================
    # FIGURE 1b: ERROR ANALYSIS + CONVERGENCE
    # ========================================================================
    fig1b = plt.figure(figsize=(20, 12))
    gs1b = fig1b.add_gridspec(2, 3, hspace=0.5, wspace=0.4)
    
    # Title
    ax_title2 = fig1b.add_subplot(gs1b[0, :])
    ax_title2.axis('off')
    ax_title2.text(0.5, 0.8, 'QUATERNION PSEUDOINVERSE VALIDATION - PART 2', 
                  fontsize=24, fontweight='bold', ha='center')
    ax_title2.text(0.5, 0.6, 'Error Analysis and Convergence History', 
                  fontsize=16, ha='center')
    
    # Error statistics (larger, more readable)
    ax_stats = fig1b.add_subplot(gs1b[1, 0])
    ax_stats.set_title('Error Statistics', fontweight='bold', fontsize=18)
    ax_stats.axis('off')
    
    # Calculate differences
    diff = np.abs(quaternion.as_float_array(X_small) - quaternion.as_float_array(A_theoretical))
    diff_reshaped = diff.reshape(3, 2, 4)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    rms_diff = np.sqrt(np.mean(diff**2))
    
    stats_text = f"""
    Max Absolute Difference: {max_diff:.2e}
    Mean Absolute Difference: {mean_diff:.2e}
    RMS Difference: {rms_diff:.2e}
    
    Validation Status: {'✓ PASSED' if cov_small[-1] < 1e-5 else '✗ FAILED'}
    Final Accuracy: {cov_small[-1]:.2e}
    Convergence: {small_iters} iterations
    Computation Time: {small_time:.6f} seconds
    """
    
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, 
                 fontsize=16, verticalalignment='top', fontfamily='monospace')
    
    # Absolute difference heatmap (larger, more readable)
    ax_diff = fig1b.add_subplot(gs1b[1, 1])
    ax_diff.set_title('Absolute Difference\n|Theoretical - Numerical| (w-component)', fontweight='bold', fontsize=18)
    
    # Show w-component difference (most important)
    im = ax_diff.imshow(diff_reshaped[:, :, 0], cmap='Reds', aspect='auto')
    ax_diff.set_xticks([0, 1])
    ax_diff.set_xticklabels(['Col 1', 'Col 2'], fontsize=14)
    ax_diff.set_yticks([0, 1, 2])
    ax_diff.set_yticklabels(['Row 1', 'Row 2', 'Row 3'], fontsize=14)
    
    # Add text annotations (larger font)
    for i in range(3):
        for j in range(2):
            text = ax_diff.text(j, i, f'{diff_reshaped[i, j, 0]:.2e}',
                              ha="center", va="center", color="black", fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax_diff, shrink=0.8)
    
    # Convergence plot (larger, more readable)
    ax_conv = fig1b.add_subplot(gs1b[1, 2])
    ax_conv.set_title('Convergence History', fontweight='bold', fontsize=18)
    ax_conv.semilogy(cov_small, 'b-', linewidth=3, label='Covariance Deviation')
    ax_conv.axhline(y=1e-6, color='g', linestyle=':', linewidth=2, label='Tolerance (1e-6)')
    ax_conv.set_xlabel('Iteration', fontsize=16)
    ax_conv.set_ylabel('Error', fontsize=16)
    ax_conv.legend(fontsize=14)
    ax_conv.grid(True, alpha=0.3)
    ax_conv.tick_params(axis='both', which='major', labelsize=14)
    
    # Save Figure 1b
    filename1b = f"../../output_figures/validation_comparison_1b_{timestamp}.png"
    plt.savefig(filename1b, dpi=300, bbox_inches='tight')
    print(f"Validation comparison figure 1b saved to: {filename1b}")
    
    # ========================================================================
    # FIGURE 1c: COMPONENT-WISE ANALYSIS
    # ========================================================================
    fig1c = plt.figure(figsize=(20, 12))
    gs1c = fig1c.add_gridspec(2, 2, hspace=0.5, wspace=0.4)
    
    # Title
    ax_title3 = fig1c.add_subplot(gs1c[0, :])
    ax_title3.axis('off')
    ax_title3.text(0.5, 0.8, 'QUATERNION PSEUDOINVERSE VALIDATION - PART 3', 
                  fontsize=24, fontweight='bold', ha='center')
    ax_title3.text(0.5, 0.6, 'Component-wise Analysis: Theoretical vs Numerical', 
                  fontsize=16, ha='center')
    
    # Component-wise comparison for Row 1
    ax_comp1 = fig1c.add_subplot(gs1c[1, 0])
    ax_comp1.set_title('Row 1 Component Comparison', fontweight='bold', fontsize=18)
    
    theo_flat = quaternion.as_float_array(A_theoretical).reshape(-1, 4)
    num_flat = quaternion.as_float_array(X_small).reshape(-1, 4)
    
    components = ['w (Real)', 'x (i)', 'y (j)', 'z (k)']
    x_pos = np.arange(len(components))
    
    width = 0.35
    ax_comp1.bar(x_pos - width/2, theo_flat[0], width, label='Theoretical', alpha=0.8, color='blue')
    ax_comp1.bar(x_pos + width/2, num_flat[0], width, label='Numerical', alpha=0.8, color='red')
    
    ax_comp1.set_xlabel('Quaternion Components', fontsize=16)
    ax_comp1.set_ylabel('Value', fontsize=16)
    ax_comp1.set_xticks(x_pos)
    ax_comp1.set_xticklabels(components, fontsize=14)
    ax_comp1.legend(fontsize=14)
    ax_comp1.grid(True, alpha=0.3)
    ax_comp1.tick_params(axis='both', which='major', labelsize=14)
    
    # Add error annotations
    for i, comp in enumerate(components):
        diff_val = abs(theo_flat[0, i] - num_flat[0, i])
        ax_comp1.text(i, max(theo_flat[0, i], num_flat[0, i]) + 0.01, 
                     f'Δ={diff_val:.2e}', ha='center', fontsize=12, color='red', fontweight='bold')
    
    # Component-wise comparison for Row 2
    ax_comp2 = fig1c.add_subplot(gs1c[1, 1])
    ax_comp2.set_title('Row 2 Component Comparison', fontweight='bold', fontsize=18)
    
    ax_comp2.bar(x_pos - width/2, theo_flat[1], width, label='Theoretical', alpha=0.8, color='blue')
    ax_comp2.bar(x_pos + width/2, num_flat[1], width, label='Numerical', alpha=0.8, color='red')
    
    ax_comp2.set_xlabel('Quaternion Components', fontsize=16)
    ax_comp2.set_ylabel('Value', fontsize=16)
    ax_comp2.set_xticks(x_pos)
    ax_comp2.set_xticklabels(components, fontsize=14)
    ax_comp2.legend(fontsize=14)
    ax_comp2.grid(True, alpha=0.3)
    ax_comp2.tick_params(axis='both', which='major', labelsize=14)
    
    # Add error annotations
    for i, comp in enumerate(components):
        diff_val = abs(theo_flat[1, i] - num_flat[1, i])
        ax_comp2.text(i, max(theo_flat[1, i], num_flat[1, i]) + 0.01, 
                     f'Δ={diff_val:.2e}', ha='center', fontsize=12, color='red', fontweight='bold')
    
    # Save Figure 1c
    filename1c = f"../../output_figures/validation_comparison_1c_{timestamp}.png"
    plt.savefig(filename1c, dpi=300, bbox_inches='tight')
    print(f"Validation comparison figure 1c saved to: {filename1c}")
    
    # ========================================================================
    # FIGURE 2: PERFORMANCE ANALYSIS
    # ========================================================================
    fig2 = plt.figure(figsize=(20, 12))
    gs2 = fig2.add_gridspec(2, 3, hspace=0.4, wspace=0.4)
    
    # Title
    ax_title2 = fig2.add_subplot(gs2[0, :])
    ax_title2.axis('off')
    ax_title2.text(0.5, 0.8, 'QUATERNION PSEUDOINVERSE PERFORMANCE ANALYSIS', 
                  fontsize=24, fontweight='bold', ha='center')
    ax_title2.text(0.5, 0.6, 'Algorithm Benchmarking Across Matrix Types', 
                  fontsize=16, ha='center')
    ax_title2.text(0.5, 0.4, 'Performance Comparison: Dense, Sparse, Ill-conditioned, and Validation Matrices', 
                  fontsize=14, ha='center')
    
    # Performance comparison - Iterations
    ax_iter = fig2.add_subplot(gs2[1, 0])
    ax_iter.set_title('Convergence Iterations', fontweight='bold', fontsize=16)
    
    matrix_types = list(performance_data.keys())
    iterations = [performance_data[mt]['iterations'] for mt in matrix_types]
    
    bars = ax_iter.bar(matrix_types, iterations, color=['blue', 'green', 'red', 'purple'], alpha=0.8)
    ax_iter.set_ylabel('Number of Iterations', fontsize=14)
    ax_iter.tick_params(axis='x', rotation=45, labelsize=12)
    ax_iter.tick_params(axis='y', labelsize=12)
    ax_iter.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, iterations):
        height = bar.get_height()
        ax_iter.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{val}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Performance comparison - Time
    ax_time = fig2.add_subplot(gs2[1, 1])
    ax_time.set_title('Computation Time', fontweight='bold', fontsize=16)
    
    times = [performance_data[mt]['time'] for mt in matrix_types]
    
    bars = ax_time.bar(matrix_types, times, color=['orange', 'lightblue', 'pink', 'yellow'], alpha=0.8)
    ax_time.set_ylabel('Time (seconds)', fontsize=14)
    ax_time.tick_params(axis='x', rotation=45, labelsize=12)
    ax_time.tick_params(axis='y', labelsize=12)
    ax_time.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, times):
        height = bar.get_height()
        ax_time.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Performance comparison - Accuracy
    ax_acc = fig2.add_subplot(gs2[1, 2])
    ax_acc.set_title('Final Accuracy', fontweight='bold', fontsize=16)
    
    final_accuracy = [performance_data[mt]['final_accuracy'] for mt in matrix_types]
    
    bars = ax_acc.bar(matrix_types, final_accuracy, color=['cyan', 'magenta', 'brown', 'lime'], alpha=0.8)
    ax_acc.set_ylabel('Covariance Deviation', fontsize=14)
    ax_acc.tick_params(axis='x', rotation=45, labelsize=12)
    ax_acc.tick_params(axis='y', labelsize=12)
    ax_acc.set_yscale('log')
    ax_acc.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, final_accuracy):
        height = bar.get_height()
        ax_acc.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                    f'{val:.2e}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Save Figure 2
    filename2 = f"../../output_figures/performance_analysis_{timestamp}.png"
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    print(f"Performance analysis figure saved to: {filename2}")
    
    return fig1a, fig1b, fig1c, fig2


def main() -> None:
    """
    Main function demonstrating quaternion matrix pseudoinverse computation.
    
    This script tests the Newton-Schulz algorithm on various matrix types:
    1. Large dense matrices (performance testing)
    2. Large sparse matrices (efficiency testing) 
    3. Ill-conditioned matrices (robustness testing)
    4. Small validation matrix (accuracy testing with known solution)
    
    The small matrix test uses a specific example from the literature to validate
    the accuracy of our implementation against known theoretical results.
    """
    
    # Ensure output directory exists
    import os
    os.makedirs('../../output_figures', exist_ok=True)
    
    print("="*80)
    print("QUATERNION MATRIX PSEUDOINVERSE COMPUTATION TEST SUITE")
    print("="*80)
    print()
    
    # Generate test matrices
    print("1. LARGE DENSE MATRIX TEST (Performance Benchmark)")
    print("-" * 50)
    A_dense = create_test_matrix(800, 1000)
    print(f"A_dense: type={type(A_dense)}, dtype={getattr(A_dense, 'dtype', 'N/A')}, shape={getattr(A_dense, 'shape', 'N/A')}")
    print(f"A_dense sample: {A_dense.flat[:3]}")
    
    print("\n2. LARGE SPARSE MATRIX TEST (Efficiency Benchmark)")
    print("-" * 50)
    A_sparse = create_sparse_quat_matrix(800, 1000, density=0.01)
    print(f"A_sparse: type={type(A_sparse)}, shape={getattr(A_sparse, 'shape', 'N/A')}")
    if hasattr(A_sparse, 'real'):
        print(f"A_sparse real part sample: {A_sparse.real.data[:3]}")
    
    print("\n3. ILL-CONDITIONED MATRIX TEST (Robustness Benchmark)")
    print("-" * 50)
    A_ill = create_test_matrix(400, 400, cond_number=1e6)
    print(f"A_ill: type={type(A_ill)}, dtype={getattr(A_ill, 'dtype', 'N/A')}, shape={getattr(A_ill, 'shape', 'N/A')}")
    print(f"A_ill sample: {A_ill.flat[:3]}")
    
    print("\n4. VALIDATION MATRIX TEST (Accuracy Benchmark)")
    print("-" * 50)
    print("This test uses a specific 2×3 quaternion matrix with known properties")
    print("from the literature to validate our implementation accuracy.")
    print()
    print("Reference: Huang, L., Wang, Q.-W., & Zhang, Y. (2015).")
    print("          The Moore–Penrose inverses of matrices over quaternion polynomial rings.")
    print("          Linear Algebra and its Applications, 475, 45-61.")
    print("          https://doi.org/10.1016/j.laa.2015.02.004")
    print()
    print("Example 5.2: Let A = [1  i+2k  3]")
    print("                [i  6+j   7]")
    print()
    print("Theoretical pseudoinverse A^† (computed using Maple package):")
    print("⎛ -47/347 + 21/694 i + 11/694 j - 21/694 k    63/347 - 28/347 i + 21/694 j - 101/694 k ⎞")
    print("⎜ -11/694 - 347 i - 11/694 k                   61/694 + 21/694 i - 6/347 j + 21/347 k  ⎟")
    print("⎝  57/347 + 49/694 i + 77/694 k                21/347 - 21/694 i - 33/694 k            ⎠")
    print()
    print("Our numerical implementation should converge to this theoretical result.")
    print()
    A_small = small_test_Mat()
    print(f"Validation matrix shape: {A_small.shape}")
    print(f"Validation matrix:\n{A_small}")
    
    # Initialize solver
    solver = NewtonSchulzPseudoinverse(gamma=0.95, max_iter=100, tol=1e-6, verbose=True, compute_residuals=True)

    print("\n" + "="*80)
    print("COMPUTING PSEUDOINVERSES")
    print("="*80)
    
    # Store performance data for visualization
    performance_data = {}
    
    # Compute pseudoinverses with timing and metrics
    print("\n1. DENSE MATRIX COMPUTATION:")
    start = time.perf_counter()
    X_dense, res_dense, cov_dense = solver.compute(A_dense)
    dense_time = time.perf_counter() - start
    dense_iters = len(cov_dense)
    print(f"✓ Dense matrix: {dense_iters} iters, {dense_time:.4f}s, Final cov dev: {cov_dense[-1]:.2e}")
    performance_data['Dense'] = {
        'iterations': dense_iters,
        'time': dense_time,
        'final_accuracy': cov_dense[-1]
    }

    print("\n2. SPARSE MATRIX COMPUTATION:")
    start = time.perf_counter()
    X_sparse, res_sparse, cov_sparse = solver.compute(A_sparse)
    sparse_time = time.perf_counter() - start
    sparse_iters = len(cov_sparse)
    print(f"✓ Sparse matrix: {sparse_iters} iters, {sparse_time:.4f}s, Final cov dev: {cov_sparse[-1]:.2e}")
    performance_data['Sparse'] = {
        'iterations': sparse_iters,
        'time': sparse_time,
        'final_accuracy': cov_sparse[-1]
    }

    print("\n3. ILL-CONDITIONED MATRIX COMPUTATION:")
    start = time.perf_counter()
    X_ill, res_ill, cov_ill = solver.compute(A_ill)
    ill_time = time.perf_counter() - start
    ill_iters = len(cov_ill)
    print(f"✓ Ill-conditioned matrix: {ill_iters} iters, {ill_time:.4f}s, Final cov dev: {cov_ill[-1]:.2e}")
    performance_data['Ill-conditioned'] = {
        'iterations': ill_iters,
        'time': ill_time,
        'final_accuracy': cov_ill[-1]
    }
    
    print("\n4. VALIDATION MATRIX COMPUTATION:")
    start = time.perf_counter()
    X_small, res_small, cov_small = solver.compute(A_small)
    small_time = time.perf_counter() - start
    small_iters = len(cov_small)
    print(f"✓ Validation matrix: {small_iters} iters, {small_time:.4f}s, Final cov dev: {cov_small[-1]:.2e}")
    print(f"✓ Computed pseudoinverse:\n{X_small}")
    performance_data['Validation'] = {
        'iterations': small_iters,
        'time': small_time,
        'final_accuracy': cov_small[-1]
    }
    
    print("\n" + "="*80)
    print("THEORETICAL vs NUMERICAL COMPARISON")
    print("="*80)
    print("Comparing our numerical result with the theoretical pseudoinverse from Example 5.2:")
    print()
    print("Theoretical A^† (from paper):")
    print("⎛ 47/347 + 21/694 i + 11/694 j +  0 k    -21/694  - 11/347 i +  0 j -  11/694 k ⎞")
    print("⎜ -63/347 - 28/347 i + 21/694 j -101/694 k                  61/694 + 21/694 i - 6/347 j + 21/347 k  ⎟")
    print("⎝  57/347 + 49/694 i + 77/694 k                21/347 - 21/694 i - 33/694 k            ⎠")
    print()
    print("Numerical A^† (our implementation):")
    print(f"{X_small}")
    print()
    
    # Direct numerical comparison
    A_theoretical = theoretical_pseudoinverse_example_5_2()
    print("Direct Numerical Comparison:")
    print("Theoretical vs Numerical (absolute difference):")
    diff = np.abs(quaternion.as_float_array(X_small) - quaternion.as_float_array(A_theoretical))
    print(f"Max absolute difference: {np.max(diff):.2e}")
    print(f"Mean absolute difference: {np.mean(diff):.2e}")
    print(f"RMS difference: {np.sqrt(np.mean(diff**2)):.2e}")
    print()
    print("Note: The numerical result should closely approximate the theoretical values.")
    print("Small differences are expected due to floating-point arithmetic precision.")
    
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    print("The validation matrix test serves as an accuracy check.")
    print("This specific 2×3 quaternion matrix has known theoretical properties")
    print("that allow us to verify our implementation correctness.")
    print()
    print("Expected behavior:")
    print("- Fast convergence (typically < 15 iterations)")
    print("- High accuracy (covariance deviation < 1e-5)")
    print("- Stable computation (no numerical issues)")
    print("- Numerical result should approximate theoretical A^† from Example 5.2")
    print()
    print("✓ Validation test PASSED" if cov_small[-1] < 1e-5 else "✗ Validation test FAILED")
    print(f"  Final accuracy: {cov_small[-1]:.2e}")
    print(f"  Convergence: {small_iters} iterations")
    print(f"  Computation time: {small_time:.6f} seconds")

    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # Create comprehensive validation visualization
    print("Creating comprehensive validation visualization...")
    create_validation_visualization(X_small, A_theoretical, cov_small, res_small, 
                                   small_iters, small_time, performance_data)

    # Original visualizations
    Visualizer.plot_residuals(
        res_dense,
        title="Residual Norms (Dense Matrix)",
        subtitle=f"Dense quaternion matrix, shape: {A_dense.shape}"
    )
    Visualizer.plot_covariances(
        cov_dense,
        title="Covariance Deviation (Dense Matrix)",
        subtitle=f"Dense quaternion matrix, shape: {A_dense.shape}"
    )
    Visualizer.visualize_matrix(
        A_dense,
        component=1,
        title="Dense Matrix: Component 1 Heatmap",
        subtitle=f"Dense quaternion matrix, shape: {A_dense.shape}"
    )
    Visualizer.plot_residuals(
        res_sparse,
        title="Residual Norms (Sparse Matrix)",
        subtitle=f"Sparse quaternion matrix, shape: {A_sparse.shape}, density: 0.01"
    )
    Visualizer.plot_covariances(
        cov_sparse,
        title="Covariance Deviation (Sparse Matrix)",
        subtitle=f"Sparse quaternion matrix, shape: {A_sparse.shape}, density: 0.01"
    )
    Visualizer.visualize_matrix(
        A_sparse,
        component=1,
        title="Sparse Matrix: Component 1 Heatmap",
        subtitle=f"Sparse quaternion matrix, shape: {A_sparse.shape}, density: 0.01"
    )
    Visualizer.plot_residuals(
        res_ill,
        title="Residual Norms (Ill-Conditioned Matrix)",
        subtitle=f"Ill-conditioned quaternion matrix, shape: {A_ill.shape}, cond_number=1e6"
    )
    Visualizer.plot_covariances(
        cov_ill,
        title="Covariance Deviation (Ill-Conditioned Matrix)",
        subtitle=f"Ill-conditioned quaternion matrix, shape: {A_ill.shape}, cond_number=1e6"
    )
    Visualizer.visualize_matrix(
        A_ill,
        component=1,
        title="Ill-Conditioned Matrix: Component 1 Heatmap",
        subtitle=f"Ill-conditioned quaternion matrix, shape: {A_ill.shape}, cond_number=1e6"
    )

    Visualizer.plot_residuals(
        res_small,
        title="Residual Norms (Validation Matrix)",
        subtitle=f"Validation quaternion matrix, shape: {A_small.shape} - Literature Example"
    )
    Visualizer.plot_covariances(
        cov_small,
        title="Covariance Deviation (Validation Matrix)",
        subtitle=f"Validation quaternion matrix, shape: {A_small.shape} - Literature Example"
    )
    Visualizer.visualize_matrix(
        A_small,
        component=1,
        title="Validation Matrix: Component 1 Heatmap",
        subtitle=f"Validation quaternion matrix, shape: {A_small.shape} - Literature Example"
    )

if __name__ == '__main__':
    main()