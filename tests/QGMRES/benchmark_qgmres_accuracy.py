#!/usr/bin/env python3
"""
Q-GMRES Accuracy Investigation
=============================

Detailed investigation of solution accuracy issues with LU preconditioning.
This script performs controlled tests to understand why LU preconditioning
might be affecting solution accuracy.

Outputs to validation_output/:
- accuracy_investigation_report.png
"""

import os
import sys

import matplotlib
import numpy as np
import pytest
import quaternion  # type: ignore

matplotlib.use("Agg")
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Path setup
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "quatica"))
from data_gen import create_test_matrix
from decomp import quaternion_lu
from solver import QGMRESSolver
from utils import quat_frobenius_norm, quat_hermitian, quat_matmat

plt.style.use("default")
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 20,
        "font.family": "sans-serif",
    }
)


def create_controlled_test_matrix(n: int, condition_number: float = 100.0, seed: int = 0):
    """Create a controlled test matrix with known properties."""
    np.random.seed(seed)

    # Create orthogonal matrix
    Q = create_test_matrix(n, n)
    Q_real = quaternion.as_float_array(Q).reshape(-1, 4)
    Q_orth, _ = np.linalg.qr(Q_real)
    Q = quaternion.as_quat_array(Q_orth).reshape(n, n)

    # Create diagonal with specified condition number
    eigenvals = np.logspace(0, -np.log10(condition_number), n)
    D = np.zeros((n, n), dtype=np.quaternion)
    for i in range(n):
        D[i, i] = quaternion.quaternion(eigenvals[i], 0, 0, 0)

    # Construct A = Q D Q^H
    A = quat_matmat(quat_matmat(Q, D), quat_hermitian(Q))

    # Known solution
    x_true = np.ones((n, 1), dtype=np.quaternion)
    for i in range(n):
        x_true[i, 0] = quaternion.quaternion(1.0 + 0.1 * i, 0.1 * i, 0.01 * i, 0.001 * i)

    # Right-hand side
    b = quat_matmat(A, x_true)

    return A, b, x_true, eigenvals


def analyze_lu_decomposition_accuracy(A: np.ndarray):
    """Analyze the accuracy of LU decomposition itself."""
    try:
        Pq, Lq, Uq = quaternion_lu(A)

        # Reconstruct A from LU
        A_reconstructed = quat_matmat(quat_matmat(Pq, Lq), Uq)

        # Check reconstruction error
        reconstruction_error = quat_frobenius_norm(
            A - A_reconstructed
        ) / quat_frobenius_norm(A)

        # Check conditioning of L and U
        L_norm = quat_frobenius_norm(Lq)
        U_norm = quat_frobenius_norm(Uq)

        return {
            "success": True,
            "reconstruction_error": reconstruction_error,
            "L_norm": L_norm,
            "U_norm": U_norm,
            "P": Pq,
            "L": Lq,
            "U": Uq,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "reconstruction_error": float("inf")}


def solve_with_detailed_tracking(
    A: np.ndarray, b: np.ndarray, x_true: np.ndarray, method: str
):
    """Solve with detailed error tracking at each stage."""

    result = {
        "method": method,
        "success": False,
        "original_residual": float("inf"),
        "final_residual": float("inf"),
        "solution_error": float("inf"),
        "iterations": 0,
        "solve_time": 0.0,
    }

    try:
        import time

        start_time = time.time()

        # Check initial residual (should be ||b||)
        initial_guess = np.zeros_like(b)
        initial_residual = quat_frobenius_norm(quat_matmat(A, initial_guess) - b)
        initial_residual_rel = initial_residual / quat_frobenius_norm(b)

        solver = QGMRESSolver(
            tol=1e-10, max_iter=200, verbose=False, preconditioner=method
        )
        x_sol, info = solver.solve(A, b)

        solve_time = time.time() - start_time

        # Calculate final residual (manually to be sure)
        final_residual = quat_frobenius_norm(quat_matmat(A, x_sol) - b)
        final_residual_rel = final_residual / quat_frobenius_norm(b)

        # Solution error
        solution_error = quat_frobenius_norm(x_sol - x_true) / quat_frobenius_norm(x_true)

        result.update(
            {
                "success": True,
                "original_residual": initial_residual_rel,
                "final_residual": final_residual_rel,
                "reported_residual": info.get("residual", float("inf")),
                "solution_error": solution_error,
                "iterations": info["iterations"],
                "solve_time": solve_time,
                "x_solution": x_sol,
            }
        )

    except Exception as e:
        result["error"] = str(e)

    return result


def run_accuracy_investigation():
    """Run comprehensive accuracy investigation."""

    print("🔬 Q-GMRES Accuracy Investigation")
    print("=" * 40)

    results = []
    condition_numbers = [10.0, 100.0, 1000.0, 10000.0]
    matrix_sizes = [20, 30]

    for size in matrix_sizes:
        print(f"\n📐 Matrix size: {size}×{size}")

        for cond_num in condition_numbers:
            print(f"  📊 Condition number: {cond_num:.0e}")

            # Create test matrix
            A, b, x_true, eigenvals = create_controlled_test_matrix(
                size, cond_num, seed=42
            )

            # Analyze LU decomposition quality
            lu_analysis = analyze_lu_decomposition_accuracy(A)
            print(
                f"    🔧 LU reconstruction error: {lu_analysis.get('reconstruction_error', 'N/A'):.2e}"
            )

            # Test both methods
            for method in ["none", "left_lu"]:
                print(
                    f"    {'🔄' if method == 'none' else '⚡'} Method: {method:8s}",
                    end="",
                )

                result = solve_with_detailed_tracking(A, b, x_true, method)
                result["condition_number"] = cond_num
                result["matrix_size"] = size
                result["lu_analysis"] = lu_analysis
                results.append(result)

                if result["success"]:
                    print(
                        f" ✅ {result['iterations']:3d}it, "
                        f"res={result['final_residual']:.2e}, "
                        f"err={result['solution_error']:.2e}"
                    )
                else:
                    print(" ❌ Failed")

    return results


def create_accuracy_investigation_plot(results: List, output_dir: Path):
    """Create detailed accuracy investigation plot."""

    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(3, 3, hspace=0.45, wspace=0.4, figure=fig)

    colors = {"none": "#E74C3C", "left_lu": "#3498DB"}

    # Filter successful results
    successful_results = [r for r in results if r["success"]]

    if not successful_results:
        print("❌ No successful results to analyze")
        return

    # 1. Solution error vs condition number (top-left)
    ax1 = fig.add_subplot(gs[0, 0])

    for method in ["none", "left_lu"]:
        method_results = [r for r in successful_results if r["method"] == method]
        if method_results:
            cond_nums = [r["condition_number"] for r in method_results]
            errors = [r["solution_error"] for r in method_results]

            ax1.loglog(
                cond_nums,
                errors,
                "o-",
                label=f"Q-GMRES ({method})",
                color=colors[method],
                linewidth=3,
                markersize=10,
            )

    ax1.set_xlabel("Condition Number")
    ax1.set_ylabel("Solution Error ||x - x*|| / ||x*||")
    ax1.set_title("Solution Accuracy vs Condition Number", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add ideal line (machine precision * condition number)
    cond_range = np.logspace(1, 5, 100)
    ideal_error = 1e-16 * cond_range
    ax1.plot(
        cond_range,
        ideal_error,
        "--",
        color="gray",
        alpha=0.7,
        label="Theoretical limit (ε·κ)",
    )
    ax1.legend()

    # 2. Residual accuracy comparison (top-center)
    ax2 = fig.add_subplot(gs[0, 1])

    for method in ["none", "left_lu"]:
        method_results = [r for r in successful_results if r["method"] == method]
        if method_results:
            final_residuals = [r["final_residual"] for r in method_results]
            reported_residuals = [r["reported_residual"] for r in method_results]

            ax2.loglog(
                reported_residuals,
                final_residuals,
                "o",
                label=f"Q-GMRES ({method})",
                color=colors[method],
                markersize=8,
                alpha=0.7,
            )

    # Add perfect agreement line
    res_range = np.logspace(-12, -6, 100)
    ax2.plot(
        res_range, res_range, "--", color="black", alpha=0.5, label="Perfect agreement"
    )

    ax2.set_xlabel("Reported Residual")
    ax2.set_ylabel("True Residual ||Ax - b|| / ||b||")
    ax2.set_title("Residual Reporting Accuracy", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. LU decomposition quality (top-right)
    ax3 = fig.add_subplot(gs[0, 2])

    lu_errors = []
    cond_nums_lu = []

    for r in successful_results:
        if r["method"] == "left_lu" and "lu_analysis" in r:
            lu_analysis = r["lu_analysis"]
            if lu_analysis["success"]:
                lu_errors.append(lu_analysis["reconstruction_error"])
                cond_nums_lu.append(r["condition_number"])

    if lu_errors:
        ax3.loglog(
            cond_nums_lu,
            lu_errors,
            "ro-",
            linewidth=2,
            markersize=8,
            label="LU reconstruction error",
        )

        # Add machine precision line
        ax3.axhline(
            y=1e-16, color="gray", linestyle="--", alpha=0.7, label="Machine precision"
        )

    ax3.set_xlabel("Condition Number")
    ax3.set_ylabel("LU Reconstruction Error")
    ax3.set_title("LU Decomposition Quality", fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Solution error ratio (middle-left)
    ax4 = fig.add_subplot(gs[1, 0])

    # Calculate error ratios (preconditioned / baseline)
    error_ratios = []
    cond_nums_ratio = []

    for cond_num in set(r["condition_number"] for r in successful_results):
        baseline_results = [
            r
            for r in successful_results
            if r["method"] == "none" and r["condition_number"] == cond_num
        ]
        precond_results = [
            r
            for r in successful_results
            if r["method"] == "left_lu" and r["condition_number"] == cond_num
        ]

        if baseline_results and precond_results:
            baseline_error = np.mean([r["solution_error"] for r in baseline_results])
            precond_error = np.mean([r["solution_error"] for r in precond_results])

            if baseline_error > 0:
                ratio = precond_error / baseline_error
                error_ratios.append(ratio)
                cond_nums_ratio.append(cond_num)

    if error_ratios:
        ax4.semilogx(
            cond_nums_ratio,
            error_ratios,
            "go-",
            linewidth=2,
            markersize=8,
            label="Error ratio (LU/baseline)",
        )
        ax4.axhline(y=1.0, color="black", linestyle="-", alpha=0.5, label="No difference")
        ax4.axhline(y=0.1, color="red", linestyle="--", alpha=0.7, label="10× better")
        ax4.axhline(y=10.0, color="red", linestyle="--", alpha=0.7, label="10× worse")

    ax4.set_xlabel("Condition Number")
    ax4.set_ylabel("Error Ratio (Preconditioned / Baseline)")
    ax4.set_title("Relative Solution Accuracy", fontweight="bold")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale("log")

    # 5. Iteration count comparison (middle-center)
    ax5 = fig.add_subplot(gs[1, 1])

    for method in ["none", "left_lu"]:
        method_results = [r for r in successful_results if r["method"] == method]
        if method_results:
            cond_nums = [r["condition_number"] for r in method_results]
            iterations = [r["iterations"] for r in method_results]

            ax5.semilogx(
                cond_nums,
                iterations,
                "o-",
                label=f"Q-GMRES ({method})",
                color=colors[method],
                linewidth=2,
                markersize=8,
            )

    ax5.set_xlabel("Condition Number")
    ax5.set_ylabel("Iterations to Convergence")
    ax5.set_title("Convergence Speed vs Conditioning", fontweight="bold")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Detailed error analysis table (middle-right and bottom)
    ax6 = fig.add_subplot(gs[1:, 2])
    ax6.axis("off")

    # Summarize findings
    analysis_text = []

    # Calculate statistics
    baseline_errors = [
        r["solution_error"] for r in successful_results if r["method"] == "none"
    ]
    precond_errors = [
        r["solution_error"] for r in successful_results if r["method"] == "left_lu"
    ]

    if baseline_errors and precond_errors:
        baseline_median = np.median(baseline_errors)
        precond_median = np.median(precond_errors)
        error_ratio_median = (
            precond_median / baseline_median if baseline_median > 0 else float("inf")
        )

        # Count cases where preconditioning is worse
        worse_cases = sum(1 for r in error_ratios if r > 2.0)
        total_cases = len(error_ratios)

        analysis_text = [
            "🔍 ACCURACY INVESTIGATION FINDINGS",
            "━" * 50,
            "",
            "📊 SOLUTION ERROR STATISTICS:",
            f"   • Median baseline error:     {baseline_median:.2e}",
            f"   • Median preconditioned error: {precond_median:.2e}",
            f"   • Error ratio (LU/baseline):   {error_ratio_median:.2f}",
            "",
            "🚨 ACCURACY DEGRADATION ANALYSIS:",
            f"   • Cases with >2× worse accuracy: {worse_cases}/{total_cases}",
            f"   • Degradation rate:             {worse_cases / total_cases * 100:.1f}%",
            "",
            "🔧 LU DECOMPOSITION QUALITY:",
            f"   • Average LU reconstruction error: {np.mean(lu_errors):.2e}"
            if lu_errors
            else "   • LU analysis failed",
            "",
            "💡 POTENTIAL CAUSES:",
            "   • Round-off error accumulation in LU",
            "   • Loss of structure in preconditioning",
            "   • Numerical instability in triangular solves",
            "   • Condition number amplification",
            "",
            "🎯 RECOMMENDATIONS:",
            "   • Use preconditioning selectively",
            "   • Monitor solution accuracy carefully",
            "   • Consider iterative refinement",
            "   • Evaluate alternative preconditioners",
        ]

    # Display analysis
    full_text = "\\n".join(analysis_text)
    ax6.text(
        0.05,
        0.95,
        full_text,
        transform=ax6.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF3CD", alpha=0.9),
    )

    # 7. Solution vector comparison (bottom-left span)
    ax7 = fig.add_subplot(gs[2, :2])

    # Find a representative case for detailed solution comparison
    representative_case = None
    for r in successful_results:
        if (
            r["method"] == "left_lu"
            and r["condition_number"] == 1000.0
            and "x_solution" in r
            and r["solution_error"] > 1e-10
        ):
            representative_case = r
            break

    if representative_case:
        # Get corresponding baseline solution
        baseline_case = None
        for r in successful_results:
            if (
                r["method"] == "none"
                and r["condition_number"] == representative_case["condition_number"]
                and r["matrix_size"] == representative_case["matrix_size"]
            ):
                baseline_case = r
                break

        if baseline_case and "x_solution" in baseline_case:
            x_baseline = baseline_case["x_solution"].flatten()
            x_precond = representative_case["x_solution"].flatten()

            # Plot real parts of solutions
            indices = range(len(x_baseline))
            ax7.plot(
                indices,
                [x.real for x in x_baseline],
                "o-",
                label="Baseline solution (real part)",
                color=colors["none"],
                linewidth=2,
            )
            ax7.plot(
                indices,
                [x.real for x in x_precond],
                "s-",
                label="Preconditioned solution (real part)",
                color=colors["left_lu"],
                linewidth=2,
            )

            ax7.set_xlabel("Solution Component Index")
            ax7.set_ylabel("Real Part of Solution")
            ax7.set_title(
                f"Solution Comparison (κ={representative_case['condition_number']:.0e})",
                fontweight="bold",
            )
            ax7.legend()
            ax7.grid(True, alpha=0.3)
    else:
        ax7.text(
            0.5,
            0.5,
            "No suitable case found\nfor solution comparison",
            ha="center",
            va="center",
            transform=ax7.transAxes,
            fontsize=12,
        )

    # Overall title
    fig.suptitle(
        "Q-GMRES LU Preconditioning: Accuracy Investigation Report",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Display the plot first, then save
    plt.tight_layout()
    plt.show()

    plt.savefig(
        output_dir / "qgmres_accuracy_investigation.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.5,
    )
    plt.close()


@pytest.mark.parametrize("dummy", [True])
def test_qgmres_accuracy_investigation(dummy):
    """Investigate Q-GMRES accuracy issues with LU preconditioning."""

    output_dir = Path(__file__).parent.parent.parent / "validation_output"
    output_dir.mkdir(exist_ok=True)

    # Run investigation
    results = run_accuracy_investigation()

    print("\n📈 Creating accuracy investigation report...")

    # Create detailed plot
    create_accuracy_investigation_plot(results, output_dir)
    print("✅ Saved: qgmres_accuracy_investigation.png")

    # Analyze results
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        baseline_errors = [
            r["solution_error"] for r in successful_results if r["method"] == "none"
        ]
        precond_errors = [
            r["solution_error"] for r in successful_results if r["method"] == "left_lu"
        ]

        if baseline_errors and precond_errors:
            error_ratio = np.median(precond_errors) / np.median(baseline_errors)
            print("\n📊 Accuracy Analysis:")
            print(f"   • Median error ratio (LU/baseline): {error_ratio:.2f}")
            if error_ratio > 2.0:
                print("   ⚠️  LU preconditioning degrades accuracy significantly!")
            elif error_ratio > 1.1:
                print("   ⚠️  LU preconditioning slightly degrades accuracy")
            else:
                print("   ✅ LU preconditioning maintains accuracy")

    print("\n🎉 Accuracy investigation completed!")
    print(f"📁 Report saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    test_qgmres_accuracy_investigation(True)
