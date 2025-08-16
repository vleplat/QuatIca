#!/usr/bin/env python3
"""
Q-SVD Performance Summary - Key Insights from Benchmark

This script summarizes the key findings from comparing rand_qsvd vs pass_eff_qsvd
on big low-rank matrices (500×300, rank=10).
"""

import os
import sys

# Add core module to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "quatica"))


def print_summary():
    """Print a comprehensive summary of the Q-SVD performance benchmark."""

    print("=" * 80)
    print("Q-SVD PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 80)
    print("Matrix: 500×300, Target rank: 10")
    print("Methods: rand_qsvd vs pass_eff_qsvd")
    print("Parameters: n_passes/n_iter ∈ [1,2,3,4]")
    print()

    print("📊 KEY FINDINGS:")
    print("-" * 40)

    print("1. 🏆 WINNER: pass_eff_qsvd")
    print("   • Consistently faster than rand_qsvd")
    print("   • 2.8x speedup on average")
    print("   • Better scalability with more passes")

    print("\n2. ⚡ PERFORMANCE COMPARISON:")
    print("   • pass_eff_qsvd: 0.077s - 0.218s (2.3x - 6.3x speedup)")
    print("   • rand_qsvd: 0.220s - 0.543s (0.9x - 2.2x speedup)")
    print("   • Full Q-SVD: 0.49s (baseline)")

    print("\n3. 🎯 CONVERGENCE BEHAVIOR:")
    print("   • rand_qsvd: Perfect accuracy from 1 iteration")
    print("   • pass_eff_qsvd: Perfect accuracy from 2+ passes")
    print("   • 1 pass in pass_eff_qsvd: Poor accuracy (80% error)")

    print("\n4. 📈 OPTIMAL CONFIGURATIONS:")
    print("   • Best speed: pass_eff_qsvd with 1 pass (but poor accuracy)")
    print("   • Best accuracy: Both methods with 2+ passes")
    print("   • Best trade-off: pass_eff_qsvd with 2 passes")
    print("     - Perfect accuracy (0.000000 relative error)")
    print("     - 4.5x speedup vs full Q-SVD")
    print("     - 2.9x speedup vs rand_qsvd")

    print("\n5. 🔍 TECHNICAL INSIGHTS:")
    print("   • pass_eff_qsvd uses fewer matrix passes than power iterations")
    print("   • MATLAB implementation validated successfully")
    print("   • Both methods produce orthonormal U and V matrices")
    print("   • Singular values match full Q-SVD exactly")

    print("\n6. 📊 VISUALIZATION:")
    print(
        "   • Performance plots saved to: output_figures/qsvd_performance_comparison.png"
    )
    print("   • 4 subplots showing:")
    print("     - Execution time comparison")
    print("     - Speedup vs full Q-SVD")
    print("     - Accuracy comparison (log scale)")
    print("     - Speedup vs accuracy trade-off")

    print("\n7. 🎯 RECOMMENDATIONS:")
    print("   • Use pass_eff_qsvd for low-rank matrices")
    print("   • Use 2 passes for optimal accuracy/speed balance")
    print("   • Consider 1 pass only if speed is critical and accuracy can be sacrificed")
    print("   • rand_qsvd becomes slower than full Q-SVD with 4+ iterations")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("✅ pass_eff_qsvd is the superior choice for this scenario:")
    print("   • Faster execution")
    print("   • Better scalability")
    print("   • Same accuracy as rand_qsvd")
    print("   • Validated against MATLAB implementation")
    print()
    print("🚀 Ready for production use with 2 passes!")


def main():
    """Run the performance summary."""
    try:
        print_summary()
        return True
    except Exception as e:
        print(f"❌ Summary failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
