"""
Visualization package for QuatIca

This package contains scripts for creating convincing visualizations
of quaternion matrix decomposition results and validation analyses.
"""

from .qsvd_reconstruction_analysis import (
    plot_reconstruction_error_vs_rank,
    plot_relative_error_summary,
    generate_validation_report
)

__all__ = [
    'plot_reconstruction_error_vs_rank',
    'plot_relative_error_summary', 
    'generate_validation_report'
] 