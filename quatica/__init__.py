from .data_gen import *
from .solver import (
    HigherOrderNewtonSchulzPseudoinverse,
    NewtonSchulzPseudoinverse,
    QGMRESSolver,
    maxvol_submatrix_quat,
)
from .optiQ import (
    BarrierParams,
    admm_barrier_fixed_mu,
    build_central_mu_instance,
    build_singleton_feasible,
    build_singleton_feasible_canonical,
    save_admm_history_plot,
    solve_barrier,
    solve_pd_mehrotra,
    # New preferred OptiQ solver names (log-det barrier Newton / barrier path)
    solve_logdet_barrier_newton,
    solve_logdet_barrier_path,
    solve_barrier_path,
)
from .utils import *
from .visualization import Visualizer

__all__ = [
    "NewtonSchulzPseudoinverse",
    "HigherOrderNewtonSchulzPseudoinverse",
    "QGMRESSolver",
    "maxvol_submatrix_quat",
    # OptiQ (optimization / SDP)
    "BarrierParams",
    "solve_barrier",
    "solve_pd_mehrotra",
    "solve_logdet_barrier_newton",
    "solve_logdet_barrier_path",
    "solve_barrier_path",
    "admm_barrier_fixed_mu",
    "save_admm_history_plot",
    "build_central_mu_instance",
    "build_singleton_feasible_canonical",
    "build_singleton_feasible",
    "Visualizer",
    # Kernel/null space functions (exported via utils import *)
    "quat_null_space",
    "quat_null_right",
    "quat_null_left",
    "quat_kernel",
]