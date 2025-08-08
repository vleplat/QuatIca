#!/usr/bin/env python3
"""
Benchmark and visualize Newton–Schulz vs Higher-Order (third-order) Newton–Schulz
for quaternion pseudoinverse.

Produces plots of residuals (E1) vs iterations and CPU time bars.
"""
import sys, os, time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core'))
import numpy as np
import quaternion  # type: ignore
import matplotlib.pyplot as plt

from solver import NewtonSchulzPseudoinverse, HigherOrderNewtonSchulzPseudoinverse


def random_quaternion_matrix(m: int, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((m, n, 4))
    return quaternion.as_quat_array(data)


def run_case(m: int, n: int, max_iter: int = 80):
    A = random_quaternion_matrix(m, n, seed=123)

    # NS damped (gamma=0.5)
    ns_damped = NewtonSchulzPseudoinverse(gamma=0.5, max_iter=max_iter, tol=1e-12, verbose=False, compute_residuals=True)
    t0 = time.time(); X_ns_d, res_ns_d, times_ns_d = ns_damped.compute(A); t_ns_d = time.time() - t0

    # NS baseline (gamma=1)
    ns_unit = NewtonSchulzPseudoinverse(gamma=1.0, max_iter=max_iter, tol=1e-12, verbose=False, compute_residuals=True)
    t1 = time.time(); X_ns_u, res_ns_u, times_ns_u = ns_unit.compute(A); t_ns_u = time.time() - t1

    hon = HigherOrderNewtonSchulzPseudoinverse(max_iter=max_iter, tol=0.0, verbose=False)
    t2 = time.time(); X_hon, res_hon, times_hon = hon.compute(A); t_hon = time.time() - t2

    return (res_ns_d['AXA-A'], t_ns_d), (res_ns_u['AXA-A'], t_ns_u), (res_hon['AXA-A'], t_hon)


def visualize(results, sizes, out_dir='output_figures'):
    import os
    # Always save in project root output_figures, regardless of cwd
    script_dir = os.path.dirname(__file__)
    root_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
    out_fig_dir = os.path.join(root_dir, 'output_figures')
    os.makedirs(out_fig_dir, exist_ok=True)
    for (m, n), (nsd_res, nsd_time), (nsu_res, nsu_time), (hon_res, hon_time) in zip(sizes, results[0], results[1], results[2]):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        # Residuals vs iterations
        ax[0].semilogy(nsd_res, label='NS (gamma=0.5)')
        ax[0].semilogy(nsu_res, label='NS (gamma=1)')
        ax[0].semilogy(hon_res, label='Higher-Order NS (3rd)')
        ax[0].set_title(f'E1 residual vs iterations ({m}x{n})')
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('E1 = ||A X A - A||_F')
        ax[0].grid(True, ls=':')
        ax[0].legend()
        # CPU time bars
        bars = ['NS 0.5', 'NS 1.0', 'HON']
        times = [nsd_time, nsu_time, hon_time]
        ax[1].bar(bars, times, color=['tab:blue', 'tab:cyan', 'tab:orange'])
        ax[1].set_title('CPU time (s)')
        for i, v in enumerate(times):
            ax[1].text(i, v, f"{v:.2f}s", ha='center', va='bottom')
        fig.suptitle('Pseudoinverse: NS vs Higher-Order NS')
        fig.tight_layout(rect=[0,0,1,0.95])
        path_hd = os.path.join(out_fig_dir, f'ns_vs_hon_{m}x{n}.png')
        fig.savefig(path_hd, dpi=300)
        # Display the figure for interactive runs
        try:
            plt.show()
        except Exception:
            pass
        plt.close(fig)
        print(f'Saved {path_hd}')


if __name__ == '__main__':
    sizes = [(60, 40), (100, 60)]
    nsd_results = []
    nsu_results = []
    hon_results = []
    for m, n in sizes:
        (nsd_res, nsd_time), (nsu_res, nsu_time), (hon_res, hon_time) = run_case(m, n, max_iter=80)
        nsd_results.append((nsd_res, nsd_time))
        nsu_results.append((nsu_res, nsu_time))
        hon_results.append((hon_res, hon_time))
    visualize((nsd_results, nsu_results, hon_results), sizes)


