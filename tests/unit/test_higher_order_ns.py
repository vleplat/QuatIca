#!/usr/bin/env python3
import os
import sys

# Match other tests: add project root and core directory so core modules using 'from utils' resolve
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "quatica"))
import time
import unittest

import numpy as np
import quaternion  # type: ignore
from solver import HigherOrderNewtonSchulzPseudoinverse, NewtonSchulzPseudoinverse


def random_quaternion_matrix(m: int, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((m, n, 4))
    return quaternion.as_quat_array(data)


class TestHigherOrderNS(unittest.TestCase):
    def test_compare_residuals_and_time(self):
        sizes = [(40, 30), (60, 40), (80, 50)]

        for m, n in sizes:
            A = random_quaternion_matrix(m, n, seed=123)

            # Baseline NS (damped)
            ns = NewtonSchulzPseudoinverse(
                gamma=0.5, max_iter=60, tol=1e-10, verbose=False, compute_residuals=True
            )
            t0 = time.time()
            X_ns, res_ns, times_ns = ns.compute(A)
            t_ns = time.time() - t0

            # Higher-order NS (third-order)
            hon = HigherOrderNewtonSchulzPseudoinverse(
                max_iter=60, tol=0.0, verbose=False
            )
            t1 = time.time()
            X_hon, res_hon, times_hon = hon.compute(A)
            t_hon = time.time() - t1

            # Compare final E1 residuals and times (no strict assertion; just sanity)
            e1_ns = res_ns["AXA-A"][-1] if res_ns["AXA-A"] else np.inf
            e1_hon = res_hon["AXA-A"][-1] if res_hon["AXA-A"] else np.inf

            self.assertTrue(np.isfinite(e1_ns))
            self.assertTrue(np.isfinite(e1_hon))
            # Both should have decreased from their initial values over iterations
            if len(res_ns["AXA-A"]) >= 2:
                self.assertLess(res_ns["AXA-A"][-1], res_ns["AXA-A"][0])
            if len(res_hon["AXA-A"]) >= 2:
                self.assertLess(res_hon["AXA-A"][-1], res_hon["AXA-A"][0])

            # Print quick summary for visibility (not required for assertions)
            print(
                f"Size {m}x{n}: NS E1={e1_ns:.2e} time={t_ns:.2f}s | HON E1={e1_hon:.2e} time={t_hon:.2f}s"
            )


if __name__ == "__main__":
    unittest.main()
    unittest.main()
