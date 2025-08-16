#!/usr/bin/env python3
"""
Unit tests for basic quaternion tensor utilities: norms and mode-n unfolding/folding.
"""

import os
import sys

import numpy as np
import quaternion  # type: ignore

# Path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "quatica"))
from tensor import tensor_entrywise_abs, tensor_fold, tensor_frobenius_norm, tensor_unfold


def random_quat_tensor(shape, seed=0):
    rng = np.random.default_rng(seed)
    comp = rng.standard_normal((*shape, 4))
    return quaternion.as_quat_array(comp)


def test_tensor_norms_and_unfold_fold():
    T = random_quat_tensor((4, 3, 5), seed=123)

    # Frobenius norm matches component-space sum of squares
    Tf = quaternion.as_float_array(T)
    F_expected = float(np.sqrt(np.sum(Tf**2)))
    F = tensor_frobenius_norm(T)
    assert np.isclose(F, F_expected, rtol=0, atol=1e-12)

    # Entrywise abs has same shape and nonnegative
    A = tensor_entrywise_abs(T)
    assert A.shape == T.shape
    assert np.all(A >= 0)

    # Unfold/fold roundtrip for all modes
    I, J, K = T.shape
    for mode in (0, 1, 2):
        M = tensor_unfold(T, mode)
        T_back = tensor_fold(M, mode, (I, J, K))
        # Exact equality should hold (no arithmetic changes)
        assert T_back.shape == T.shape
        assert np.all(quaternion.as_float_array(T_back) == quaternion.as_float_array(T))
        assert np.all(quaternion.as_float_array(T_back) == quaternion.as_float_array(T))
