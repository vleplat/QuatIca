from typing import Tuple

import numpy as np
import quaternion  # type: ignore

__all__ = [
    "tensor_frobenius_norm",
    "tensor_entrywise_abs",
    "tensor_unfold",
    "tensor_fold",
]


def tensor_frobenius_norm(T: np.ndarray) -> float:
    """
    Frobenius-like norm of a quaternion tensor of arbitrary order.

    Computes ||T||_F = sqrt(sum of squares) over all four quaternion components
    (w, x, y, z) across all tensor entries.

    Parameters:
    -----------
    T : np.ndarray
        Quaternion tensor of arbitrary shape

    Returns:
    --------
    float
        Frobenius norm of the tensor

    Notes:
    ------
    This extends the matrix Frobenius norm to quaternion tensors by treating
    each quaternion entry as a 4-component vector and summing all squared
    components across the entire tensor.
    """
    Tf = quaternion.as_float_array(T)
    return float(np.sqrt(np.sum(Tf**2)))


def tensor_entrywise_abs(T: np.ndarray) -> np.ndarray:
    """
    Return entrywise quaternion magnitudes |T| for a quaternion tensor.

    Computes the modulus |q| = sqrt(w² + x² + y² + z²) for each quaternion
    entry in the tensor, returning a real-valued tensor.

    Parameters:
    -----------
    T : np.ndarray
        Quaternion tensor of arbitrary shape

    Returns:
    --------
    np.ndarray
        Real tensor with same shape as T containing element-wise magnitudes

    Notes:
    ------
    The output is a real ndarray with the same shape as the input tensor,
    where each entry contains the magnitude of the corresponding quaternion.
    """
    Tf = quaternion.as_float_array(T)
    # Last axis are the 4 components (w,x,y,z)
    return np.sqrt(np.sum(Tf**2, axis=-1))


def tensor_unfold(T: np.ndarray, mode: int) -> np.ndarray:
    """
    Mode-n unfolding (matricization) for an order-3 quaternion tensor.

    Converts a 3rd-order tensor into a matrix by arranging fibers along a
    specified mode. This is a fundamental operation in tensor decompositions.

    Parameters:
    -----------
    T : np.ndarray
        Quaternion tensor of shape (I, J, K)
    mode : int
        Unfolding mode (0, 1, or 2)

    Returns:
    --------
    np.ndarray
        Unfolded matrix of shape (dims[mode], prod(other dims))

    Raises:
    -------
    ValueError
        If T is not an order-3 quaternion tensor or mode is invalid

    Notes:
    ------
    - Mode 0: unfolding along first dimension → shape (I, J*K)
    - Mode 1: unfolding along second dimension → shape (J, I*K)
    - Mode 2: unfolding along third dimension → shape (K, I*J)
    """
    if T.ndim != 3 or T.dtype != np.quaternion:
        raise ValueError("tensor_unfold: T must be an order-3 quaternion tensor")
    I, J, K = T.shape
    if mode == 0:
        return T.reshape(I, J * K)
    elif mode == 1:
        # bring axis 1 front: (J, I, K) -> (J, I*K)
        return np.transpose(T, (1, 0, 2)).reshape(J, I * K)
    elif mode == 2:
        # bring axis 2 front: (K, I, J) -> (K, I*J)
        return np.transpose(T, (2, 0, 1)).reshape(K, I * J)
    else:
        raise ValueError("mode must be 0, 1, or 2")


def tensor_fold(M: np.ndarray, mode: int, shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Inverse of mode-n unfolding for an order-3 quaternion tensor.

    Converts an unfolded matrix back into its original tensor form by reversing
    the matricization operation. This is the inverse of tensor_unfold.

    Parameters:
    -----------
    M : np.ndarray
        Unfolded quaternion matrix
    mode : int
        Folding mode (0, 1, or 2) corresponding to the original unfolding mode
    shape : tuple of int
        Target tensor shape (I, J, K)

    Returns:
    --------
    np.ndarray
        Folded quaternion tensor of shape `shape`

    Raises:
    -------
    ValueError
        If the matrix shape is incompatible with the folding mode and target shape

    Notes:
    ------
    The mode must match the mode used in the original tensor_unfold operation
    for the folding to correctly reconstruct the tensor structure.
    """
    I, J, K = shape
    if mode == 0:
        if M.shape != (I, J * K):
            raise ValueError("fold shape mismatch for mode 0")
        return M.reshape(I, J, K)
    elif mode == 1:
        if M.shape != (J, I * K):
            raise ValueError("fold shape mismatch for mode 1")
        return np.transpose(M.reshape(J, I, K), (1, 0, 2))
    elif mode == 2:
        if M.shape != (K, I * J):
            raise ValueError("fold shape mismatch for mode 2")
        return np.transpose(M.reshape(K, I, J), (1, 2, 0))
    else:
        raise ValueError("mode must be 0, 1, or 2")
