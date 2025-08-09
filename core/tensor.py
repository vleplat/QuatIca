import numpy as np
import quaternion  # type: ignore
from typing import Tuple

__all__ = [
    "tensor_frobenius_norm",
    "tensor_entrywise_abs",
    "tensor_unfold",
    "tensor_fold",
]


def tensor_frobenius_norm(T: np.ndarray) -> float:
    """Frobenius-like norm of a quaternion tensor of arbitrary order.

    Computes sqrt(sum of squares) over all four components across all entries.
    """
    Tf = quaternion.as_float_array(T)
    return float(np.sqrt(np.sum(Tf ** 2)))


def tensor_entrywise_abs(T: np.ndarray) -> np.ndarray:
    """Return entrywise quaternion magnitudes |T| for a quaternion tensor.

    Output is a real ndarray with the same shape as T.
    """
    Tf = quaternion.as_float_array(T)
    # Last axis are the 4 components (w,x,y,z)
    return np.sqrt(np.sum(Tf ** 2, axis=-1))


def tensor_unfold(T: np.ndarray, mode: int) -> np.ndarray:
    """Mode-n unfolding (matricization) for an order-3 quaternion tensor.

    Parameters
    ----------
    T : np.ndarray (quaternion)
        Quaternion tensor of shape (I, J, K)
    mode : int
        Unfolding mode (0, 1, or 2)

    Returns
    -------
    np.ndarray (quaternion)
        Unfolded matrix of shape (dims[mode], prod(other dims))
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
    """Inverse of mode-n unfolding for an order-3 quaternion tensor.

    Parameters
    ----------
    M : np.ndarray (quaternion)
        Unfolded matrix
    mode : int
        Folding mode (0, 1, or 2)
    shape : (I, J, K)
        Target tensor shape

    Returns
    -------
    np.ndarray (quaternion)
        Folded quaternion tensor of shape `shape`.
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
