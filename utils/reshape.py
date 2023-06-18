import numpy as np
from numba import jit


@jit(nopython=True)
def reshape_to_1d(array: np.ndarray) -> np.ndarray:
    """
    Reshape an array into a 1D array. Uses Numba for JIT compilation and
    faster execution.
    """

    # Make the array contiguous
    array = np.ascontiguousarray(array)
    return array.reshape(-1,)


@jit(nopython=True)
def reshape_to_2d(array: np.ndarray) -> np.ndarray:
    """
    Reshape an array into a 2D column array. Uses Numba for JIT compilation and
    faster execution.
    """

    # Make the array contiguous
    array = np.ascontiguousarray(array)
    return array.reshape(-1, 1)


def reshape_to_1d_array(array: np.ndarray) -> np.ndarray:
    if array.size == 0:
        raise ValueError("Input array must not be empty.")

    if array.ndim > 2 or (array.ndim == 2 and array.shape[1] != 1):
        raise ValueError("Input array must have shape (n,) or (n, 1).")

    if array.ndim == 1:
        return array

    return reshape_to_1d(array)


def reshape_to_2d_array(array: np.ndarray) -> np.ndarray:
    if array.size == 0:
        raise ValueError("Input array must not be empty.")

    if array.ndim > 2 or (array.ndim == 2 and array.shape[1] != 1):
        raise ValueError("Input array must have shape (n,) or (n, 1).")

    if array.ndim == 2:
        return array

    return reshape_to_2d(array)
