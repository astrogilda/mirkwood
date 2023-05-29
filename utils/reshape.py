import numpy as np
from numba import jit


@jit(nopython=True)
def reshape_array(array: np.ndarray) -> np.ndarray:
    """
    Reshape an array into a 1D array. Uses Numba for JIT compilation and
    faster execution.
    """
    return array.reshape(-1,)
