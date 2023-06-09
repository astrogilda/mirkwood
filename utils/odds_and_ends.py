from sklearn.utils.validation import check_consistent_length
from typing import List
import numpy as np
from numba import jit


@jit(nopython=True)
def reshape_array(array: np.ndarray) -> np.ndarray:
    """
    Reshape an array into a 1D array. Uses Numba for JIT compilation and
    faster execution.
    """
    # Make the array contiguous
    array = np.ascontiguousarray(array)
    return array.reshape(-1,)


@jit(nopython=True)
def numba_resample(idx: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Fast resampling function using Numba.

    Parameters
    ----------
    idx : np.ndarray
        1D array of indices to be sampled.
    n_samples : int
        Number of samples to draw.

    Returns
    -------
    np.ndarray
        1D array of sampled indices.
    """
    return np.random.choice(idx, size=n_samples, replace=True)


def resample_data(self, *arrays: np.ndarray) -> List[np.ndarray]:
    """
    Perform resampling of data.

    Parameters
    ----------
    arrays : np.ndarray
        Data arrays to resample.

    Returns
    -------
    List[np.ndarray]
        List of resampled data arrays.
    """
    # Check if all arrays have the same number of samples
    check_consistent_length(*arrays)

    n_samples = int(self.frac_samples_best * len(arrays[0]))
    idx_res = numba_resample(np.arange(len(arrays[0])), n_samples)

    resampled_arrays = [arr[idx_res] for arr in arrays]
    return resampled_arrays
