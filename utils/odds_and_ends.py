import logging
from sklearn.utils.validation import check_consistent_length
from typing import Tuple
import numpy as np
from numba import jit


@jit(nopython=True)
def reshape_to_1d_array(array: np.ndarray) -> np.ndarray:
    """
    Reshape an array into a 1D array. Uses Numba for JIT compilation and
    faster execution.
    """
    # Make the array contiguous
    array = np.ascontiguousarray(array)
    return array.reshape(-1,)


@jit(nopython=True)
def reshape_to_2d_array(array: np.ndarray) -> np.ndarray:
    """
    Reshape an array into a 2D column array. Uses Numba for JIT compilation and
    faster execution.
    """
    # Make the array contiguous
    array = np.ascontiguousarray(array)
    return array.reshape(-1, 1)


@jit(nopython=True)
def _numba_resample_and_oob(idx: np.ndarray, frac_samples: float, seed: int = 0, replace: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast resampling function using Numba, also returns out-of-bag samples.
    This is an internal function, use resample_and_oob instead.

    Parameters
    ----------
    idx : np.ndarray
        1D array of indices to be sampled.
    frac_samples : float
        Fraction of samples to draw.
    seed : int
        Seed for the random number generator.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        1D array of sampled indices, and 1D array of out-of-bag indices.
    """
    n_samples = int(frac_samples * len(idx))
    np.random.seed(seed)

    resampled_idx = np.random.choice(idx, size=n_samples, replace=replace)

    oob_mask = np.zeros_like(idx, dtype=np.bool_)
    for i in range(len(idx)):
        oob_mask[i] = idx[i] not in resampled_idx
    oob_idx = idx[oob_mask]
    return resampled_idx, oob_idx


def resample_data(*arrays: np.ndarray, frac_samples: float = 1, seed: int = 0, replace: bool = True) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
    """
    Perform resampling of data and also return out-of-bag samples.

    Parameters
    ----------
    arrays : np.ndarray
        Data arrays to resample.
    frac_samples : float
        Fraction of samples to draw for resampling. Must be between 0 and 1 inclusive.

    Returns
    -------
    Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]
        Tuple of resampled data arrays, and tuple of out-of-bag data arrays.
    """
    # Check if all arrays have the same number of samples
    check_consistent_length(*arrays)

    if not (0 < frac_samples <= 1):
        err_msg = 'frac_samples must be greater than 0 and less than or equal to 1.'
        logging.error(err_msg)
        raise ValueError(err_msg)

    if len(arrays[0]) < 2:
        err_msg = 'At least two samples are required for resampling.'
        logging.error(err_msg)
        raise ValueError(err_msg)

    logging.info(f'Resampling to {frac_samples} percentage samples.')

    # Check for unique elements and set replace accordingly
    idx = np.arange(len(arrays[0]))
    unique_elements = len(set(arrays[0]))
    print(f"input_array: {arrays[0]}")
    print(f"idx: {idx}")
    print(f"unique_elements: {unique_elements}")

    if unique_elements == 1:
        if frac_samples < 1:
            # If all elements are the same, manually calculate in-bootstrap and out-of-bootstrap indices
            n_resampled = int(frac_samples * len(idx))
            resampled_idx = idx[:n_resampled]
            oob_idx = idx[n_resampled:]
            resampled_arrays = tuple(arr[resampled_idx] for arr in arrays)
            oob_arrays = tuple(arr[oob_idx] for arr in arrays)
        else:
            #  return input arrays as resampled data and empty arrays for out-of-bag data
            resampled_arrays = tuple(arr for arr in arrays)
            oob_arrays = tuple(np.array([], dtype=arr.dtype) for arr in arrays)
    else:
        idx_res, idx_oob = _numba_resample_and_oob(
            idx, frac_samples, seed, replace)
        resampled_arrays = tuple(arr[idx_res] for arr in arrays)
        oob_arrays = tuple(arr[idx_oob] for arr in arrays)

    logging.info(
        f'Resampled {len(resampled_arrays)} arrays, obtained {len(oob_arrays)} out-of-bag arrays.')

    return resampled_arrays, oob_arrays
