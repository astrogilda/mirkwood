from pydantic import BaseModel, validator
from typing import List, Tuple
from numba import jit
import numpy as np
from pydantic import BaseModel, validator, confloat, Field, conint
from sklearn.utils import check_consistent_length


@jit(nopython=True)
def _numba_resample_and_oob(idx: np.ndarray, frac_samples: float, seed: int, replace: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast resampling function using Numba, also returns out-of-bag samples.

    Parameters
    ----------
    idx : np.ndarray
        1D array of indices to be sampled.
    frac_samples : float
        Fraction of samples to draw.
    seed : int
        Seed for the random number generator.
    replace : bool
        Whether or not to allow sampling of the same index more than once.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        1D array of sampled indices, and 1D array of out-of-bag indices.
    """
    n_samples = int(frac_samples * len(idx))

    if n_samples == 0:
        raise ValueError(
            "Fraction of samples is too small, resulting in zero samples. Increase 'frac_samples'.")

    np.random.seed(seed)
    resampled_idx = np.random.choice(idx, size=n_samples, replace=replace)

    # Create an array to hold the count of occurrences for each index in resampled_idx
    counts = np.zeros_like(idx)
    for i in range(len(resampled_idx)):
        counts[resampled_idx[i]] += 1

    # Indices that are not in resampled_idx will have count = 0
    oob_mask = counts == 0

    if replace and frac_samples == 1:
        oob_idx = np.empty(0, dtype=idx.dtype)
    else:
        oob_idx = idx[oob_mask]

    return resampled_idx, oob_idx


class ResamplerConfig(BaseModel):
    # TODO: explain why "strict" was set to True
    frac_samples: confloat(gt=0, le=1, strict=True) = Field(
        default=1, description='Fraction of samples to draw for resampling.')
    seed: conint(ge=0, lt=2**32) = Field(
        default=1, description='Seed for the random number generator.')
    replace: bool = True


class Resampler:
    def __init__(self, params: ResamplerConfig):
        self.params = params
        if not isinstance(self.params, ResamplerConfig):
            raise ValueError(
                f"Expected params to be a ResamplingConfig, but got {type(self.params)}")

    def resample_data(self, arrays: List[np.ndarray]) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
        """
        Perform resampling of data and also return out-of-bag samples.

        Parameters
        ----------
        arrays : List[np.ndarray]
            Data arrays to resample.

        Returns
        -------
        Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]
            Tuple of resampled data arrays, tuple of out-of-bag data arrays, tuple of in-bag indices, tuple of out-of-bag indices.
        """
        check_consistent_length(*arrays)

        data_size = len(arrays[0])

        if data_size < 2:
            raise ValueError(
                "Resampling would be nonsensical with fewer than two elements per array.")

        unique_elements_counts = [len(np.unique(arr)) for arr in arrays]
        if all(count == 1 for count in unique_elements_counts):
            raise ValueError(
                "Resampling would be nonsensical when all arrays have only one unique element.")

        min_frac_samples = 1 / data_size
        if self.params.frac_samples < min_frac_samples:
            raise ValueError(
                f"Given 'frac_samples' of {self.params.frac_samples} and data size of {data_size}, resampling results in fewer than one sample. Increase 'frac_samples'.")

        idx = np.arange(len(arrays[0]))
        idx_ib, idx_oob = _numba_resample_and_oob(
            idx, self.params.frac_samples, self.params.seed, self.params.replace)

        ib_arrays = tuple(arr[idx_ib] for arr in arrays)
        oob_arrays = tuple(arr[idx_oob] for arr in arrays)

        return ib_arrays, oob_arrays, idx_ib, idx_oob
