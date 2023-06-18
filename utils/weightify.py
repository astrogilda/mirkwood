from pydantic import BaseModel, Field
from typing import Union
from numba import jit
from enum import Enum
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import gaussian_kde
from scipy.ndimage import convolve1d, gaussian_filter1d
import numpy as np
from sklearn.utils.validation import check_is_fitted
from utils.reshape import reshape_to_1d_array


class Style(str, Enum):
    INV = "inv"
    SQRT_INV = "sqrt_inv"
    SPECIAL = "special"
    DIR = "dir"


class WeightifyConfig(BaseModel):
    style: Style = Field(Style.DIR, description="Style of weighting to apply.")
    lds_ks: int = Field(
        1, description="Kernel size for local density smoothing.", le=10, gt=0)
    n_bins: int = Field(
        50, description="Number of bins for kernel density estimation.", ge=10)
    beta: float = Field(
        0.9, description="Beta parameter for special style weighting.", ge=0, le=1)
    bw_method: Union[float, str] = Field(
        3, description="Bandwidth estimation method for kernel density estimation.", ne=0)
    lds_sigma: float = Field(
        1, description="Standard deviation for local density smoothing.", gt=0.01, le=100)


# move the jit accelerated functions outside of the class
@jit(nopython=True)
def calc_weights_jit(smoothed_value: np.ndarray) -> np.ndarray:
    """
    Calculate weights using Numba-accelerated function.

    Parameters
    ----------
    smoothed_value : numpy array
        Smoothed values.

    Returns
    -------
    weights : numpy array
        Calculated weights.
    """
    return np.sqrt(1.0 / smoothed_value)


@jit(nopython=True)
def inv_weights(samples_per_bin: np.ndarray) -> np.ndarray:
    """
    Calculate weights using inverse style.

    Parameters
    ----------
    samples_per_bin : numpy array
        Samples per bin.

    Returns
    -------
    weights : numpy array
        Calculated weights.
    """
    weights = 1.0 / samples_per_bin
    scaling = len(weights) / np.sum(weights)
    weights *= scaling
    return weights


@jit(nopython=True)
def sqrt_inv_weights(samples_per_bin: np.ndarray) -> np.ndarray:
    """
    Calculate weights using square root inverse style.

    Parameters
    ----------
    samples_per_bin : numpy array
        Samples per bin.

    Returns
    -------
    weights : numpy array
        Calculated weights.
    """
    weights = np.sqrt(1.0 / samples_per_bin)
    scaling = len(weights) / np.sum(weights)
    weights *= scaling
    return weights


@jit(nopython=True)
def special_weights(beta: float, samples_per_bin: np.ndarray) -> np.ndarray:
    """
    Calculate weights using special style.

    Parameters
    ----------
    beta : float
        The beta parameter for special style weighting.

    samples_per_bin : numpy array
        Samples per bin.

    Returns
    -------
    weights : numpy array
        Calculated weights.
    """
    samples_idx = np.arange(len(samples_per_bin))
    samples_per_bin = samples_per_bin / samples_per_bin.min()
    effective_num = 1.0 - np.power(beta, samples_per_bin)
    weights = np.where(effective_num != 0, (1.0 - beta) / effective_num, 0)
    weights = weights / np.sum(weights) * len(samples_per_bin)
    samples_weights = np.array(
        [weights[i] for i in samples_idx], dtype=samples_per_bin.dtype)
    return samples_weights


class Weightify(BaseEstimator, TransformerMixin):
    def __init__(self, config: WeightifyConfig = WeightifyConfig()) -> None:
        """
        Initialize the Weightify transformer with the given configuration.

        Parameters
        ----------
        config : WeightifyConfig
            The configuration to use.
        """
        self.config = config

    def get_lds_kernel_window(self, ks: int, sigma: float) -> np.ndarray:
        """
        Get the kernel window for local density smoothing.

        Parameters
        ----------
        ks : int
            Kernel size.
        sigma : float
            Standard deviation for smoothing.

        Returns
        -------
        kernel_window : numpy array
            Kernel window for local density smoothing.
        """
        half_ks = (ks - 1) // 2
        base_kernel = [0.0] * half_ks + [1.0] + [0.0] * half_ks
        numerator_kernel_window = gaussian_filter1d(base_kernel, sigma=sigma)
        denominator_kernel_window = max(numerator_kernel_window)
        kernel_window = numerator_kernel_window / denominator_kernel_window
        return kernel_window

    def dir_weights(self, samples_per_bin: np.ndarray) -> np.ndarray:
        """
        Calculate weights using direct style.

        Parameters
        ----------
        samples_per_bin : numpy array
            Samples per bin.

        Returns
        -------
        weights : numpy array
            Calculated weights.
        """
        lds_kernel_window = self.get_lds_kernel_window(
            ks=self.config.lds_ks, sigma=self.config.lds_sigma)
        smoothed_value = convolve1d(
            samples_per_bin, weights=lds_kernel_window, mode="constant")
        weights = calc_weights_jit(smoothed_value)
        scaling = len(weights) / np.sum(weights)
        weights *= scaling
        return weights

    style_methods = {
        Style.DIR: dir_weights,
        Style.INV: inv_weights,
        Style.SQRT_INV: sqrt_inv_weights,
        Style.SPECIAL: lambda self, samples_per_bin: special_weights(self.config.beta, samples_per_bin),
    }

    def calculate_weights(self, y: np.ndarray) -> np.ndarray:
        """
        Calculate weights based on the specified style.

        Parameters
        ----------
        y : numpy array of shape [n_samples]
            Input data.

        Returns
        -------
        weights : numpy array
            Calculated weights.
        """
        # if all elements are equal, return an array of 1s
        if np.all(y == y[0]):
            return np.full(len(y), 1, dtype=y.dtype)
        else:
            kernel = gaussian_kde(y, bw_method=self.config.bw_method)
            kernel.set_bandwidth(bw_method=kernel.factor / self.config.n_bins)
            samples_per_bin = kernel(y)
            return self.style_methods[self.config.style](self, samples_per_bin).reshape(y.shape)

    def fit(self, y: np.ndarray, sub_size: int = 100_000, poly_order: int = 2) -> "Weightify":
        """
        Fit the transformer to the input data.

        Parameters
        ----------
        y : numpy array of shape [n_samples]
            Input data.
        sub_size : int, default=100_000
            Size of subset to use for polynomial fitting.
        poly_order : int, default=2
            Order of the polynomial to be estimated.

        Returns
        -------
        self : Weightify
            The fitted Weightify transformer.
        """
        y = reshape_to_1d_array(y)

        if sub_size < len(y):
            indices = np.random.permutation(len(y))[:sub_size]
            y_sub = y[indices]
            weights_sub = self.calculate_weights(y_sub)
            self.poly_coeffs_ = np.polyfit(
                x=y_sub, y=weights_sub, deg=poly_order)
            del y_sub, weights_sub
        else:
            weights = self.calculate_weights(y)
            self.poly_coeffs_ = np.polyfit(x=y, y=weights, deg=poly_order)

        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Transform the input data using the calculated weights.

        Parameters
        ----------
        y : numpy array of shape [n_samples]
            Input data. Weights are calculated based on the new data.
            Otherwise, previously calculated weights during fit method are used.

        Returns
        -------
        sample_weights : numpy array
            Transformed version of the input data.
        """
        check_is_fitted(self, 'poly_coeffs_')

        y = reshape_to_1d_array(y)

        poly_order = len(self.poly_coeffs_) - 1
        transposed_y = np.vstack([y ** (poly_order - i)
                                 for i in range(poly_order + 1)]).T
        sample_weights = np.dot(transposed_y, self.poly_coeffs_)
        sample_weights = sample_weights.reshape(y.shape)
        sample_weights = np.clip(sample_weights, 0.1, 10)

        return sample_weights

    def fit_transform(self, y: np.ndarray, sub_size: int = 100_000, poly_order: int = 2) -> np.ndarray:
        """
        Fit the transformer to the input data and transform it.

        Parameters
        ----------
        y : numpy array of shape [n_samples]
            Input data.
        sub_size : int, default=100_000
            Size of subset to use for polynomial fitting.
        poly_order : int, default=2
            Order of the polynomial to be estimated.

        Returns
        -------
        sample_weights : numpy array
            Transformed version of the input data.
        """
        return self.fit(y, sub_size, poly_order).transform(y)
