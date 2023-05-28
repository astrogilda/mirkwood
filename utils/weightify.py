from numba import jit
from enum import Enum
from typing import Union
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import gaussian_kde
from scipy.ndimage import convolve1d, gaussian_filter1d
import numpy as np
from sklearn.utils.validation import check_is_fitted


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


class Style(str, Enum):
    INV = "inv"
    SQRT_INV = "sqrt_inv"
    SPECIAL = "special"
    DIR = "dir"


class Weightify(BaseEstimator, TransformerMixin):
    def __init__(self,
                 style: Style = Style.DIR,
                 lds_ks: int = 1,
                 n_bins: int = 50,
                 beta: float = 0.9,
                 bw_method: Union[float, str] = 3,
                 lds_sigma: float = 1) -> None:
        """
        Initialize the Weightify transformer.

        Parameters
        ----------
        style : Style, default=Style.DIR
            Style of weighting to apply.
        lds_ks : int, default=1
            Kernel size for local density smoothing.
        n_bins : int, default=50
            Number of bins for kernel density estimation.
        beta : float, default=0.9
            Beta parameter for special style weighting.
        bw_method : float or str, default=3
            Bandwidth estimation method for kernel density estimation.
        lds_sigma : float, default=1
            Standard deviation for local density smoothing.

        Raises
        ------
        ValueError
            If the input parameters violate the specified constraints.
        """
        self.style = style
        self.lds_ks = lds_ks
        self.n_bins = n_bins
        self.beta = beta
        self.bw_method = bw_method
        self.lds_sigma = lds_sigma
        self._validate_params()

    def _validate_params(self) -> None:
        """
        Validate the input parameters.

        Raises
        ------
        ValueError
            If any of the input parameters violate the specified constraints.
        """
        if abs(self.lds_ks) > 10:
            raise ValueError("Input a value <= 10")
        if self.n_bins < 10:
            raise ValueError("Use at least 10 bins")
        if not 0 <= self.beta <= 1:
            raise ValueError("beta must lie in [0,1]")
        if self.bw_method == 0:
            raise ValueError("bw_method cannot be 0")
        if np.isclose(self.lds_sigma, 0.01) or self.lds_sigma > 100:
            raise ValueError("standard deviation must be in range (0.01, 100]")

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
            ks=self.lds_ks, sigma=self.lds_sigma)
        smoothed_value = convolve1d(
            samples_per_bin, weights=lds_kernel_window, mode="constant")
        weights = calc_weights_jit(smoothed_value)
        scaling = len(weights) / np.sum(weights)
        weights *= scaling
        return weights

    @jit(nopython=True)
    def inv_weights(self, samples_per_bin: np.ndarray) -> np.ndarray:
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
    def sqrt_inv_weights(self, samples_per_bin: np.ndarray) -> np.ndarray:
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
    def special_weights(self, samples_per_bin: np.ndarray) -> np.ndarray:
        """
        Calculate weights using special style.

        Parameters
        ----------
        samples_per_bin : numpy array
            Samples per bin.
        Returns
        -------
        weights : numpy array
            Calculated weights.
        """
        samples_idx = np.arange(len(samples_per_bin))
        samples_per_bin = samples_per_bin / samples_per_bin.min()
        effective_num = 1.0 - np.power(self.beta, samples_per_bin)
        weights = np.where(effective_num != 0, (1.0 - self.beta)
                           * 1 / effective_num, 0)
        weights = weights / np.sum(weights) * len(samples_per_bin)
        samples_weights = np.array(
            [weights[i] for i in samples_idx], dtype=samples_per_bin.dtype)
        return samples_weights

    style_methods = {
        Style.DIR: dir_weights,
        Style.INV: inv_weights,
        Style.SQRT_INV: sqrt_inv_weights,
        Style.SPECIAL: special_weights,
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
        if len(set(y)) == 1:
            return np.array([1]*len(y), dtype=y.dtype)
        else:
            kernel = gaussian_kde(y, bw_method=self.bw_method)
            kernel.set_bandwidth(bw_method=kernel.factor / self.n_bins)
            samples_per_bin = kernel(y)
            return self.style_methods[self.style](self, samples_per_bin=samples_per_bin)

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

        Raises
        ------
        NotImplementedError
            If the output y is not 1D (i.e., scalar).

        Returns
        -------
        self : Weightify
            The fitted Weightify transformer.
        """
        if y.ndim > 1:
            raise NotImplementedError("Output y must be 1D (i.e. scalar)")

        if sub_size < len(y):
            indices = np.random.permutation(len(y))[:sub_size]
            y_sub = y[indices]
            weights_sub = self.calculate_weights(y_sub)
            poly_coeffs = np.polyfit(x=y_sub, y=weights_sub, deg=poly_order)
            del y_sub, weights_sub
            transposed_y = np.vstack([y ** (poly_order - i)
                                      for i in range(poly_order + 1)]).T
            sample_weights = np.dot(transposed_y, poly_coeffs)
            del transposed_y
        else:
            sample_weights = self.calculate_weights(y)

        sample_weights = np.clip(sample_weights, 0.1, 10)
        self.sample_weights_ = sample_weights
        return self

    def transform(self, X=None) -> np.ndarray:
        """
        Transform the input data using the calculated weights.

        Parameters
        ----------
        X : array-like, default=None
            Unused argument.

        Returns
        -------
        sample_weights : numpy array
            Transformed version of the input data.
        """
        check_is_fitted(self, 'sample_weights_')
        return self.sample_weights_

    def fit_transform(self, y, sub_size: int = 100_000, poly_order: int = 2) -> np.ndarray:
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
        return self.fit(y, sub_size, poly_order).transform()
