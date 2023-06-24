from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_array
from pydantic import BaseModel, Field
from typing import Union, Optional
from numba import jit
from enum import Enum
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import gaussian_kde
from scipy.ndimage import convolve1d, gaussian_filter1d
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone

from utils.reshape import reshape_to_1d_array


# TODO: Add support for special style weighting.
class Style(str, Enum):
    INV = "inv"
    SQRT_INV = "sqrt_inv"
    # SPECIAL = "special"
    DIR = "dir"


class WeightifyConfig(BaseModel):
    style: Style = Field(Style.DIR, description="Style of weighting to apply.")
    lds_ks: int = Field(
        1, description="Kernel size for local density smoothing.", le=10, gt=0)
    lds_sigma: float = Field(
        1, description="Standard deviation for local density smoothing.", gt=0.01, le=100)
    n_bins: int = Field(
        50, description="Number of bins for kernel density estimation.", ge=10)
    bw_method: Union[float, str] = Field(
        3, description="Bandwidth estimation method for kernel density estimation.", ne=0)
    beta: float = Field(
        0.9, description="Beta parameter for special style weighting.", ge=0, le=1)

    class Config:
        arbitrary_types_allowed: bool = True


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
    config : WeightifyConfig
        Weightify configuration.
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
    config : WeightifyConfig
        Weightify configuration.
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


EPS = 1e-6


def special_weights(config: WeightifyConfig, samples_per_bin: np.ndarray) -> np.ndarray:
    """
    Calculate weights using special style.

    Parameters
    ----------
    config : WeightifyConfig
        Weightify configuration.
    samples_per_bin : numpy array
        Samples per bin.

    Returns
    -------
    weights : numpy array
        Calculated weights.
    """
    # Create an array of indices
    samples_idx = np.arange(len(samples_per_bin))
    # Normalize the samples_per_bin array by its smallest value. This adjusts the scale of the samples, making the smallest sample equal to 1 and larger samples proportionally larger.
    samples_per_bin = MinMaxScaler().fit_transform(samples_per_bin.reshape(-1, 1))
    # samples_per_bin = samples_per_bin / samples_per_bin.min()
    # Create a new array which is a function of samples_per_bin and config.beta. This operation resembles a decay function where bins with fewer samples have a larger effective number.
    effective_num = 1.0 - np.power(config.beta, samples_per_bin)
    effective_num += EPS
    # Calculate weights based on effective_num, setting weight to 0 if effective_num is 0. This ensures bins with more samples (lower effective numbers) get smaller weights and bins with fewer samples get larger weights.
    weights = np.where(effective_num != 0,
                       (1.0 - config.beta) / effective_num, 0)
    # Normalize the weights such that they sum up to the number of samples. This keeps the total sum of weights consistent regardless of their individual distribution.
    weights = weights / np.sum(weights) * len(samples_per_bin)
    # Reorder weights according to the original sample indices.
    samples_weights = np.array(
        [weights[i] for i in samples_idx], dtype=samples_per_bin.dtype)

    return samples_weights


def get_lds_kernel_window(ks: int, sigma: float) -> np.ndarray:
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


def dir_weights(config: WeightifyConfig, samples_per_bin: np.ndarray) -> np.ndarray:
    """
    Calculate weights using direct style.

    Parameters
    ----------
    config : WeightifyConfig
        Weightify configuration.
    samples_per_bin : numpy array
        Samples per bin.

    Returns
    -------
    weights : numpy array
        Calculated weights.
    """
    lds_kernel_window = get_lds_kernel_window(
        ks=config.lds_ks, sigma=config.lds_sigma)
    smoothed_value = convolve1d(
        samples_per_bin, weights=lds_kernel_window, mode="constant")
    weights = calc_weights_jit(smoothed_value)
    scaling = len(weights) / np.sum(weights)
    weights *= scaling
    return weights


default_weightify_config = WeightifyConfig()

style_methods = {
    Style.DIR: dir_weights,
    Style.INV: inv_weights,
    Style.SQRT_INV: sqrt_inv_weights,
    # Style.SPECIAL: special_weights,
}


class Weightify(BaseEstimator, TransformerMixin):
    def __init__(self,
                 style: Style = default_weightify_config.style,
                 lds_ks: int = default_weightify_config.lds_ks,
                 lds_sigma: float = default_weightify_config.lds_sigma,
                 n_bins: int = default_weightify_config.n_bins,
                 bw_method: Union[float,
                                  str] = default_weightify_config.bw_method,
                 beta: float = default_weightify_config.beta):
        self.style = style
        self.lds_ks = lds_ks
        self.lds_sigma = lds_sigma
        self.n_bins = n_bins
        self.bw_method = bw_method
        self.beta = beta

    def fit(self, X: np.ndarray, y=None, sub_size: int = 100_000, poly_order: int = 2):
        """
        Fit the transformer to the input data.

        Parameters
        ----------
        X : numpy array of shape [n_samples,1]
            Input data.
        y : None. Provided for compatibility with scikit-learn.
        sub_size : int, default=100_000
            Size of subset to use for polynomial fitting.
        poly_order : int, default=2
            Order of the polynomial to be estimated.

        Returns
        -------
        self : Weightify
            The fitted Weightify transformer.
        """
        if y is not None:
            raise ValueError("y must be None")

        config = WeightifyConfig(
            style=self.style,
            lds_ks=self.lds_ks,
            lds_sigma=self.lds_sigma,
            n_bins=self.n_bins,
            bw_method=self.bw_method,
            beta=self.beta
        )

        X = reshape_to_1d_array(X)

        if sub_size < len(X):
            indices = np.random.permutation(len(X))[:sub_size]
            X_sub = X[indices]
            weights_sub = self.calculate_weights(X_sub, config)
            self.poly_coeffs_ = np.polyfit(
                x=X_sub, y=weights_sub, deg=poly_order)
            del X_sub, weights_sub
        else:
            weights = self.calculate_weights(X, config)
            self.poly_coeffs_ = np.polyfit(x=X, y=weights, deg=poly_order)

        self.fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data using the calculated weights.

        Parameters
        ----------
        y : numpy array of shape [n_samples, 1]
            Input data. Weights are calculated based on the new data.
            Otherwise, previously calculated weights during fit method are used.

        Returns
        -------
        sample_weights : numpy array
            Transformed version of the input data.
        """

        check_is_fitted(self, 'fitted_')

        poly_order = len(self.poly_coeffs_) - 1
        X = reshape_to_1d_array(X)
        transposed_X = np.vstack([X ** (poly_order - i)
                                 for i in range(poly_order + 1)]).T
        sample_weights = np.dot(transposed_X, self.poly_coeffs_)
        sample_weights = sample_weights.reshape(X.shape)
        sample_weights = np.clip(sample_weights, 0.1, 10)

        return sample_weights

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the transformer to the input data and transform it.

        Parameters
        ----------
        X : numpy array of shape [n_samples,1]
            Input data.

        Returns
        -------
        sample_weights : numpy array
            Transformed version of the input data.
        """
        return self.fit(X).transform(X)

    def calculate_weights(self, X: np.ndarray, config: WeightifyConfig) -> np.ndarray:
        """
        Calculate weights based on the specified style.

        Parameters
        ----------
        X : numpy array of shape [n_samples]
            Input data.
        config : WeightifyConfig
            The configuration to use.

        Returns
        -------
        weights : numpy array
            Calculated weights.
        """

        X = reshape_to_1d_array(X)

        if np.all(X == X[0]):
            return np.full(len(X), 1, dtype=X.dtype)
        else:
            kernel = gaussian_kde(X, bw_method=config.bw_method)
            kernel.set_bandwidth(bw_method=kernel.factor / config.n_bins)
            samples_per_bin = kernel(X)
            if config.style == Style.DIR:  # or config.style == Style.SPECIAL:
                return style_methods[config.style](config, samples_per_bin).reshape(X.shape)
            else:
                return style_methods[config.style](samples_per_bin).reshape(X.shape)

    def set_params(self, **params):
        """
        Set the parameters of the Weightify transformer.

        Parameters
        ----------
        **params : dict
            Weightify parameters to set.
        """
        super().set_params(**params)
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_params(self, deep=True):
        """
        Get the parameters of the Weightify transformer.

        Parameters
        ----------
        deep : bool, default=True
            Unused, here for compatibility with scikit-learn.

        Returns
        -------
        params : dict
            Weightify parameters.
        """
        params = super().get_params(deep=deep)
        weightify_config_params = {
            key: getattr(self, key) for key in vars(self)}
        params.update(weightify_config_params)
        return params
