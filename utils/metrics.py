from numba import jit, prange
import numpy as np
from pydantic import BaseModel, Field, confloat
from functools import lru_cache
import math
from typing import Tuple
import scipy.stats as stats

import warnings
from numba.core.errors import NumbaExperimentalFeatureWarning

# Ignore the experimental feature warning
warnings.filterwarnings('ignore', category=NumbaExperimentalFeatureWarning)

# TODO: Add miscalibration_area and PICP metrics


def calculate_z_score(confidence_level: float) -> float:
    """
    Calculate the z-score.
    """
    alpha = 1 - confidence_level
    z_score = stats.norm.ppf(1 - alpha / 2)
    return z_score


@jit(nopython=True)
def erf_numba(x):
    # constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    # Save the sign of x
    sign = 1 if x >= 0 else -1
    x = abs(x)

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t

    return sign*(1 - y * np.exp(-x*x))


EPS = 1e-6


@jit(nopython=True, parallel=True)
def calculate_iqr(yt: np.ndarray) -> float:
    """ Calculate the interquartile range plus EPS """
    return np.quantile(yt, 0.95) - np.quantile(yt, 0.05) + EPS


@jit(nopython=True, parallel=True)
def calculate_nrmse(yt: np.ndarray, yp: np.ndarray, iqr: float) -> float:
    """ Calculate normalized root mean square error """
    return np.sqrt(np.mean((yt - yp) ** 2)) / iqr


@jit(nopython=True, parallel=True)
def calculate_nmae(yt: np.ndarray, yp: np.ndarray, iqr: float) -> float:
    """ Calculate normalized mean absolute error """
    return np.mean(np.abs(yt - yp)) / iqr


@jit(nopython=True, parallel=True)
def calculate_medianae(yt: np.ndarray, yp: np.ndarray) -> float:
    """ Calculate median absolute error """
    return np.median(np.abs(yt - yp))


@jit(nopython=True, parallel=True)
def calculate_mape(yt: np.ndarray, yp: np.ndarray) -> float:
    """ Calculate mean absolute percentage error """
    return np.mean(np.abs((yt - yp) / (yt + EPS)))


@jit(nopython=True, parallel=True)
def calculate_bias(yt: np.ndarray, yp: np.ndarray) -> float:
    """ Calculate bias """
    return np.mean(np.where(yp >= yt, 1.0, -1.0))


@jit(nopython=True, parallel=True)
def calculate_nbe(yt: np.ndarray, yp: np.ndarray, iqr: float) -> float:
    """ Calculate normalized bias error """
    return np.mean(yp - yt) / iqr


class ErrorMetricsBase(BaseModel):
    """Base class for encapsulating all error metrics functions"""

    yt: np.ndarray = Field(..., description="Actual values")
    yp: np.ndarray = Field(..., description="Predicted values")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.yt = np.asarray(self.yt).flatten()
        self.yp = np.asarray(self.yp).flatten()
        self.yt, self.yp = self._equalize_shape(self.yt, self.yp)

    # @lru_cache(maxsize=None)
    def get_iqr(self):
        """ Calculate the interquartile range plus EPS """
        return calculate_iqr(self.yt)

    @staticmethod
    def _equalize_shape(*arrays) -> Tuple[np.ndarray, ...]:
        max_shape = max(arr.shape for arr in arrays)
        return tuple(np.broadcast_to(arr, max_shape).flatten() for arr in arrays)


class DeterministicErrorMetrics(ErrorMetricsBase):
    """Deterministic error metrics"""

    def nrmse(self) -> float:
        """ Calculate normalized root mean square error """
        return calculate_nrmse(self.yt, self.yp, self.get_iqr())

    def nmae(self) -> float:
        """ Calculate normalized mean absolute error """
        return calculate_nmae(self.yt, self.yp, self.get_iqr())

    def medianae(self) -> float:
        """ Calculate median absolute error """
        return calculate_medianae(self.yt, self.yp)

    def mape(self) -> float:
        """ Calculate mean absolute percentage error """
        return calculate_mape(self.yt, self.yp)

    def bias(self) -> float:
        """ Calculate bias """
        return calculate_bias(self.yt, self.yp)

    def nbe(self) -> float:
        """ Calculate normalized bias error """
        return calculate_nbe(self.yt, self.yp, self.get_iqr())


#### Probabilistic metrics ####

@jit(nopython=True, parallel=True)
def calculate_ace(yt: np.ndarray, yp_lower: np.ndarray, yp_upper: np.ndarray, confint: float) -> float:
    """ Calculate the average coverage error (ACE) for confidence intervals. """
    alpha = 1 - confint
    c = np.logical_and(yt >= yp_lower, yt <= yp_upper)
    ace = np.nanmean(c) - (1 - alpha)
    # ACE can be negative when empirical coverage is less than nominal coverage
    # Negative ACE implies that the prediction intervals are too narrow
    # Positive ACE implies that the prediction intervals are too wide
    return ace


@jit(nopython=True, parallel=True)
def calculate_pinaw(yp_upper: np.ndarray, yp_lower: np.ndarray, iqr: float) -> float:
    """ Calculate the prediction interval normalized average width (PINAW). """
    pinaw = np.mean(yp_upper - yp_lower) / iqr
    assert pinaw >= 0, "PINAW should be non-negative"
    return pinaw


@jit(nopython=True)
def calculate_cdf_normdist(yt, loc, scale):
    """ Calculate the cumulative distribution function (CDF) of the normal distribution. """
    u = (1.0 + erf_numba((yt - loc) / (scale * np.sqrt(2.0)))) / 2.0
    return u


@jit(nopython=True, parallel=True)
def calculate_interval_sharpness(yt: np.ndarray, yp: np.ndarray, yp_lower: np.ndarray, yp_upper: np.ndarray, confint: float) -> float:
    """ 
    Calculate the interval sharpness of probabilistic predictions.

    Parameters
    ----------
    yt : np.ndarray
        True target values.
    yp : np.ndarray
        Predicted target values.
    yp_lower : np.ndarray
        Lower bounds of the prediction intervals.
    yp_upper : np.ndarray
        Upper bounds of the prediction intervals.
    confint : float
        Confidence interval level.

    Returns
    -------
    float
        Interval sharpness.
    """

    # Transform true targets into percentiles
    for i in prange(yt.shape[0]):
        yt[i] = calculate_cdf_normdist(
            yt[i], yp[i], 0.5 * (yp_upper[i] - yp_lower[i]))

    # Define the lower and upper bounds of the prediction intervals
    alpha = 1 - confint
    yp_lower = np.ones_like(yp_lower) * (0.5 - confint / 2)
    yp_upper = np.ones_like(yp_upper) * (0.5 + confint / 2)

    # Compute the differences between the bounds
    delta_alpha = yp_upper - yp_lower

    # Compute the sharpness
    intsharp = np.nanmean(np.where(yt >= yp_upper,
                                   np.abs(-2 * alpha * delta_alpha -
                                          4 * (yt - yp_upper)),
                                   np.where(yp_lower >= yt,
                                            np.abs(-2 * alpha * delta_alpha -
                                                   4 * (yp_lower - yt)),
                                            np.abs(-2 * alpha * delta_alpha))))

    assert intsharp >= 0, "Interval sharpness must be non-negative."
    return intsharp


# @vectorize([float64(float64, float64, float64)], target='parallel')
@jit(nopython=True)
def _gaussian_crps(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Compute the Continuous Ranked Probability Score (CRPS) for arrays with the help of Numba.

    Parameters
    ----------
    y_true: np.ndarray
        The array of true values.
    mu: np.ndarray
        The array of means of the predicted Gaussian distributions.
    sigma: np.ndarray
        The array of standard deviations of the predicted Gaussian distributions.

    Returns
    -------
    np.ndarray
        The array of CRPS of the predictions.
    """
    # normalization for the Gaussian distribution
    y_true_normalized = (y_true - mu) / sigma

    # the cumulative distribution function (CDF) of the Gaussian distribution
    phi = 0.5 * (1 + erf_numba(y_true_normalized / np.sqrt(2)))

    # the probability density function (PDF) of the Gaussian distribution
    pdf = math.exp(-0.5 * y_true_normalized ** 2) / np.sqrt(2 * np.pi)

    # the CRPS in terms of the CDF and PDF of the normalized Gaussian distribution
    crps = sigma * (y_true_normalized * (2 * phi - 1) +
                    2 * pdf - 1 / np.sqrt(np.pi))

    return crps


@jit(nopython=True, parallel=True)
def calculate_gaussian_crps(yt: np.ndarray, yp: np.ndarray, yp_lower: np.ndarray, yp_upper: np.ndarray) -> float:
    """ Calculate Gaussian CRPS. """
    sigma = 0.5 * (yp_upper - yp_lower)
    crps_array = np.ones_like(yt)
    for i in prange(yt.shape[0]):
        crps_array[i] = _gaussian_crps(yt[i], yp[i], sigma[i])

    mean_crps = np.nanmean(crps_array)
    assert mean_crps >= 0, "CRPS must be non-negative."
    return mean_crps


class ProbabilisticErrorMetrics(ErrorMetricsBase):
    """Probabilistic error metrics"""

    yp_lower: np.ndarray = Field(...,
                                 description="Lower bound of predicted values")
    yp_upper: np.ndarray = Field(...,
                                 description="Upper bound of predicted values")
    confidence_level: confloat(gt=0, le=1) = Field(
        0.6827, description="Confidence level")

    def __init__(self, **data):
        super().__init__(**data)
        # reshape the additional arrays
        self.yp_lower = np.asarray(self.yp_lower).flatten()
        self.yp_upper = np.asarray(self.yp_upper).flatten()
        self.yt, self.yp, self.yp_lower, self.yp_upper = self._equalize_shape(
            self.yt, self.yp, self.yp_lower, self.yp_upper)
        assert np.all(self.yp_upper >
                      self.yp), "yp_upper should be greater than yp"
        assert np.all(
            self.yp > self.yp_lower), "yp should be greater than yp_lower"

    def ace(self) -> float:
        """
        Calculate the average coverage error (ACE) for confidence intervals.

        Parameters
        ----------
        confint : float, optional
            Confidence interval, by default 0.6827

        Returns
        -------
        float
            Average coverage error.
        """
        return calculate_ace(self.yt, self.yp_lower, self.yp_upper, confint=self.confidence_level)

    def pinaw(self) -> float:
        """
        Calculate the prediction interval normalized average width (PINAW).

        Returns
        -------
        float
            Prediction interval normalized average width.
        """
        return calculate_pinaw(self.yp_upper, self.yp_lower, self.get_iqr())

    def interval_sharpness(self) -> float:
        """
        Calculate the interval sharpness.

        Parameters
        ----------
        confint : float, optional
            Confidence interval, by default 0.6827

        Returns
        -------
        float
            Interval sharpness.
        """
        return calculate_interval_sharpness(self.yt, self.yp, self.yp_lower, self.yp_upper, confint=self.confidence_level)

    def gaussian_crps(self) -> float:
        return calculate_gaussian_crps(self.yt, self.yp, self.yp_lower, self.yp_upper)
