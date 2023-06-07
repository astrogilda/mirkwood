from numba import jit, vectorize, float32, float64
import numpy as np
from scipy.special import erf
from pydantic import BaseModel, Field
from functools import lru_cache
import math

EPS = 1e-6


class ErrorMetricsBase(BaseModel):
    """Base class for encapsulating all error metrics functions"""

    yt: np.ndarray = Field(..., description="Actual values")
    yp: np.ndarray = Field(..., description="Predicted values")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        # flatten the arrays once, to optimize memory usage
        self.yt = np.asarray(self.yt).flatten()
        self.yp = np.asarray(self.yp).flatten()

    @lru_cache(maxsize=None)
    def get_iqr(self) -> float:
        """ Calculate the interquartile range plus EPS """
        return np.quantile(self.yt, 0.95) - np.quantile(self.yt, 0.05) + EPS


class DeterministicErrorMetrics(ErrorMetricsBase):
    """Deterministic error metrics"""

    # All the methods are decorated with jit for optimization
    @jit(nopython=True, parallel=True)
    def nrmse(self) -> float:
        """ Calculate normalized root mean square error """
        return np.sqrt(np.mean((self.yt - self.yp) ** 2)) / self.get_iqr()

    @jit(nopython=True, parallel=True)
    def nmae(self) -> float:
        """ Calculate normalized mean absolute error """
        return np.mean(np.abs(self.yt - self.yp)) / self.get_iqr()

    @jit(nopython=True, parallel=True)
    def medianae(self) -> float:
        """ Calculate median absolute error """
        return np.median(np.abs(self.yt - self.yp))

    @jit(nopython=True, parallel=True)
    def mape(self) -> float:
        """ Calculate mean absolute percentage error """
        return np.mean(np.abs((self.yt - self.yp) / (self.yt + EPS)))

    @jit(nopython=True, parallel=True)
    def bias(self) -> float:
        """ Calculate bias """
        return np.mean(np.where(self.yp >= self.yt, 1.0, -1.0))

    @jit(nopython=True, parallel=True)
    def nbe(self) -> float:
        """ Calculate normalized bias error """
        return np.mean(self.yp - self.yt) / self.get_iqr()


class ProbabilisticErrorMetrics(ErrorMetricsBase):
    """Probabilistic error metrics"""

    yp_lower: np.ndarray = Field(...,
                                 description="Lower bound of predicted values")
    yp_upper: np.ndarray = Field(...,
                                 description="Upper bound of predicted values")

    def __init__(self, **data):
        super().__init__(**data)
        # reshape the additional arrays
        self.yp_lower = np.asarray(self.yp_lower).flatten()
        self.yp_upper = np.asarray(self.yp_upper).flatten()

    @jit(nopython=True, parallel=True)
    def ace(self, confint: float = 0.6827) -> float:
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
        alpha = 1 - confint
        c = np.logical_and(self.yt >= self.yp_lower, self.yt <= self.yp_upper)
        return np.nanmean(c) - (1 - alpha)

    @jit(nopython=True, parallel=True)
    def pinaw(self) -> float:
        """
        Calculate the prediction interval normalized average width (PINAW).

        Returns
        -------
        float
            Prediction interval normalized average width.
        """
        iqr = self.get_iqr()
        return np.mean(self.yp_upper - self.yp_lower) / iqr

    @jit(nopython=True, parallel=True)
    def cdf_normdist(self, loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """
        Calculate the cumulative distribution function (CDF) of the normal distribution.

        Parameters
        ----------
        loc : np.ndarray
            Mean of the normal distribution.
        scale : np.ndarray
            Standard deviation of the normal distribution.

        Returns
        -------
        np.ndarray
            CDF values.
        """
        u = (1.0 + erf((self.yt - loc) / (scale * np.sqrt(2.0)))) / 2.0
        return u

    @jit(nopython=True, parallel=True)
    def interval_sharpness(self, confint: float = 0.6827) -> float:
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
        yt = self.cdf_normdist(loc=self.yp, scale=0.5 *
                               (self.yp_upper - self.yp_lower))
        alpha = 1 - confint
        yp_lower = np.ones_like(self.yp_lower) * (0.5 - confint / 2)
        yp_upper = np.ones_like(self.yp_upper) * (0.5 + confint / 2)
        delta_alpha = yp_upper - yp_lower
        intsharp = np.nanmean(np.where(yt >= yp_upper, -2 * alpha * delta_alpha - 4 * (yt - yp_upper),
                                       np.where(yp_lower >= yt, -2 * alpha * delta_alpha - 4 * (yp_lower - yt), -2 * alpha * delta_alpha)))
        return intsharp

    @staticmethod
    @vectorize([float64(float64, float64, float64)], target='parallel')
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
        phi = 0.5 * (1 + math.erf(y_true_normalized / np.sqrt(2)))

        # the probability density function (PDF) of the Gaussian distribution
        pdf = math.exp(-0.5 * y_true_normalized ** 2) / np.sqrt(2 * np.pi)

        # the CRPS in terms of the CDF and PDF of the normalized Gaussian distribution
        crps = sigma * (y_true_normalized * (2 * phi - 1) +
                        2 * pdf - 1 / np.sqrt(np.pi))

        return crps

    @jit(nopython=True, parallel=True)
    def gaussian_crps(self) -> float:
        sigma = 0.5 * (self.yp_upper - self.yp_lower)
        crps_array = self._gaussian_crps(self.yt, self.yp, sigma)
        mean_crps = np.nanmean(crps_array)
        return mean_crps
