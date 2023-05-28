from pathlib import Path
from numba import njit, jit
from pydantic import BaseModel, Field, validator
from sklearn.base import TransformerMixin
from typing import Callable, List, Optional, Tuple
import numpy as np
from src.model_handler import ModelHandler
from pydantic_numpy import NDArray, NDArrayFp32

# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")


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


# Suppress all warnings
warnings.filterwarnings("ignore")


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


class BootstrapHandler(BaseModel):
    """
    BootstrapHandler class for resampling and reversing transformation and function application.

    Attributes
    ----------
    x : np.ndarray
        2D array of x data.
    y : np.ndarray
        1D array of y data or 2D array with second dimension 1.
    frac_samples_best : float
        Maximum fraction of samples to draw, defaults to 1.0 (meaning the entire dataset is sampled).
    """
    x: NDArrayFp32
    y: NDArrayFp32
    frac_samples_best: float = Field(1.0, gt=0, lte=1)

    @validator('x')
    def _check_x_dimension(cls, v: np.ndarray) -> np.ndarray:
        """Validate if the input x array is two-dimensional"""
        if len(v.shape) != 2:
            raise ValueError("x should be 2-dimensional")
        return v

    @validator('y', pre=True)
    def _check_y_dimension(cls, v: np.ndarray) -> np.ndarray:
        """Validate if the input y array is one-dimensional or two-dimensional with second dimension 1"""
        if len(v.shape) == 1:
            v = v.reshape(-1, 1)
        elif len(v.shape) != 2 or (len(v.shape) == 2 and v.shape[1] != 1):
            raise ValueError(
                "y should be 1-dimensional or 2-dimensional with second dimension 1")
        return v

    @validator('frac_samples_best')
    def _check_frac_samples_best(cls, v: float) -> float:
        """Validate if frac_samples_best is in (0, 1] range"""
        if not (0 < v <= 1):
            raise ValueError("frac_samples_best should be in (0, 1] range")
        return v

    def resample_data(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform resampling of x and y data.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            x_resampled, y_resampled, y_resampled_weights
        """
        n_samples = int(self.frac_samples_best * len(x))
        idx_res = numba_resample(np.arange(len(x)), n_samples)
        x_res, y_res = x[idx_res], y[idx_res]
        y_res_weights = np.ones_like(y_res)
        return x_res, y_res, y_res_weights

    @staticmethod
    def apply_inverse_transform(y_pred_upper: np.ndarray, y_pred_lower: np.ndarray, y_pred_mean: np.ndarray, list_of_fitted_transformers: List[TransformerMixin]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply inverse transform on y predictions.

        Parameters
        ----------
        y_pred_upper : np.ndarray
            Upper bound predictions.
        y_pred_lower : np.ndarray
            Lower bound predictions.
        y_pred_mean : np.ndarray
            Mean predictions.
        list_of_fitted_transformers : List[TransformerMixin]
            List of fitted transformers.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            y_pred_upper_transformed, y_pred_lower_transformed, y_pred_mean_transformed
        """
        if list_of_fitted_transformers:
            for ytr in reversed(list_of_fitted_transformers):
                y_pred_upper = ytr.inverse_transform(
                    y_pred_upper.reshape(-1, 1)).reshape(-1,)
                y_pred_lower = ytr.inverse_transform(
                    y_pred_lower.reshape(-1, 1)).reshape(-1,)
                y_pred_mean = ytr.inverse_transform(
                    y_pred_mean.reshape(-1, 1)).reshape(-1,)
        return y_pred_upper, y_pred_lower, y_pred_mean

    @staticmethod
    def apply_reversify(y_pred_upper: np.ndarray, y_pred_lower: np.ndarray, y_pred_mean: np.ndarray, reversifyfn: Optional[Callable[[np.ndarray], np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply a reverse function to y predictions.

        Parameters
        ----------
        y_pred_upper : np.ndarray
            Upper bound predictions.
        y_pred_lower : np.ndarray
            Lower bound predictions.
        y_pred_mean : np.ndarray
            Mean predictions.
        reversifyfn : Callable[[np.ndarray], np.ndarray], optional
            Function to reverse predictions, by default None

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            y_pred_upper_reversed, y_pred_lower_reversed, y_pred_mean_reversed
        """
        if reversifyfn is not None:
            y_pred_upper = reversifyfn(y_pred_upper)
            y_pred_lower = reversifyfn(y_pred_lower)
            y_pred_mean = reversifyfn(y_pred_mean)
        return y_pred_upper, y_pred_lower, y_pred_mean

    def bootstrap_func_mp(self, model_handler: ModelHandler, iteration_num: int,
                          property_name: Optional[str] = None, testfoldnum: int = 0) -> Tuple[np.ndarray,
                                                                                              np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform bootstrapping for model training and prediction.

        Parameters
        ----------
        model_handler : ModelHandler
            Model handler object.
        iteration_num : int
            Iteration number.
        property_name : str, optional
            Property name, by default None.
        testfoldnum : int, optional
            Test fold number, by default 0.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Tuple of prediction mean, std, lower, upper, and mean SHAP values.
        """
        x_res, y_res, y_res_weights = self.resample_data(
            model_handler.x, model_handler.y)
        model_handler.x = x_res
        model_handler.y = y_res
        model_handler.y_weights = y_res_weights
        model_handler.transform_data()

        file_path = model_handler.file_path / \
            f'ngb_prop={property_name}_fold={testfoldnum}_bag={iteration_num}.pkl'
        shap_file_path = model_handler.shap_file_path / \
            f'shap_prop={property_name}_fold={testfoldnum}_bag={iteration_num}.pkl'

        model_handler.file_path = file_path
        model_handler.shap_file_path = shap_file_path

        estimator = model_handler.fit_or_load_estimator()
        y_pred_mean, y_pred_std, y_pred_lower, y_pred_upper, shap_values_mean = model_handler.compute_prediction_bounds_and_shap_values(
            x_res)
        y_pred_upper, y_pred_lower, y_pred_mean = BootstrapHandler.apply_inverse_transform(
            y_pred_upper, y_pred_lower, y_pred_mean, model_handler.y_transformer)
        y_pred_upper, y_pred_lower, y_pred_mean = BootstrapHandler.apply_reversify(
            y_pred_upper, y_pred_lower, y_pred_mean, model_handler.reversifyfn)

        y_pred_std = (np.ma.masked_invalid(y_pred_upper) -
                      np.ma.masked_invalid(y_pred_lower))/2
        return np.ma.masked_invalid(y_pred_mean), np.ma.masked_invalid(y_pred_std), np.ma.masked_invalid(y_pred_lower), np.ma.masked_invalid(y_pred_upper), np.ma.masked_invalid(shap_values_mean)
