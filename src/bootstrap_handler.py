from numba import jit
from pydantic import BaseModel, Field, validator
from typing import Optional, Tuple
import numpy as np
from src.model_handler import ModelHandler
from pydantic_numpy import NDArrayFp64
from src.data_handler import DataHandler, GalaxyProperty
from utils.odds_and_ends import resample_data


class BootstrapHandler(BaseModel):
    """
    BootstrapHandler class for resampling and reversing transformation and function application.

    Attributes
    ----------
    model_handler : ModelHandler
        Model handler object for accessing x and y data.
    frac_samples_best : float
        Maximum fraction of samples to draw, defaults to 1.0 (meaning the entire dataset is sampled).
    """
    model_handler: ModelHandler
    frac_samples_best: float = Field(0.8, gt=0, le=1)
    galaxy_property: GalaxyProperty = Field(GalaxyProperty.STELLAR_MASS)
    z_score: float = 1.96

    @validator('frac_samples_best')
    def _check_frac_samples_best(cls, v: float) -> float:
        """Validate if frac_samples_best is in (0, 1] range"""
        if not (0 < v <= 1):
            raise ValueError("frac_samples_best should be in (0, 1] range")
        return v

    def bootstrap_func_mp(self, iteration_num: int, property_name: Optional[str] = None, testfoldnum: int = 0) -> Tuple[np.ndarray,
                                                                                                                        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform bootstrapping for model training and prediction.

        Parameters
        ----------
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
        X_res, y_res = resample_data(
            self.model_handler.X_train, self.model_handler.y_train)
        self.model_handler.X_train = X_res
        self.model_handler.y_train = y_res

        file_path = self.model_handler.file_path / \
            f'ngb_prop={property_name}_fold={testfoldnum}_bag={iteration_num}.pkl'
        shap_file_path = self.model_handler.shap_file_path / \
            f'shap_prop={property_name}_fold={testfoldnum}_bag={iteration_num}.pkl'

        self.model_handler.file_path = file_path
        self.model_handler.shap_file_path = shap_file_path

        self.model_handler.fit()
        y_pred_mean, y_pred_std = self.model_handler.predict(X_test=X_res)
        shap_values_mean = self.model_handler.calculate_shap_values(
            X_test=X_res)
        y_pred_lower, y_pred_upper = y_pred_mean - self.z_score * \
            y_pred_std, y_pred_mean + self.z_score * y_pred_std

        y_pred_upper, y_pred_lower, y_pred_mean = DataHandler.postprocess_y(
            ys=(y_pred_upper, y_pred_lower, y_pred_mean), prop=self.galaxy_property)

        y_pred_std = (np.ma.masked_invalid(y_pred_upper) -
                      np.ma.masked_invalid(y_pred_lower))/2
        return np.ma.masked_invalid(y_pred_mean), np.ma.masked_invalid(y_pred_std), np.ma.masked_invalid(y_pred_lower), np.ma.masked_invalid(y_pred_upper), np.ma.masked_invalid(shap_values_mean)
