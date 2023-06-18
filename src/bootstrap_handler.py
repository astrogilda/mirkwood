from pydantic import BaseModel, Field, confloat
from typing import Optional, Tuple
import numpy as np
from src.model_handler import ModelHandler
from src.data_handler import DataHandler, GalaxyProperty
from utils.odds_and_ends import resample_data, reshape_to_2d_array
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    frac_samples_best: float = Field(default=0.8, gt=0, le=1)

    def bootstrap_func_mp(self, iteration_num: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform bootstrapping for model training and prediction.

        Parameters
        ----------
        iteration_num : int
            Iteration number for current bootstrap iteration.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Tuple of prediction mean, std, lower, upper, and mean SHAP values.
        """
        msg = "Iteration number must be a non-negative integer."

        if not isinstance(iteration_num, int):
            logger.error(msg)
            raise TypeError(msg)
        elif iteration_num < 0:
            logger.error(msg)
            raise ValueError(msg)

        (X_res, y_res), (X_oob, y_oob) = resample_data(
            self.model_handler.X_train, self.model_handler.y_train, frac_samples=self.frac_samples_best, seed=iteration_num, replace=True)
        self.model_handler.X_train = X_res
        self.model_handler.y_train = y_res

        # file_path = self.model_handler.file_path / \
        #    f'ngb_prop={property_name}_fold={testfoldnum}_bag={iteration_num}.pkl'
        # shap_file_path = self.model_handler.shap_file_path / \
        #    f'shap_prop={property_name}_fold={testfoldnum}_bag={iteration_num}.pkl'

        # self.model_handler.file_path = file_path
        # self.model_handler.shap_file_path = shap_file_path

        X_test = self.model_handler.X_val if self.model_handler.X_val is not None else X_oob
        y_test = self.model_handler.y_val if self.model_handler.y_val is not None else y_oob

        self.model_handler.fit()
        y_pred_mean, y_pred_std = self.model_handler.predict(
            X_test=X_test, return_std=True)
        shap_values_mean = self.model_handler.calculate_shap_values(
            X_test=X_test)

        # Create mask for invalid values
        mask = np.ma.masked_invalid

        return reshape_to_2d_array(mask(y_test)), reshape_to_2d_array(mask(y_pred_mean)), reshape_to_2d_array(mask(y_pred_std)), reshape_to_2d_array(mask(shap_values_mean))
