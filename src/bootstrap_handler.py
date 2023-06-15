from pydantic import BaseModel, Field, confloat
from typing import Optional, Tuple
import numpy as np
from src.model_handler import ModelHandler
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
    frac_samples_best: float = Field(default=0.8, gt=0, le=1)
    galaxy_property: GalaxyProperty = Field(
        default=GalaxyProperty.STELLAR_MASS)
    z_score: confloat(gt=0, le=5) = Field(
        default=1.96,
        description="The z-score for the confidence interval. Defaults to 1.96, which corresponds to a 95 per cent confidence interval."
    )

    def bootstrap_func_mp(self, iteration_num: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        X_res, y_res = resample_data(
            self.model_handler.X_train, self.model_handler.y_train, frac_samples=self.frac_samples_best)
        self.model_handler.X_train = X_res
        self.model_handler.y_train = y_res

        # file_path = self.model_handler.file_path / \
        #    f'ngb_prop={property_name}_fold={testfoldnum}_bag={iteration_num}.pkl'
        # shap_file_path = self.model_handler.shap_file_path / \
        #    f'shap_prop={property_name}_fold={testfoldnum}_bag={iteration_num}.pkl'

        # self.model_handler.file_path = file_path
        # self.model_handler.shap_file_path = shap_file_path

        self.model_handler.fit()
        y_pred_mean, y_pred_std = self.model_handler.predict(
            X_test=X_res, return_std=True)
        shap_values_mean = self.model_handler.calculate_shap_values(
            X_test=X_res)
        y_pred_lower, y_pred_upper = y_pred_mean - self.z_score * \
            y_pred_std, y_pred_mean + self.z_score * y_pred_std
        print(y_pred_lower)
        print(y_pred_upper)
        # Postprocess prediction values
        y_pred_upper, y_pred_lower, y_pred_mean = DataHandler().postprocess_y(
            (y_pred_upper, y_pred_lower, y_pred_mean), prop=self.galaxy_property)

        # Create mask for invalid values
        mask = np.ma.masked_invalid

        # Re-calculate std after postprocessing
        y_pred_std = (mask(y_pred_upper) -
                      mask(y_pred_lower))/2

        return mask(y_res), mask(y_pred_mean), mask(y_pred_std), mask(y_pred_lower), mask(y_pred_upper), mask(shap_values_mean)
