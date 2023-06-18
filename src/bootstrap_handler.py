from typing import Tuple
from pydantic import BaseModel, Field
from src.model_handler import ModelHandler
from utils.resample import Resampler, ResamplerConfig
from utils.reshape import reshape_to_2d_array
import numpy as np
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BootstrapConfig(BaseModel):
    """
    BootstrapConfig class for setting up bootstrapping process.

    Attributes
    ----------
    frac_samples : float
        Maximum fraction of samples to draw, defaults to 1.0 (meaning the entire dataset is sampled).
    seed : int
        Seed for the random number generator. Used for reproducibility.
    """
    frac_samples: float = Field(default=0.8, gt=0, le=1)
    seed: int = Field(default=None, ge=0, le=2**32-1)


class BootstrapHandler(BaseModel):
    """
    BootstrapHandler class for resampling and reversing transformation and function application.

    Attributes
    ----------
    model_handler : ModelHandler
        Model handler object for accessing x and y data.
    bootstrap_config : BootstrapConfig
        Bootstrapping configuration object.
    """
    model_handler: ModelHandler
    bootstrap_config: BootstrapConfig

    def __init__(self, model_handler: ModelHandler, bootstrap_config: BootstrapConfig):
        self.model_handler = model_handler
        self.bootstrap_config = bootstrap_config

    def _resample(self):
        (X_ib, y_ib), (X_oob, y_oob), (X_ib_idx, y_ib_idx), (X_oob_idx, y_oob_idx) = Resampler(
            ResamplerConfig(
                frac_samples=self.bootstrap_config.frac_samples, seed=self.bootstrap_config.seed, replace=True)
        ).resample_data(self.model_handler._config.X_train, self.model_handler._config.y_train)

        self.model_handler._config.X_train = X_ib
        self.model_handler._config.y_train = y_ib

        return X_ib, y_ib, X_oob, y_oob, X_ib_idx, y_ib_idx, X_oob_idx, y_oob_idx

    def bootstrap(self, iteration_num: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logger.info(f"Starting bootstrap iteration: {iteration_num}")

        X_ib, y_ib, X_oob, y_oob, X_ib_idx, y_ib_idx, X_oob_idx, y_oob_idx = self._resample(
            iteration_num)

        X_test = self.model_handler._config.X_val if self.model_handler._config.X_val is not None else X_oob
        y_test = self.model_handler._config.y_val if self.model_handler._config.y_val is not None else y_oob

        logger.info("Fitting the model...")
        self.model_handler.fit()

        logger.info("Making predictions...")
        y_pred_mean, y_pred_std = self.model_handler.predict(X_test=X_test)

        logger.info("Calculating SHAP values...")
        shap_values_mean = self.model_handler.calculate_shap_values(
            X_test=X_test)

        mask = np.ma.masked_invalid

        return reshape_to_2d_array(mask(y_test)), reshape_to_2d_array(mask(y_pred_mean)), reshape_to_2d_array(mask(y_pred_std)), reshape_to_2d_array(mask(shap_values_mean))
