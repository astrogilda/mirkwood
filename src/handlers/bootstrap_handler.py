from typing import Tuple
from pydantic import BaseModel, Field
import numpy as np
import logging
from sklearn.utils import check_X_y
from pathlib import Path

from src.handlers.model_handler import ModelHandler
from utils.resample import Resampler, ResamplerConfig
from utils.reshape import reshape_to_2d_array, reshape_to_1d_array
from joblib import dump
# In module1.py

from utils.logger import LoggingUtility

logger = LoggingUtility.get_logger(
    __name__, log_file='logs/bootstrap_handler.log')
logger.info('Saving logs from bootstrap_handler.py')


class BootstrapHandlerConfig(BaseModel):
    """
    BootstrapHandlerConfig class for setting up bootstrapping process.

    Attributes
    ----------
    frac_samples : float
        Maximum fraction of samples to draw, defaults to 1.0 (meaning the entire dataset is sampled).
    replace : bool
        Whether to sample with replacement or not, defaults to True.
    """
    frac_samples: float = Field(default=0.8, gt=0, le=1)
    replace: bool = Field(default=True)


class BootstrapHandler():
    """
    BootstrapHandler class for resampling and reversing transformation and function application.
    """

    def __init__(self, model_handler: ModelHandler, bootstrap_config: BootstrapHandlerConfig):
        """
        Parameters
        model_handler : ModelHandler
            Model handler object for accessing x and y data.
        bootstrap_config : BootstrapHandlerConfig
            Bootstrapping configuration object.
        """
        self.model_handler = model_handler
        self.bootstrap_config = bootstrap_config
        if not isinstance(self.model_handler, ModelHandler):
            raise TypeError(
                "model_handler must be of type ModelHandler or a subclass of it")
        if not isinstance(self.bootstrap_config, BootstrapHandlerConfig):
            raise TypeError(
                "bootstrap_config must be of type BootstrapHandlerConfig or a subclass of it")

    def _resample(self, seed: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (X_ib, y_ib), (X_oob, y_oob), ib_idx, oob_idx = Resampler(
            ResamplerConfig(
                frac_samples=self.bootstrap_config.frac_samples, seed=seed, replace=self.bootstrap_config.replace)
        ).resample_data([self.model_handler._config.X_train, self.model_handler._config.y_train])

        X_ib, y_ib = check_X_y(
            X_ib, reshape_to_1d_array(y_ib), force_all_finite=True, y_numeric=True)

        self.model_handler._config.X_train = X_ib
        self.model_handler._config.y_train = y_ib

        return X_ib, y_ib, X_oob, y_oob, ib_idx, oob_idx

    def bootstrap(self, seed: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Parameters
        seed : int
            Seed for the random number generator. Used for reproducibility.
        """
        if not isinstance(seed, int):
            raise TypeError("seed must be of type int")
        if seed < 0 or seed > 2**32-1:
            raise ValueError("seed must be between 0 and 2**32-1")

        logger.info("Starting bootstrap...")

        X_ib, y_ib, X_oob, y_oob, ib_idx, oob_idx = self._resample(seed=seed)

        X_test = self.model_handler._config.X_val if self.model_handler._config.X_val is not None else X_oob
        y_test = self.model_handler._config.y_val if self.model_handler._config.y_val is not None else y_oob

        # Save indices only if we are saving the estimator
        if self.model_handler._config.file_path is not None:
            idx_file_path = self.model_handler._config.file_path.parent.joinpath(
                f'idx_bootstrap_{seed}.pkl')
            try:
                dump({
                    "ib_idx": ib_idx,
                    "oob_idx": oob_idx
                }, idx_file_path)
            except (ValueError, IOError) as e:
                logger.warning(
                    f"Failed to save the indices for bootstrap_{seed}: {e}")

        # Modify file names
        if self.model_handler._config.file_path is not None:
            self.model_handler._config.file_path = Path(str(self.model_handler._config.file_path).replace(
                ".pkl", f"_bootstrap_{seed}.pkl"))
        if self.model_handler._config.shap_file_path is not None:
            self.model_handler._config.shap_file_path = Path(str(self.model_handler._config.shap_file_path).replace(
                ".pkl", f"_bootstrap_{seed}.pkl"))

        logger.info("Fitting the estimator...")
        self.model_handler.fit()

        logger.info("Making predictions...")
        y_pred_mean, y_pred_std = self.model_handler.predict(X_test=X_test)

        logger.info("Fitting the explainer...")
        self.model_handler.create_explainer()

        logger.info("Calculating SHAP values...")
        shap_values_mean = self.model_handler.calculate_shap_values(
            X_test=X_test)

        mask = np.ma.masked_invalid

        return reshape_to_2d_array(mask(y_test).data), reshape_to_2d_array(mask(y_pred_mean).data), reshape_to_2d_array(mask(y_pred_std).data), mask(shap_values_mean).data
        # , reshape_to_2d_array(mask(ib_idx).data), reshape_to_2d_array(mask(oob_idx).data)
