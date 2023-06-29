import logging
from joblib import dump, load
from sklearn.exceptions import NotFittedError
from typing import Dict, Any
import numpy as np

from src.regressors.customtransformedtarget_regressor import (
    CustomTransformedTargetRegressor, create_estimator)
from src.handlers.model_handler import ModelHandlerConfig
from utils.validate import validate_file_path

from utils.logger import LoggingUtility

logger = LoggingUtility.get_logger(
    __name__, log_file='logs/estimator_handler.log')
logger.info('Saving logs from estimator_handler.py')

# TODO: currently if fitting_mode is False, it will load the estimator from the file_path. If the file_path does not exist, it will raise an error. We should provide an option to use a precreated estimator instead. If both are provided, we'll have to figure out which one to use.


class EstimatorHandler:
    """
    A class to handle loading, saving, and managing the estimator.
    """

    def __init__(self, config: ModelHandlerConfig) -> None:
        self._config = config
        self._estimator = None
        self._fitted = False

    @property
    def estimator(self) -> CustomTransformedTargetRegressor:
        """
        Accessor for the estimator. Raises an exception if the estimator has not been created.
        Returns:
            The estimator object.
        Raises:
            NotFittedError: If the estimator has not been created.
        """
        if self._estimator is None:
            if self._config.precreated_estimator is not None:
                self._estimator = self._config.precreated_estimator
            else:
                raise NotFittedError(
                    "Estimator is not created. Use fit() to create it.")
        return self._estimator

    @estimator.setter
    def estimator(self, value: CustomTransformedTargetRegressor) -> None:
        self._estimator = value

    @property
    def is_fitted(self) -> bool:
        """
        Check if the estimator is fitted.
        Returns:
            True if the estimator is fitted, False otherwise.
        """
        return self._fitted

    def fit(self) -> None:
        """
        Fit the estimator.
        1. If fitting_mode is True, it fits the estimator. Additionally, if model_handler.file_path is provided, it saves the fitted estimator there.
        2. If fitting_mode is False, it loads a saved estimator from the file_path.
        """
        validate_file_path(self._config.file_path,
                           self._config.fitting_mode)

        if self._config.fitting_mode:
            self._create_and_fit_estimator()
        else:
            self._load_estimator()

    def _create_and_fit_estimator(self) -> None:
        """
        1. Check if the estimator has already been created. If so, use it.
        2. If not, create the estimator.
        3. Fit the estimator based on the model configuration and transformers in the model handler.
        4. Save the fitted estimator if a file_path is provided.
        """
        if self._config.precreated_estimator is not None:
            self.estimator = self._config.precreated_estimator
        else:
            self.estimator = create_estimator(
                model_config=self._config.model_config,
                X_transformer=self._config.X_transformer,
                y_transformer=self._config.y_transformer,
                weightifier=self._config.weightifier)

        fit_params = self._get_fit_params()

        self.estimator.fit(self._config.X,
                           self._config.y, **fit_params)

        self._fitted = True
        self._save_estimator()

    def _save_estimator(self):
        if self._config.file_path is not None:
            if self.is_fitted:
                try:
                    dump({
                        "estimator": self.estimator,
                        "is_fitted": self.is_fitted
                    }, self._config.file_path)
                except (ValueError, IOError) as e:
                    logger.warning(f"Failed to save the estimator: {e}")
            else:
                raise NotFittedError("The estimator has not been fitted.")
        else:
            logger.warning("No filename provided. Skipping save.")

    def _load_estimator(self) -> None:
        """
        Load the estimator from a file specified by file_path.
        Raises:
            NotFittedError: If the loaded estimator is not fitted.
        """
        # Load both the estimator and its fitted status
        data = load(self._config.file_path)
        self.estimator = data['estimator']
        self._fitted = data['is_fitted']

        # Check if the loaded estimator is indeed fitted
        if not self._fitted:
            raise NotFittedError("The loaded estimator is not fitted.")

    def _get_fit_params(self) -> Dict[str, Any]:
        """
        Generate the parameters for fitting the estimator.
        Returns:
            A dictionary of fit parameters.
        """
        fit_params = {'weight_flag': self._config.weight_flag}
        if self._config.X_test is not None:
            fit_params['X_val'] = self._config.X_test
            fit_params['y_val'] = self._config.y_test
        return fit_params
