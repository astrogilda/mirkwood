import logging
from joblib import dump, load
from sklearn.exceptions import NotFittedError
from shap import Explainer, TreeExplainer
from src.model_handler import ModelHandlerConfig
from utils.validate import validate_file_path, is_estimator_fitted
from sklearn.base import BaseEstimator
import numpy as np
import shap

logger = logging.getLogger(__name__)


class ShapHandler:
    """
    A class to handle loading, saving, and managing the SHAP Explainer.
    """

    def __init__(self, config: ModelHandlerConfig) -> None:
        self._config = config
        self._explainer = None

    @property
    def explainer(self) -> TreeExplainer:
        """
        Accessor for the SHAP Explainer. Raises an exception if the explainer has not been created.
        Returns:
            The SHAP Explainer.
        Raises:
            NotFittedError: If the explainer has not been created.
        """
        if self._explainer is None:
            if self._config.precreated_explainer is not None:
                self._explainer = self._config.precreated_explainer
            else:
                raise NotFittedError(
                    "SHAP Explainer is not created. Use calculate() to create it.")
        return self._explainer

    @explainer.setter
    def explainer(self, value: TreeExplainer) -> None:
        self._explainer = value

    def create(self, fitted_base_estimator: BaseEstimator) -> None:
        """
        Calculate the SHAP Explainer.
        1. If fitting_mode is True, it creates the explainer. Additionally, if model_handler.file_path is provided, it saves the explainer there.
        2. If fitting_mode is False, it loads a saved explainer from the file_path.
        """
        validate_file_path(self._config.shap_file_path,
                           self._config.fitting_mode)

        if self._config.fitting_mode:
            self._create_explainer(fitted_base_estimator)
        else:
            self._load_explainer()

    def _create_explainer(self, fitted_base_estimator: BaseEstimator) -> None:
        """
        1. Check if the explainer has already been created. If so, use it.
        2. If not, create the explainer.
        3. Save the explainer if a file_path is provided.
        """
        if self._explainer is not None:
            logger.warning("SHAP Explainer has already been created.")
            return

        if not hasattr(fitted_base_estimator, "fit"):
            logger.error(
                "The provided estimator has no 'fit' method. Cannot calculate SHAP values.")
            raise ValueError("Invalid estimator: 'fit' method not found.")

        assert is_estimator_fitted(fitted_base_estimator)
        '''
        if not hasattr(fitted_base_estimator, "estimators_"):
            logger.error(
                "The provided estimator has not been fitted. Cannot calculate SHAP values."
            )
            raise ValueError(
                "Invalid estimator: estimator has not been fitted.")
        '''

        if self._config.precreated_explainer is not None:
            self.explainer = self._config.precreated_explainer
        else:
            data_train = self._get_shap_data(self._config.X_train)
            if not hasattr(fitted_base_estimator, "predict_std") or not hasattr(fitted_base_estimator, "predict_dist"):
                self.explainer = shap.TreeExplainer(
                    fitted_base_estimator, data=data_train, feature_names=self._config.feature_names)
            else:
                self.explainer = shap.TreeExplainer(
                    fitted_base_estimator, data=data_train, model_output=0, feature_names=self._config.feature_names)

        logger.info("SHAP explainer created")

        self._save_explainer()

    def _save_explainer(self) -> None:
        """
        Save the explainer to a file specified by file_path.
        """
        if self._explainer is None:
            raise NotFittedError(
                "SHAP Explainer is not created. Use calculate() to create it.")

        if self._config.shap_file_path is not None:
            try:
                dump(self.explainer, self._config.shap_file_path)
            except (ValueError, IOError) as e:
                logger.warning(f"Failed to save the explainer: {e}")
        else:
            logger.warning("No filename provided. Skipping save.")

    def _load_explainer(self) -> None:
        """
        Load the explainer from a file specified by file_path.
        """
        self.explainer = load(self._config.shap_file_path)

    @staticmethod
    def _get_shap_data(X: np.ndarray) -> np.ndarray:
        if X.shape[0] > 200:
            data = shap.kmeans(X, 100).data
        else:
            data = X
        return data
