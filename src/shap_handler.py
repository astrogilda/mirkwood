
from numpy import ndarray
from pydantic import BaseModel, validator
from typing import Optional, Tuple
import logging
from joblib import dump, load
from sklearn.exceptions import NotFittedError
from shap import Explainer, TreeExplainer
from src.model_handler import ModelHandlerConfig
from utils.validate import validate_file_path
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
            raise NotFittedError(
                "SHAP Explainer is not created. Use calculate() to create it.")
        return self._explainer

    @explainer.setter
    def explainer(self, value: TreeExplainer) -> None:
        self._explainer = value

    def create(self, fitted_base_estimator: BaseEstimator) -> None:
        """
        Calculate the SHAP Explainer.
        1. If calculation_mode is True, it creates the explainer. Additionally, if model_handler.file_path is provided, it saves the explainer there.
        2. If calculation_mode is False, it loads a saved explainer from the file_path.
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
        if self._config.precreated_explainer is not None:
            self.explainer = self._config.precreated_explainer
        else:
            data_train = self._get_shap_data(self._config.X_train)
            self.explainer = shap.TreeExplainer(
                fitted_base_estimator, data=data_train, model_output=0, feature_names=self._config.feature_names)
            logger.info("SHAP explainer created")

        self._save_explainer()

    def _save_explainer(self) -> None:
        """
        Save the explainer to a file specified by file_path.
        """
        dump(self.explainer, self._config.shap_file_path)

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


logger = logging.getLogger(__name__)


class ModelHandlerConfig(BaseModel):
    """
    Configuration class for ModelHandler.
    """
    X_train: Optional[ndarray]
    y_train: Optional[ndarray]
    X_val: Optional[ndarray]
    y_val: Optional[ndarray]
    estimator_file_path: str
    shap_file_path: str
    shap_values_path: str

    @validator('X_val', 'y_val', always=True)
    def check_val(cls, value, values, field):
        if values.get('X_val') is not None and values.get('y_val') is None:
            raise ValueError(
                f"{field.name} is required when X_val is provided")
        return value

    @validator('X_train', 'y_train', 'X_val', 'y_val', always=True)
    def check_lengths(cls, value, values, field):
        if values.get('X_train') is not None and values.get('y_train') is not None:
            if len(values['X_train']) != len(values['y_train']):
                raise ValueError("Lengths of X_train and y_train must match")
        if values.get('X_val') is not None and values.get('y_val') is not None:
            if len(values['X_val']) != len(values['y_val']):
                raise ValueError("Lengths of X_val and y_val must match")
        return value

    class Config:
        arbitrary_types_allowed = True


class ModelHandler:
    """
    A model handler for performing various tasks including data transformation,
    fitting/loading an estimator, and computing prediction bounds and SHAP values.
    """

    def __init__(self, config: ModelHandlerConfig) -> None:
        self._config = config
        self._estimator = None
        self._shap_handler = None
        self._shap_values = None

    @property
    def estimator(self) -> BaseEstimator:
        if self._estimator is None:
            try:
                self._estimator = load(self._config.estimator_file_path)
            except Exception as e:
                logger.error(
                    f"Error occurred while loading the estimator: {e}")
        return self._estimator

    @estimator.setter
    def estimator(self, value: BaseEstimator) -> None:
        self._estimator = value
        self._save_estimator()

    @property
    def explainer(self) -> shap.TreeExplainer:
        if self._shap_handler is None:
            self._shap_handler = ShapHandler(self._config)
        return self._shap_handler.explainer

    @explainer.setter
    def explainer(self, value: shap.TreeExplainer) -> None:
        if self._shap_handler is None:
            self._shap_handler = ShapHandler(self._config)
        self._shap_handler.explainer = value

    @property
    def shap_values(self) -> ndarray:
        if self._shap_values is None:
            self._shap_values = self._shap_handler.load_shap_values(
                self._config.shap_values_path)
        return self._shap_values

    def fit(self) -> None:
        if self._config.X_train is None or self._config.y_train is None:
            raise ValueError(
                "X_train or y_train is not set in the configuration.")
        # Fit the estimator with the training data.
        self.estimator.fit(self._config.X_train, self._config.y_train)

    def predict(self, X: Optional[ndarray]) -> Optional[ndarray]:
        return self.estimator.predict(X) if self.estimator is not None else None

    def create_explainer(self) -> None:
        # Assumes estimator is a Tree-based model.
        self.explainer = shap.TreeExplainer(self.estimator)

    def calculate_and_save_shap_values(self, X: Optional[ndarray]) -> None:
        self._shap_values = self.explainer.shap_values(X)
        self._shap_handler.save_shap_values(
            self._config.shap_values_path, self._shap_values)

    def _save_estimator(self) -> None:
        try:
            dump(self.estimator, self._config.estimator_file_path)
        except Exception as e:
            logger.error(f"Error occurred while saving the estimator: {e}")
