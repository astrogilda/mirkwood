from typing import Optional, Dict
from pydantic import BaseModel, validator
import logging
import warnings
from joblib import dump, load
from numpy import ndarray
from pathlib import Path
from pydantic import BaseModel, Field, root_validator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from typing import Any, Optional, Tuple, List
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from utils.validate import validate_input
from utils.reshape import reshape_to_1d_array
from utils.custom_transformers_and_estimators import (
    ModelConfig, XTransformer, YTransformer, create_estimator,
    CustomTransformedTargetRegressor, PostProcessY)
from src.data_handler import GalaxyProperty
import numpy as np
import shap
from sklearn.utils.validation import check_array

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ModelHandlerConfig(BaseModel):
    """
    Configuration class for ModelHandler.
    """
    X_train: ndarray
    y_train: ndarray
    feature_names: List[str]
    galaxy_property: Optional[GalaxyProperty] = Field(
        default=GalaxyProperty.STELLAR_MASS)
    X_val: Optional[ndarray] = None
    y_val: Optional[ndarray] = None
    weight_flag: bool = Field(
        False, description="Flag to indicate whether to use weights in the loss function")
    fitting_mode: bool = Field(
        True, description="Flag to indicate whether to fit the model or not")
    file_path: Optional[Path] = None
    shap_file_path: Optional[Path] = None
    shap_values_path: Optional[Path] = None
    model_config: ModelConfig = ModelConfig()
    X_transformer: XTransformer = XTransformer()
    y_transformer: YTransformer = YTransformer()
    precreated_estimator: Optional[CustomTransformedTargetRegressor] = None
    precreated_explainer: Optional[shap.TreeExplainer] = None

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def _validate_X_y_pair(X_name: str, y_name: str, values: Dict[str, Any]) -> None:
        if X_name in values and y_name in values:
            if (values[X_name] is None) != (values[y_name] is None):
                raise ValueError(
                    f"{X_name} and {y_name} must be both None or not None")
            if values[X_name] is not None and values[X_name].shape[0] != len(values[y_name]):
                raise ValueError(
                    f"{X_name} and {y_name} must have the same number of elements")

    @validator('X_val', 'y_val', 'X_train', 'y_train', pre=True)
    def check_pairs(cls, value, values, field):
        if field.name in ['X_val', 'y_val']:
            cls._validate_X_y_pair('X_val', 'y_val', values)
        elif field.name in ['X_train', 'y_train']:
            cls._validate_X_y_pair('X_train', 'y_train', values)
        return value

    @validator('X_train', 'X_val', pre=True, always=True)
    def validate_X_arrays(cls, value):
        if value is not None:
            value = check_array(value)
        return value

    @validator('y_train', 'y_val', pre=True, always=True)
    def validate_y_arrays(cls, value):
        if value is not None:
            value = check_array(value, ensure_2d=False)
            if value.ndim != 1:
                raise ValueError("y arrays must be 1-dimensional")
        return value

    @validator('feature_names', always=True)
    def validate_feature_names(cls, value, values):
        if 'X_train' in values and values['X_train'] is not None:
            X_train = values['X_train']
            if len(value) != X_train.shape[1]:
                raise ValueError(
                    f"The length of 'feature_names' ({len(value)}) must be the same as the number of columns in 'X_train' ({X_train.shape[1]})")
        return value

    @root_validator
    def validate(cls, values):
        cls._validate_X_y_pair('X_train', 'y_train', values)
        cls._validate_X_y_pair('X_val', 'y_val', values)
        return values


class ModelHandler:
    """
    A model handler for performing various tasks including data transformation,
    fitting/loading an estimator, and computing prediction bounds and SHAP values.
    """

    def __init__(self, config: ModelHandlerConfig) -> None:

        self._config = config
        from src.estimator_handler import EstimatorHandler
        from src.shap_handler import ShapHandler

        self._estimator_handler = EstimatorHandler(config)
        self._shap_handler = ShapHandler(config)

    @property
    def estimator(self) -> Optional[CustomTransformedTargetRegressor]:
        """
        Accessor for the estimator from the estimator handler.
        Returns:
            The estimator object.
        """
        return self._estimator_handler.estimator

    @property
    def explainer(self) -> Optional[shap.TreeExplainer]:
        """
        Accessor for the estimator from the estimator handler.
        Returns:
            The estimator object.
        """
        return self._shap_handler.explainer

    def fit(self) -> None:
        """
        Fit the model by invoking the fit method of the estimator handler.
        """
        self._estimator_handler.fit()

    def predict(self, X_test: Optional[ndarray]) -> Tuple[ndarray, ndarray]:
        """
        Predict target variable and uncertainty given features.
        Args:
            X_test: Array of test features.
        Returns:
            Tuple of arrays: predicted target variable and uncertainty.
        """
        validate_input(np.ndarray, arg1=X_test)
        y_pred = self._predict_with_estimator(X_test)
        y_pred_mean, y_pred_std = PostProcessY(prop=self._config.galaxy_property).transform(
            y_pred)
        return y_pred_mean, y_pred_std

    def create_explainer(self) -> None:
        """
        Fit the SHAP explainer using the fitted estimator.
        Args:
            X: Array of training features. If None, X_train from the config will be used.
        """
        if not self._estimator_handler.is_fitted:
            raise NotFittedError(
                "This ModelHandler instance is not fitted yet. Call 'fit' with appropriate arguments before using this explainer.")
        self._shap_handler.create(
            self.estimator.regressor_.named_steps['regressor'])

    def calculate_shap_values(self, X_test: Optional[ndarray]) -> Optional[ndarray]:
        """
        Calculate SHAP values for the given test features.
        Args:
            X_test: Array of test features.
        Returns:
            Array of SHAP values.
        """
        validate_input(np.ndarray, arg1=X_test)
        shap_pred = self._calculate_shap_values_with_explainer(X_test)
        return shap_pred

    def _calculate_shap_values_with_explainer(self, X_test: Optional[ndarray]) -> Optional[ndarray]:

        if not self._estimator_handler.is_fitted:
            raise NotFittedError(
                "This ModelHandler instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        X = X_test if X_test is not None else self._config.X_val
        shap_pred = self.explainer.shap_values(X_test, check_additivity=False)
        return shap_pred

    def _predict_with_estimator(self, X_test: Optional[ndarray]) -> Tuple[ndarray, ndarray]:
        """
        Perform prediction using the estimator.
        Args:
            X_test: Array of test features.
        Returns:
            Tuple of arrays: predicted target variable and uncertainty.
        """
        if not self._estimator_handler.is_fitted:
            raise NotFittedError(
                "This ModelHandler instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        X = X_test if X_test is not None else self._config.X_val
        y_pred = self.estimator.predict(X)
        y_pred_std = self.estimator.predict_std(X)

        if self._config.galaxy_property is not None:
            y_pred, y_pred_std = self._convert_to_original_scale(
                y_pred, y_pred_std)

        return reshape_to_1d_array(y_pred), reshape_to_1d_array(y_pred_std)

    def _convert_to_original_scale(self, predicted_mean: ndarray, predicted_std: Optional[ndarray]) -> Tuple[ndarray, Optional[ndarray]]:
        """
        Convert the predicted values and uncertainties to the original scale.
        Args:
            predicted_mean: Predicted mean values.
            predicted_std: Predicted standard deviations.
        Returns:
            Tuple of arrays: converted predicted mean values and standard deviations.
        """
        post_processor = PostProcessY(self._config.galaxy_property)
        return post_processor.transform(predicted_mean), post_processor.transform(predicted_std) if predicted_std is not None else None
