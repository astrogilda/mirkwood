from utils.logger import LoggingUtility
from typing import Optional, Dict
from pydantic import BaseModel, validator
import warnings
from pathlib import Path
from pydantic import BaseModel, Field, root_validator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from typing import Any, Optional, Tuple, List, Union
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import numpy as np
import shap
from sklearn.utils.validation import check_array
from sklearn.preprocessing import StandardScaler


from utils.validate import validate_input
from utils.reshape import reshape_to_1d_array
from utils.weightify import Weightify
from src.regressors.customngb_regressor import ModelConfig
from src.transformers.xandy_transformers import XTransformer, YTransformer, TransformerConfig
from src.regressors.customtransformedtarget_regressor import CustomTransformedTargetRegressor, create_estimator
from src.transformers.yscaler import GalaxyProperty, YScaler

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)

# Set up logging
logger = LoggingUtility.get_logger(
    __name__, log_file='logs/model_handler.log')
logger.info('Saving logs from model_handler.py')

# TODO: isolate common validations in TrainPredictHandlerConfig and ModelHandlerConfig into a separate class. Then have TrainPredictHandlerConfig and ModelHandlerConfig inherit from that class. This will avoid having to repeat the same validations in both classes. Ensure that attribute names are common between the two classes.

# TODO: more robust way to handle assignment of new X_transformer, y_transformer, of weightifier, after modelhandlerconfig has been instantiated. so one can continue to leverage pydantic's validations


class ModelHandlerBaseConfig(BaseModel):
    """
    Configuration class for ModelHandler.
    """
    X: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    feature_names: List[str]
    # Field(default=GalaxyProperty.STELLAR_MASS)
    galaxy_property: Optional[GalaxyProperty] = None
    X_test: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    weight_flag: bool = Field(
        False, description="Flag to indicate whether to use weights in the loss function")
    fitting_mode: bool = Field(
        True, description="Flag to indicate whether to fit the model or not")
    file_path: Optional[Path] = None
    shap_file_path: Optional[Path] = None
    shap_values_path: Optional[Path] = None
    model_config: ModelConfig = ModelConfig()
    X_transformer: XTransformer = XTransformer(transformers=None)
    y_transformer: YTransformer = YTransformer(
        transformers=[TransformerConfig(name="rescale_y", transformer=YScaler()), TransformerConfig(name="standard_scaler", transformer=StandardScaler())])
    weightifier: Weightify = Weightify()

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

    @validator('X_test', 'y_test', 'X', 'y', pre=True)
    def check_pairs(cls, value, values, field):
        if field.name in ['X_test', 'y_test']:
            cls._validate_X_y_pair('X_test', 'y_test', values)
        elif field.name in ['X', 'y']:
            cls._validate_X_y_pair('X', 'y', values)
        return value

    @validator('X', 'X_test', pre=True, always=True)
    def validate_X_arrays(cls, value, values):
        if 'fitting_mode' in values and values['fitting_mode'] and value is None:
            raise ValueError(
                'X cannot be None when fitting_mode is True')
        if value is not None:
            value = check_array(value)
        return value

    @validator('y', 'y_test', pre=True, always=True)
    def validate_y_arrays(cls, value, values):
        if 'fitting_mode' in values and values['fitting_mode'] and value is None:
            raise ValueError(
                'y cannot be None when fitting_mode is True')
        if value is not None:
            value = check_array(value, ensure_2d=False)
            if not ((value.ndim == 1) or (value.ndim == 2 and value.shape[1] == 1)):
                raise ValueError("y arrays must be 1-dimensional")
        return value

    @validator('feature_names', always=True)
    def validate_feature_names(cls, value, values):
        if 'X' in values and values['X'] is not None:
            X = values['X']
            if len(value) != X.shape[1]:
                raise ValueError(
                    f"The length of 'feature_names' ({len(value)}) must be the same as the number of columns in 'X' ({X.shape[1]})")
        return value

    @root_validator
    def validate(cls, values):
        if 'fitting_mode' in values and values['fitting_mode']:
            if 'X' not in values or values['X'] is None:
                raise ValueError(
                    'X cannot be None when fitting_mode is True')
            if 'y' not in values or values['y'] is None:
                raise ValueError(
                    'y cannot be None when fitting_mode is True')
        cls._validate_X_y_pair('X', 'y', values)
        cls._validate_X_y_pair('X_test', 'y_test', values)
        return values

    @validator('file_path', pre=True)
    def validate_file_path(cls, value, values):
        if not values.get('fitting_mode') and not value.exists():
            raise FileNotFoundError(f"File at {value} not found.")
        return value

    def __str__(self):
        """
        This will return a string representing the configuration object.
        """
        return f"ModelHandlerBaseConfig({self.dict()})"


class ModelHandlerConfig(ModelHandlerBaseConfig):
    precreated_estimator: Optional[CustomTransformedTargetRegressor] = None
    precreated_explainer: Optional[shap.TreeExplainer] = None

    class Config:
        arbitrary_type_allowed = True

    def __str__(self):
        """
        This will return a string representing the configuration object.
        """
        return f"ModelHandlerConfig({self.dict()})"


class ModelHandler:
    """
    A model handler for performing various tasks including data transformation,
    fitting/loading an estimator, and computing prediction bounds and SHAP values.
    """

    def __init__(self, config: ModelHandlerConfig) -> None:

        self._config = config
        from src.handlers.estimator_handler import EstimatorHandler
        from src.handlers.shap_handler import ShapHandler

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

    def predict(self, X_test: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict target variable and uncertainty given features.
        Args:
            X_test: Array of test features.
        Returns:
            Tuple of arrays: predicted target variable and uncertainty.
        """
        if X_test is not None:
            validate_input(np.ndarray, arg1=X_test)
        y_pred_mean, y_pred_std = self._predict_with_estimator(X_test)

        return y_pred_mean, y_pred_std

    def create_explainer(self) -> None:
        """
        Fit the SHAP explainer using the fitted estimator.
        Args:
            X: Array of training features. If None, X from the config will be used.
        """
        if not self._estimator_handler.is_fitted:
            raise NotFittedError(
                "This ModelHandler instance is not fitted yet. Call 'fit' with appropriate arguments before using this explainer.")
        self._shap_handler.create(
            self.estimator.regressor_.named_steps['regressor'])

    def calculate_shap_values(self, X_test: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Calculate SHAP values for the given test features.
        Args:
            X_test: Array of test features.
        Returns:
            Array of SHAP values.
        """
        if X_test is not None:
            validate_input(np.ndarray, arg1=X_test)
        shap_pred = self._calculate_shap_values_with_explainer(X_test)
        return shap_pred

    def _calculate_shap_values_with_explainer(self, X_test: Optional[np.ndarray]) -> Optional[np.ndarray]:

        if not self._estimator_handler.is_fitted:
            raise NotFittedError(
                "This ModelHandler instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        if X_test is None and self._config.X_test is None:
            raise ValueError(
                "X_test cannot be None when X_test in the config is None.")

        X = X_test if X_test is not None else self._config.X_test
        shap_pred = self.explainer.shap_values(X_test, check_additivity=False)
        return shap_pred

    def _predict_with_estimator(self, X_test: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
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

        if X_test is None and self._config.X_test is None:
            raise ValueError(
                "X_test cannot be None when X_test in the config is None.")
        X = X_test if X_test is not None else self._config.X_test
        y_pred = self.estimator.predict(X)
        y_pred_std = self.estimator.predict_std(X)

        return reshape_to_1d_array(y_pred), reshape_to_1d_array(y_pred_std) if y_pred_std is not None else None
