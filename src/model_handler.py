

# Suppress deprecation warnings
from utils.weightify import Weightify
from utils.odds_and_ends import reshape_to_1d_array, reshape_to_2d_array
from utils.custom_transformers_and_estimators import ModelConfig, XTransformer, YTransformer, create_estimator, CustomTransformedTargetRegressor
import shap
from pydantic_numpy import NDArrayFp64
from pydantic import BaseModel, Field, validator
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from joblib import dump, load
import numpy as np
from typing import Any, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)


class ModelHandler(BaseModel):
    """
    A model handler for performing various tasks including data transformation,
    fitting/loading an estimator, and computing prediction bounds and SHAP values.
    """
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None
    weight_flag: bool = Field(False, alias="WEIGHT_FLAG")
    fitting_mode: bool = True
    file_path: Optional[Path] = None
    shap_file_path: Optional[Path] = None
    estimator: Optional[CustomTransformedTargetRegressor] = None
    model_config: ModelConfig = ModelConfig()
    X_transformer: XTransformer = XTransformer()  # Default XTransformer
    y_transformer: YTransformer = YTransformer()  # Default YTransformer

    class Config:
        arbitrary_types_allowed: bool = True

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.estimator is None:
            self.estimator = create_estimator(
                model_config=self.model_config, X_transformer=self.X_transformer, y_transformer=self.y_transformer)

    @validator('file_path', 'shap_file_path', pre=True, always=True)
    def default_file_path(cls, v: Optional[Path]) -> Path:
        """Default file path to be used when none is provided."""
        return v or Path.home() / 'desika'

    @validator('estimator', pre=True)
    def validate_estimator(cls, v: Optional[CustomTransformedTargetRegressor]) -> Optional[CustomTransformedTargetRegressor]:
        """Ensure provided estimator is of type CustomTransformedTargetRegressor"""
        if v is not None and not isinstance(v, CustomTransformedTargetRegressor):
            raise ValueError('Invalid estimator')
        return v

    def fit(self) -> None:
        """Fit the estimator to the data, and save to a file if provided. If fitting_mode is False, load the estimator from a file."""
        if self.fitting_mode:

            fit_params = {'weight_flag': self.weight_flag}
            if self.X_val is not None and self.y_val is not None:
                fit_params.update(
                    {'X_val': self.X_val, 'y_val': self.y_val})

            self.estimator.fit(self.X_train, self.y_train, **fit_params)

            if self.file_path is not None:
                dump(self.estimator, self.file_path)

        else:
            if self.file_path is not None:
                self.estimator = load(self.file_path)
            else:
                warnings.warn(
                    "No file path provided. Using the default estimator.")

    def predict(self, X_test: Optional[np.ndarray], return_bounds: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X = X_test if X_test is not None else self.X_val
        predicted_mean = reshape_to_1d_array(self.estimator.predict(X))
        predicted_std = reshape_to_1d_array(
            self.estimator.regressor_.named_steps['regressor'].predict_std(X)) if return_bounds else None
        return predicted_mean, predicted_std

    def calculate_shap_values(self, explainer: Any, X_test: Optional[np.ndarray]) -> np.ndarray:
        """Calculate SHAP values, based on whether fitting mode is enabled or not. If not in fitting mode, load the SHAP explainer from a file."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            base_model = self.estimator.regressor_.named_steps['regressor'].base_model
            X = X_test if X_test is not None else self.X_val

            if self.fitting_mode:
                explainer_mean = shap.TreeExplainer(
                    base_model, data=shap.kmeans(self.X_train, 100), model_output=0)
                shap_values_mean = explainer_mean.shap_values(
                    X, check_additivity=False)
                if self.shap_file_path is not None:
                    dump(explainer_mean, self.shap_file_path)

            else:
                if self.shap_file_path is None or not self.shap_file_path.exists():
                    raise FileNotFoundError(
                        f"File at {self.shap_file_path or 'provided path'} not found.")
                explainer_mean = load(self.shap_file_path)
                shap_values_mean = explainer_mean.shap_values(
                    X, check_additivity=False)

            return shap_values_mean
