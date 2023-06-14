

# Suppress deprecation warnings
from pydantic import BaseModel, Field, validator
from sklearn.exceptions import NotFittedError
from utils.odds_and_ends import reshape_to_1d_array
from utils.custom_transformers_and_estimators import ModelConfig, XTransformer, YTransformer, create_estimator, CustomTransformedTargetRegressor
import shap

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from joblib import dump, load
import numpy as np
from typing import Any, Optional, Tuple, List
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
    fitting_mode: bool = True  # common for both prediction and shap values
    file_path: Optional[Path] = None
    shap_file_path: Optional[Path] = None
    estimator: Optional[CustomTransformedTargetRegressor] = None
    model_config: ModelConfig = ModelConfig()
    X_transformer: XTransformer = XTransformer()
    y_transformer: YTransformer = YTransformer()

    class Config:
        arbitrary_types_allowed: bool = True

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.estimator is None:
            self.estimator = create_estimator(
                model_config=self.model_config, X_transformer=self.X_transformer, y_transformer=self.y_transformer)

    @property
    def is_fitted(self) -> bool:
        try:
            # Check if the estimator has been fit
            self.estimator.predict(self.X_train[0:1])
            return True
        except NotFittedError:
            return False

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
            if self.file_path is None or not self.file_path.exists():
                raise FileNotFoundError(
                    f"File at {self.file_path or 'provided path'} not found.")
            self.estimator = load(self.file_path)

    def predict(self, X_test: Optional[np.ndarray], return_bounds: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.fitting_mode and not self.is_fitted:
            raise NotFittedError("Estimator is not fitted.")

        X = X_test if X_test is not None else self.X_val
        predicted_mean = reshape_to_1d_array(self.estimator.predict(X))
        predicted_std = reshape_to_1d_array(
            self.estimator.predict_std(X)) if return_bounds else None
        return predicted_mean, predicted_std

    def calculate_shap_values(self, X_test: Optional[np.ndarray], feature_names: List[str]) -> np.ndarray:
        """Calculate SHAP values, based on whether fitting mode is enabled or not. If not in fitting mode, load the SHAP explainer from a file."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.fitting_mode and not self.is_fitted:
                raise NotFittedError("Estimator is not fitted.")

            X = X_test if X_test is not None else self.X_val

            if self.fitting_mode:
                base_model = self.estimator.regressor_.named_steps['regressor'].base_model
                if self.X_train.shape[0] > 200:
                    data = shap.kmeans(self.X_train, 100)
                else:
                    data = self.X_train
                print(f"data.shape = {data.shape}")
                explainer_mean = shap.TreeExplainer(
                    base_model, data=data, model_output=0, feature_names=feature_names)
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
