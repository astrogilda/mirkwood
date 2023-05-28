
# Suppress all warnings
from typing import Optional, Tuple, Union, List, Dict, Callable, Any
from pydantic import BaseModel, Field, validator, parse_obj_as
from pydantic_numpy import NDArray, NDArrayFp32
from sklearn.base import TransformerMixin
from pathlib import Path
from joblib import dump, load
import numpy as np
from warnings import catch_warnings, simplefilter
import shap
from numba import njit, jit
from ngboost import NGBRegressor
from utils.custom_pydantic_classes import TransformerMixinField, NGBRegressorField
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class ModelConfig(BaseModel):
    """
    Model configuration for the NGBRegressor. We use this to have a
    centralized place for model parameters which enhances readability
    and maintainability.
    """
    n_estimators: int = 500
    learning_rate: float = 0.04
    col_sample: float = 1.0
    minibatch_frac: float = 1.0
    verbose: bool = False
    natural_gradient: bool = True


class ModelHandler(BaseModel):
    """
    A model handler for performing various tasks including data transformation,
    fitting/loading an estimator, and computing prediction bounds and SHAP values.
    """
    x: NDArrayFp32
    y: NDArrayFp32

    x_transformer: Optional[Any] = None
    y_transformer: Optional[List[Any]] = None
    fitting_mode: bool = True
    file_path: Optional[Path] = None
    shap_file_path: Optional[Path] = None
    estimator: Optional[Any] = None
    x_noise: Optional[NDArrayFp32] = None
    y_weights: Optional[NDArrayFp32] = None
    model_config: ModelConfig = ModelConfig()  # incorporate the model config
    arbitrary_types_allowed: bool = True

    @validator('x_transformer', 'y_transformer', 'estimator', pre=True)
    def validate_transformers_and_estimator(cls, v: Any) -> Any:
        if isinstance(v, list):
            if not all(isinstance(elem, TransformerMixin) for elem in v):
                raise ValueError('Invalid transformer')
        elif v is not None and not (isinstance(v, TransformerMixin) or isinstance(v, NGBRegressor)):
            raise ValueError('Invalid transformer or estimator')
        return v

    @validator('y_weights')
    def validate_y_weights_shape(cls, v, values):
        if v is not None and v.shape != values['y'].shape:
            raise ValueError("y and y_weights must have the same shape")
        return v

    @validator('y_transformer', pre=True)
    def make_list(cls, v):
        if v is None:
            return v
        return v if isinstance(v, list) else [v]

    @validator('file_path', pre=True, always=True)
    def default_file_path(cls, v: Optional[Path]) -> Path:
        """Default file path to be used when none is provided."""
        return v or Path.home() / 'desika'

    @validator('shap_file_path', pre=True, always=True)
    def default_shap_file_path(cls, v: Optional[Path]) -> Path:
        """Default shap file path to be used when none is provided."""
        return v or Path.home() / 'desika'

    def transform_data(self) -> Tuple[np.ndarray, np.ndarray, List[TransformerMixin]]:
        """
        Transforms input data using the provided transformers.
        The transformers are also fitted if required.
        """
        if self.x_transformer is not None:
            self.x_transformer = self.x_transformer.fit(self.x)
            self.x = self.x_transformer.transform(self.x)

        list_of_fitted_transformers = []
        if self.y_transformer is not None:
            for ytr in self.y_transformer:
                ytr = ytr.fit(self.y.reshape(-1, 1))
                self.y = ytr.transform(self.y.reshape(-1, 1)).reshape(-1,)
                list_of_fitted_transformers.append(ytr)

        return self.x, self.y, list_of_fitted_transformers

    def fit_or_load_estimator(self) -> NGBRegressor:
        """
        Fits the estimator if in fitting mode, otherwise loads it from disk.
        Uses Joblib for efficient disk caching.
        """
        if self.fitting_mode:
            if self.x is None or self.y is None:
                raise ValueError("Both x and y need to be set for fitting.")

            # Instantiate a new estimator if it's None
            if self.estimator is None:
                self.estimator = NGBRegressor(
                    n_estimators=self.model_config.n_estimators,
                    learning_rate=self.model_config.learning_rate,
                    col_sample=self.model_config.col_sample,
                    minibatch_frac=self.model_config.minibatch_frac,
                    verbose=self.model_config.verbose,
                    natural_gradient=self.model_config.natural_gradient,
                )

            if self.y_weights is None:
                self.y_weights = np.ones_like(self.y)

            fitted_estimator = self.estimator.fit(
                self.x, self.y, X_noise=self.x_noise, sample_weight=self.y_weights,
            )
            # Joblib dump for efficiency
            dump(fitted_estimator, self.file_path)
        else:
            if not self.file_path.exists():
                raise FileNotFoundError(f"File at {self.file_path} not found.")

            self.estimator = load(self.file_path)  # Joblib load for efficiency

        return self.estimator

    @staticmethod
    @jit(nopython=True)
    def _reshape_array(array: np.ndarray) -> np.ndarray:
        """
        Reshape an array into a 1D array. Uses Numba for JIT compilation and
        faster execution.
        """
        return array.reshape(-1,)

    def compute_prediction_bounds_and_shap_values(self, x_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the prediction bounds and SHAP values of the model. If in fitting mode,
        the SHAP values are calculated and saved to disk. Otherwise, they're loaded from disk.
        """

        if not hasattr(self.estimator, 'init_params') or self.estimator.init_params is None:
            if self.fitting_mode:
                self.estimator = self.fit_or_load_estimator()
            else:
                raise ValueError("Model is not fitted.")

        y_pred = self.estimator.pred_dist(x_val)
        y_pred_mean = self._reshape_array(y_pred.loc)
        y_pred_std = self._reshape_array(y_pred.scale)
        y_pred_upper = self._reshape_array(y_pred_mean + y_pred_std)
        y_pred_lower = self._reshape_array(y_pred_mean - y_pred_std)

        with catch_warnings():
            simplefilter("ignore")
            if self.fitting_mode:
                explainer_mean = shap.TreeExplainer(
                    self.estimator, data=shap.kmeans(self.x, 100), model_output=0)
                shap_values_mean = explainer_mean.shap_values(
                    x_val, check_additivity=False)
                # Joblib dump for efficiency
                dump(shap_values_mean, self.shap_file_path)
            else:
                if not self.shap_file_path.exists():
                    raise FileNotFoundError(
                        f"File at {self.shap_file_path} not found.")
                # Joblib load for efficiency
                shap_values_mean = load(self.shap_file_path)

        return y_pred_mean, y_pred_std, y_pred_lower, y_pred_upper, shap_values_mean
