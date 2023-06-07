
from utils.weightify import Weightify
from utils.reshape import reshape_array
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import TransformerMixin
from pydantic_numpy import NDArrayFp64
from pydantic import BaseModel, Field, validator
from ngboost.scores import LogScore
from ngboost.distns import Normal
from ngboost import NGBRegressor
from joblib import dump, load
import numpy as np
from typing import Any, List, Optional, Tuple
from pathlib import Path
import warnings
import shap
from sklearn.compose import TransformedTargetRegressor
from typing import List, Optional, Dict
import numpy as np
from sklearn.base import TransformerMixin
from ngboost import NGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import TransformedTargetRegressor
from utils.custom_transformers_and_estimators import CustomNGBRegressor, MultipleTransformer


class ModelConfig(BaseModel):
    """
    Model configuration for the NGBRegressor. We use this to have a
    centralized place for model parameters which enhances readability
    and maintainability.
    """
    Base: Any = Field(
        DecisionTreeRegressor(
            criterion='friedman_mse',
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_leaf_nodes=31,
            splitter='best'),
        description="Base learner for NGBRegressor"
    )
    Dist: Any = Normal
    Score: Any = LogScore
    n_estimators: int = 500
    learning_rate: float = 0.04
    col_sample: float = 1.0
    minibatch_frac: float = 1.0
    verbose: bool = False
    natural_gradient: bool = True
    early_stopping_rounds: Optional[int] = 10

    class Config:
        arbitrary_types_allowed: bool = True

    @validator('Base')
    def validate_learner(cls, v):
        if not isinstance(v, DecisionTreeRegressor):
            raise TypeError(
                'Base learner must be an instance of DecisionTreeRegressor')
        return v


class ModelHandler(BaseModel):
    """
    A model handler for performing various tasks including data transformation,
    fitting/loading an estimator, and computing prediction bounds and SHAP values.
    """
    X_train: NDArrayFp64
    y_train: NDArrayFp64
    X_val: Optional[NDArrayFp64] = None
    y_val: Optional[NDArrayFp64] = None
    weight_flag: bool = Field(False, alias="WEIGHT_FLAG")
    fitting_mode: bool = True
    file_path: Optional[Path] = None
    shap_file_path: Optional[Path] = None
    estimator: TransformedTargetRegressor
    X_noise: Optional[NDArrayFp64] = None
    model_config: ModelConfig = ModelConfig()

    class Config:
        arbitrary_types_allowed: bool = True

    @validator('estimator', pre=True)
    def validate_estimator(cls, v: Any) -> Any:
        if not isinstance(v, TransformedTargetRegressor):
            raise ValueError('Invalid estimator')
        return v

    @validator('file_path', 'shap_file_path', pre=True, always=True)
    def default_file_path(cls, v: Optional[Path]) -> Path:
        """Default file path to be used when none is provided."""
        return v or Path.home() / 'desika'

    @staticmethod
    def calculate_weights(y_train: np.ndarray, y_val: Optional[np.ndarray] = None, weight_flag: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        weightifier = Weightify()
        train_weights = weightifier.fit_transform(
            y_train) if weight_flag else np.ones_like(y_train)
        val_weights = weightifier.transform(y_val) if weight_flag and y_val is not None else np.ones_like(
            y_val) if y_val is not None else None

        return reshape_array(train_weights), reshape_array(val_weights)

    def _load_estimator(self) -> NGBRegressor:
        if not self.file_path.exists():
            raise FileNotFoundError(f"File at {self.file_path} not found.")
        return load(self.file_path)

    def _fit_estimator(self) -> TransformedTargetRegressor:
        if self.estimator is None:
            transformations_X = [("standard_scaler", StandardScaler())]
            transformations_y = [
                ("log_transform", FunctionTransformer(
                    np.log1p, inverse_func=np.expm1)),
                ("sqrt_transform", FunctionTransformer(
                    np.sqrt, inverse_func=np.square)),
            ]

            # Create pipelines for X and y
            pipeline_X = Pipeline(transformations_X)
            pipeline_y = MultipleTransformer(transformations_y)

            feature_pipeline = Pipeline([
                ('preprocessing', pipeline_X),
                ('regressor', CustomNGBRegressor(**self.model_config.dict()))
            ])

            self.estimator = TransformedTargetRegressor(
                regressor=feature_pipeline,
                transformer=pipeline_y
            )

        y_train_weights, y_val_weights = self.calculate_weights()

        return self.estimator.fit(
            X=self.X_train, y=self.y_train, X_val=self.X_val, y_val=self.y_val, X_noise=self.X_noise, sample_weight=y_train_weights, val_sample_weight=y_val_weights
        )

    def fit_or_load_estimator(self) -> NGBRegressor:
        """
        Fits the estimator if in fitting mode, otherwise loads it from disk.
        Uses Joblib for efficient disk caching.
        """
        if self.fitting_mode:
            self.estimator = self._fit_estimator()
            dump(self.estimator, self.file_path)
        else:
            self.estimator = self._load_estimator()

        return self.estimator

    def compute_prediction_bounds_and_shap_values(self, X_val: np.ndarray, z_score: float = 1.96) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the prediction bounds and SHAP values of the model. If in fitting mode,
        the SHAP values are calculated and saved to disk. Otherwise, they're loaded from disk.
        """
        self.estimator = self.fit_or_load_estimator()

        y_val_pred_mean = self.estimator.predict(X_val)
        y_val_pred_std = self.estimator.regressor_.named_steps['regressor'].predict_std(
            X_val)
        y_val_pred_mean = reshape_array(y_val_pred_mean)
        y_val_pred_std = reshape_array(y_val_pred_std)
        y_val_pred_upper = reshape_array(
            y_val_pred_mean + z_score * y_val_pred_std)
        y_val_pred_lower = reshape_array(
            y_val_pred_mean - z_score * y_val_pred_std)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Access base model for SHAP explainer
            base_model = self.estimator.regressor_.named_steps['regressor'].base_model

            if self.fitting_mode:
                explainer_mean = shap.TreeExplainer(
                    base_model, data=shap.kmeans(self.X_train, 100), model_output=0)
                shap_values_mean = explainer_mean.shap_values(
                    X_val, check_additivity=False)
                dump(shap_values_mean, self.shap_file_path)
            else:
                if not self.shap_file_path.exists():
                    raise FileNotFoundError(
                        f"File at {self.shap_file_path} not found.")
                shap_values_mean = load(self.shap_file_path)

        return y_val_pred_mean, y_val_pred_std, y_val_pred_lower, y_val_pred_upper, shap_values_mean
