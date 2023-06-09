
from pydantic import BaseModel, Field
from typing import Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from pydantic_numpy import NDArrayFp64
from pydantic import BaseModel, Field, validator
from utils.weightify import Weightify
from utils.odds_and_ends import reshape_array
from utils.custom_transformers_and_estimators import CustomNGBRegressor, MultipleTransformer, ReshapeTransformer, YTransformer, XTransformer, ModelConfig, CustomTransformedTargetRegressor
from joblib import dump, load
import shap
from pydantic import root_validator
import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)


def create_estimator(model_config: ModelConfig = ModelConfig(),
                     x_transformer: XTransformer = XTransformer(),
                     y_transformer: YTransformer = YTransformer()) -> TransformedTargetRegressor:

    pipeline_X = Pipeline([(transformer.name, transformer.transformer)
                          for transformer in x_transformer.transformers])
    pipeline_y = MultipleTransformer(y_transformer.transformers)

    if x_transformer.transformers:
        feature_pipeline = Pipeline([
            ('preprocessing', pipeline_X),
            ('regressor', CustomNGBRegressor(**model_config.dict()))
        ])
    else:
        feature_pipeline = Pipeline([
            ('regressor', CustomNGBRegressor(**model_config.dict()))
        ])

    return CustomTransformedTargetRegressor(
        regressor=feature_pipeline,
        transformer=pipeline_y
    )


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
    estimator: Optional[TransformedTargetRegressor] = None
    model_config: ModelConfig = ModelConfig()

    class Config:
        arbitrary_types_allowed: bool = True

    @validator('estimator', pre=True)
    def validate_estimator(cls, v: Any) -> Any:
        if v is not None and not isinstance(v, TransformedTargetRegressor):
            raise ValueError('Invalid estimator')
        return v

    @validator('file_path', 'shap_file_path', pre=True, always=True)
    def default_file_path(cls, v: Optional[Path]) -> Path:
        """Default file path to be used when none is provided."""
        return v or Path.home() / 'desika'

    def __init__(self, **data):
        super().__init__(**data)
        if self.estimator is None:
            self.estimator = create_estimator(
                model_config=self.model_config)

    @staticmethod
    def calculate_weights(y_train: np.ndarray, y_val: Optional[np.ndarray] = None, weight_flag: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        weightifier = Weightify()
        train_weights = weightifier.fit_transform(
            y_train) if weight_flag else np.ones_like(y_train)
        val_weights = weightifier.transform(
            y_val) if weight_flag and y_val is not None else None
        return reshape_array(train_weights), reshape_array(val_weights)

    def fit(self) -> None:
        if self.fitting_mode:
            # Calculate weights for training and validation data
            train_weights, val_weights = self.calculate_weights(
                self.y_train, self.y_val, self.weight_flag)

            # Fit the estimator
            if self.X_val is not None and self.y_val is not None:
                self.estimator.fit(
                    self.X_train, self.y_train, self.X_val, self.y_val,
                    sample_weight=train_weights,
                    val_sample_weight=val_weights
                )
            else:
                self.estimator.fit(
                    self.X_train, self.y_train,
                    sample_weight=train_weights
                )

            # Save the fitted estimator
            if self.file_path is not None:
                dump(self.estimator, self.file_path)

        else:
            if self.file_path is not None:
                self.estimator = load(self.file_path)
            else:
                warnings.warn(
                    "No file path provided. Using the default estimator.")

    def predict(self, X_test: Optional[np.ndarray], return_bounds: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        predicted_mean = reshape_array(self.estimator.predict(
            X_test if X_test is not None else self.X_val))
        predicted_std = reshape_array(self.estimator.regressor_.named_steps['regressor'].predict_std(
            X_test if X_test is not None else self.X_val)) if return_bounds else None
        return predicted_mean, predicted_std

    def calculate_shap_values(self, explainer: Any, X_test: Optional[np.ndarray]) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Access base model for SHAP explainer
            base_model = self.estimator.regressor_.named_steps['regressor'].base_model

            if self.fitting_mode:
                explainer_mean = shap.TreeExplainer(
                    base_model, data=shap.kmeans(self.X_train, 100), model_output=0)
                shap_values_mean = explainer_mean.shap_values(
                    X_test if X_test is not None else self.X_val, check_additivity=False)
                if self.shap_file_path is not None:
                    dump(explainer_mean, self.shap_file_path)

            else:
                if not self.shap_file_path.exists():
                    raise FileNotFoundError(
                        f"File at {self.shap_file_path} not found.")
                elif self.shap_file_path is None:
                    raise FileNotFoundError(
                        "No file path provided.")
                else:
                    explainer_mean = load(self.shap_file_path)

                shap_values_mean = explainer_mean.shap_values(
                    X_test if X_test is not None else self.X_val, check_additivity=False)

            return shap_values_mean
