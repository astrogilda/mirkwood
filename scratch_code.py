
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer
from ngboost.distns import Normal
from ngboost.scores import LogScore
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
import numpy as np
from utils.custom_transformers_and_estimators import TransformerTuple, TransformerConfig, XTransformer, YTransformer, MultipleTransformer, ReshapeTransformer, CustomNGBRegressor, CustomTransformedTargetRegressor, create_estimator
from ngboost import NGBRegressor
from utils.weightify import Weightify

from handlers.data_handler import DataHandler, DataHandlerConfig, DataSet, TrainData, GalaxyProperty
from handlers.model_handler import ModelConfig
from hypothesis.strategies import floats, lists, sampled_from, just, one_of
import pytest
from hypothesis import given, assume, settings
from pydantic import ValidationError
import numpy as np
from typing import List
import pandas as pd


def trace_calls():
    import sys
    from inspect import stack
    from copy import deepcopy

    def tracer(frame, event, arg):
        if event == "call":
            call_function = stack()[1][3]
            print(f"{call_function} called")
    sys.settrace(tracer)


# trace_calls()
create_estimator()


# Load the data
# X_y = DataHandler().get_data(train_data=TrainData.SIMBA)
# X, y = np.log10(1+X_y[0].values), X_y[1]['log_stellar_mass']
X, y = load_diabetes(return_X_y=True)


print(X.shape, y.shape)

y_transformer = YTransformer()
pipeline_y = MultipleTransformer(y_transformer)
pipeline_y.fit(X, y)

ce = create_estimator()
ce.fit(X, y)
ce.predict_std(X).shape, y.shape

# Define your transformations
transformations_X = XTransformer(transformers=TransformerTuple([
    TransformerConfig(name="standard_scaler", transformer=StandardScaler()),
    TransformerConfig(name="robust_scaler", transformer=RobustScaler())
]))
transformations_X = XTransformer(transformation=None)


transformations_Y = YTransformer(transformers=TransformerTuple([
    TransformerConfig(name="standard_scaler", transformer=StandardScaler()),
    TransformerConfig(name="reshape_transform0",
                      transformer=ReshapeTransformer()),
]))

# Create pipelines for X and y
pipeline_X = Pipeline([(transformer.name, transformer.transformer)
                       for transformer in transformations_X.transformers])
pipeline_y = MultipleTransformer(transformations_Y.transformers)


if transformations_X.transformers:
    feature_pipeline = Pipeline([
        ('preprocessing', pipeline_X),
        ('regressor', CustomNGBRegressor(Dist=Normal, Score=LogScore))
    ])
else:
    feature_pipeline = Pipeline([
        ('regressor', CustomNGBRegressor(Dist=Normal, Score=LogScore))
    ])


model = CustomTransformedTargetRegressor(
    regressor=feature_pipeline,
    transformer=pipeline_y
)

# Sequentially transform y
y_transformed = pipeline_y.fit_transform(y)

y_transformed.shape
y.shape
X.shape

model.fit(X, y)

y_pred_mean = model.predict(X)

y_pred_std = model.predict_std(X)


# Initialize your Weightify transformer
weightify_transformer = Weightify()

# Calculate weights for y
y_weights = weightify_transformer.fit_transform(y)

# Fit the model
model.fit(X, y, regressor__sample_weight=y_weights)

y_pred_mean_weighted = model.predict(X)

y_pred_std_weighted = model.predict_std(X)


# Sequentially transform y
y_transformed = pipeline_y.fit_transform(y)
y_transformed = pipeline_y.transform(y)

# Initialize your Weightify transformer
weightify_transformer = Weightify()

# Calculate weights for y
y_train_weights = weightify_transformer.fit_transform(y_train_transformed)
y_val_weights = weightify_transformer.transform(y_val_transformed)

# Fit the model
model.fit(X_train, y_train, regressor__sample_weight=y_train_weights, regressor__X_val=X_val,
          regressor__y_val=y_val, regressor__val_sample_weight=y_val_weights)

# Make predictions
# The predictions are automatically inverse-transformed by TransformedTargetRegressor
y_test_pred = model.predict(X_test)


class TrainPredictHandlerConfig(ModelHandlerBaseConfig, HPOHandlerBaseConfig, BootstrapHandlerConfig):
    """
    TrainPredictHandler class for training and predicting an estimator using
    cross-validation, bootstrapping, and parallel computing.
    """
    '''
    # ModelHandlerBaseConfig
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    galaxy_property: Optional[GalaxyProperty] = None
    X_test: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    weight_flag: bool = Field(False, alias="WEIGHT_FLAG")
    fitting_mode: bool = True
    file_path: Optional[Path] = None
    shap_file_path: Optional[Path] = None
    model_config: ModelConfig = ModelConfig()
    X_transformer: XTransformer = XTransformer(
        transformers=None)
    y_transformer: YTransformer = YTransformer(
        transformers=[TransformerConfig(name="rescale_y", transformer=YScaler()), TransformerConfig(name="standard_scaler", transformer=StandardScaler())])
    weightifier: Weightify = Weightify()
    # HPOHandlerBaseConfig
    confidence_level: float = Field(0.67, gt=0, le=1)
    num_jobs_hpo: Optional[int] = Field(
        default=os.cpu_count(), gt=0, le=os.cpu_count(), alias="n_jobs_hpo", description="Number of HPO jobs to run in parallel")
    num_trials_hpo: conint(ge=10) = Field(default=100, alias="n_trials_hpo")
    timeout_hpo: Optional[conint(gt=0)] = Field(default=30*60)
    # BootstrapHandlerConfig
    frac_samples: float = Field(0.8, gt=0, le=1)
    replace: bool = Field(default=True)
    '''
    #
    X_noise_percent: float = Field(default=0, ge=0, le=1)
    num_folds_outer: int = Field(default=5, ge=2, le=20, alias="n_folds_outer")
    num_folds_inner: int = Field(
        default=5, ge=2, le=20, alias="n_folds_innter")
    num_bs_inner: int = Field(50, alias="n_bs_inner")
    num_bs_outer: int = Field(50, alias="n_bs_outer")

    class Config:
        arbitrary_types_allowed: bool = True

    @validator('file_path', pre=True)
    def validate_file_path(cls, value, values):
        if not values.get('fitting_mode') and not value.exists():
            raise FileNotFoundError(f"File at {value} not found.")
        return value

    @validator('X', 'X_test')
    def _check_X_dimension(cls, v: np.ndarray) -> np.ndarray:
        """Validate if the input X array is two-dimensional"""
        if v is not None and len(v.shape) != 2:
            raise ValueError("X should be 2-dimensional")
        return v

    @validator('y', 'y_test', pre=True)
    def _check_y_dimension(cls, v: np.ndarray) -> np.ndarray:
        """Validate if the input y array is one-dimensional or two-dimensional with second dimension 1"""
        if v is not None:
            if len(v.shape) == 1:
                v = v.reshape(-1, 1)
            elif len(v.shape) != 2 or (len(v.shape) == 2 and v.shape[1] != 1):
                raise ValueError(
                    "y should be 1-dimensional or 2-dimensional with second dimension 1")
        return v

    @validator('X_transformer', 'y_transformer', pre=True)
    def validate_transformers(cls, v, values, **kwargs):
        if not isinstance(v, (XTransformer, YTransformer)):
            raise ValueError("Invalid transformer provided")
        return v

    @validator('model_config', pre=True)
    def validate_model_config(cls, v, values, **kwargs):
        if not isinstance(v, ModelConfig):
            raise ValueError("Invalid model configuration provided")
        return v

    @root_validator
    def validate_array_lengths(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        X, y, X_test, y_test = values.get('X'), values.get(
            'y'), values.get('X_test'), values.get('y_test')

        # Check if X and y have the same number of samples
        if X is not None and y is not None and X.shape[0] != y.shape[0]:
            raise ValueError("X and y should have the same number of samples")

        # Check if X_test and y_test have the same number of samples
        if X_test is not None and y_test is not None and X_test.shape[0] != y_test.shape[0]:
            raise ValueError(
                "X_test and y_test should have the same number of samples")

        return values

    @validator('galaxy_property', pre=True)
    def validate_galaxy_property(cls, v, values, **kwargs):
        if not isinstance(v, GalaxyProperty):
            raise ValueError("Invalid galaxy property provided")
        return v

    @validator('X_noise_percent', pre=True)
    def validate_X_noise_percent(cls, v, values, **kwargs):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("X_noise_percent should be from 0 to 1")
        return v

    def __str__(self):
        """
        This will return a string representing the configuration object.
        """
        # Customize the string representation of the object.
        return f"TrainPredictHandlerConfig({self.dict()})"
