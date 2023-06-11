from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
import numpy as np
from ngboost import NGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ngboost.distns import Normal, Distn
from ngboost.scores import LogScore, Score

from pydantic import BaseModel, Field, validator, root_validator, conint, confloat
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from typing import Any, List, Optional, Union

from utils.odds_and_ends import reshape_array

from sklearn.pipeline import Pipeline


class ModelConfig(BaseModel):
    """
    Model configuration for the NGBRegressor. We use this to have a
    centralized place for model parameters which enhances readability
    and maintainability.
    """
    Base: DecisionTreeRegressor = Field(
        default=DecisionTreeRegressor(
            criterion='friedman_mse',
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_leaf_nodes=31,
            splitter='best'),
        description="Base learner for NGBRegressor"
    )
    Dist: Distn = Normal
    Score: Score = LogScore
    n_estimators: conint(gt=0) = 500
    learning_rate: confloat(gt=0, le=1) = Field(
        default=0.04,
        description="The learning rate for the NGBRegressor. Must be greater than 0 and less than or equal to 1."
    )
    col_sample: confloat(gt=0, le=1) = Field(
        default=1.0,
        description="The column sample rate. Must be greater than 0 and less than or equal to 1."
    )
    minibatch_frac: confloat(gt=0, le=1) = Field(
        default=1.0,
        description="The minibatch fraction for NGBRegressor. Must be greater than 0 and less than or equal to 1."
    )
    verbose: bool = False
    natural_gradient: bool = True
    early_stopping_rounds: Optional[conint(gt=0)] = Field(
        default=None, description="Early stopping rounds for NGBRegressor")

    class Config:
        arbitrary_types_allowed: bool = True


class TransformerConfig(BaseModel):
    """
    Config for a transformer.
    """
    name: str
    transformer: TransformerMixin

    class Config:
        arbitrary_types_allowed: bool = True


class TransformerTuple(List[TransformerConfig]):
    """
    This class represents a list of transformers.
    """
    pass


class XTransformer(BaseModel):
    """
    This class handles transformers for X.
    """
    transformers: Optional[Union[TransformerTuple, TransformerConfig]] = Field(
        default=None
    )

    @root_validator(pre=True)
    def validate_transformers(cls, values):
        transformers = values.get("transformers")

        if transformers is None:
            transformers = TransformerTuple([
                TransformerConfig(name="standard_scaler",
                                  transformer=StandardScaler())
            ])
        elif isinstance(transformers, TransformerConfig):
            transformers = [transformers]

        values["transformers"] = transformers
        return values


class ReshapeTransformer(TransformerMixin):
    """
    A transformer that reshapes an array from 2D to 1D or from 1D to 2D, depending on the target dimension.
    """

    def __init__(self, target_dim: int = 0):
        """
        Parameters
        ----------
        target_dim: int
            The target dimension of the transformation. 0 for (n, 1) to (n,) and 1 for (n,) to (n, 1)
        """
        self.target_dim = target_dim
        self.shapes_ = []

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'ReshapeTransformer':
        """Do nothing and return the transformer unchanged."""
        # Since we don't really fit anything here, we'll just set a dummy attribute
        # to mark the transformer as fitted.
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Reshape X to target dimension and store its shape, if necessary."""
        check_is_fitted(self, "is_fitted_")
        self.shapes_.append(X.shape)
        if self.target_dim == 1 and len(X.shape) == 1:
            return X.reshape(-1, 1)
        elif self.target_dim == 0 and len(X.shape) == 2 and X.shape[1] == 1:
            return X.reshape(-1)
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        """Reshape X back to its original shape, if necessary."""
        original_shape = self.shapes_.pop()
        if len(original_shape) == 1 or (len(original_shape) == 2 and original_shape[1] == 1):
            return X.reshape(original_shape)
        return X


class YTransformer(BaseModel):
    """
    This class handles transformers for y.
    """
    transformers: Optional[Union[TransformerTuple, TransformerConfig]] = Field(
        default=None
    )

    @root_validator(pre=True)
    def ensure_reshape_transformer(cls, values):
        transformers = values.get("transformers")

        if transformers is None:
            transformers = TransformerTuple([
                TransformerConfig(name="reshape_transform1",
                                  transformer=ReshapeTransformer(target_dim=1)),
                TransformerConfig(name="standard_scaler",
                                  transformer=StandardScaler()),
                TransformerConfig(name="reshape_transform0",
                                  transformer=ReshapeTransformer(target_dim=0)),
            ])

        # Handle the case when transformers is a TransformerConfig instance
        if isinstance(transformers, TransformerConfig):
            transformers = [transformers]

        # If transformers is None or empty, add ReshapeTransformer(target_dim=0).
        if not transformers:
            transformers = [
                TransformerConfig(name="reshape_transform0",
                                  transformer=ReshapeTransformer(target_dim=0))
            ]

        else:
            first_transformer = transformers[0]
            last_transformer = transformers[-1]

            if not isinstance(first_transformer.transformer, ReshapeTransformer) or first_transformer.transformer.target_dim != 1:
                transformers.insert(0, TransformerConfig(
                    name="reshape_transform1", transformer=ReshapeTransformer(target_dim=1)))

            if not isinstance(last_transformer.transformer, ReshapeTransformer) or last_transformer.transformer.target_dim != 0:
                transformers.append(TransformerConfig(
                    name="reshape_transform0", transformer=ReshapeTransformer(target_dim=0)))

        values["transformers"] = transformers
        return values


class _MultipleTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies a list of transformers sequentially.
    This class is for internal use and should not be instantiated directly.
    Please use the create_estimator function instead.

    Parameters
    ----------
    transformers : list of TransformerConfig
        List of TransformerConfig to be applied sequentially.
    """

    def __init__(self, transformers: Union[TransformerConfig, TransformerTuple, list]):
        if isinstance(transformers, list):
            for t in transformers:
                if not isinstance(t, TransformerConfig):
                    raise TypeError(
                        f"Expected items in list to be of type TransformerConfig, got {type(t).__name__}")

        elif not isinstance(transformers, (TransformerConfig, TransformerTuple)):
            raise TypeError(
                f"Expected TransformerConfig or TransformerTuple or list of TransformerConfig, got {type(transformers).__name__}")

        if isinstance(transformers, TransformerConfig):
            transformers = [transformers]

        self.transformers = [config.transformer for config in transformers]
        self.ensure_reshape_transformer()

    def ensure_reshape_transformer(self):
        """
        Ensure that the first transformer is ReshapeTransformer with target_dim=1
        and the last transformer is ReshapeTransformer with target_dim=0.
        """
        first_transformer = self.transformers[0]
        last_transformer = self.transformers[-1]

        if not isinstance(first_transformer, ReshapeTransformer) or first_transformer.target_dim != 1:
            self.transformers.insert(0, ReshapeTransformer(target_dim=1))

        if not isinstance(last_transformer, ReshapeTransformer) or last_transformer.target_dim != 0:
            self.transformers.append(ReshapeTransformer(target_dim=0))

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> '_MultipleTransformer':
        """Fit all transformers using X and y."""
        for transformer in self.transformers:
            transformer.fit(X, y)
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X using all transformers."""
        check_is_fitted(self, "is_fitted_")
        result = X
        for transformer in self.transformers:
            result = transformer.transform(result)
        return result

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform X using all transformers."""
        check_is_fitted(self, "is_fitted_")
        result = X
        for transformer in reversed(self.transformers):
            result = transformer.inverse_transform(result)
        return result

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {"transformers": self.transformers}

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self


class CustomNGBRegressor(NGBRegressor):
    """
    A custom NGBRegressor class compatible with scikit-learn Pipeline.
    Inherits from NGBRegressor and overrides the fit and predict methods to work with Pipeline.
    """

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None, sample_weight: Optional[np.ndarray] = None,
            val_sample_weight: Optional[np.ndarray] = None) -> 'CustomNGBRegressor':
        """Fit the model according to the given training data."""
        return super().fit(X=X, Y=y, X_val=X_val, Y_val=y_val,
                           sample_weight=sample_weight, val_sample_weight=val_sample_weight)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the base model and return the mean of the predicted distribution."""
        dist = super().pred_dist(X).loc
        return dist  # np.squeeze(reshape_array(dist.loc))

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """Predict using the base model and return the standard deviation of the predicted distribution."""
        dist = super().pred_dist(X).scale
        return dist  # np.squeeze(reshape_array(dist.scale))

    @property
    def base_model(self):
        return self


class CustomTransformedTargetRegressor(TransformedTargetRegressor):
    def predict_std(self, X):
        check_is_fitted(self)

        X_trans = self.transformer.transform(X)

        # Extract the underlying regressor
        underlying_regressor = self.regressor_.named_steps['regressor']

        # Check if predict_std method exists in the regressor
        if hasattr(underlying_regressor, 'predict_std'):
            y_pred_std = underlying_regressor.predict_std(X_trans)
            return self.transformer_.inverse_transform(y_pred_std)
        else:
            raise AttributeError(
                f"The underlying regressor does not have 'predict_std' method.")


def create_estimator(model_config: Optional[ModelConfig] = None,
                     x_transformer: Optional[XTransformer] = None,
                     y_transformer: Optional[YTransformer] = None) -> CustomTransformedTargetRegressor:

    if model_config is None:
        model_config = ModelConfig()

    if x_transformer is None:
        x_transformer = XTransformer()

    if y_transformer is None:
        y_transformer = YTransformer()

    pipeline_X = Pipeline([(transformer.name, transformer.transformer)
                          for transformer in x_transformer.transformers])
    pipeline_y = _MultipleTransformer(y_transformer.transformers)

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


'''


from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer
from ngboost.distns import Normal
from ngboost.scores import LogScore
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
import numpy as np
from utils.custom_transformers_and_estimators import TransformerTuple, TransformerConfig, XTransformer, YTransformer, _MultipleTransformer, ReshapeTransformer, CustomNGBRegressor, CustomTransformedTargetRegressor, create_estimator
from ngboost import NGBRegressor
from utils.weightify import Weightify

from src.data_handler import DataHandler, DataHandlerConfig, DataSet, TrainData, GalaxyProperty
from src.model_handler import ModelConfig
from hypothesis.strategies import floats, lists, sampled_from, just, one_of
import pytest
from hypothesis import given, assume, settings
from pydantic import ValidationError
import numpy as np
from typing import List
import pandas as pd

# Load the data
X_y = DataHandler().get_data(train_data=TrainData.SIMBA)
X, y = np.log10(1+X_y[0].values), X_y[1]['log_stellar_mass']
#X, y = load_diabetes(return_X_y=True)

print(X.shape, y.shape)

# Define your transformations
transformations_X = XTransformer(transformers=TransformerTuple([
    TransformerConfig(name="standard_scaler", transformer=StandardScaler()),
    TransformerConfig(name="robust_scaler", transformer=RobustScaler())
]))
transformations_X = XTransformer(transformation=None)


transformations_Y = YTransformer(transformers=TransformerTuple([
    TransformerConfig(name="standard_scaler", transformer=StandardScaler()),
    TransformerConfig(name="reshape_transform0", transformer=ReshapeTransformer()),
]))

# Create pipelines for X and y
pipeline_X = Pipeline([(transformer.name, transformer.transformer)
                          for transformer in transformations_X.transformers])
pipeline_y = _MultipleTransformer(transformations_Y.transformers)


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
'''
