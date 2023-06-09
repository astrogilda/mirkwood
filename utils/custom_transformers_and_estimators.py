from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
import numpy as np
from ngboost import NGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ngboost.distns import Normal
from ngboost.scores import LogScore

from pydantic import BaseModel, Field, validator, root_validator, conint
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from typing import Any, List, Optional, Union

from utils.odds_and_ends import reshape_array


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
    Dist: Any = Normal
    Score: Any = LogScore
    n_estimators: int = 500
    learning_rate: float = 0.04
    col_sample: float = 1.0
    minibatch_frac: float = 1.0
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
        default=TransformerTuple([
            TransformerConfig(name="standard_scaler",
                              transformer=StandardScaler())
        ])
    )

    @root_validator(pre=True)
    def validate_transformers(cls, values):
        transformers = values.get("transformers")

        if transformers is None:
            transformers = []
        elif isinstance(transformers, TransformerConfig):
            transformers = [transformers]

        values["transformers"] = transformers
        return values


class ReshapeTransformer(TransformerMixin):
    """
    A transformer that reshapes a 2D array of shape (n, 1) to a 1D array of shape (n,).
    """

    def __init__(self):
        self.shapes_ = []

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'ReshapeTransformer':
        """Do nothing and return the transformer unchanged."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Reshape X to 1D array and store its shape, if necessary."""
        self.shapes_.append(X.shape)
        if len(X.shape) == 2 and X.shape[1] == 1:
            return X.reshape(-1)
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reshape X back to its original shape, if necessary."""
        original_shape = self.shapes_.pop()
        if len(original_shape) == 2 and original_shape[1] == 1:
            return X.reshape(original_shape)
        return X


class YTransformer(BaseModel):
    """
    This class handles transformers for y.
    """
    transformers: Optional[Union[TransformerTuple, TransformerConfig]] = Field(
        default=TransformerTuple([
            TransformerConfig(name="log_transform", transformer=FunctionTransformer(
                np.log1p, inverse_func=np.expm1)),
            TransformerConfig(name="sqrt_transform", transformer=FunctionTransformer(
                np.sqrt, inverse_func=np.square)),
            TransformerConfig(name="reshape_transform",
                              transformer=ReshapeTransformer()),
        ])
    )

    @root_validator(pre=True)
    def ensure_reshape_transformer(cls, values):
        """
        Ensure that the last transformer is ReshapeTransformer.
        If it's not, append it to the transformers list.
        """
        transformers = values.get("transformers")

        # Handle the case when transformers is a TransformerConfig instance
        if isinstance(transformers, TransformerConfig):
            transformers = [transformers]

        # Check the last transformer
        if transformers and not isinstance(transformers[-1].transformer, ReshapeTransformer):
            # If it's not ReshapeTransformer, append it
            transformers.append(
                TransformerConfig(name="reshape_transform",
                                  transformer=ReshapeTransformer())
            )

        # If transformers is None or empty, add ReshapeTransformer
        if not transformers:
            transformers = [
                TransformerConfig(name="reshape_transform",
                                  transformer=ReshapeTransformer())
            ]

        values["transformers"] = transformers
        return values


class MultipleTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies a list of transformers sequentially.

    Parameters
    ----------
    transformers : TransformerTuple or list of TransformerMixin
        A TransformerTuple object or a list of transformer instances to be applied sequentially.
    """

    def __init__(self, transformers):
        """
        We check if each object in the transformers list is an instance of TransformerMixin (which would be true if we received a list of transformer instances, like when the object is cloned). If it is, we take the list as it is. If not, we assume we have a list of TransformerConfig objects and extract the transformers from them.
        """
        if all(isinstance(t, TransformerMixin) for t in transformers):
            self.transformers = transformers
        else:
            self.transformers = [config.transformer for config in transformers]

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'MultipleTransformer':
        """Fit all transformers using X and y."""
        for transformer in self.transformers:
            transformer.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X using all transformers."""
        result = X
        for transformer in self.transformers:
            result = transformer.transform(result)
        return result

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform X using all transformers."""
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


'''


from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer
from ngboost.distns import Normal
from ngboost.scores import LogScore
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
import numpy as np
from utils.custom_transformers_and_estimators import TransformerTuple, TransformerConfig, XTransformer, YTransformer, MultipleTransformer, ReshapeTransformer, CustomNGBRegressor, CustomTransformedTargetRegressor
from ngboost import NGBRegressor
from utils.weightify import Weightify

# Load the data
X, y = load_diabetes(return_X_y=True)

print(X.shape, y.shape)

# Define your transformations
transformations_X = XTransformer(transformers=TransformerTuple([
    TransformerConfig(name="standard_scaler", transformer=StandardScaler()),
    TransformerConfig(name="robust_scaler", transformer=RobustScaler())
]))

transformations_Y = YTransformer(transformers=TransformerTuple([
    TransformerConfig(name="log_transform", transformer=FunctionTransformer(np.log1p, inverse_func=np.expm1)),
    TransformerConfig(name="reshape_transform", transformer=ReshapeTransformer()),
]))
transformations_Y = YTransformer(transformation=None)

# Create pipelines for X and y
pipeline_X = Pipeline([(transformer.name, transformer.transformer)
                          for transformer in transformations_X.transformers])
pipeline_y = MultipleTransformer(transformations_Y.transformers)

feature_pipeline = Pipeline([
    ('preprocessing', pipeline_X),
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
