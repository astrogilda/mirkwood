from copy import deepcopy
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
        print(
            f"Storing original shape before transforming using reshape_transformer = {X.shape}")
        print(f"self.shapes_ = {self.shapes_}")

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

        # If transformers is an empty list , add ReshapeTransformer(target_dim=0).
        elif not transformers:
            transformers = [
                TransformerConfig(name="reshape_transform0",
                                  transformer=ReshapeTransformer(target_dim=0))
            ]

        else:
            # Handle the case when transformers is a TransformerConfig instance
            if isinstance(transformers, TransformerConfig):
                transformers = TransformerTuple([transformers])

            else:  # the only remaining case is where transformers is a TransformerTuple
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
    Please use the create_estimator function, or YTransformer, instead.

    Parameters
    ----------
    y_transformer : YTransformer
    """

    def __init__(self, y_transformer: YTransformer):
        if not isinstance(y_transformer, YTransformer):
            raise ValueError(
                f"y_transformer should be an instance of YTransformer, but got {type(y_transformer)}")

        self.y_transformer = y_transformer
        self.transformers = y_transformer.transformers

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "_MultipleTransformer":
        """Fit all transformers using X"""
        if len(self.transformers) > 1:
            first_transformer = self.transformers[0].transformer
            print(f"First transformer is {self.transformers[0].name}")

            first_transformer.fit(X, y)
            print(f"First transformer, {self.transformers[0].name}, fitted")

            X = first_transformer.transform(X)
            print(
                f"shape after first transformer, {self.transformers[0].name}: {X.shape}")

        for transformer_config in self.transformers[1:]:
            transformer = transformer_config.transformer
            transformer.fit(X, y)
            print(f"{transformer_config.name} fitted")

        # Inverse transform back
        if len(self.transformers) > 1:
            X = first_transformer.inverse_transform(X)
            print(
                f"shape after inverse transform using first transformer, {self.transformers[0].name}: {X.shape}")

        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X using all transformers."""
        check_is_fitted(self, "is_fitted_")
        result = X
        for transformer_config in self.transformers:
            transformer = transformer_config.transformer
            result = transformer.transform(result)
        return result

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform X using all transformers."""
        check_is_fitted(self, "is_fitted_")
        result = X
        for transformer_config in reversed(self.transformers):
            transformer = transformer_config.dict()['transformer']
            result = transformer.inverse_transform(result)
        return result

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {"y_transformer": self.y_transformer}

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
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CustomTransformedTargetRegressor':

        # Also explicitly call fit on `_MultipleTransformer`
        self.transformer.fit(y)
        y = self.transformer.transform(y)
        print(f"Transformed y shape: {y.shape}")
        print("\n\n")
        self.regressor.fit(X, y)
        self.regressor_ = deepcopy(self.regressor)
        self.transformer_ = deepcopy(self.transformer)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'regressor_')
        check_is_fitted(self, 'transformer_')

        # Extract the underlying regressor
        underlying_regressor = self.regressor_.named_steps['regressor']

        # Check if predict_std method exists in the regressor
        if hasattr(underlying_regressor, 'predict'):
            y_pred_mean = underlying_regressor.predict(X)
            # print(y_pred_mean)
            # print(y_pred_mean.shape)
            return self.transformer_.inverse_transform(y_pred_mean)
        else:
            raise AttributeError(
                f"The underlying regressor does not have 'predict' method.")

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'regressor_')
        check_is_fitted(self, 'transformer_')

        # X_trans = self.regressor.transform(X)

        # Extract the underlying regressor
        underlying_regressor = self.regressor_.named_steps['regressor']

        # Check if predict_std method exists in the regressor
        if hasattr(underlying_regressor, 'predict_std'):
            y_pred_std = underlying_regressor.predict_std(X)
            # print(y_pred_std)
            # print(y_pred_std.shape)
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
    pipeline_y = _MultipleTransformer(y_transformer=y_transformer)

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
