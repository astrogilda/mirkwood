from copy import deepcopy
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
from typing import Any, List, Optional, Union, Tuple

from utils.odds_and_ends import reshape_to_1d_array, reshape_to_2d_array

from sklearn.pipeline import Pipeline
from utils.weightify import Weightify


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


class TransformerTuple(list):
    """
    This class represents a list of transformers.
    """

    def __init__(self, transformers):
        if isinstance(transformers, TransformerConfig):
            transformers = [transformers]

        super().__init__(transformers)


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


class YTransformer(BaseModel):
    """
    This class handles transformers for y.
    """
    transformers: Optional[Union[TransformerTuple, TransformerConfig]] = Field(
        default=None
    )

    @root_validator(pre=True)
    def validate_transformers(cls, values):
        transformers = values.get("transformers")

        # If transformers is an instance of a TransformerMixin but not a TransformerConfig, raise an error
        if isinstance(transformers, TransformerMixin) and not isinstance(transformers, TransformerConfig):
            raise ValueError(
                "transformers must be of type TransformerConfig or TransformerTuple")

        if transformers is None:
            transformers = TransformerTuple([
                TransformerConfig(name="standard_scaler",
                                  transformer=StandardScaler()),
            ])

        # If transformers is an empty list , do nothing.
        elif not transformers:
            transformers = transformers

        else:
            # Handle the case when transformers is a TransformerConfig instance or an instance of a valid TransformerMixin subclass
            transformers = TransformerTuple(transformers)

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

        for transformer_config in self.transformers:
            transformer = transformer_config.transformer
            try:
                transformer.fit(X.ravel(), y)
            except ValueError:
                transformer.fit(X.reshape(-1, 1), y)

            print(f"{transformer_config.name} fitted")

        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X using all transformers."""
        check_is_fitted(self, "is_fitted_")
        result = X
        for transformer_config in self.transformers:
            transformer = transformer_config.transformer
            try:
                result = transformer.transform(result.ravel())
            except ValueError:
                result = transformer.transform(result.reshape(-1, 1))

        return result

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform X using all transformers."""
        check_is_fitted(self, "is_fitted_")
        result = X
        for transformer_config in reversed(self.transformers):
            transformer = transformer_config.transformer
            try:
                result = transformer.inverse_transform(result.ravel())
            except ValueError:
                result = transformer.inverse_transform(result.reshape(-1, 1))
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
        if y_val is not None:
            y_val = y_val.ravel()
        if sample_weight is not None:
            sample_weight = sample_weight.ravel()
        if val_sample_weight is not None:
            val_sample_weight = val_sample_weight.ravel()

        return super().fit(X=X, Y=y.ravel(), X_val=X_val, Y_val=y_val,
                           sample_weight=sample_weight, val_sample_weight=val_sample_weight)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the base model and return the mean of the predicted distribution."""
        dist = super().pred_dist(X).loc
        return dist

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """Predict using the base model and return the standard deviation of the predicted distribution."""
        dist = super().pred_dist(X).scale
        return dist

    @property
    def base_model(self):
        return self


class CustomTransformedTargetRegressor(TransformedTargetRegressor):

    @staticmethod
    def calculate_weights(y_train: np.ndarray, y_val: Optional[np.ndarray] = None, weight_flag: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        weightifier = Weightify()
        train_weights = weightifier.fit_transform(
            y_train) if weight_flag else np.ones_like(y_train)
        val_weights = weightifier.transform(
            y_val) if weight_flag and y_val is not None else None
        return reshape_to_1d_array(train_weights), reshape_to_1d_array(val_weights)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None, weight_flag: bool = False) -> 'CustomTransformedTargetRegressor':

        # Explicitly call fit on `_MultipleTransformer`
        self.transformer.fit(X=y)
        y = self.transformer.transform(X=y)
        if y_val is not None:
            y_val = self.transformer.transform(X=y_val)

        # calculate weights for training and validation, to be passed to the regressor
        train_weights, val_weights = CustomTransformedTargetRegressor.calculate_weights(
            y, y_val, weight_flag)

        # fit the preprocessor in the regressor
        self.regressor.named_steps['preprocessor'].fit(X)
        X = self.regressor.named_steps['preprocessor'].transform(X)
        if X_val is not None:
            X_val = self.regressor.named_steps['preprocessor'].transform(X_val)

        # fit the regressor
        self.regressor.named_steps['regressor'].fit(
            X=X, y=y, X_val=X_val, y_val=y_val, sample_weight=train_weights, val_sample_weight=val_weights)

        # inverse transform y, y_val, X, X_val
        y = self.transformer.inverse_transform(X=y)
        if y_val is not None:
            y_val = self.transformer.inverse_transform(X=y_val)
        X = self.regressor.named_steps['preprocessor'].inverse_transform(X)
        if X_val is not None:
            X_val = self.regressor.named_steps['preprocessor'].inverse_transform(
                X_val)

        # self.regressor.fit(X, y)

        self.regressor_ = deepcopy(self.regressor)
        self.transformer_ = deepcopy(self.transformer)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'regressor_')
        check_is_fitted(self, 'transformer_')

        # Preprocess the input data
        X_trans = self.regressor_.named_steps['preprocessor'].transform(X)

        # Extract the underlying regressor
        underlying_regressor = self.regressor_.named_steps['regressor']

        # Check if predict method exists in the regressor
        if hasattr(underlying_regressor, 'predict'):
            y_pred_mean = underlying_regressor.predict(X_trans)
            return self.transformer_.inverse_transform(y_pred_mean).ravel()
        else:
            raise AttributeError(
                f"The underlying regressor does not have 'predict' method.")

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'regressor_')
        check_is_fitted(self, 'transformer_')

        # Preprocess the input data
        X_trans = self.regressor_.named_steps['preprocessor'].transform(X)

        # Extract the underlying regressor
        underlying_regressor = self.regressor_.named_steps['regressor']

        # Check if predict_std method exists in the regressor
        if hasattr(underlying_regressor, 'predict_std'):
            y_pred_std = underlying_regressor.predict_std(X_trans)
            return self.transformer_.inverse_transform(y_pred_std).ravel()
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
            ('preprocessor', pipeline_X),
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
