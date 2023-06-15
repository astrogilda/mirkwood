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
    n_estimators: conint(gt=0) = Field(
        default=500, description="Number of estimators for NGBRegressor")
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


class TransformerBase(BaseModel):
    """
    Base class for Transformers handling common validations.
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
                                  transformer=StandardScaler())
            ])
        elif isinstance(transformers, TransformerConfig):
            transformers = [transformers]

        values["transformers"] = transformers
        return values


class XTransformer(TransformerBase):
    """
    This class handles transformers for X.
    """
    pass


class YTransformer(TransformerBase):
    """
    This class handles transformers for y.
    """
    pass


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
        self._transformers = []
        self._is_fitted = False

    def apply_transform(self, X, transform_method, transformer, y=None):
        """A helper function that reshapes data and applies transformation"""
        try:
            if y is None:
                transformed_data = transform_method(reshape_to_1d_array(X))
            else:
                transformed_data = transform_method(reshape_to_1d_array(X), y)
        except Exception as e:
            try:
                if y is None:
                    transformed_data = transform_method(reshape_to_2d_array(X))
                else:
                    transformed_data = transform_method(
                        reshape_to_2d_array(X), y)
            except:
                raise ValueError(
                    f"Failed to transform data with {transformer.__class__.__name__}. Original error: {e}")
        return transformed_data

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "_MultipleTransformer":
        """Fit all transformers using X"""
        X_copy = deepcopy(X)
        for transformer in self.y_transformer.transformers:
            new_transformer = deepcopy(transformer.transformer)
            self.apply_transform(
                X_copy, new_transformer.fit, new_transformer, y)
            self._transformers.append(new_transformer)
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X using all transformers."""
        check_is_fitted(self, ["_is_fitted"])
        X_copy = deepcopy(X)
        for transformer in self._transformers:
            X_copy = self.apply_transform(
                X_copy, transformer.transform, transformer)
        return X_copy

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform X using all transformers."""
        check_is_fitted(self, ["_is_fitted"])
        X_copy = deepcopy(X)
        for transformer in reversed(self._transformers):
            X_copy = self.apply_transform(
                X_copy, transformer.inverse_transform, transformer)
        return X_copy

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
            y_val = reshape_to_1d_array(y_val)
        if sample_weight is not None:
            sample_weight = reshape_to_1d_array(sample_weight)
        if val_sample_weight is not None:
            val_sample_weight = reshape_to_1d_array(val_sample_weight)

        return super().fit(X=X, Y=reshape_to_1d_array(y), X_val=X_val, Y_val=y_val,
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
    """
    A Custom Transformed Target Regressor
    """

    @staticmethod
    def calculate_weights(y_train: np.ndarray, y_val: Optional[np.ndarray] = None, weight_flag: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Calculate weights for training and validation.
        """
        weightifier = Weightify()
        train_weights = weightifier.fit_transform(
            y_train) if weight_flag else np.ones_like(y_train)
        val_weights = weightifier.transform(
            y_val) if weight_flag and y_val is not None else None
        return reshape_to_1d_array(train_weights), reshape_to_1d_array(val_weights) if val_weights is not None else None

    def preprocess_data(self, preprocessor: Pipeline, data: np.ndarray) -> np.ndarray:
        """
        Preprocess data using the provided preprocessor.
        """
        preprocessor.fit(data)
        return preprocessor.transform(data)

    def inverse_transform_data(self, transformer: Union[Pipeline, _MultipleTransformer], data: np.ndarray) -> np.ndarray:
        """
        Inverse transform data using the provided transformer.
        """
        return transformer.inverse_transform(data)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None, weight_flag: bool = False) -> 'CustomTransformedTargetRegressor':
        """
        Fit the regressor.
        """
        self.transformer.fit(X=y)
        y = self.transformer.transform(X=y)
        y_val = self.transformer.transform(
            X=y_val) if y_val is not None else None

        train_weights, val_weights = self.calculate_weights(
            y, y_val, weight_flag)

        preprocessor = self.regressor.named_steps['preprocessor']
        X = self.preprocess_data(preprocessor, X)
        X_val = self.preprocess_data(
            preprocessor, X_val) if X_val is not None else None

        regressor = self.regressor.named_steps['regressor']
        regressor.fit(X=X, y=y, X_val=X_val, y_val=y_val,
                      sample_weight=train_weights, val_sample_weight=val_weights)

        y = self.inverse_transform_data(self.transformer, y)
        y_val = self.inverse_transform_data(
            self.transformer, y_val) if y_val is not None else None
        X = self.inverse_transform_data(preprocessor, X)
        X_val = self.inverse_transform_data(
            preprocessor, X_val) if X_val is not None else None

        self.regressor_ = deepcopy(self.regressor)
        self.transformer_ = deepcopy(self.transformer)

        return self

    def predict_with_check(self, X: np.ndarray, method_name: str) -> np.ndarray:
        """
        Predict with provided method, with checks.
        """
        check_is_fitted(self, 'regressor_')
        check_is_fitted(self, 'transformer_')

        X_trans = self.preprocess_data(
            self.regressor_.named_steps['preprocessor'], X)
        underlying_regressor = self.regressor_.named_steps['regressor']

        if hasattr(underlying_regressor, method_name):
            return underlying_regressor.__getattribute__(method_name)(X_trans)
        else:
            raise AttributeError(
                f"The underlying regressor does not have '{method_name}' method.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target variable.
        """
        y_pred = self.predict_with_check(X, method_name='predict')
        return reshape_to_1d_array(self.inverse_transform_data(self.transformer_, y_pred))

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the standard deviation of the target variable.
        """
        y_pred_std = self.predict_with_check(X, method_name='predict_std')
        y_pred_mean = self.predict_with_check(X, method_name='predict')

        y_pred_upper = y_pred_mean + y_pred_std
        y_pred_lower = y_pred_mean - y_pred_std

        y_pred_upper_inverse_transformed = self.inverse_transform_data(
            self.transformer_, y_pred_upper)
        y_pred_lower_inverse_transformed = self.inverse_transform_data(
            self.transformer_, y_pred_lower)

        y_pred_std_inverse_transformed = (
            y_pred_upper_inverse_transformed - y_pred_lower_inverse_transformed) / 2

        return reshape_to_1d_array(y_pred_std_inverse_transformed)


def create_estimator(model_config: Optional[ModelConfig] = None,
                     X_transformer: Optional[XTransformer] = None,
                     y_transformer: Optional[YTransformer] = None) -> CustomTransformedTargetRegressor:

    if not isinstance(model_config, ModelConfig) and model_config is not None:
        raise TypeError(
            f'model_config must be an instance of ModelConfig or None. Got {type(model_config)} instead.')

    if not isinstance(X_transformer, XTransformer) and X_transformer is not None:
        raise TypeError(
            f'X_transformer must be an instance of XTransformer or None. Got {type(X_transformer)} instead.')

    if not isinstance(y_transformer, YTransformer) and y_transformer is not None:
        raise TypeError(
            f'y_transformer must be an instance of YTransformer or None. Got {type(y_transformer)} instead.')

    if model_config is None:
        model_config = ModelConfig()

    if X_transformer is None:
        X_transformer = XTransformer()

    if y_transformer is None:
        y_transformer = YTransformer()

    pipeline_X = Pipeline([(transformer.name, transformer.transformer)
                          for transformer in X_transformer.transformers])
    pipeline_y = _MultipleTransformer(y_transformer=y_transformer)

    if X_transformer.transformers:
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
