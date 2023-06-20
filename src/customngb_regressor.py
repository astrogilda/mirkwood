
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import BaseEstimator, RegressorMixin
from utils.reshape import reshape_to_1d_array, reshape_to_2d_array
from typing import Any, List, Optional, Union, Tuple, Dict, Callable
from sklearn.tree import DecisionTreeRegressor
from pydantic import BaseModel, Field, validator, root_validator, conint, confloat
from ngboost.scores import LogScore, Score
from ngboost.distns import Normal, Distn
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.base import BaseEstimator, TransformerMixin, clone
from ngboost import NGBRegressor
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EPS = 1e-6

# Centralizing the model parameters in a model config class for better readability and maintainability


class ModelConfig(BaseModel):
    """
    Model configuration for the NGBRegressor. We use this to have a
    centralized place for model parameters which enhances readability
    and maintainability.
    """
    Base: Optional[DecisionTreeRegressor] = Field(
        default=None,
        description="Base learner for NGBRegressor")
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

    @validator("Base", pre=True, always=True)
    def set_base(cls, v):
        return v or DecisionTreeRegressor(
            criterion='friedman_mse',
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_leaf_nodes=31,
            splitter='best'
        )


default_model_config = ModelConfig()


class CustomNGBRegressor(NGBRegressor, BaseEstimator, RegressorMixin):
    def __init__(self,
                 Base=default_model_config.Base,
                 Dist=default_model_config.Dist,
                 Score=default_model_config.Score,
                 n_estimators=default_model_config.n_estimators,
                 learning_rate=default_model_config.learning_rate,
                 col_sample=default_model_config.col_sample,
                 minibatch_frac=default_model_config.minibatch_frac,
                 verbose=default_model_config.verbose,
                 natural_gradient=default_model_config.natural_gradient,
                 early_stopping_rounds=default_model_config.early_stopping_rounds):
        super().__init__()
        self.Base = Base
        self.Dist = Dist
        self.Score = Score
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.col_sample = col_sample
        self.minibatch_frac = minibatch_frac
        self.verbose = verbose
        self.natural_gradient = natural_gradient
        self.early_stopping_rounds = early_stopping_rounds

    def fit(self, X: np.ndarray, y: np.ndarray, *args, **kwargs):
        params = {key: getattr(self, key) for key in vars(self)}
        super().set_params(**params)

        X, y = check_X_y(X, y, accept_sparse=True,
                         force_all_finite='allow-nan')
        super().fit(X, y, *args, **kwargs)
        self.fitted_ = True
        return self

    def predict_dist(self, X: np.ndarray):
        check_is_fitted(self, "fitted_")
        X = check_array(X, accept_sparse=True, force_all_finite='allow-nan')
        y_pred_dist = super().pred_dist(X)
        return y_pred_dist

    def predict(self, X: np.ndarray):
        y_pred_mean = self.predict_dist(X=X).loc
        return y_pred_mean

    def predict_std(self, X: np.ndarray):
        y_pred_std = self.predict_dist(X=X).scale
        return y_pred_std

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        # model_config_params = {key: getattr(self, key) for key in vars(self)}
        # params.update(model_config_params)
        return params

    def set_params(self, **params):
        super().set_params(**params)
        # for key, value in params.items():
        #    setattr(self, key, value)
        return self
