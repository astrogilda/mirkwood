
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import BaseEstimator, RegressorMixin
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

from utils.reshape import reshape_to_1d_array, reshape_to_2d_array

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
        default=0.01,
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
    verbose: bool = Field(
        False, description=" Flag indicating whether output should be printed during fitting")
    natural_gradient: bool = Field(
        True, description="Flag indicating whether to use natural gradient")
    early_stopping_rounds: Optional[conint(gt=0)] = Field(
        default=10, description="Early stopping rounds for NGBRegressor")
    verbose_eval: conint(gt=0) = Field(
        10, description="Increment (in boosting iterations) at which output should be printed")
    tol: confloat(gt=0, le=1e-2) = Field(1e-4,
                                         description="Numerical tolerance to be used in optimization")
    random_state: conint(gt=0, lt=2**32-1) = Field(
        1, description="Seed for reproducibility. See https://stackoverflow.com/questions/28064634/random-state-pseudo-random-number-in-scikit-learn")
    validation_fraction: confloat(gt=0, le=1) = Field(
        0.1, description="Proportion of training data to set aside as validation data for early stopping")

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
            max_depth=3,
            splitter='best'
        )


default_model_config = ModelConfig()


class CustomNGBRegressor(NGBRegressor):
    def __init__(self,
                 Base=ModelConfig().Base,
                 Dist=ModelConfig().Dist,
                 Score=ModelConfig().Score,
                 n_estimators=ModelConfig().n_estimators,
                 learning_rate=ModelConfig().learning_rate,
                 col_sample=ModelConfig().col_sample,
                 minibatch_frac=ModelConfig().minibatch_frac,
                 verbose=ModelConfig().verbose,
                 natural_gradient=ModelConfig().natural_gradient,
                 early_stopping_rounds=ModelConfig().early_stopping_rounds,
                 verbose_eval=ModelConfig().verbose_eval,
                 tol=ModelConfig().tol,
                 random_state=ModelConfig().random_state,
                 validation_fraction=ModelConfig().validation_fraction):
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
        self.verbose_eval = verbose_eval
        self.tol = tol
        self.random_state = random_state
        self.validation_fraction = validation_fraction

    def fit(self, X: np.ndarray, y: np.ndarray, *args, **kwargs):
        params = {key: getattr(self, key) for key in vars(self)}
        super().set_params(**params)

        y = reshape_to_1d_array(y)
        X, y = check_X_y(X, y, accept_sparse=True,
                         force_all_finite=True, ensure_2d=True, y_numeric=True)

        # NGBoost is weird in that it names the y_val argument as Y_val
        if 'y_val' in kwargs:
            kwargs['Y_val'] = kwargs.pop('y_val')
        super().fit(X, y, *args, **kwargs)

        self.fitted_ = True
        return self

    def pred_dist(self, X: np.ndarray):
        check_is_fitted(self, "fitted_")
        X = check_array(X, accept_sparse=True,
                        force_all_finite=True, ensure_2d=True)
        y_pred_dist = super().pred_dist(X)
        return y_pred_dist

    def predict(self, X: np.ndarray):
        y_pred_mean = self.pred_dist(X=X).loc
        return y_pred_mean

    def predict_std(self, X: np.ndarray):
        y_pred_std = self.pred_dist(X=X).scale
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
