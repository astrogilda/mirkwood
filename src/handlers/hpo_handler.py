from sklearn.base import clone
from pydantic import BaseModel, Field, ValidationError
from optuna import Study
from typing import Union
import numpy as np
import optuna
from optuna.trial import Trial
from pydantic import BaseModel, Field, conint, validator, confloat
from typing import Callable, List, Optional, Tuple, Union

from handlers.model_handler import ModelConfig
from regressors.customtransformedtarget_regressor import CustomTransformedTargetRegressor, create_estimator
from transformers.xandy_transformers import XTransformer, YTransformer
from utils.metrics import ProbabilisticErrorMetrics

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# TODO: add option for loss to be None, which necessitates that the estimator has a score method. ie add such a method in custom_estimators_and_transformers.py
# TODO: add limits for the distributions in ParamGridConfig


def crps_scorer(y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_std: np.ndarray, z_score: float) -> float:

    y_lower = y_pred_mean - z_score * y_pred_std
    y_upper = y_pred_mean + z_score * y_pred_std

    crps_metrics = ProbabilisticErrorMetrics(
        yt=y_true, yp=y_pred_mean, yp_lower=y_lower, yp_upper=y_upper)
    crps_score = crps_metrics.gaussian_crps()
    # Since Optuna tries to maximize the score, we return the negative CRPS
    return -crps_score


class ParamGridConfig(BaseModel):
    """
    Pydantic model for the configuration of the parameter grid.
    """
    regressor__regressor__learning_rate: optuna.distributions.BaseDistribution = optuna.distributions.FloatDistribution(
        0.01, 0.3)
    regressor__regressor__n_estimators: optuna.distributions.BaseDistribution = optuna.distributions.IntDistribution(
        100, 1000)
    regressor__regressor__minibatch_frac: optuna.distributions.BaseDistribution = optuna.distributions.FloatDistribution(
        0.1, 1.0)
    regressor__regressor__Base__max_depth: optuna.distributions.BaseDistribution = optuna.distributions.IntDistribution(
        1, 10)
    regressor__regressor__Base__max_leaf_nodes: optuna.distributions.BaseDistribution = optuna.distributions.IntDistribution(
        20, 100)

    class Config:
        arbitrary_types_allowed: bool = True


class HPOHandlerConfig(BaseModel):
    """
    Pydantic model for the parameters of HPOHandler.
    """
    param_grid: ParamGridConfig = Field(
        ParamGridConfig(),
        description="Parameter grid for hyperparameter optimization"
    )
    n_trials: conint(ge=10) = Field(100)
    timeout: Optional[conint(gt=0)] = Field(30*60)
    n_jobs: Optional[int] = Field(
        default=-1, ge=-1, description="Number of jobs to run in parallel")
    loss: Optional[Callable] = Field(
        None, description="Loss function to optimize")
    estimator: Optional[CustomTransformedTargetRegressor] = Field(
        default=None,
        description="Estimator to be used for hyperparameter optimization"
    )
    cv: List[Tuple[np.ndarray, np.ndarray]
             ] = Field(..., description="Cross validation splits")
    z_score: confloat(gt=0, le=5) = Field(
        default=1.96, description="The z-score for the confidence interval. Defaults to 1.96, which corresponds to a 95 per cent confidence interval.")

    class Config:
        arbitrary_types_allowed: bool = True

    @validator('n_jobs')
    def check_n_jobs(cls, v):
        if v == 0:
            raise ValueError('n_jobs cannot be 0')
        return v

    @validator('estimator', pre=True, always=True)
    def set_default_estimator(cls, v):
        return v or create_estimator(model_config=ModelConfig(), X_transformer=XTransformer(), y_transformer=YTransformer())

    @validator('loss', pre=True, always=True)
    def set_default_loss(cls, v):
        return v or crps_scorer


class HPOHandler:
    """
    Handler for hyperparameter optimization.
    """

    def __init__(self, config: HPOHandlerConfig, best_trial: Optional[Trial] = None, weight_flag: bool = False):
        if not isinstance(config, HPOHandlerConfig):
            raise ValueError(
                f"config must be of type HPOHandlerConfig, got {type(config)} instead")
        if not isinstance(best_trial, (Trial, type(None))):
            raise ValueError(
                f"best_trial must be of type Trial or None, got {type(best_trial)} instead")
        if not isinstance(weight_flag, bool):
            raise ValueError(
                f"weight_flag must be of type bool, got {type(weight_flag)} instead")
        self.best_trial = best_trial
        self.weight_flag = weight_flag
        self.config = config

    class Config:
        arbitrary_types_allowed: bool = True

    @property
    def is_model_fitted(self):
        if self.best_trial is None:
            raise ValueError("You must call fit() before predict()")
        return True

    def train_model(self, params: dict, X_train: np.ndarray, y_train: np.ndarray, weight_flag: bool) -> CustomTransformedTargetRegressor:
        """
        Creates a new instance of the estimator, sets its parameters, fits it and returns it.
        """
        model = clone(self.config.estimator)
        model.set_params(**params)
        model.fit(X_train, y_train, weight_flag=weight_flag)
        return model

    def objective(self, trial: Trial, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """
        Objective function for Optuna hyperparameter optimization.
        """
        # get parameter ranges from param_grid
        param_grid = self.config.param_grid.dict()

        params = {}
        for key, distribution in param_grid.items():
            if isinstance(distribution, optuna.distributions.FloatDistribution):
                params[key] = trial.suggest_float(
                    key, distribution.low, distribution.high)
            elif isinstance(distribution, optuna.distributions.IntDistribution):
                params[key] = trial.suggest_int(
                    key, distribution.low, distribution.high)
            else:
                raise ValueError(f"Unsupported distribution: {distribution}")

        # Clone and fit a new pipeline for this trial
        pipeline = self.train_model(params, X_train, y_train, self.weight_flag)

        scores = []
        for train_index, val_index in self.config.cv:
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

            pipeline.fit(X=X_train_fold, y=y_train_fold,
                         weight_flag=self.weight_flag)

            y_val_fold_pred_mean = pipeline.predict(X_val_fold)
            y_val_fold_pred_std = pipeline.predict_std(
                X_val_fold)

            score = self.config.loss(y_true=y_val_fold, y_pred_mean=y_val_fold_pred_mean,
                                     y_pred_std=y_val_fold_pred_std, z_score=self.config.z_score)
            scores.append(score)

        return np.mean(scores)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target variable for the given data.
        """
        if self.is_model_fitted:
            return self.config.estimator.predict(X)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the standard deviation of the target variable for the given data.
        """
        if self.is_model_fitted:
            return self.config.estimator.predict_std(X)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the pipeline to the training data.
        """
        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: self.objective(trial, X_train, y_train),
                           n_trials=self.config.n_trials, timeout=self.config.timeout, n_jobs=self.config.n_jobs)

            self.best_trial = study.best_trial

            # Extract only the parameters that exist in the estimator
            best_params = {param: value for param, value in self.best_trial.config.items()
                           if param in self.config.estimator.get_params().keys()}

            # Use the new train_model method to fit the final model
            self.config.estimator = self.train_model(
                best_params, X_train, y_train, self.weight_flag)

        except Exception as e:
            # This will log any exception that is thrown during the fit process.
            # Make sure to import logging at the start of your file.
            logger.error(
                "An error occurred during the fit process:", exc_info=True)
            raise
