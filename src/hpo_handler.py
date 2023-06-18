from pydantic import BaseModel, Field, ValidationError
from optuna import Study
from typing import Union
import numpy as np
import optuna
from optuna.trial import Trial
from pydantic import BaseModel, Field, conint, validator, confloat
from typing import Callable, List, Optional, Tuple, Union

from src.model_handler import ModelConfig
from utils.custom_transformers_and_estimators import XTransformer, YTransformer, CustomTransformedTargetRegressor, create_estimator
from utils.metrics import ProbabilisticErrorMetrics

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
        crps_scorer, description="Loss function to optimize")
    estimator: CustomTransformedTargetRegressor = Field(
        default=create_estimator(model_config=ModelConfig(
        ), X_transformer=XTransformer(), y_transformer=YTransformer()),
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


class HPOHandler(BaseModel):
    """
    Handler for hyperparameter optimization.
    """
    params: HPOHandlerConfig = Field(...,
                                     description="Parameters of HPOHandler")
    # Do not use this directly, use is_model_fitted instead
    best_trial: Optional[Trial] = None
    weight_flag: bool = False

    class Config:
        arbitrary_types_allowed: bool = True

    @property
    def is_model_fitted(self):
        if self.best_trial is None:
            raise ValueError("You must call fit() before predict()")
        return True

    def objective(self, trial: Trial, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """
        Objective function for Optuna hyperparameter optimization.
        """
        # get parameter ranges from param_grid
        param_grid = self.params.param_grid.dict()

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

        # update parameters of the entire CustomTransformedTargetRegressor pipeline
        pipeline = self.params.estimator.set_params(**params)

        scores = []
        for train_index, val_index in self.params.cv:
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

            pipeline.fit(X=X_train_fold, y=y_train_fold,
                         weight_flag=self.weight_flag)

            y_val_fold_pred_mean = pipeline.predict(X_val_fold)
            y_val_fold_pred_std = pipeline.predict_std(
                X_val_fold)

            score = self.params.loss(y_true=y_val_fold, y_pred_mean=y_val_fold_pred_mean,
                                     y_pred_std=y_val_fold_pred_std, z_score=self.params.z_score)
            scores.append(score)

        return np.mean(scores)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target variable for the given data.
        """
        if self.is_model_fitted:
            return self.params.estimator.predict(X)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the standard deviation of the target variable for the given data.
        """
        if self.is_model_fitted:
            return self.params.estimator.predict_std(X)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the pipeline to the training data.
        """
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.objective(trial, X_train, y_train),
                       n_trials=self.params.n_trials, timeout=self.params.timeout, n_jobs=self.params.n_jobs)

        try:
            self.best_trial = study.best_trial
        except ValueError as e:
            raise ValueError("No trials are completed yet.") from e

        # Extract only the parameters that exist in the estimator
        best_params = {param: value for param, value in self.best_trial.params.items()
                       if param in self.params.estimator.get_params().keys()}

        self.params.estimator.set_params(**best_params)
        self.params.estimator.fit(
            X_train, y_train, weight_flag=self.weight_flag)
