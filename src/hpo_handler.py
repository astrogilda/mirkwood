import numpy as np
import optuna
from optuna.trial import Trial
from pydantic import BaseModel, Field, validator
from sklearn.compose import TransformedTargetRegressor
from typing import Callable, Optional, Tuple

from src.model_handler import ModelConfig, create_estimator
from utils.custom_cv import CustomCV
from utils.custom_transformers_and_estimators import CustomNGBRegressor, CustomTransformedTargetRegressor
from utils.metrics import ProbabilisticErrorMetrics


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
    regressor__regressor__learning_rate: optuna.distributions.BaseDistribution = optuna.distributions.UniformDistribution(
        0.01, 0.3)
    regressor__regressor__n_estimators: optuna.distributions.BaseDistribution = optuna.distributions.IntUniformDistribution(
        100, 1000)
    regressor__regressor__minibatch_frac: optuna.distributions.BaseDistribution = optuna.distributions.UniformDistribution(
        0.1, 1.0)
    # add configurations for other stages in the pipeline as needed


class PipelineConfig(BaseModel):
    """
    Pydantic model for the configuration of the pipeline.
    """
    estimator: CustomTransformedTargetRegressor = Field(
        default=create_estimator(model_config=ModelConfig()))

    class Config:
        arbitrary_types_allowed: bool = True

    @validator("estimator")
    def check_model(cls, v):
        """
        Validator for estimator. Checks that estimator is an instance of TransformedTargetRegressor.
        """
        if not isinstance(v, TransformedTargetRegressor):
            raise ValueError(
                "estimator must be an instance of TransformedTargetRegressor")
        return v


class HPOHandlerParams(BaseModel):
    """
    Pydantic model for the parameters of HPOHandler.
    """
    param_grid: ParamGridConfig = Field(
        ParamGridConfig(),
        description="Parameter grid for hyperparameter optimization"
    )
    n_trials: int = Field(100, gt=0)
    timeout: Optional[int] = Field(30*60, gt=0)
    loss: Optional[Callable] = crps_scorer
    # Use the TransformedTargetRegressor with default parameters from PipelineConfig
    estimator: PipelineConfig = PipelineConfig()
    cv: CustomCV  # Accepts a cross-validator of type CustomCV
    z_score: float = 1.96

    class Config:
        arbitrary_types_allowed: bool = True


class HPOHandler:
    """
    Handler for hyperparameter optimization.
    """

    def __init__(self, params: HPOHandlerParams):
        """
        Initialize the handler with the given parameters.
        """
        self.params = params
        self.best_trial = None

    def objective(self, trial: Trial, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> float:
        """
        Objective function for Optuna hyperparameter optimization.
        """
        # get parameter ranges from param_grid
        param_grid = self.params.param_grid.dict()

        params = {}
        for key, distribution in param_grid.items():
            if isinstance(distribution, optuna.distributions.UniformDistribution):
                params[key] = trial.suggest_float(
                    key, distribution.low, distribution.high)
            elif isinstance(distribution, optuna.distributions.IntUniformDistribution):
                params[key] = trial.suggest_int(
                    key, distribution.low, distribution.high)
            else:
                raise ValueError(f"Unsupported distribution: {distribution}")

        # update parameters of the entire pipeline
        pipeline = self.params.estimator.estimator.set_params(**params)

        scores = []
        for train_index, val_index in self.params.cv:
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            weights_fold = weights[train_index]

            pipeline.fit(X_train_fold, y_train_fold,
                         sample_weight=weights_fold)

            y_val_fold_pred_mean = pipeline.predict(X_val_fold)
            y_val_fold_pred_std = pipeline.regressor_.named_steps['regressor'].predict_std(
                X_val_fold)

            score = self.params.loss(
                y_val_fold, y_val_fold_pred_mean, y_val_fold_pred_std, self.params.z_score)
            scores.append(score)

        return np.mean(scores)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> None:
        """
        Fit the pipeline to the training data.
        """
        study = optuna.create_study(
            direction='maximize')  # Change 'minimize' to 'maximize' if higher scores are better
        study.optimize(lambda trial: self.objective(
            trial, X_train, y_train, weights), n_trials=self.params.n_trials)

        self.best_trial = study.best_trial

        best_params = self.best_trial.params
        self.params.estimator.estimator.set_params(**best_params)
        self.params.estimator.estimator.fit(
            X_train, y_train, sample_weight=weights)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target variable for the given data.
        """
        if self.best_trial is None:
            raise ValueError("You must call fit() before predict()")
        return self.params.estimator.estimator.predict(X)

    def predict_dist(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the target variable for the given data using the pred_dist method.
        This returns the mean and standard deviation of the predicted distribution,
        as well as the upper and lower bounds.
        """
        if self.best_trial is None:
            raise ValueError("You must call fit() before predict_dist()")

        y_pred = self.params.estimator.estimator.predict(X)
        y_pred_std = self.params.estimator.estimator.regressor_.named_steps['regressor'].predict_std(
            X)
        y_pred_upper = y_pred + self.z_score * y_pred_std
        y_pred_lower = y_pred - self.z_score * y_pred_std

        return y_pred, y_pred_std, y_pred_lower, y_pred_upper
