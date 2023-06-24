from utils.logger import LoggingUtility
from pydantic import ValidationError
from typing import Sequence
import logging
import numpy as np
import optuna
from optuna import Trial
from optuna.distributions import BaseDistribution, FloatDistribution, IntDistribution
from pydantic import BaseModel, Field, confloat, conint, validator
from sklearn.base import clone
from typing import Callable, List, Optional, Tuple, Union

from src.handlers.model_handler import ModelConfig
from src.regressors.customtransformedtarget_regressor import CustomTransformedTargetRegressor, create_estimator
from src.transformers.xandy_transformers import XTransformer, YTransformer
from utils.metrics import ProbabilisticErrorMetrics, calculate_z_score
from utils.validate import validate_X_y
import os

# Suppressing the following warning: OMP: Info #270: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.
os.environ['OMP_DISPLAY_ENV'] = 'FALSE'

# Setting the logging level WARNING, the INFO logs are suppressed.
optuna.logging.set_verbosity(optuna.logging.WARNING)


logger = LoggingUtility.get_logger(
    __name__, log_file='logs/hpo_handler.log')
logger.info('Saving logs from hpo_handler.py')


def crps_scorer(y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_std: np.ndarray, confidence_level: float) -> float:
    """
    Custom Scoring Function.
    """
    z_score = calculate_z_score(confidence_level)
    y_lower = y_pred_mean - z_score * y_pred_std
    y_upper = y_pred_mean + z_score * y_pred_std

    crps_metrics = ProbabilisticErrorMetrics(
        yt=y_true, yp=y_pred_mean, yp_lower=y_lower, yp_upper=y_upper, confidence_level=confidence_level)
    crps_score = crps_metrics.gaussian_crps()
    return -crps_score


class ParamGridConfig(BaseModel):
    """
    Pydantic model for the configuration of the parameter grid.
    """
    regressor__regressor__learning_rate: BaseDistribution = Field(
        default=FloatDistribution(0.01, 0.05))
    regressor__regressor__n_estimators: BaseDistribution = Field(
        default=IntDistribution(100, 1000))
    regressor__regressor__minibatch_frac: BaseDistribution = Field(
        default=FloatDistribution(0.1, 1.0))
    regressor__regressor__Base__max_depth: BaseDistribution = Field(
        default=IntDistribution(1, 5))
    regressor__regressor__Base__max_leaf_nodes: BaseDistribution = Field(
        default=IntDistribution(10, 40))

    class Config:
        arbitrary_types_allowed = True


class HPOHandlerConfig(BaseModel):
    """
    Pydantic model for the parameters of HPOHandler.
    """
    param_grid: ParamGridConfig = Field(default=ParamGridConfig())
    n_trials: conint(ge=10) = Field(default=100)
    timeout: Optional[conint(gt=0)] = Field(default=30*60)
    n_jobs: Optional[int] = Field(default=None, gt=0, le=os.cpu_count())
    loss: Optional[Callable] = Field(default=None)
    estimator: Optional[CustomTransformedTargetRegressor] = Field(default=None)
    cv: List[Tuple[Union[np.ndarray, list], Union[np.ndarray, list]]
             ] = Field(..., min_items=1)
    confidence_level: confloat(gt=0, le=5) = Field(default=1.96)

    @validator('n_jobs')
    def check_n_jobs(cls, v):
        if v is None:
            return os.cpu_count()

    @validator('estimator', pre=True, always=True)
    def set_default_estimator(cls, v):
        return v or create_estimator(model_config=ModelConfig(), X_transformer=XTransformer(), y_transformer=YTransformer())

    @validator('loss', pre=True, always=True)
    def set_default_loss(cls, v):
        return v or crps_scorer

    @validator('cv')
    def check_cv(cls, v):
        for item in v:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValidationError('Each element of cv must be a two-tuple')
            for subitem in item:
                if not isinstance(subitem, (np.ndarray, list)):
                    raise ValidationError(
                        'Each item in the tuple must be a 1D numpy array or list')
                if isinstance(subitem, np.ndarray) and subitem.ndim != 1:
                    raise ValidationError(
                        'Each numpy array in the tuple must be 1D')
        return v

    class Config:
        arbitrary_types_allowed: bool = True


class HPOHandler:
    """
    Handler for hyperparameter optimization.
    """

    def __init__(self, config: HPOHandlerConfig, best_trial: Optional[Trial] = None, weight_flag: bool = False):
        self.best_trial = best_trial
        self.weight_flag = weight_flag
        self.config = config

    def train_model(self, params: dict, X_train: np.ndarray, y_train: np.ndarray) -> CustomTransformedTargetRegressor:
        """
        Creates a new instance of the estimator, sets its parameters, fits it and returns it.
        """
        model = clone(self.config.estimator)
        model.set_params(**params)
        model.fit(X_train, y_train, weight_flag=self.weight_flag)
        return model

    def objective(self, trial: Trial, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """
        Objective function for Optuna hyperparameter optimization.
        """
        # get parameter ranges from param_grid
        X_train, y_train = validate_X_y(X_train, y_train)
        param_grid = self.config.param_grid.dict()

        params = {}
        for key, distribution in param_grid.items():
            if isinstance(distribution, FloatDistribution):
                params[key] = trial.suggest_float(
                    key, distribution.low, distribution.high)
            elif isinstance(distribution, IntDistribution):
                params[key] = trial.suggest_int(
                    key, distribution.low, distribution.high)
            else:
                raise ValueError(f"Unsupported distribution: {distribution}")

        # Clone and fit a new pipeline for this trial
        pipeline = self.train_model(params, X_train, y_train)

        scores = []
        for train_index, val_index in self.config.cv:
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

            pipeline.fit(X=X_train_fold, y=y_train_fold,
                         weight_flag=self.weight_flag)

            y_val_fold_pred_mean = pipeline.predict(X_val_fold)
            y_val_fold_pred_std = pipeline.predict_std(X_val_fold)

            score = self.config.loss(y_true=y_val_fold, y_pred_mean=y_val_fold_pred_mean,
                                     y_pred_std=y_val_fold_pred_std, confidence_level=self.config.confidence_level)
            scores.append(score)

        return np.mean(scores)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target variable for the given data.
        """
        if self.best_trial is None:
            raise ValueError("You must call fit() before predict()")
        return self.config.estimator.predict(X)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the standard deviation of the target variable for the given data.
        """
        if self.best_trial is None:
            raise ValueError("You must call fit() before predict()")
        return self.config.estimator.predict_std(X)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the pipeline to the training data.
        """
        X_train, y_train = validate_X_y(X_train, y_train)
        try:
            study = optuna.create_study(
                direction='maximize')
            study.optimize(lambda trial: self.objective(trial, X_train, y_train),
                           n_trials=self.config.n_trials, timeout=self.config.timeout, n_jobs=self.config.n_jobs)

            self.best_trial = study.best_trial

            # Extract only the parameters that exist in the estimator
            best_params = {param: value for param, value in self.best_trial.params.items(
            ) if param in self.config.estimator.get_params().keys()}

            # Use the new train_model method to fit the final model
            self.config.estimator = self.train_model(
                best_params, X_train, y_train)

        except Exception as e:
            logger.error(
                "An error occurred during the fit process:", exc_info=True)
            raise e
