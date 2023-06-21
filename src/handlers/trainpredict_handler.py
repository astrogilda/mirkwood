from handlers.data_handler import GalaxyProperty
from pydantic import BaseModel, Field, validator, root_validator
from sklearn.utils import check_X_y
from typing import Optional, List, Tuple, Any, Dict
from multiprocessing import Pool
import numpy as np
from pathlib import Path
import scipy.stats as stats
import pandas as pd
import logging

from handlers.bootstrap_handler import BootstrapHandler, BootstrapHandlerConfig
from handlers.model_handler import ModelHandler, ModelHandlerConfig, ModelConfig
from handlers.data_handler import GalaxyProperty, DataHandler
from handlers.hpo_handler import HPOHandler, HPOHandlerConfig
from utils.custom_cv import CustomCV
from src.transformers.xandy_transformers import XTransformer, YTransformer
from src.regressors.customtransformedtarget_regressor import create_estimator
from utils.metrics import calculate_z_score


class TrainPredictHandlerConfig(BaseModel):
    """
    TrainPredictHandler class for training and predicting an estimator using
    cross-validation, bootstrapping, and parallel computing.
    """

    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    X_test: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    confidence_level: float = Field(0.67, gt=0, le=1)
    n_folds_outer: int = Field(default=5, ge=2, le=20)
    n_folds_inner: int = Field(default=5, ge=2, le=20)
    num_bs_inner: int = Field(50, alias="NUM_BS_INNER")
    num_bs_outer: int = Field(50, alias="NUM_BS_OUTER")
    n_jobs_hpo: Optional[int] = Field(
        default=-1, ge=-1, description="Number of HPO jobs to run in parallel")
    X_noise_percent: float = Field(default=None, ge=0, le=1)
    X_transformer: XTransformer = XTransformer()  # Default XTransformer
    y_transformer: YTransformer = YTransformer()  # Default YTransformer
    model_config: ModelConfig = ModelConfig()
    frac_samples: float = Field(0.8, gt=0, le=1)
    galaxy_property: GalaxyProperty = Field(GalaxyProperty.STELLAR_MASS)
    property_name: Optional[str] = None
    testfoldnum: int = 0
    fitting_mode: bool = True
    weight_flag: bool = Field(False, alias="WEIGHT_FLAG")
    file_path: Optional[Path] = None
    shap_file_path: Optional[Path] = None

    class Config:
        arbitrary_types_allowed: bool = True

    @validator('n_jobs_hpo')
    def check_n_jobs(cls, v):
        if v == 0:
            raise ValueError('n_jobs_hpo cannot be 0')
        return v

    @validator('file_path', pre=True)
    def validate_file_path(cls, value, values):
        if not values.get('fitting_mode') and not value.exists():
            raise FileNotFoundError(f"File at {value} not found.")
        return value

    @validator('X', 'X_test')
    def _check_X_dimension(cls, v: np.ndarray) -> np.ndarray:
        """Validate if the input X array is two-dimensional"""
        if v is not None and len(v.shape) != 2:
            raise ValueError("X should be 2-dimensional")
        return v

    @validator('y', 'y_test', pre=True)
    def _check_y_dimension(cls, v: np.ndarray) -> np.ndarray:
        """Validate if the input y array is one-dimensional or two-dimensional with second dimension 1"""
        if v is not None:
            if len(v.shape) == 1:
                v = v.reshape(-1, 1)
            elif len(v.shape) != 2 or (len(v.shape) == 2 and v.shape[1] != 1):
                raise ValueError(
                    "y should be 1-dimensional or 2-dimensional with second dimension 1")
        return v

    @validator('X_transformer', 'y_transformer', pre=True)
    def validate_transformers(cls, v, values, **kwargs):
        if not isinstance(v, (XTransformer, YTransformer)):
            raise ValueError("Invalid transformer provided")
        return v

    @validator('model_config', pre=True)
    def validate_model_config(cls, v, values, **kwargs):
        if not isinstance(v, ModelConfig):
            raise ValueError("Invalid model configuration provided")
        return v

    @root_validator
    def validate_array_lengths(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        X, y, X_test, y_test = values.get('X'), values.get(
            'y'), values.get('X_test'), values.get('y_test')

        # Check if X and y have the same number of samples
        if X is not None and y is not None and X.shape[0] != y.shape[0]:
            raise ValueError("X and y should have the same number of samples")

        # Check if X_test and y_test have the same number of samples
        if X_test is not None and y_test is not None and X_test.shape[0] != y_test.shape[0]:
            raise ValueError(
                "X_test and y_test should have the same number of samples")

        return values

    @validator('galaxy_property', pre=True)
    def validate_galaxy_property(cls, v, values, **kwargs):
        if not isinstance(v, GalaxyProperty):
            raise ValueError("Invalid galaxy property provided")
        return v

    @validator('X_noise_percent', pre=True)
    def validate_X_noise_percent(cls, v, values, **kwargs):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("X_noise_percent should be between 0 and 1")
        return v

    def __str__(self):
        """
        This will return a string representing the configuration object.
        """
        # Customize the string representation of the object.
        return f"TrainPredictHandlerConfig({self.dict()})"


'''
# Example usage:
config = TrainPredictHandlerConfig(
    X=np.random.rand(100, 5),
    y=np.random.rand(100),
    feature_names=["a", "b", "c", "d", "e"]
)
print(config)

'''


class TrainPredictHandler:
    def __init__(self, config: TrainPredictHandlerConfig):
        if not isinstance(config, TrainPredictHandlerConfig):
            raise ValueError('Invalid config object')
        self._config = config

    @staticmethod
    def create_noisy_X(X: np.ndarray, X_noise_percent: float) -> np.ndarray:
        noise = X_noise_percent * np.random.default_rng().normal(size=(X.shape))
        X_noisy_nonlog = (10**X - 1)*(1+noise)
        X_noisy = np.log10(1 + np.clip(a=X_noisy_nonlog, a_min=0., a_max=None))
        return X_noisy

    def train_predict(self) -> Tuple:
        """
        Train and predict using the pipeline.
        Return a tuple of lists for predictions, std, upper and lower bounds, epis std, actuals, and mean shap.
        """
        # Prepare the estimator and cross-validation indices
        estimator = create_estimator(
            model_config=self._config.model_config, x_transformer=self._config.X_transformer, y_transformer=self._config.y_transformer)
        metrics_df = pd.DataFrame()
        cv_outer = CustomCV(
            self._config.y, n_folds=self._config.n_folds_outer).get_indices()
        cv_inner = CustomCV(
            self._config.y, n_folds=self._config.n_folds_inner).get_indices()

        yval_lists = [np.array([]) for _ in range(7)]

        for i, (train_idx, val_idx) in enumerate(cv_outer):
            logging.info('CV fold %d of %d', i+1, len(cv_outer))

            # Ensure train_idx and val_idx are integer arrays and not empty
            train_idx = np.array(train_idx, dtype=int)
            val_idx = np.array(val_idx, dtype=int)
            if not train_idx.size or not val_idx.size:
                raise ValueError('Training or validation index array is empty')

            X_train, X_val = self._config.X[train_idx], self._config.X[val_idx]
            y_train, y_val = self._config.y[train_idx], self._config.y[val_idx]

            z_score = calculate_z_score(
                confidence_level=self._config.confidence_level)

            # Hyperparameter tuning
            hpo_handler_config = HPOHandlerConfig(estimator=estimator, cv=cv_inner, z_score=z_score,
                                                  n_jobs=self._config.n_jobs_hpo, timeout=60*60, n_trials=100)
            hpo_handler = HPOHandler(
                params=hpo_handler_config, weight_flag=self._config.weight_flag)
            hpo_handler.fit(X_train=X_train, y_train=y_train)
            best_estimator = hpo_handler.params.estimator

            # Prepare the ModelHandler and BootstrapHandler
            model_handler = self._prepare_model_handler(
                X_train, y_train, X_val, y_val, best_estimator)
            bootstrap_handler_config = BootstrapHandlerConfig(
                frac_samples=self._config.frac_samples, replace=True)
            bootstrap_handler = BootstrapHandler(
                model_handler=model_handler, bootstrap_config=bootstrap_handler_config)

            # Perform bootstrapping
            with Pool(self._config.num_bs_inner) as p:
                args = ((bootstrap_handler, j)
                        for j in range(self._config.num_bs_inner))
                concat_output = p.starmap(BootstrapHandler.bootstrap, args)

            yval_outputs = TrainPredictHandler.process_concat_output(
                concat_output)
            y_val = DataHandler.postprocess_y(
                y_val, prop=self._config.galaxy_property)

            for idx, arr in enumerate(yval_outputs):
                yval_lists[idx] = np.concatenate([yval_lists[idx], arr])

            yval_lists[5] = np.concatenate([yval_lists[5], y_val])  # Actuals

        return tuple(yval_lists)

    def _prepare_model_handler(self, X_train, y_train, X_val, y_val, best_estimator):
        return ModelHandler(
            X_train=X_train,
            y_train=y_train,
            feature_names=self._config.feature_names,
            X_val=X_val,
            y_val=y_val,
            estimator=best_estimator,
            fitting_mode=self._config.fitting_mode,
            file_path=self._config.file_path,
            shap_config=self._config.shap_config,
            model_config=self._config.model_config,
            X_transformer=self._config.X_transformer,
            y_transformer=self._config.y_transformer,
        )

    @staticmethod
    def process_concat_output(concat_output: List[Tuple[np.ndarray, ...]]) -> List[np.ndarray]:
        """
        Process the concatenated output from the bootstrapping process.
        """
        y_val_agg = [np.zeros_like(concat_output[0][0]) for _ in range(7)]

        for output in concat_output:
            for idx, arr in enumerate(output):
                y_val_agg[idx] += arr

        for idx in range(7):
            y_val_agg[idx] /= len(concat_output)

        return y_val_agg
