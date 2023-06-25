from pydantic import BaseModel, Field, validator, root_validator, conint
from typing import Optional, List, Tuple, Any, Dict
from multiprocessing import Pool
import numpy as np
from pathlib import Path
import scipy.stats as stats
import pandas as pd
import logging
import os
from sklearn.preprocessing import StandardScaler

from src.handlers.bootstrap_handler import BootstrapHandler, BootstrapHandlerConfig
from src.handlers.model_handler import ModelHandler, ModelHandlerConfig, ModelConfig
from src.handlers.data_handler import DataHandler
from src.transformers.yscaler import GalaxyProperty, YScaler
from src.handlers.hpo_handler import HPOHandler, HPOHandlerConfig
from src.transformers.xandy_transformers import XTransformer, YTransformer, TransformerConfig
from src.regressors.customtransformedtarget_regressor import create_estimator
from utils.custom_cv import CustomCV
from utils.weightify import Weightify
from utils.metrics import ProbabilisticErrorMetrics, calculate_z_score


class TrainPredictHandlerConfig(BaseModel):
    """
    TrainPredictHandler class for training and predicting an estimator using
    cross-validation, bootstrapping, and parallel computing.
    """
    # ModelHandlerBaseConfig
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    galaxy_property: Optional[GalaxyProperty] = None
    X_test: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    weight_flag: bool = Field(False, alias="WEIGHT_FLAG")
    fitting_mode: bool = True
    file_path: Optional[Path] = None
    shap_file_path: Optional[Path] = None
    model_config: ModelConfig = ModelConfig()
    X_transformer: XTransformer = XTransformer(
        transformers=None)
    y_transformer: YTransformer = YTransformer(
        transformers=[TransformerConfig(name="rescaley", transformer=YScaler()), TransformerConfig(name="standard_scaler", transformer=StandardScaler())])
    weightifier: Weightify = Weightify()
    # HPOHandlerBaseConfig
    confidence_level: float = Field(0.67, gt=0, le=1)
    num_jobs_hpo: Optional[int] = Field(
        default=os.cpu_count(), gt=0, le=os.cpu_count(), alias="n_jobs_hpo", description="Number of HPO jobs to run in parallel")
    num_trials_hpo: conint(ge=10) = Field(default=100, alias="n_trials_hpo")
    timeout_hpo: Optional[conint(gt=0)] = Field(default=30*60)
    # BootstrapHandlerBaseConfig
    frac_samples: float = Field(0.8, gt=0, le=1)
    replace: bool = Field(default=True)
    #
    X_noise_percent: float = Field(default=0, ge=0, le=1)
    num_folds_outer: int = Field(default=5, ge=2, le=20, alias="n_folds_outer")
    num_folds_inner: int = Field(
        default=5, ge=2, le=20, alias="n_folds_innter")
    num_bs_inner: int = Field(50, alias="n_bs_inner")
    num_bs_outer: int = Field(50, alias="n_bs_outer")

    class Config:
        arbitrary_types_allowed: bool = True

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
            raise ValueError("X_noise_percent should be from 0 to 1")
        return v

    def __str__(self):
        """
        This will return a string representing the configuration object.
        """
        # Customize the string representation of the object.
        return f"TrainPredictHandlerConfig({self.dict()})"


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

    def train(self):  # -> List:
        """
        Train and predict using the pipeline.
        Return a tuple of lists for predictions, std, upper and lower bounds, epis std, actuals, and mean shap.
        """
        # Add noise to X if X_noise_percent is not 0
        if self._config.X_noise_percent > 0:
            self._config.X = self.create_noisy_X(
                X=self._config.X, X_noise_percent=self._config.X_noise_percent)

        # Prepare the estimator and cross-validation indices
        estimator = create_estimator(
            model_config=self._config.model_config, X_transformer=self._config.X_transformer, y_transformer=self._config.y_transformer)
        cv_outer = CustomCV(
            self._config.y, n_folds=self._config.num_folds_outer).get_indices()

        bootstrap_outputs_list = []
        best_estimator_list = []
        for i, (train_idx, val_idx) in enumerate(cv_outer):
            logging.info('CV fold %d of %d', i+1, len(cv_outer))

            # Ensure train_idx and val_idx are integer arrays and not empty
            train_idx = np.array(train_idx, dtype=int)
            val_idx = np.array(val_idx, dtype=int)
            if not train_idx.size or not val_idx.size:
                raise ValueError('Training or validation index array is empty')

            X_train, X_val = self._config.X[train_idx], self._config.X[val_idx]
            y_train, y_val = self._config.y[train_idx], self._config.y[val_idx]

            # Hyperparameter tuning
            logging.info('Hyperparameter tuning')
            cv_inner = CustomCV(
                y_train, n_folds=self._config.num_folds_inner).get_indices()
            hpo_handler = self._prepare_hpo_handler(
                estimator=estimator, cv=cv_inner)
            hpo_handler.fit(X_train=X_train, y_train=y_train)
            logging.info('Hyperparameter tuning complete')

            # Prepare the ModelHandler and BootstrapHandler
            # For this inner bootstrap, we do not need to save the model, explainer, or indices (saved within bootstrap_handler if file_path is not None)
            logging.info('Inner bootstrap')
            model_handler = self._prepare_model_handler(
                X_train, y_train, X_val, y_val, best_estimator=hpo_handler.config.estimator, file_path=None, shap_file_path=None)
            bootstrap_handler = self._prepare_bootstrap_handler(
                model_handler=model_handler)

            # Perform bootstrapping
            with Pool(os.cpu_count()) as p:
                args = ((bootstrap_handler, j+1)
                        for j in range(self._config.num_bs_inner))
                concat_output = p.starmap(BootstrapHandler.bootstrap, args)
            bootstrap_outputs = TrainPredictHandler.process_concat_output(
                concat_output)
            # Append bootstrap outputs to list
            bootstrap_outputs_list.append(bootstrap_outputs)
            # Append best estimator to list
            best_estimator_list.append(hpo_handler.config.estimator)
            logging.info('Inner bootstrap complete')

        metrics_df = self._calculate_metrics_from_predictions(
            bootstrap_outputs_list)

        # Use the best estimator to bootstrap on the entire dataset
        best_estimator_idx = np.argmin(metrics_df['gaussian_crps'].values)
        best_estimator = best_estimator_list[best_estimator_idx]
        # For the outer bootstrap, we save the model, explainer, and indices (saved within bootstrap_handler if file_path is not None)
        logging.info('Outer bootstrap')
        best_model_handler = self._prepare_model_handler(
            X_train=self._config.X, y_train=self._config.y, X_val=self._config.X_test, y_val=self._config.y_test, best_estimator=best_estimator, file_path=self._config.file_path, shap_file_path=self._config.shap_file_path)
        bootstrap_handler = self._prepare_bootstrap_handler(
            model_handler=best_model_handler)

        with Pool(os.cpu_count()) as p:
            args = ((bootstrap_handler, j)
                    for j in range(self._config.num_bs_outer))
            concat_output_outer = p.starmap(BootstrapHandler.bootstrap, args)

        # We do not need to process the outer bootstrap output since we are only interested in the fitted and saved estimators, explainers, and indices
        logging.info('Outer bootstrap complete')

    def _prepare_model_handler(self, X_train, y_train, X_val, y_val, best_estimator, file_path, shap_file_path):
        model_handler_config = ModelHandlerConfig(
            X_train=X_train,
            y_train=y_train,
            feature_names=self._config.feature_names,
            galaxy_property=self._config.galaxy_property,
            X_val=X_val,
            y_val=y_val,
            weight_flag=self._config.weight_flag,
            fitting_mode=self._config.fitting_mode,
            file_path=file_path,
            shap_file_path=shap_file_path,
            model_config=self._config.model_config,
            X_transformer=self._config.X_transformer,
            y_transformer=self._config.y_transformer,
            weightifier=self._config.weightifier,
            precreated_estimator=best_estimator,
        )
        model_handler = ModelHandler(config=model_handler_config)
        return model_handler

    def _prepare_bootstrap_handler(self, model_handler):
        bootstrap_handler_config = BootstrapHandlerConfig(
            frac_samples=self._config.frac_samples, replace=self._config.replace)
        bootstrap_handler = BootstrapHandler(
            model_handler=model_handler, bootstrap_config=bootstrap_handler_config)
        return bootstrap_handler

    def _prepare_hpo_handler(self, estimator, cv):
        hpo_handler_config = HPOHandlerConfig(estimator=estimator, cv=cv, confidence_level=self._config.confidence_level,
                                              n_jobs=self._config.num_jobs_hpo, timeout=self._config.timeout_hpo, n_trials=self._config.num_trials_hpo)
        hpo_handler = HPOHandler(
            config=hpo_handler_config, weight_flag=self._config.weight_flag)
        return hpo_handler

    @staticmethod
    def process_concat_output(concat_output: List[Tuple[np.ndarray, ...]]) -> List[np.ndarray]:
        """
        Process the concatenated output from the bootstrapping process.

        :param concat_output: A list of tuples. Each tuple contains 4 numpy arrays.
                            The first three elements are of shape (n, 1) and the fourth element is of shape (n, m).
                            n: number of samples, m: number of features, len(concat_output): number of bootstraps

        :return: A tuple of 5 numpy arrays. They are calculated by taking mean, square root mean square,
                and standard deviation across the bootstrap dimension respectively.
        """

        # Input validation
        assert isinstance(concat_output, list), "Input must be a list."
        assert all(isinstance(x, tuple) and len(x) == 4 for x in concat_output), \
            "Each element of the input list must be a tuple of 4 elements."

        n, m = concat_output[0][0].shape[0], concat_output[0][3].shape[1]
        assert all(x[0].shape == (n, 1) and x[1].shape == (n, 1) and x[2].shape == (n, 1) and x[3].shape == (n, m)
                   for x in concat_output), "Elements of the tuples have incorrect shape."

        # Split concat_output into four lists and convert them to arrays
        arrays = [np.stack(x, axis=0) for x in zip(*concat_output)]

        # Calculate the mean, root mean square, and standard deviation across the bootstrap dimension
        y_val = np.mean(arrays[0], axis=0)
        y_pred_mean = np.mean(arrays[1], axis=0)
        # Square, mean, then sqrt for root mean square
        y_pred_std = np.sqrt(np.mean(arrays[2]**2, axis=0))
        # Std deviation of second array (not third) - verify logic here
        y_pred_std_epis = np.std(arrays[1], axis=0)
        shap_mean = np.mean(arrays[3], axis=0)

        return [y_val, y_pred_mean, y_pred_std, y_pred_std_epis, shap_mean]

    def _calculate_metrics_from_predictions(self, bootstrap_concat_preds: List[List[np.ndarray]]) -> pd.DataFrame:
        """
        Calculate the metrics from the predictions.
        """
        metrics_df = pd.DataFrame(
            columns=['ace', 'pinaw', 'interval_sharpness', 'gaussian_crps'])
        metrics_df.index.name = 'cv_index'
        z_score = calculate_z_score(
            confidence_level=self._config.confidence_level)
        for cv_idx, bootstrap_preds in enumerate(bootstrap_concat_preds):
            assert len(
                bootstrap_preds) == 5, "Each bootstrap prediction should have 5 elements."
            perm = ProbabilisticErrorMetrics(
                yt=bootstrap_preds[0],
                yp=bootstrap_preds[1],
                yp_lower=bootstrap_preds[1] - z_score *
                np.sqrt(bootstrap_preds[2]**2 + bootstrap_preds[3]**2),
                yp_upper=bootstrap_preds[1] + z_score *
                np.sqrt(bootstrap_preds[2]**2 + bootstrap_preds[3]**2),
                confidence_interval=self._config.confidence_level)
            metrics_df.loc[cv_idx] = perm.ace(), perm.pinaw(
            ), perm.interval_sharpness(), perm.gaussian_crps()

        return metrics_df
