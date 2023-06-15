from src.data_handler import GalaxyProperty
from pydantic import BaseModel, Field, validator, PrivateAttr, root_validator
from typing import Optional, List, Tuple, Union, Any
from sklearn.utils import check_X_y
from typing import Optional, List, Tuple, Any
from multiprocessing import Pool
import numpy as np
from src.bootstrap_handler import BootstrapHandler
from src.model_handler import ModelHandler
from pydantic_numpy import NDArrayFp64
from pathlib import Path
from src.data_handler import GalaxyProperty, DataHandler
from src.hpo_handler import HPOHandler, HPOHandlerParams, PipelineConfig
import scipy.stats as stats
from pydantic.fields import ModelField
from utils.custom_cv import CustomCV
from src.model_handler import ModelConfig, ModelHandler
from utils.custom_transformers_and_estimators import XTransformer, YTransformer, create_estimator

import pandas as pd
from pydantic import BaseModel, Field, root_validator


class TrainPredictHandler(BaseModel):
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
    X_noise_percent: float = Field(default=None, ge=0, le=1)
    X_transformer: XTransformer = XTransformer()  # Default XTransformer
    y_transformer: YTransformer = YTransformer()  # Default YTransformer
    model_config: ModelConfig = ModelConfig()
    frac_samples_best: float = Field(0.8, gt=0, le=1)
    galaxy_property: GalaxyProperty = Field(GalaxyProperty.STELLAR_MASS)
    property_name: Optional[str] = None
    testfoldnum: int = 0
    fitting_mode: bool = True
    weight_flag: bool = Field(False, alias="WEIGHT_FLAG")
    n_jobs_bs: Optional[int] = Field(
        default=-1, ge=-1, description="Number of bootstrap jobs to run in parallel")
    n_jobs_hpo: Optional[int] = Field(
        default=-1, ge=-1, description="Number of HPO jobs to run in parallel")
    file_path: Optional[Path] = None
    shap_file_path: Optional[Path] = None

    class Config:
        arbitrary_types_allowed: bool = True

    @validator('n_jobs_bs', 'n_jobs_hpo')
    def check_n_jobs(cls, v):
        if v == 0:
            raise ValueError('n_jobs_bs or n_jobs_hpo cannot be 0')
        return v

    @validator('file_path', pre=True)
    def validate_file_path(cls, value, values):
        if not values.get('fitting_mode') and not value.exists():
            raise FileNotFoundError(f"File at {value} not found.")
        return value

    @validator('X', 'y', 'X_test', 'y_test')
    def validate_array_length(cls, v):
        if v is not None and len(v) < 100:
            raise ValueError('The array should have at least 100 elements')
        return v

    @validator('X', 'y', 'X_test', 'y_test', pre=True)
    def _check_same_length(cls, value: np.ndarray, values: dict, config, field: ModelField):
        if field.name in ['y', 'y_test'] and 'X' in values:
            X = values['X']
            X, y = check_X_y(X, value)
            values['X'] = X
            return y
        elif field.name in ['y', 'y_test'] and 'X_test' in values:
            X = values['X_test']
            X, y = check_X_y(X, value)
            values['X_test'] = X
            return y
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
    def validate_transformers(cls, v: Any) -> Any:
        """Validate if the input v is an instance of the corresponding Transformer class"""
        if not isinstance(v, (XTransformer, YTransformer)):
            raise ValueError('Invalid transformer object')
        return v

    @root_validator(pre=True)
    def apply_noisy_X(cls, values: dict) -> dict:
        if 'X' in values and values.get('X_noise_percent') is not None:
            values['X'] = cls.create_noisy_X(
                values['X'], values['X_noise_percent'])
        return values

    @staticmethod
    def create_noisy_X(X: np.ndarray, X_noise_percent: float) -> np.ndarray:
        noise = X_noise_percent * np.random.default_rng().normal(size=(X.shape))
        X_noisy_nonlog = (10**X - 1)*(1+noise)
        X_noisy = np.log10(1 + np.clip(a=X_noisy_nonlog, a_min=0., a_max=None))
        return X_noisy

    def calculate_z_score(self) -> float:
        """
        Calculate the z-score.
        """
        alpha = 1 - self.confidence_level
        z_score = stats.norm.ppf(1 - alpha / 2)
        return z_score

    def train_predict(self) -> Tuple:
        """
        Train and predict using the pipeline.
        Return a tuple of lists for predictions, std, upper and lower bounds, epis std, actuals, and mean shap.
        """

        estimator = create_estimator(
            model_config=self.model_config, x_transformer=self.X_transformer, y_transformer=self.y_transformer)

        metrics_df = pd.DataFrame()

        cv_outer = CustomCV(self.y, n_folds=self.n_folds_outer).get_indices()
        cv_inner = CustomCV(self.y, n_folds=self.n_folds_inner).get_indices()

        for i, (train_idx, val_idx) in enumerate(cv_outer):
            yval_lists = [np.array([]) for _ in range(
                7)]  # 7 different predictions returned

            # ensure the arrays/lists are not empty
            assert len(train_idx) > 0, "Training index array is empty"
            assert len(val_idx) > 0, "Validation index array is empty"

            print('CV fold %d of %d' % (i+1, len(cv_outer)))
            print(f"train_idx: {train_idx}, type: {type(train_idx)}")
            print(f"val_idx: {val_idx}, type: {type(val_idx)}")

            # ensure train_idx and val_idx are integer arrays
            train_idx = np.array(train_idx, dtype=int)
            val_idx = np.array(val_idx, dtype=int)

            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            # carry out hyperparameter tuning
            hpo_handler = HPOHandler(params=HPOHandlerParams(estimator=estimator,
                                                             cv=cv_inner, z_score=self.calculate_z_score(), n_jobs=self.n_jobs_hpo).dict())
            hpo_handler.fit(X_train=X_train, y_train=y_train)
            best_estimator = hpo_handler.params.estimator

            # setup the model handler with the best estimator found using HPO, and the right training and validation data

            model_handler = ModelHandler(
                X_train=X_train,
                y_train=y_train,
                feature_names=self.feature_names,
                X_val=X_val,
                y_val=y_val,
                estimator=best_estimator,
                fitting_mode=self.fitting_mode,
                file_path=self.file_path,
                shap_file_path=self.shap_file_path,
                weight_flag=self.weight_flag,
            )

            # calculate the z-score
            z_score = self.calculate_z_score()

            # create a BootstrapHandler for each fold
            bootstrap_handler = BootstrapHandler(
                model_handler=model_handler, frac_samples_best=self.frac_samples_best, galaxy_property=self.galaxy_property, z_score=z_score)

            with Pool(self.n_jobs_bs) as p:
                args = ((bootstrap_handler, j, self.property_name,
                        self.testfoldnum) for j in range(self.num_bs_inner))
                concat_output = p.starmap(
                    BootstrapHandler.bootstrap_func_mp, args)

            yval_outputs = self.process_concat_output(z_score, concat_output)

            y_val = DataHandler.postprocess_y(y_val, prop=self.galaxy_property)

            for idx, arr in enumerate(yval_outputs):
                yval_lists[idx] = np.concatenate([yval_lists[idx], arr])

            yval_lists[5] = np.concatenate([yval_lists[5], y_val])  # Actuals

        return tuple(yval_lists)

    @staticmethod
    def process_concat_output(z_score: float, concat_output: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process the output from the bootstrap handler.
        Returns tuples of prediction mean, std, lower, upper, epis std, and mean shap.
        """
        mu_array, std_array, lower_array, upper_array, shap_mu_array = np.array(
            concat_output).T

        # avoid infs. from std_array. repeat for mu_array just in case.
        mu_array = np.ma.masked_invalid(mu_array)
        std_array = np.ma.masked_invalid(std_array)
        lower_array = np.ma.masked_invalid(lower_array)
        upper_array = np.ma.masked_invalid(upper_array)
        shap_mu_array = np.ma.masked_invalid(shap_mu_array)

        yval_pred_mean = np.ma.mean(mu_array, axis=0)
        # Squaring and sqrt operation removed
        yval_pred_std = np.ma.sqrt(np.ma.mean(std_array**2, axis=0))
        yval_pred_std_epis = np.ma.std(mu_array, axis=0)

        yval_pred_lower = yval_pred_mean - z_score * yval_pred_std
        yval_pred_upper = yval_pred_mean + z_score * yval_pred_std
        yval_shap_mean = np.ma.mean(shap_mu_array, axis=0)

        return yval_pred_mean, yval_pred_std, yval_pred_lower, yval_pred_upper, yval_pred_std_epis, yval_shap_mean
