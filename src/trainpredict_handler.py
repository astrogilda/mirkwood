from src.data_handler import GalaxyProperty
from pydantic_numpy import NDArrayFp64
from pydantic import BaseModel, Field, validator, PrivateAttr
from typing import Optional, List, Tuple, Union, Any
from sklearn.utils import check_X_y
from typing import Optional, List, Tuple, Any
from sklearn.base import TransformerMixin
from multiprocessing import Pool
import numpy as np
from src.bootstrap_handler import BootstrapHandler
from src.model_handler import ModelHandler
from pydantic_numpy import NDArrayFp32, NDArrayFp64
from pathlib import Path
from data_handler import GalaxyProperty, DataHandler
from src.hpo_handler import HPOHandler, HPOHandlerParams

import scipy.stats as stats

from pydantic import BaseModel, Field, validator, ValidationError, PrivateAttr
from pydantic.fields import ModelField

from utils.custom_cv import CustomCV
from utils.custom_transformers_and_estimators import MultipleTransformer, CustomNGBRegressor

from sklearn.pipeline import Pipeline

from src.model_handler import ModelConfig, ModelHandler

from sklearn.compose import TransformedTargetRegressor


# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")


class TransformerTuple(Tuple[str, TransformerMixin]):
    """
    This class is just a placeholder for a type hint. It indicates a tuple that consists of
    a string (name of transformer) and an instance of a TransformerMixin (the transformer itself).
    """
    pass


class TrainPredictHandler(BaseModel):
    """
    TrainPredictHandler class for training and predicting a model using
    cross-validation, bootstrapping, and parallel computing.
    """

    X: NDArrayFp64
    y: NDArrayFp64
    confidence_level: float = Field(0.67, gt=0, le=1)
    n_folds_outer: int = Field(default=5, ge=2, le=20)
    n_folds_inner: int = Field(default=5, ge=2, le=20)
    X_noise: Optional[NDArrayFp64] = None
    X_transformer: Optional[Union[TransformerTuple,
                                  List[TransformerTuple]]] = None
    y_transformer: Optional[Union[TransformerTuple,
                                  List[TransformerTuple]]] = None
    estimator: Optional[Any] = CustomNGBRegressor()
    frac_samples_best: float = Field(0.8, gt=0, le=1)
    galaxy_property: GalaxyProperty = Field(GalaxyProperty.Mass)
    property_name: Optional[str] = None
    testfoldnum: int = 0
    fitting_mode: bool = True
    num_bs: int = Field(10, alias="NUM_BS")
    weight_flag: bool = Field(False, alias="WEIGHT_FLAG")
    n_workers: int = 1  # control the number of workers
    file_path: Optional[Path] = None
    shap_file_path: Optional[Path] = None
    model_config: ModelConfig = ModelConfig()

    _model_handler: Optional[ModelHandler] = PrivateAttr(None)

    class Config:
        arbitrary_types_allowed: bool = True

    @validator('file_path', pre=True)
    def validate_file_path(cls, value, values):
        if not values.get('fitting_mode') and not value.exists():
            raise FileNotFoundError(f"File at {value} not found.")
        return value

    @validator('X', 'y')
    def validate_array_length(cls, v):
        if len(v) < 100:
            raise ValueError('The array should have at least 100 elements')
        return v

    @validator('X', 'y', pre=True)
    def _check_same_length(cls, value: np.ndarray, values: dict, config, field: ModelField):
        if 'X' in values and field.name == 'y':
            X = values['X']
            X, y = check_X_y(X, value)
            values['X'] = X
            return y
        return value

    @validator('X')
    def _check_X_dimension(cls, v: np.ndarray) -> np.ndarray:
        """Validate if the input X array is two-dimensional"""
        if len(v.shape) != 2:
            raise ValueError("X should be 2-dimensional")
        return v

    @validator('y', pre=True)
    def _check_y_dimension(cls, v: np.ndarray) -> np.ndarray:
        """Validate if the input y array is one-dimensional or two-dimensional with second dimension 1"""
        if len(v.shape) == 1:
            v = v.reshape(-1, 1)
        elif len(v.shape) != 2 or (len(v.shape) == 2 and v.shape[1] != 1):
            raise ValueError(
                "y should be 1-dimensional or 2-dimensional with second dimension 1")
        return v

    @validator('X_transformer', 'y_transformer', pre=True)
    def validate_transformers(cls, v: Any) -> Any:
        """Validate if the input v is a list. If it is, we iterate over each element of the list, checking if each element is an instance of TransformerTuple. If v is not a list, we check if it's a TransformerTuple. If it is, we return it enclosed in a list. Otherwise, we raise a ValueError."""
        if isinstance(v, list):
            for elem in v:
                if not isinstance(elem, TransformerTuple):
                    raise ValueError(
                        'Invalid transformer or transformer identifier')
        elif v is not None and not isinstance(v, TransformerTuple):
            raise ValueError('Invalid transformer or transformer identifier')
        return v if isinstance(v, list) else [v]

    @validator('estimator', pre=True)
    def validate_estimator(cls, v: Any) -> Any:
        if not isinstance(v, CustomNGBRegressor):
            raise ValueError('Invalid estimator')
        return v

    @property
    def model_handler(self) -> ModelHandler:
        """
        Singleton for the ModelHandler. If an instance doesn't exist, we create one.
        """
        if self._model_handler is None:
            self._model_handler = ModelHandler(
                X_train=self.X,
                y_train=self.y,
                X_transformer=self.X_transformer,
                y_transformer=self.y_transformer,
                fitting_mode=self.fitting_mode,
                file_path=self.file_path,
                shap_file_path=self.shap_file_path,
                X_noise=self.X_noise,
                weight_flag=self.weight_flag,
            )
        return self._model_handler

    def calculate_z_score(self) -> float:
        """
        Calculate the z-score of the model.
        """
        alpha = 1 - self.confidence_level
        z_score = stats.norm.ppf(1 - alpha / 2)
        return z_score

    def train_predict(self) -> Tuple:
        """
        Train and predict using the pipeline.
        Return a tuple of lists for predictions, std, upper and lower bounds, epis std, actuals, and mean shap.
        """
        yval_lists = [np.array([]) for _ in range(
            7)]  # 7 different predictions returned

        cv_outer = CustomCV(self.y, n_folds=self.n_folds_outer).get_indices()
        cv_inner = CustomCV(self.y, n_folds=self.n_folds_inner).get_indices()

        for i, (train_idx, val_idx) in enumerate(cv_outer):
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

            # Create pipelines for X and y
            pipeline_X = Pipeline(self.X_transformer)
            pipeline_y = MultipleTransformer(self.y_transformer)

            feature_pipeline = Pipeline([
                ('preprocessing', pipeline_X),
                ('regressor', CustomNGBRegressor(**self.model_config.dict()))
            ])

            model = TransformedTargetRegressor(
                regressor=feature_pipeline,
                transformer=pipeline_y
            )

            # carry out hyperparameter tuning

            hpo_handler = HPOHandler(params=HPOHandlerParams(
                cv=cv_inner, z_score=self.calculate_z_score()))
            weights_train = ModelHandler.calculate_weights(
                y_train=y_train, y_val=y_val, weight_flag=self.weight_flag)
            hpo_handler.fit(X_train=X_train, y_train=y_train,
                            weights=weights_train)
            best_model = hpo_handler.model

            # setup the model handler with the best model found using HPO, and the right training data
            self.model_handler.estimator = best_model
            self.model_handler.X_train = X_train
            self.model_handler.y_train = y_train
            self.model_handler.X_val = X_val
            self.model_handler.y_val = y_val

            # create a BootstrapHandler for each fold
            bootstrap_handler = BootstrapHandler(
                X=X_train, y=y_train, frac_samples_best=self.frac_samples_best, galaxy_property=self.galaxy_property,)

            with Pool(self.n_workers) as p:
                args = ((bootstrap_handler, self.model_handler, j, self.property_name,
                        self.testfoldnum) for j in range(self.num_bs))
                concat_output = p.starmap(
                    BootstrapHandler.bootstrap_func_mp, args)

            z_score = self.calculate_z_score()

            yval_outputs = self.process_concat_output(z_score, concat_output)

            y_val = DataHandler.postprocess_y(y_val, prop=self.galaxy_property)

            for idx, arr in enumerate(yval_outputs):
                yval_lists[idx] = np.concatenate([yval_lists[idx], arr])

            yval_lists[5] = np.concatenate([yval_lists[5], y_val])  # Actuals

        return tuple(yval_lists)

    @staticmethod
    def process_concat_output(z_score: float, concat_output: List[np.ndarray]) -> Tuple:
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
