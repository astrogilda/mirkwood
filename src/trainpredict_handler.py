from sklearn.utils import check_X_y
from typing import Callable, Optional, List, Tuple, Any
from sklearn.base import TransformerMixin
from multiprocessing import Pool
import numpy as np
from src.bootstrap_handler import BootstrapHandler
from src.model_handler import ModelHandler
from utils.weightify import Weightify
from pydantic_numpy import NDArray, NDArrayFp32
from pathlib import Path

import os

from pydantic import BaseModel, Field, validator, ValidationError, PrivateAttr
from pydantic.fields import ModelField

from utils.custom_cv import CustomCV

# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")


class TrainPredictHandler(BaseModel):
    """
    TrainPredictHandler class for training and predicting a model using
    cross-validation, bootstrapping, and parallel computing.
    """

    x: NDArrayFp32  # = Field(..., min_items=100)
    y: NDArrayFp32  # = Field(..., min_items=100)
    n_folds: int = Field(default=5, ge=2, le=20)
    x_noise: Optional[NDArrayFp32] = None
    x_transformer: Optional[Any] = None
    y_transformer: Optional[Any] = None
    frac_samples_best: float = Field(0.8, gt=0, le=1)
    weight_bins: int = 10
    reversifyfn: Optional[Callable[[NDArrayFp32], NDArrayFp32]] = None
    property_name: Optional[str] = None
    testfoldnum: int = 0
    fitting_mode: bool = True
    num_bs: int = Field(10, alias="NUM_BS")
    weight_flag: bool = Field(False, alias="WEIGHT_FLAG")
    n_workers: int = 1  # control the number of workers
    file_path: Optional[Path] = None
    shap_file_path: Optional[Path] = None

    arbitrary_types_allowed: bool = True

    _model_handler: Optional[ModelHandler] = PrivateAttr(None)

    # if `file_path` does not exist, and the model is in loading mode (i.e., not in fitting mode), it'll throw an error
    @validator('file_path', pre=True)
    def validate_file_path(cls, value, values):
        if not values.get('fitting_mode') and not value.exists():
            raise FileNotFoundError(f"File at {value} not found.")
        return value

    @validator('x', 'y')
    def validate_array_length(cls, v):
        if len(v) < 100:
            raise ValueError('The array should have at least 100 elements')
        return v

    @validator('x', 'y', pre=True)
    def _check_same_length(cls, value: np.ndarray, values: dict, config, field: ModelField):
        if 'x' in values and field.name == 'y':
            x = values['x']
            x, y = check_X_y(x, value)
            values['x'] = x
            return y
        return value

    @validator('x')
    def _check_x_dimension(cls, v: np.ndarray) -> np.ndarray:
        """Validate if the input x array is two-dimensional"""
        if len(v.shape) != 2:
            raise ValueError("x should be 2-dimensional")
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

    @property
    def model_handler(self) -> ModelHandler:
        """
        Singleton for the ModelHandler. If an instance doesn't exist, we create one.
        """
        if self._model_handler is None:
            self._model_handler = ModelHandler(
                x=self.x,
                y=self.y,
                x_transformer=self.x_transformer,
                y_transformer=self.y_transformer,
                fitting_mode=self.fitting_mode,
                file_path=self.file_path,
                shap_file_path=self.shap_file_path,
                x_noise=self.x_noise,
            )
        return self._model_handler

    @validator('x_transformer', 'y_transformer', pre=True)
    def validate_transformers_and_estimator(cls, v: Any) -> Any:
        if isinstance(v, list):
            if not all(isinstance(elem, TransformerMixin) for elem in v):
                raise ValueError('Invalid transformer')
        elif v is not None and not isinstance(v, TransformerMixin):
            raise ValueError('Invalid transformer or estimator')
        return v

    def train_predict(self) -> Tuple:
        """
        Train and predict using the model.
        Return a tuple of lists for predictions, std, upper and lower bounds, epis std, actuals, and mean shap.
        """
        yval_lists = [np.array([]) for _ in range(
            7)]  # 7 different metrics returned

        cv = CustomCV(self.y, n_folds=self.n_folds).get_indices()

        for i, (train_idx, val_idx) in enumerate(cv):
            # ensure the arrays/lists are not empty
            assert len(train_idx) > 0, "Training index array is empty"
            assert len(val_idx) > 0, "Validation index array is empty"

            print('CV fold %d of %d' % (i+1, len(cv)))
            print(f"train_idx: {train_idx}, type: {type(train_idx)}")
            print(f"val_idx: {val_idx}, type: {type(val_idx)}")

            # ensure train_idx and val_idx are integer arrays
            train_idx = np.array(train_idx, dtype=int)
            val_idx = np.array(val_idx, dtype=int)

            x_train, x_val = self.x[train_idx], self.x[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            if self.weight_flag:
                y_train_weights = Weightify(n_bins=self.weight_bins).fit_transform(
                    y=y_train).reshape(-1, 1)
                print(y_train_weights.shape)
            else:
                y_train_weights = np.ones(y_train.shape)

            # setup the model handler with the right training data
            self.model_handler.x = x_train
            self.model_handler.y = y_train
            self.model_handler.y_weights = y_train_weights

            # create a BootstrapHandler for each fold
            bootstrap_handler = BootstrapHandler(
                x=x_train, y=y_train, frac_samples_best=self.frac_samples_best)

            with Pool(self.n_workers) as p:
                args = ((bootstrap_handler, self.model_handler, j, self.property_name,
                        self.testfoldnum) for j in range(self.num_bs))
                concat_output = p.starmap(
                    BootstrapHandler.bootstrap_func_mp, args)

            yval_outputs = self.process_concat_output(concat_output)

            if self.reversifyfn is not None:
                y_val = self.reversifyfn(y_val)

            for idx, arr in enumerate(yval_outputs):
                yval_lists[idx] = np.concatenate([yval_lists[idx], arr])

            yval_lists[5] = np.concatenate([yval_lists[5], y_val])  # Actuals

        return tuple(yval_lists)

    @staticmethod
    def process_concat_output(concat_output: List[np.ndarray]) -> Tuple:
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
        yval_pred_lower = yval_pred_mean - yval_pred_std
        yval_pred_upper = yval_pred_mean + yval_pred_std
        yval_shap_mean = np.ma.mean(shap_mu_array, axis=0)

        return yval_pred_mean, yval_pred_std, yval_pred_lower, yval_pred_upper, yval_pred_std_epis, yval_shap_mean
