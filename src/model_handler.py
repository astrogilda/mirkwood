from pydantic import BaseModel, Field
from sklearn.exceptions import NotFittedError
from utils.odds_and_ends import reshape_to_1d_array
from utils.custom_transformers_and_estimators import ModelConfig, XTransformer, YTransformer, create_estimator, CustomTransformedTargetRegressor
import shap
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from joblib import dump, load
import numpy as np
from typing import Any, Optional, Tuple, List
from pathlib import Path
import logging
import warnings

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)


class ModelHandler(BaseModel):
    """
    A model handler for performing various tasks including data transformation,
    fitting/loading an estimator, and computing prediction bounds and SHAP values.
    """
    X_train: np.ndarray
    y_train: np.ndarray
    feature_names: List[str]
    X_val: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None
    weight_flag: bool = Field(False, alias="WEIGHT_FLAG")
    fitting_mode: bool = True  # common for both prediction and shap values
    file_path: Optional[Path] = None
    shap_file_path: Optional[Path] = None
    estimator: Optional[CustomTransformedTargetRegressor] = None
    model_config: ModelConfig = ModelConfig()
    X_transformer: XTransformer = XTransformer()
    y_transformer: YTransformer = YTransformer()

    class Config:
        arbitrary_types_allowed: bool = True

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._create_or_load_estimator()

    @property
    def is_fitted(self) -> bool:
        return self._check_if_estimator_is_fitted()

    def fit(self) -> None:
        self._fit_and_save_estimator()

    def predict(self, X_test: Optional[np.ndarray], return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self._predict_with_estimator(X_test, return_std)

    def calculate_shap_values(self, X_test: Optional[np.ndarray]) -> np.ndarray:
        return self._calculate_and_save_shap_values(X_test)

    def _create_or_load_estimator(self):
        if self.estimator is None:
            self.estimator = create_estimator(
                model_config=self.model_config, X_transformer=self.X_transformer, y_transformer=self.y_transformer)

    def _check_if_estimator_is_fitted(self) -> bool:
        try:
            # Check if the estimator has been fit
            self.estimator.predict(self.X_train[0:1])
            return True
        except NotFittedError:
            return False

    def _fit_and_save_estimator(self):
        if self.fitting_mode:
            fit_params = self._get_fit_params()
            self.estimator.fit(self.X_train, self.y_train, **fit_params)
            self._save_estimator()

        else:
            self._load_estimator()

    def _get_fit_params(self):
        fit_params = {'weight_flag': self.weight_flag}
        if self.X_val is not None and self.y_val is not None:
            fit_params.update(
                {'X_val': self.X_val, 'y_val': self.y_val})

        return fit_params

    def _save_estimator(self):
        if self.file_path is not None:
            dump(self.estimator, self.file_path)

    def _load_estimator(self):
        if self.file_path is None or not self.file_path.exists():
            raise FileNotFoundError(
                f"File at {self.file_path or 'provided path'} not found.")
        self.estimator = load(self.file_path)

    def _predict_with_estimator(self, X_test: Optional[np.ndarray], return_std: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.fitting_mode and not self.is_fitted:
            raise NotFittedError("Estimator is not fitted.")

        X = X_test if X_test is not None else self.X_val
        predicted_mean = reshape_to_1d_array(self.estimator.predict(X))
        predicted_std = reshape_to_1d_array(
            self.estimator.predict_std(X)) if return_std else None

        # logging scales for debugging
        logger.info(
            f'Predicted mean scale: {np.mean(predicted_mean)}, standard deviation scale: {np.mean(predicted_std) if predicted_std is not None else None}')
        return predicted_mean, predicted_std

    def _calculate_and_save_shap_values(self, X_test: Optional[np.ndarray]) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.fitting_mode and not self.is_fitted:
                raise NotFittedError("Estimator is not fitted.")

            X = X_test if X_test is not None else self.X_val

            if self.fitting_mode:
                shap_values, explainer = self._calculate_shap_values(X)
                self._save_shap_model(explainer)
            else:
                shap_values, explainer = self._load_shap_model(X)

            return shap_values

    def _calculate_shap_values(self, X):
        base_model = self.estimator.regressor_.named_steps['regressor']
        data = self._get_shap_data()
        explainer_mean = shap.TreeExplainer(
            base_model, data=data, model_output=0, feature_names=self.feature_names)
        shap_values_mean = explainer_mean.shap_values(
            X, check_additivity=False)
        return shap_values_mean, explainer_mean

    def _get_shap_data(self):
        if self.X_train.shape[0] > 200:
            data = shap.kmeans(self.X_train, 100)
        else:
            data = self.X_train
        return data

    def _save_shap_model(self, explainer):
        if self.shap_file_path is not None:
            dump(explainer, self.shap_file_path)

    def _load_shap_model(self, X):
        if self.shap_file_path is None or not self.shap_file_path.exists():
            raise FileNotFoundError(
                f"File at {self.shap_file_path or 'provided path'} not found.")
        explainer_mean = load(self.shap_file_path)
        shap_values_mean = explainer_mean.shap_values(
            X, check_additivity=False)
        return shap_values_mean, explainer_mean
