import os
from pathlib import Path
import numpy as np
import pytest
from hypothesis import given, strategies as st
from typing import List
from sklearn.preprocessing import StandardScaler
from ngboost import NGBRegressor
from src.model_handler import ModelHandler
from utils.custom_transformers_and_estimators import TransformerConfig


@pytest.fixture(scope='module')
def dummy_model_handler():
    """Return a ModelHandler instance with dummy data."""
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([1, 2, 3])
    return ModelHandler(X_train=X, y_train=y, X_transformer=TransformerConfig(name="standard_scaler", transformer=StandardScaler()), y_transformer=TransformerConfig(name="standard_scaler", transformer=StandardScaler()), estimator=None)


def test_fit(dummy_model_handler: ModelHandler):
    """Test whether the fit method works properly."""
    dummy_model_handler.fit()

    assert hasattr(dummy_model_handler.estimator, 'predict')


def test_predict(dummy_model_handler: ModelHandler):
    """
    Test the predict method.
    """
    x_val = np.random.randn(10, 1)
    results = dummy_model_handler.predict(X_test=x_val, return_bounds=False)

    assert len(results) == 2
    assert isinstance(results[0], np.ndarray)
    assert results[1] is None

    results = dummy_model_handler.predict(X_test=x_val, return_bounds=True)

    assert len(results) == 2
    assert isinstance(results[0], np.ndarray)
    assert isinstance(results[1], np.ndarray)


def test_fit_load_estimator_file_not_exists(dummy_model_handler: ModelHandler):
    """
    Test the fit method when the file doesn't exist.
    """
    dummy_model_handler.file_path = Path("not_exist_file_path")
    dummy_model_handler.fitting_mode = False
    with pytest.raises(FileNotFoundError):
        dummy_model_handler.fit()


def test_fit_load_estimator_model_fitted(dummy_model_handler: ModelHandler):
    """
    Test the fit method when the model is fitted.
    """
    dummy_model_handler.fitting_mode = True
    dummy_model_handler.fit()
    assert hasattr(dummy_model_handler.estimator, 'predict')


def test_calculate_shap_values_shap_file_not_exists(dummy_model_handler: ModelHandler):
    """
    Test the calculate_shap_values method when the shap file doesn't exist.
    """
    x_val = np.random.randn(10, 1)
    dummy_model_handler.fitting_mode = True
    dummy_model_handler.fit()
    dummy_model_handler.fitting_mode = False
    with pytest.raises(FileNotFoundError):
        dummy_model_handler.calculate_shap_values(None, x_val)


def test_calculate_shap_values_model_not_fitted(dummy_model_handler: ModelHandler):
    """
    Test the calculate_shap_values method when the model is not fitted.
    """
    x_val = np.random.randn(10, 1)
    dummy_model_handler.fitting_mode = False
    with pytest.raises(ValueError):
        dummy_model_handler.calculate_shap_values(None, x_val)


def test_calculate_shap_values(dummy_model_handler: ModelHandler):
    pass
    # shap_values = self.handler.calculate_shap_values(X)
    # assert np.array_equal(shap_values, np.array([0.5, 0.5]))


def test_calculate_weights(dummy_model_handler: ModelHandler):
    pass
    # weights = self.handler.calculate_weights(X)
    # assert np.array_equal(weights, np.array([1, 1, 1]))
