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
from sklearn.exceptions import NotFittedError


@pytest.fixture(scope='function')
def dummy_model_handler():
    """Return a ModelHandler instance with dummy data."""
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([1, 2, 3])
    return ModelHandler(X_train=X, y_train=y, X_transformer=TransformerConfig(name="standard_scaler", transformer=StandardScaler()), y_transformer=TransformerConfig(name="standard_scaler", transformer=StandardScaler()), estimator=None)


def test_fit(dummy_model_handler: ModelHandler):
    """Test whether the fit method works properly."""
    dummy_model_handler.fit()

    assert dummy_model_handler.is_fitted == True


def test_predict_not_fitted(dummy_model_handler: ModelHandler):
    """
    Test the predict method before calling fit
    """
    x_val = np.random.randn(10, 3)
    with pytest.raises(NotFittedError):
        dummy_model_handler.predict(X_test=x_val, return_bounds=False)


def test_predict_fitted(dummy_model_handler: ModelHandler):
    """
    Test the predict method after calling fit
    """
    x_val = np.random.randn(10, 3)
    dummy_model_handler.fit()

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
    x_val = np.random.randn(10, 3)
    dummy_model_handler.shap_file_path = Path("not_exist_file_path")
    dummy_model_handler.fitting_mode = False
    with pytest.raises(FileNotFoundError):
        dummy_model_handler.calculate_shap_values(
            x_val, feature_names=["a", "b", "c"])


def test_calculate_shap_values_model_not_fitted(dummy_model_handler: ModelHandler):
    """
    Test the calculate_shap_values method when the model is not fitted but is expected to be fit
    """
    x_val = np.random.randn(10, 3)
    dummy_model_handler.fitting_mode = True
    with pytest.raises(NotFittedError):
        dummy_model_handler.calculate_shap_values(
            x_val, feature_names=["a", "b", "c"])


def test_calculate_shap_values_model_fitted(dummy_model_handler: ModelHandler):
    """
    Test the calculate_shap_values method when the model is fitted and is also expected to be fit
    """
    x_val = np.random.randn(100, 3)
    dummy_model_handler.fitting_mode = True
    dummy_model_handler.fit()
    shap_values_val = dummy_model_handler.calculate_shap_values(
        x_val, feature_names=["a", "b", "c"])
    assert isinstance(shap_values_val, np.ndarray)
    assert shap_values_val.shape == (100, 3)


def test_save_and_load_estimator(dummy_model_handler: ModelHandler):
    """
    Test saving and loading of the estimator.
    """
    dummy_model_handler.fitting_mode = True
    dummy_model_handler.file_path = Path("test_estimator_file")
    dummy_model_handler.fit()

    # Assert that the file was created
    assert dummy_model_handler.file_path.exists()

    # Test loading from the saved file
    dummy_model_handler_loaded = ModelHandler(
        X_train=dummy_model_handler.X_train,
        y_train=dummy_model_handler.y_train,
        fitting_mode=False,
        file_path=dummy_model_handler.file_path
    )
    dummy_model_handler_loaded.fit()

    # Check if the loaded model has been fit
    assert dummy_model_handler_loaded.is_fitted == True

    # Delete the file after the test
    os.remove(dummy_model_handler.file_path)


def test_save_and_load_shap_explainer(dummy_model_handler: ModelHandler):
    """
    Test saving and loading of the SHAP explainer.
    """
    x_val = np.random.randn(100, 3)
    dummy_model_handler.fitting_mode = True
    dummy_model_handler.shap_file_path = Path("test_shap_file")
    dummy_model_handler.fit()
    dummy_model_handler.calculate_shap_values(
        x_val, feature_names=["a", "b", "c"])

    # Assert that the file was created
    assert dummy_model_handler.shap_file_path.exists()

    # Test loading from the saved file
    dummy_model_handler_loaded = ModelHandler(
        X_train=dummy_model_handler.X_train,
        y_train=dummy_model_handler.y_train,
        fitting_mode=False,
        shap_file_path=dummy_model_handler.shap_file_path
    )
    dummy_model_handler_loaded.calculate_shap_values(
        x_val, feature_names=["a", "b", "c"])

    # Delete the file after the test
    os.remove(dummy_model_handler.shap_file_path)
