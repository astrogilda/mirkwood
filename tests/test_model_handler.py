import os
from hypothesis import given, strategies as st
from src.model_handler import ModelHandler
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import numpy as np
import pytest
from typing import List
from ngboost import NGBRegressor

os.environ[
    "HYPOTHESIS_ARBITRARY_TYPES_ALLOWED"
] = "true"


@pytest.fixture
def dummy_model_handler():
    """
    Pytest fixture for a ModelHandler instance with dummy data.
    """
    x = np.random.randn(100, 1)
    y = np.random.randn(100, 1)
    return ModelHandler(x=x, y=y, x_transformer=StandardScaler(), y_transformer=StandardScaler(), estimator=NGBRegressor())


def test_transform_data(dummy_model_handler):
    """
    Test the transform_data method.
    """
    x_transformed, y_transformed, transformers = dummy_model_handler.transform_data()
    assert x_transformed.shape == (100, 1)
    assert y_transformed.shape == (100,)
    assert [type(transformer)
            for transformer in transformers] == [type(StandardScaler())]


def test_fit_or_load_estimator(dummy_model_handler):
    """
    Test the fit_or_load_estimator method.
    """
    estimator = dummy_model_handler.fit_or_load_estimator()
    assert isinstance(estimator, NGBRegressor)


def test_compute_prediction_bounds_and_shap_values(dummy_model_handler, tmp_path):
    """
    Test the compute_prediction_bounds_and_shap_values method.
    """
    x_val = np.random.randn(10, 1)
    shap_file_path = tmp_path / "shap_values.pkl"
    results = dummy_model_handler.compute_prediction_bounds_and_shap_values(
        x_val, shap_file_path)

    assert len(results) == 5
    assert isinstance(results[0], np.ndarray)
    assert isinstance(results[1], np.ndarray)
    assert isinstance(results[2], np.ndarray)
    assert isinstance(results[3], np.ndarray)
    assert isinstance(results[4], np.ndarray)


@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1))
def test_default_file_path(test_list: List[float]):
    """
    Test the default_file_path method using hypothesis for property based testing.
    """
    model_handler = ModelHandler(x=np.array(test_list), y=np.array(test_list))
    assert model_handler.file_path == Path.home() / 'desika'


def test_fit_or_load_estimator_file_not_exists(dummy_model_handler):
    """
    Test the fit_or_load_estimator method when the file doesn't exist.
    """
    dummy_model_handler.file_path = Path("not_exist_file_path")
    with pytest.raises(FileNotFoundError):
        dummy_model_handler.fit_or_load_estimator()


def test_fit_or_load_estimator_model_fitted(dummy_model_handler):
    """
    Test the fit_or_load_estimator method when the model is fitted.
    """
    dummy_model_handler.fitting_mode = True
    estimator = dummy_model_handler.fit_or_load_estimator()
    assert isinstance(estimator, NGBRegressor)


def test_compute_prediction_bounds_and_shap_values_shap_file_not_exists(dummy_model_handler, tmp_path):
    """
    Test the compute_prediction_bounds_and_shap_values method when the shap file doesn't exist.
    """
    x_val = np.random.randn(10, 1)
    shap_file_path = tmp_path / "shap_values.pkl"
    dummy_model_handler.fitting_mode = True
    dummy_model_handler.estimator = dummy_model_handler.fit_or_load_estimator()
    dummy_model_handler.fitting_mode = False
    with pytest.raises(FileNotFoundError):
        dummy_model_handler.compute_prediction_bounds_and_shap_values(
            x_val, shap_file_path)


def test_compute_prediction_bounds_and_shap_values_model_not_fitted(dummy_model_handler, tmp_path):
    """
    Test the compute_prediction_bounds_and_shap_values method when the model is not fitted.
    """
    x_val = np.random.randn(10, 1)
    shap_file_path = tmp_path / "shap_values.pkl"
    dummy_model_handler.fitting_mode = False
    with pytest.raises(ValueError):
        dummy_model_handler.compute_prediction_bounds_and_shap_values(
            x_val, shap_file_path)


def test_compute_prediction_bounds_and_shap_values_output_shape(dummy_model_handler, tmp_path):
    """
    Test the compute_prediction_bounds_and_shap_values method to check the shape of output.
    """
    x_val = np.random.randn(10, 1)
    shap_file_path = tmp_path / "shap_values.pkl"
    results = dummy_model_handler.compute_prediction_bounds_and_shap_values(
        x_val, shap_file_path)

    # The last result is shap_values_mean which may have different size
    for result in results[:-1]:
        assert result.shape == (10,)
