import os
import random
import string
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
from src.model_handler import ModelHandler
from utils.custom_transformers_and_estimators import TransformerConfig
from pathlib import Path


FEATURE_NAMES = [''.join(random.choices(
    string.ascii_letters + string.digits, k=10)) for _ in range(3)]


@pytest.fixture(scope='function')
def dummy_model_handler():
    """Return a ModelHandler instance with dummy data."""
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([1, 2, 3])
    return ModelHandler(X_train=X, y_train=y, X_transformer=TransformerConfig(name="standard_scaler", transformer=StandardScaler()), y_transformer=TransformerConfig(name="standard_scaler", transformer=StandardScaler()), estimator=None, feature_names=FEATURE_NAMES)


def test_fit(dummy_model_handler: ModelHandler):
    dummy_model_handler.fit()
    assert dummy_model_handler._estimator_handler.is_fitted == True


@pytest.mark.parametrize("fit_before_predict", [True, False])
def test_predict(dummy_model_handler: ModelHandler, fit_before_predict):
    x_val = np.random.randn(10, len(FEATURE_NAMES))

    if fit_before_predict:
        dummy_model_handler.fit()

    if not fit_before_predict:
        with pytest.raises(NotFittedError):
            dummy_model_handler.predict(X_test=x_val, return_std=False)
    else:
        results = dummy_model_handler.predict(X_test=x_val, return_std=False)
        assert len(results) == 2
        assert isinstance(results[0], np.ndarray)
        assert results[1] is None

        results = dummy_model_handler.predict(X_test=x_val, return_std=True)
        assert len(results) == 2
        assert isinstance(results[0], np.ndarray)
        assert isinstance(results[1], np.ndarray)


@pytest.mark.parametrize("fit_before_predict", [True, False])
def test_calculate_shap_values(dummy_model_handler: ModelHandler, fit_before_predict):
    x_val = np.random.randn(100, len(FEATURE_NAMES))
    dummy_model_handler.shap_file_path = Path("test_shap_file")

    if fit_before_predict:
        dummy_model_handler.fit()

    if not fit_before_predict:
        with pytest.raises(NotFittedError):
            dummy_model_handler.calculate_shap_values(x_val)
    else:
        shap_values_val = dummy_model_handler.calculate_shap_values(x_val)
        assert isinstance(shap_values_val, np.ndarray)
        assert shap_values_val.shape == (100, len(FEATURE_NAMES))


@pytest.mark.parametrize("fit_before_predict, X_test, expected_exception", [
    (False, np.random.randn(10, len(FEATURE_NAMES)), NotFittedError),
    (True, 'invalid_input', TypeError),
    (True, np.random.randn(10, len(FEATURE_NAMES) + 1), ValueError)
])
def test_predict_exception(dummy_model_handler: ModelHandler, fit_before_predict, X_test, expected_exception):
    if fit_before_predict:
        dummy_model_handler.fit()

    with pytest.raises(expected_exception):
        dummy_model_handler.predict(X_test=X_test, return_std=False)


@pytest.mark.parametrize("fit_before_shap, X_test, expected_exception", [
    (False, np.random.randn(100, len(FEATURE_NAMES)), NotFittedError),
    (True, 'invalid_input', TypeError),
    (True, np.random.randn(100, len(FEATURE_NAMES) + 1), ValueError)
])
def test_calculate_shap_values_exception(dummy_model_handler: ModelHandler, fit_before_shap, X_test, expected_exception):
    if fit_before_shap:
        dummy_model_handler.fit()

    with pytest.raises(expected_exception):
        dummy_model_handler.calculate_shap_values(X_test)


@pytest.mark.parametrize("model_type, valid_file", [
    ('estimator', True),
    ('estimator', False),
    ('shap', True),
    ('shap', False)
])
def test_save_and_load(dummy_model_handler: ModelHandler, model_type, valid_file):
    dummy_model_handler.fitting_mode = True

    if model_type == 'estimator':
        dummy_model_handler.file_path = Path("test_estimator_file")
    else:
        dummy_model_handler.shap_file_path = Path("test_shap_file")

    if not valid_file:
        dummy_model_handler.file_path = 'invalid_path'

    dummy_model_handler.fit()

    if model_type == 'shap':
        x_val = np.random.randn(100, len(FEATURE_NAMES))
        dummy_model_handler.calculate_shap_values(x_val)

    # Assert that the file was created if it's a valid file
    if valid_file:
        if model_type == 'estimator':
            assert dummy_model_handler.file_path.exists()
        else:
            assert dummy_model_handler.shap_file_path.exists()

        # Test loading from the saved file
        dummy_model_handler_loaded = ModelHandler(
            X_train=dummy_model_handler.X_train,
            y_train=dummy_model_handler.y_train,
            fitting_mode=False,
            file_path=dummy_model_handler.file_path if model_type == 'estimator' else None,
            shap_file_path=dummy_model_handler.shap_file_path if model_type == 'shap' else None,
            feature_names=dummy_model_handler.feature_names,
        )
        dummy_model_handler_loaded.fit()

        if model_type == 'shap':
            dummy_model_handler_loaded.calculate_shap_values(x_val)

        # Check if the loaded model has been fit
        assert dummy_model_handler_loaded._estimator_handler.is_fitted == True

    # Delete the file after the test if it's valid
    if valid_file:
        if model_type == 'estimator':
            os.remove(dummy_model_handler.file_path)
        else:
            os.remove(dummy_model_handler.shap_file_path)


def test_fit_load_estimator_file_not_exists(dummy_model_handler: ModelHandler):
    dummy_model_handler.file_path = Path("not_exist_file_path")
    dummy_model_handler.fitting_mode = False
    with pytest.raises(FileNotFoundError):
        dummy_model_handler.fit()
