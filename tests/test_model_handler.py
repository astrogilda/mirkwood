import os
import random
import string
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
from src.model_handler import ModelHandler, ModelHandlerConfig
from utils.custom_transformers_and_estimators import TransformerConfig
from pathlib import Path
from pydantic import ValidationError

FEATURE_NAMES = [''.join(random.choices(
    string.ascii_letters + string.digits, k=10)) for _ in range(3)]


@pytest.fixture(scope='function')
def dummy_model_handler():
    """Return a ModelHandler instance with dummy data."""
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([1, 2, 3])
    return ModelHandler(
        config=ModelHandlerConfig(
            X_train=X,
            y_train=y,
            X_transformer=TransformerConfig(
                name="standard_scaler", transformer=StandardScaler()),
            y_transformer=TransformerConfig(
                name="standard_scaler", transformer=StandardScaler()),
            estimator=None,
            feature_names=FEATURE_NAMES
        )
    )


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
            dummy_model_handler.predict(X_test=x_val)
    else:
        results = dummy_model_handler.predict(X_test=x_val)
        assert len(results) == 2
        assert isinstance(results[0], np.ndarray)
        assert results[1] is None

        results = dummy_model_handler.predict(X_test=x_val)
        assert len(results) == 2
        assert isinstance(results[0], np.ndarray)
        assert isinstance(results[1], np.ndarray)


@pytest.mark.parametrize("fit_before_predict, X_test, expected_exception", [
    (False, np.random.randn(10, len(FEATURE_NAMES)), NotFittedError),
    (True, 'invalid_input', TypeError),
    (True, np.random.randn(10, len(FEATURE_NAMES) + 1), ValueError)
])
def test_predict_exception(dummy_model_handler: ModelHandler, fit_before_predict, X_test, expected_exception):
    if fit_before_predict:
        dummy_model_handler.fit()

    with pytest.raises(expected_exception):
        dummy_model_handler.predict(X_test=X_test)


def test_fit_load_estimator_file_not_exists(dummy_model_handler: ModelHandler):
    dummy_model_handler._config.file_path = Path("not_exist_file_path")
    dummy_model_handler._config.fitting_mode = False
    with pytest.raises(FileNotFoundError):
        dummy_model_handler.fit()


# ModelHandlerConfig tests

def test_empty_X_train():
    X = np.array([[], [], []])
    y = np.array([1, 2, 3])
    with pytest.raises(ValidationError):
        ModelHandlerConfig(
            X_train=X,
            y_train=y,
            feature_names=["feature1", "feature2", "feature3"]
        )


def test_unequal_X_y_length():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([1, 2, 3])
    with pytest.raises(ValidationError):
        ModelHandlerConfig(
            X_train=X,
            y_train=y,
            feature_names=["feature1", "feature2", "feature3"]
        )


def test_non_matching_feature_names():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([1, 2, 3])
    with pytest.raises(ValidationError):
        ModelHandlerConfig(
            X_train=X,
            y_train=y,
            feature_names=["feature1", "feature2"]
        )


def test_invalid_fit_params():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([1, 2, 3])

    with pytest.raises(ValidationError):
        ModelHandlerConfig(
            X_train=X,
            y_train=y,
            feature_names=["feature1", "feature2", "feature3"],
            weight_flag="qwerty",
            model_config="abc"
        )
