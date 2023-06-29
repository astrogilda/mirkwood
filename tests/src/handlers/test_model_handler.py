import pickle
import os
import random
import string
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
from pathlib import Path
from pydantic import ValidationError
import tempfile

from src.handlers.model_handler import ModelHandler, ModelHandlerConfig
from src.transformers.xandy_transformers import TransformerConfig

FEATURE_NAMES = [''.join(random.choices(
    string.ascii_letters + string.digits, k=10)) for _ in range(3)]


@pytest.fixture(scope='function')
def dummy_model_handler():
    """Return a ModelHandler instance with dummy data."""
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([1, 2, 3])
    return ModelHandler(
        config=ModelHandlerConfig(
            X=X,
            y=y,
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
    X_test = np.random.randn(10, len(FEATURE_NAMES))

    if not fit_before_predict:
        with pytest.raises(NotFittedError):
            dummy_model_handler.predict(X_test=X_test)
    else:
        dummy_model_handler.fit()
        results = dummy_model_handler.predict(X_test=X_test)
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


@pytest.mark.parametrize("fit_before_predict, X_test, expected_exception", [
    (False, np.random.randn(10, len(FEATURE_NAMES)), NotFittedError),
    (True, 'invalid_input', TypeError),
    (True, np.random.randn(10, len(FEATURE_NAMES) + 1), ValueError)
])
def test_calculate_shap_values_exception(dummy_model_handler: ModelHandler, fit_before_predict, X_test, expected_exception):
    if fit_before_predict:
        dummy_model_handler.fit()

    with pytest.raises(expected_exception):
        dummy_model_handler.calculate_shap_values(X_test=X_test)


@pytest.mark.parametrize("fit_before_create_explainer", [True, False])
def test_create_explainer(dummy_model_handler: ModelHandler, fit_before_create_explainer):
    if not fit_before_create_explainer:
        with pytest.raises(NotFittedError):
            dummy_model_handler.create_explainer()
    else:
        dummy_model_handler.fit()
        dummy_model_handler.create_explainer()
        assert dummy_model_handler.explainer is not None


# ModelHandlerConfig tests

def test_empty_X_train():
    X = np.array([[], [], []])
    y = np.array([1, 2, 3])
    with pytest.raises(ValidationError):
        ModelHandlerConfig(
            X=X,
            y=y,
            feature_names=["feature1", "feature2", "feature3"]
        )


def test_unequal_X_y_length():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        ModelHandlerConfig(
            X=X,
            y=y,
            feature_names=["feature1", "feature2", "feature3"]
        )


def test_non_matching_feature_names():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([1, 2, 3])
    with pytest.raises(ValidationError):
        ModelHandlerConfig(
            X=X,
            y=y,
            feature_names=["feature1", "feature2"]
        )


def test_invalid_fit_params():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([1, 2, 3])

    with pytest.raises(ValidationError):
        ModelHandlerConfig(
            X=X,
            y=y,
            feature_names=["feature1", "feature2", "feature3"],
            weight_flag="qwerty",
            model_config="abc"
        )


def test_model_handler_config_mismatched_val_arrays():
    X_test = np.array([[1, 2, 3], [4, 5, 6]])
    y_test = np.array([1, 2])
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        ModelHandlerConfig(
            X=X,
            y=y,
            X_test=X_test,
            y_test=y_test
        )


def test_model_handler_config_invalid_array():
    X = 'invalid_array'
    y = np.array([1, 2, 3])

    with pytest.raises(ValidationError):
        ModelHandlerConfig(
            X=X,
            y=y,
            feature_names=["feature1", "feature2", "feature3"]
        )


def test_model_handler_config_non_2d_X_train():
    X = np.array([1, 2, 3])
    y = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        ModelHandlerConfig(
            X=X,
            y=y,
            feature_names=["feature1", "feature2", "feature3"]
        )


def test_model_handler_config_non_1d_y_train():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    with pytest.raises(ValueError):
        ModelHandlerConfig(
            X=X,
            y=y,
            feature_names=["feature1", "feature2", "feature3"]
        )


def test_model_handler_config_missing_X_train():
    """
    Test if a ValidationError is raised when X is missing and fitting_mode is True
    """
    y = np.array([1, 2, 3])

    with pytest.raises(ValidationError):
        ModelHandlerConfig(
            X=None,
            y=y,
            feature_names=["feature1", "feature2", "feature3"],
            fitting_mode=True
        )


def test_model_handler_config_missing_y_train():
    """
    Test if a ValidationError is raised when y is missing and fitting_mode is True
    """
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    with pytest.raises(ValidationError):
        ModelHandlerConfig(
            X=X,
            y=None,
            feature_names=["feature1", "feature2", "feature3"],
            fitting_mode=True
        )


def test_model_handler_config_non_2d_X_test():
    """
    Test if a ValueError is raised when X_test is not 2-dimensional
    """
    X_test = np.array([1, 2, 3])
    y_test = np.array([1, 2, 3])
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        ModelHandlerConfig(
            X=X,
            y=y,
            X_test=X_test,
            y_test=y_test,
            feature_names=["feature1", "feature2", "feature3"]
        )


def test_model_handler_config_non_1d_y_val():
    """
    Test if a ValueError is raised when y_test is not 1-dimensional
    """
    X_test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y_test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        ModelHandlerConfig(
            X=X,
            y=y,
            X_test=X_test,
            y_test=y_test,
            feature_names=["feature1", "feature2", "feature3"]
        )


class DummyEstimator:
    def __init__(self):
        self.estimator = 'dummy'
        self.is_fitted = True


def test_model_handler_config_fit_with_none_X_y_train():
    """
    Test if the fit function works when X and y are None and fitting_mode is False
    """

    dummy_model_handler = ModelHandler(
        config=ModelHandlerConfig(
            X=None,
            y=None,
            X_transformer=TransformerConfig(
                name="standard_scaler", transformer=StandardScaler()),
            y_transformer=TransformerConfig(
                name="standard_scaler", transformer=StandardScaler()),
            estimator=None,
            feature_names=FEATURE_NAMES,
            fitting_mode=False,
            precreated_estimator=None,
            file_path=None,
        )
    )

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False, mode="wb") as temp:
        print(f"temp file path: {temp.name}")
        dummy_model_handler._config.file_path = Path(temp.name)

        # create a dummy object with `estimator` and `is_fitted` attributes
        dummy_object = DummyEstimator()

        # dump the dummy object into the temporary file
        pickle.dump(dummy_object.__dict__, temp.file)

        # Close the file to ensure it's written and not locked
        temp.file.close()

        # try:
        dummy_model_handler.fit()
        loaded_object = dummy_model_handler._estimator_handler

        assert loaded_object.estimator == 'dummy'
        assert loaded_object.is_fitted == True
        # except Exception:
        #    pytest.fail("ModelHandler.fit() raised an Exception unexpectedly!")
