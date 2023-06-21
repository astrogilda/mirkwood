import pytest
from src.handlers.estimator_handler import EstimatorHandler
from src.handlers.model_handler import ModelHandlerConfig
from src.regressors.customtransformedtarget_regressor import (
    CustomTransformedTargetRegressor, create_estimator)
from sklearn.exceptions import NotFittedError
from pathlib import Path
import os
import numpy as np
from pydantic import ValidationError
import logging
import tempfile


@pytest.fixture(scope='function')
def dummy_estimator_handler():
    """Return an EstimatorHandler instance with dummy data."""
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([1, 2, 3])
    config = ModelHandlerConfig(X_train=X, y_train=y, feature_names=[
                                "feature1", "feature2", "feature3"])
    return EstimatorHandler(config)


@pytest.mark.parametrize("fit,valid_filepath", [(True, True), (False, True), (True, False), (False, False)])
def test_save_and_load_estimator(dummy_estimator_handler, fit, valid_filepath, caplog):
    model = create_estimator(dummy_estimator_handler._config.model_config,
                             dummy_estimator_handler._config.X_transformer,
                             dummy_estimator_handler._config.y_transformer)
    if valid_filepath:
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temp:
            dummy_estimator_handler._config.file_path = temp.name
            if fit:
                model.fit(dummy_estimator_handler._config.X_train,
                          dummy_estimator_handler._config.y_train)
                dummy_estimator_handler._estimator = model
                dummy_estimator_handler._fitted = True

                dummy_estimator_handler._save_estimator()
                assert Path(dummy_estimator_handler._config.file_path).exists()

                loaded_estimator_handler = EstimatorHandler(
                    dummy_estimator_handler._config)
                loaded_estimator_handler._load_estimator()
                assert isinstance(loaded_estimator_handler.estimator,
                                  CustomTransformedTargetRegressor)
                assert loaded_estimator_handler.is_fitted
            else:
                with pytest.raises(NotFittedError):
                    dummy_estimator_handler._save_estimator()

                # Since the estimator has not been fitted, it has not been saved, and thus not be loaded
                with pytest.raises(EOFError):
                    loaded_estimator_handler = EstimatorHandler(
                        dummy_estimator_handler._config)
                    loaded_estimator_handler._load_estimator()
    else:
        dummy_estimator_handler._config.file_path = None
        with caplog.at_level(logging.WARNING):
            dummy_estimator_handler._save_estimator()
        assert "No filename provided. Skipping save." in caplog.text


'''
def test_save_and_load_estimator(dummy_estimator_handler):
    dummy_estimator_handler.fit()
    file_path = Path("test_estimator_file")

    # Save the estimator
    dummy_estimator_handler._config.file_path = file_path
    dummy_estimator_handler._create_and_fit_estimator()
    assert file_path.exists()

    # Load the estimator
    dummy_estimator_handler_loaded = EstimatorHandler(
        dummy_estimator_handler._config)
    dummy_estimator_handler_loaded._load_estimator()
    assert isinstance(dummy_estimator_handler_loaded.estimator,
                      CustomTransformedTargetRegressor)
    assert dummy_estimator_handler_loaded.is_fitted

    # Clean up
    file_path.unlink()
'''


def test_load_estimator_file_not_exists(dummy_estimator_handler):
    dummy_estimator_handler._config.file_path = Path("not_exist_file_path")
    with pytest.raises(FileNotFoundError):
        dummy_estimator_handler._load_estimator()


# Edge cases


def test_precreated_estimator(dummy_estimator_handler):
    precreated_estimator = create_estimator(
        dummy_estimator_handler._config.model_config,
        dummy_estimator_handler._config.X_transformer,
        dummy_estimator_handler._config.y_transformer)
    precreated_estimator.fit(dummy_estimator_handler._config.X_train,
                             dummy_estimator_handler._config.y_train)

    config = ModelHandlerConfig(
        X_train=dummy_estimator_handler._config.X_train,
        y_train=dummy_estimator_handler._config.y_train,
        feature_names=dummy_estimator_handler._config.feature_names,
        precreated_estimator=precreated_estimator
    )
    precreated_estimator_handler = EstimatorHandler(config)
    precreated_estimator_handler.fit()
    assert precreated_estimator_handler.is_fitted


def test_fit(dummy_estimator_handler):
    dummy_estimator_handler.fit()
    assert isinstance(dummy_estimator_handler.estimator,
                      CustomTransformedTargetRegressor)
    assert dummy_estimator_handler.is_fitted


def test_fit_with_validation_data(dummy_estimator_handler):
    X_val = np.array([[1, 2, 3]])
    y_val = np.array([1])

    config = ModelHandlerConfig(
        X_train=dummy_estimator_handler._config.X_train,
        y_train=dummy_estimator_handler._config.y_train,
        feature_names=dummy_estimator_handler._config.feature_names,
        X_val=X_val,
        y_val=y_val
    )
    validation_data_estimator_handler = EstimatorHandler(config)
    validation_data_estimator_handler.fit()
    assert validation_data_estimator_handler.is_fitted
