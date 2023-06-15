import pytest
import numpy as np
from src.model_handler import ModelHandler, ModelConfig
from src.bootstrap_handler import BootstrapHandler
from pydantic import ValidationError
from src.data_handler import GalaxyProperty
import random
import string
from numpy.testing import assert_almost_equal

# Test Data
X_train = np.random.rand(50, 3).astype(np.float64)
y_train = np.random.rand(50).astype(np.float64)
X_val = np.random.rand(30, 3).astype(np.float64)
y_val = np.random.rand(30).astype(np.float64)
feature_names = []
for _ in range(20):
    random_string = ''.join(random.choices(
        string.ascii_letters + string.digits, k=10))
    feature_names.append(random_string)

# ModelHandler for tests
model_handler = ModelHandler(
    X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, feature_names=feature_names)


def test_BootstrapHandler_init():
    with pytest.raises(ValidationError):
        # This should fail as frac_samples_best should be in (0, 1]
        BootstrapHandler(model_handler=model_handler, frac_samples_best=1.5)

    # This should pass
    BootstrapHandler(model_handler=model_handler, frac_samples_best=0.8)


def test_BootstrapHandler_Validation():
    with pytest.raises(ValidationError):
        # Test should fail due to the value of z_score is outside the valid range
        BootstrapHandler(model_handler=model_handler, z_score=10)

    with pytest.raises(ValidationError):
        # Test should fail due to invalid GalaxyProperty
        BootstrapHandler(model_handler=model_handler,
                         galaxy_property='Invalid')

    # Test should pass with valid z_score and galaxy_property
    BootstrapHandler(model_handler=model_handler, z_score=1.96,
                     galaxy_property=GalaxyProperty.STELLAR_MASS)


def test_bootstrap_func_mp():
    bootstrap_handler = BootstrapHandler(model_handler=model_handler)

    y_train_resampled, y_pred_mean, y_pred_std, y_pred_lower, y_pred_upper, shap_values_mean = bootstrap_handler.bootstrap_func_mp(
        iteration_num=0)

    # Assert the shapes are consistent
    assert y_pred_mean.shape == y_train_resampled.shape
    assert y_pred_std.shape == y_train_resampled.shape
    assert y_pred_lower.shape == y_train_resampled.shape
    assert y_pred_upper.shape == y_train_resampled.shape
    assert shap_values_mean.shape == (
        y_train_resampled.shape[0], X_train.shape[1])

    # Assert that the predicted means are within the expected range
    assert np.all(y_pred_mean >= y_pred_lower)
    assert np.all(y_pred_mean <= y_pred_upper)

    # Assert that the confidence intervals have the expected relationship
    assert_almost_equal(y_pred_upper - y_pred_mean,
                        y_pred_mean - y_pred_lower)

    # Assert that SHAP values are finite
    assert np.all(np.isfinite(shap_values_mean))
