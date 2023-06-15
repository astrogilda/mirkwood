import pytest
import numpy as np
from src.model_handler import ModelHandler, ModelConfig
from src.bootstrap_handler import BootstrapHandler
from pydantic import ValidationError
from src.data_handler import GalaxyProperty
import random
import string
from numpy.testing import assert_almost_equal, assert_array_equal
from utils.metrics import DeterministicErrorMetrics, ProbabilisticErrorMetrics
from utils.metrics import calculate_z_score

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

    y_val, y_pred_mean, y_pred_std, shap_values_mean = bootstrap_handler.bootstrap_func_mp(
        iteration_num=0)

    # Assert the shapes are consistent
    assert_array_equal(y_val, bootstrap_handler.model_handler.y_val)
    assert y_pred_mean.shape == y_val.shape
    assert y_pred_std.shape == y_val.shape
    assert shap_values_mean.shape == (
        y_val.shape[0], X_train.shape[1])

    # Assert that the standard deviations are positive
    assert np.all(y_pred_std >= 0)

    # Assert that SHAP values are finite
    assert np.all(np.isfinite(shap_values_mean))


def test_bootstrap_func_mp_metrics():
    # Test that the metrics are calculated correctly
    bootstrap_handler = BootstrapHandler(model_handler=model_handler)
    y_val, y_pred_mean, y_pred_std, shap_values_mean = bootstrap_handler.bootstrap_func_mp(
        iteration_num=0)

    confidence_level = 0.95
    z_score = calculate_z_score(confidence_level=confidence_level)

    deterministic_error_metrics_handler = DeterministicErrorMetrics(
        yp=y_pred_mean, yt=y_val)
    assert isinstance(deterministic_error_metrics_handler.nrmse(), float)
    assert isinstance(deterministic_error_metrics_handler.nmae(), float)
    assert isinstance(deterministic_error_metrics_handler.medianae(), float)
    assert isinstance(deterministic_error_metrics_handler.mape(), float)
    assert isinstance(deterministic_error_metrics_handler.bias(), float)
    assert isinstance(deterministic_error_metrics_handler.nbe(), float)

    probabilistic_error_metrics_handler = ProbabilisticErrorMetrics(
        yp=y_pred_mean, yt=y_val, yp_lower=y_pred_mean - z_score * y_pred_std, yp_upper=y_pred_mean + z_score * y_pred_std)
    assert isinstance(
        probabilistic_error_metrics_handler.gaussian_crps(), float)
    assert isinstance(probabilistic_error_metrics_handler.ace(), float)
    assert isinstance(probabilistic_error_metrics_handler.pinaw(), float)
    assert isinstance(
        probabilistic_error_metrics_handler.interval_sharpness(), float)
