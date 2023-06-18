import numpy as np
import pytest
from pathlib import Path
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import RandomForestRegressor
from src.shap_handler import ShapHandler
from src.model_handler import ModelHandlerConfig
import shap
import logging
from sklearn.base import BaseEstimator

# Replace FEATURE_NAMES with the actual feature names in your dataset
FEATURE_NAMES = ["feature1", "feature2", "feature3"]

# Configure logger for test file
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.fixture
def dummy_shap_handler_config():
    return ModelHandlerConfig(
        X_train=np.random.randn(100, len(FEATURE_NAMES)),
        y_train=np.random.randn(100),
        fitting_mode=True,
        file_path=None,
        shap_file_path=None,
        feature_names=FEATURE_NAMES,
    )


@pytest.fixture
def dummy_shap_handler(dummy_shap_handler_config):
    return ShapHandler(dummy_shap_handler_config)


@pytest.mark.parametrize(
    "model, should_pass",
    [
        (RandomForestRegressor(), True),
        (BaseEstimator(), False),
        (None, False),
    ],
)
def test_create_explainer(dummy_shap_handler: ShapHandler, dummy_shap_handler_config: ModelHandlerConfig, model, should_pass):
    """
    Test creating a SHAP explainer.
    """
    if should_pass:
        dummy_shap_handler.create(model.fit(
            dummy_shap_handler_config.X_train, dummy_shap_handler_config.y_train))
        assert isinstance(dummy_shap_handler.explainer, shap.TreeExplainer)
    else:
        with pytest.raises(ValueError):
            dummy_shap_handler.create(model)


@pytest.mark.parametrize("fit", [True, False])
def test_save_and_load_explainer(dummy_shap_handler: ShapHandler, dummy_shap_handler_config: ModelHandlerConfig, fit):
    """
    Test saving and loading the SHAP explainer.
    """
    model = RandomForestRegressor()
    if not fit:
        with pytest.raises(ValueError):
            dummy_shap_handler.create(model)
    else:
        model.fit(dummy_shap_handler_config.X_train,
                  dummy_shap_handler_config.y_train)
        dummy_shap_handler.create(model)

        dummy_shap_handler_config.shap_file_path = "test_explainer.joblib"
        dummy_shap_handler._save_explainer()
        assert Path(dummy_shap_handler_config.shap_file_path).exists()

        loaded_shap_handler = ShapHandler(dummy_shap_handler_config)
        loaded_shap_handler._load_explainer()
        assert isinstance(loaded_shap_handler.explainer, shap.TreeExplainer)

        # Cleanup
        Path(dummy_shap_handler_config.shap_file_path).unlink()


def test_not_fitted_error(dummy_shap_handler: ShapHandler):
    """
    Test raising of NotFittedError when trying to access the explainer before it's created.
    """
    with pytest.raises(NotFittedError):
        dummy_shap_handler.explainer


def test_get_shap_data(dummy_shap_handler: ShapHandler):
    """
    Test the _get_shap_data function for different input sizes.
    """
    X_small = np.random.randn(100, len(FEATURE_NAMES))
    X_large = np.random.randn(300, len(FEATURE_NAMES))

    data_small = dummy_shap_handler._get_shap_data(X_small)
    data_large = dummy_shap_handler._get_shap_data(X_large)

    assert np.array_equal(data_small, X_small)
    assert data_large.shape == (100, len(FEATURE_NAMES))
