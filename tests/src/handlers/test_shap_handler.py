
import re
from pydantic import ValidationError, parse_obj_as
from sklearn.datasets import make_regression
import tempfile
import os
import numpy as np
import pytest
from pathlib import Path
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import RandomForestRegressor
from src.handlers.shap_handler import ShapHandler
from src.handlers.model_handler import ModelHandlerConfig
from src.transformers.xandy_transformers import XTransformer, YTransformer, TransformerConfig
import shap
import logging
from sklearn.base import BaseEstimator
from utils.validate import check_estimator_compliance
from copy import copy
# Replace FEATURE_NAMES with the actual feature names in your dataset
FEATURE_NAMES = ["feature1", "feature2", "feature3"]

# Configure logger for test file
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.fixture
def dummy_shap_handler_config():
    return ModelHandlerConfig(
        X=np.random.randn(100, len(FEATURE_NAMES)),
        y=np.random.randn(100),
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
            dummy_shap_handler_config.X, dummy_shap_handler_config.y))
        assert isinstance(dummy_shap_handler.explainer, shap.TreeExplainer)
    else:
        with pytest.raises(ValueError):
            dummy_shap_handler.create(model)


# Test failing and passing cases for saving and loading explainer
@pytest.mark.parametrize("fit,valid_filepath", [(True, True), (False, True), (True, False), (False, False)])
def test_save_and_load_explainer(dummy_shap_handler: ShapHandler, dummy_shap_handler_config: ModelHandlerConfig, fit, valid_filepath, caplog):
    model = RandomForestRegressor()
    if fit:
        model.fit(dummy_shap_handler_config.X,
                  dummy_shap_handler_config.y)
        dummy_shap_handler.create(model)

    if valid_filepath:
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temp:
            dummy_shap_handler_config.shap_file_path = temp.name
            if fit:
                dummy_shap_handler._save_explainer()
                assert Path(dummy_shap_handler_config.shap_file_path).exists()

                loaded_shap_handler = ShapHandler(dummy_shap_handler_config)
                loaded_shap_handler._load_explainer()
                assert isinstance(
                    loaded_shap_handler.explainer, shap.TreeExplainer)
            else:
                # Since the base estimator has not been fitted, the explainer should not be created
                with pytest.raises(NotFittedError):
                    dummy_shap_handler._save_explainer()

                # Since the base estimator has not been fitted, the explainer should not be created, and thus not be loaded
                with pytest.raises(EOFError):
                    loaded_shap_handler = ShapHandler(
                        dummy_shap_handler_config)
                    loaded_shap_handler._load_explainer()

            # Cleanup
            os.unlink(dummy_shap_handler_config.shap_file_path)
    else:
        dummy_shap_handler_config.shap_file_path = None
        if fit:
            # with pytest.raises(IOError):
            dummy_shap_handler._save_explainer()
            assert "No filename provided. Skipping save." in caplog.text
        else:
            with pytest.raises(NotFittedError):
                dummy_shap_handler._save_explainer()
                assert "No filename provided. Skipping save." in caplog.text


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


# Test edge cases for precreated_explainer
# Precreate some data
X, y = make_regression()

# Precreate and fit a RandomForestRegressor
model = RandomForestRegressor()
model.fit(X, y)


@pytest.mark.parametrize("precreated_explainer", [
    None,
    (lambda: shap.TreeExplainer(model)),
    "Not an explainer"
])
def test_precreated_explainer(dummy_shap_handler_config: ModelHandlerConfig, precreated_explainer):
    if callable(precreated_explainer):
        precreated_explainer = precreated_explainer()

    dummy_shap_handler_config_dict = dummy_shap_handler_config.dict()
    dummy_shap_handler_config_dict['precreated_explainer'] = precreated_explainer

    if dummy_shap_handler_config_dict['X_transformer']['transformers'] is not None:
        dummy_shap_handler_config_dict['X_transformer'] = XTransformer(transformers=[TransformerConfig(
            name=i['name'], transformer=i['transformer']) for i in dummy_shap_handler_config_dict['X_transformer']['transformers']])
    else:
        dummy_shap_handler_config_dict['X_transformer'] = XTransformer(
            transformers=None)
    if dummy_shap_handler_config_dict['y_transformer']['transformers'] is not None:
        dummy_shap_handler_config_dict['y_transformer'] = YTransformer(transformers=[TransformerConfig(
            name=i['name'], transformer=i['transformer']) for i in dummy_shap_handler_config_dict['y_transformer']['transformers']])
    else:
        dummy_shap_handler_config_dict['y_transformer'] = YTransformer(
            transformers=None)

    if precreated_explainer is not "Not an explainer":
        # Create a new config with the precreated explainer, so pydantic validations can be leveraged
        dummy_shap_handler_config = ModelHandlerConfig(
            **dummy_shap_handler_config_dict)
        shap_handler = ShapHandler(dummy_shap_handler_config)
        if precreated_explainer is not None:
            assert shap_handler.explainer is precreated_explainer
        else:
            with pytest.raises(NotFittedError, match=re.escape("SHAP Explainer is not created. Use calculate() to create it.")):
                shap_handler.explainer
    else:
        with pytest.raises(ValidationError):
            # Create a new config with the precreated explainer, so pydantic validations can be leveraged
            ModelHandlerConfig(**dummy_shap_handler_config_dict)
