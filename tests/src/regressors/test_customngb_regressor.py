import numpy as np
import pytest
from hypothesis import given, strategies as st, settings

from ngboost.distns import Normal
from ngboost.scores import LogScore
from sklearn.preprocessing import FunctionTransformer, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
from pydantic.error_wrappers import ValidationError

from src.regressors.customngb_regressor import ModelConfig, CustomNGBRegressor


@st.composite
def array_1d_and_2d(draw):
    """Strategy to generate a pair of 1D and 2D numpy arrays with shared elements"""
    n_elements = draw(st.integers(10, 100))
    unique_ratio = draw(st.floats(min_value=0.1, max_value=0.99))
    n_unique = round(n_elements * unique_ratio)
    n_repeat = n_elements - n_unique

    # feel free to adjust the range and size of this pool
    elements_pool = np.random.uniform(-100, 100, 5000)
    unique_elements = draw(st.lists(st.sampled_from(
        elements_pool), min_size=n_unique, max_size=n_unique, unique=True))

    repeat_elements = np.random.choice(unique_elements, size=n_repeat).tolist()
    elements = unique_elements + repeat_elements
    np.random.shuffle(elements)
    array_1d = np.array(elements)

    # Generate array_2d
    n_rows = array_1d.shape[0]
    n_columns = draw(st.integers(1, 50))
    array_2d = np.zeros((n_rows, n_columns + 1))
    array_2d[:, 0] = array_1d

    for i in range(n_columns):
        array_2d[:, i + 1] = np.random.permutation(array_1d)

    return array_1d, array_2d


@st.composite
def array_1d(draw):
    """Strategy to generate a 1D numpy array"""
    array_1d, _ = draw(array_1d_and_2d())
    return array_1d


@st.composite
def array_2d(draw):
    """Strategy to generate a 2D numpy array"""
    _, array_2d = draw(array_1d_and_2d())
    return array_2d


def test_model_config_defaults():
    """Test the default configuration of ModelConfig class"""
    config = ModelConfig()
    assert isinstance(config.Base, DecisionTreeRegressor)
    assert config.Dist == Normal
    assert config.Score == LogScore
    assert config.n_estimators == 500
    assert config.learning_rate == 0.04
    assert config.col_sample == 1.0
    assert config.minibatch_frac == 1.0
    assert config.verbose is False
    assert config.natural_gradient is True
    assert config.early_stopping_rounds == 10
    assert config.verbose_eval == 10
    assert config.tol == 1e-4
    assert config.random_state == 1
    assert config.validation_fraction == .1


def test_invalid_model_config():
    """Test ModelConfig's response to invalid configuration"""
    with pytest.raises(ValidationError):
        ModelConfig(n_estimators=-100)
    with pytest.raises(ValidationError):
        ModelConfig(n_estimators="abc")
    with pytest.raises(ValidationError):
        ModelConfig(learning_rate="1.1")  # should be float in range (0,1]


def test_model_config_wrong_base():
    """Test ModelConfig's response to incorrect base model type"""
    # This should raise a TypeError because ModelConfig expects a DecisionTreeRegressor as the base model
    with pytest.raises(ValidationError):
        ModelConfig(Base=StandardScaler())


def test_customngbregressor_init():
    """Test the initialization of the CustomNGBRegressor class"""
    model_config = ModelConfig()
    ngb = CustomNGBRegressor(**vars(model_config))
    assert isinstance(ngb, CustomNGBRegressor)
    assert isinstance(ngb.Base, DecisionTreeRegressor)
    assert ngb.Dist == Normal
    assert ngb.Score == LogScore
    assert ngb.n_estimators == 500
    assert ngb.learning_rate == 0.04
    assert ngb.col_sample == 1.0
    assert ngb.minibatch_frac == 1.0
    assert ngb.verbose is False
    assert ngb.natural_gradient is True
    assert ngb.early_stopping_rounds == 10
    assert ngb.verbose_eval == 10
    assert ngb.tol == 1e-4
    assert ngb.random_state == 1
    assert ngb.validation_fraction == .1


@given(array_1d_and_2d())
@settings(
    deadline=None, max_examples=10
)
def test_customngbregressor_fit_and_predict(arrays):
    """Test fitting and predicting of the CustomNGBRegressor class"""
    y, X = arrays
    X = np.nan_to_num(X)  # replace infinities with large finite numbers
    y = np.nan_to_num(y)  # replace infinities with large finite numbers
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42)
    sample_weight = np.random.uniform(low=0.1, high=1.0, size=X_train.shape[0])
    val_sample_weight = np.random.uniform(
        low=0.1, high=1.0, size=X_val.shape[0])

    model_config = ModelConfig()
    ngb = CustomNGBRegressor(**vars(model_config))
    ngb.fit(X_train, y_train, sample_weight=sample_weight, X_val=X_val,
            Y_val=y_val, val_sample_weight=val_sample_weight)

    preds = ngb.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]

    preds_std = ngb.predict_std(X)
    assert isinstance(preds_std, np.ndarray)
    assert preds_std.shape[0] == X.shape[0]


def test_fit_with_different_sized_X_y():
    """Test that fit raises a ValueError if X and y have different sizes."""
    model_config = ModelConfig()
    ngb = CustomNGBRegressor(**vars(model_config))
    X = np.random.rand(10, 2)
    y = np.random.rand(11)
    with pytest.raises(ValueError):
        ngb.fit(X, y)


def test_predict_before_fit():
    """Test that predict raises a NotFittedError if called before fit."""
    model_config = ModelConfig()
    ngb = CustomNGBRegressor(**vars(model_config))
    X = np.random.rand(10, 2)
    with pytest.raises(NotFittedError):
        ngb.predict(X)


@given(array_2d())
@settings(
    deadline=None, max_examples=10
)
def test_customngbregressor_fit_and_predict_with_pandas_dataframe(X):
    """Test fitting and predicting with pandas DataFrame."""
    import pandas as pd
    X = pd.DataFrame(X)
    y = pd.Series(np.random.rand(X.shape[0]))
    model_config = ModelConfig()
    ngb = CustomNGBRegressor(**vars(model_config))
    ngb.fit(X, y)
    preds = ngb.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]
