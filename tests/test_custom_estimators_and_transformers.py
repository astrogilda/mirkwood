from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from random import shuffle
from utils.custom_transformers_and_estimators import *
from ngboost.distns import Normal
from ngboost.scores import LogScore
from sklearn.preprocessing import FunctionTransformer, StandardScaler, RobustScaler
from sklearn.tree import DecisionTreeRegressor
from pydantic.error_wrappers import ValidationError
from utils.custom_transformers_and_estimators import _MultipleTransformer


@st.composite
def array_1d_and_2d(draw):
    """Strategy to generate a pair of 1D and 2D numpy arrays with shared elements"""
    n_elements = draw(st.integers(100, 1000))
    unique_ratio = draw(st.floats(min_value=0.1, max_value=0.99))
    n_unique = round(n_elements * unique_ratio)
    n_repeat = n_elements - n_unique

    # feel free to adjust the range and size of this pool
    elements_pool = np.linspace(-1000, 1000, 50000)
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
    assert config.early_stopping_rounds is None


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


@given(array_1d_and_2d())
@settings(
    deadline=None, max_examples=10
)
def test_customngbregressor_fit_and_predict(arrays):
    """Test fitting and predicting of the CustomNGBRegressor class"""
    y, X = arrays
    X = np.nan_to_num(X)  # replace infinities with large finite numbers
    y = np.nan_to_num(y)  # replace infinities with large finite numbers

    ngb = CustomNGBRegressor()
    ngb.fit(X, y)
    preds = ngb.predict(X)

    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]


@given(array_2d())
@settings(deadline=None, max_examples=10)
def test_xtransformer_passing(array):
    # Define the transformer configuration
    x_transformer = XTransformer()

    # Fit and transform
    transformed_array = x_transformer.transformers[0].transformer.fit_transform(
        array)

    # Validate if StandardScaler is effectively applied
    assert_almost_equal(transformed_array.mean(axis=0), 0)
    assert_almost_equal(transformed_array.std(axis=0), 1)

    # Test inverse transform
    inversed_array = x_transformer.transformers[0].transformer.inverse_transform(
        transformed_array)
    assert_almost_equal(inversed_array, array)


def test_xtransformer_failing():
    # Define the transformer configuration
    x_transformer = XTransformer()

    # Test for NotFittedError
    with pytest.raises(NotFittedError):
        x_transformer.transformers[0].transformer.transform(array_2d())


@given(array_1d())
@settings(deadline=None)
def test_ytransformer_passing(array):
    # Define the transformer configuration
    y_transformer = YTransformer(transformers=TransformerTuple(
        [
            TransformerConfig(name="robust_scaler",
                              transformer=RobustScaler()),
            TransformerConfig(name="standard_scaler",
                              transformer=StandardScaler())
        ]))

    array = array.reshape(-1, 1)
    # Fit and transform

    robust_array = y_transformer.transformers[0].transformer.fit_transform(
        array)
    scaled_array = y_transformer.transformers[1].transformer.fit_transform(
        robust_array)

    # Validate if reshaping, StandardScaler and reshaping again are effectively applied
    assert_almost_equal(scaled_array.mean(), 0)
    assert_almost_equal(scaled_array.std(), 1)

    # Test inverse transform
    inversed_array = y_transformer.transformers[0].transformer.inverse_transform(
        y_transformer.transformers[1].transformer.inverse_transform(scaled_array))
    assert_almost_equal(inversed_array.reshape(-1, 1), array)


def test_ytransformer_failing():
    # Define the transformer configuration
    y_transformer = YTransformer()

    # Test for NotFittedError
    with pytest.raises(NotFittedError):
        y_transformer.transformers[0].transformer.transform(array_1d())


def test_multiple_transformer_wrong_input():
    """Test MultipleTransformer's response to invalid input -- single transformer"""
    # This should raise a TypeError because MultipleTransformer expects instances of YTransformer
    with pytest.raises(ValueError):
        _MultipleTransformer([StandardScaler(
        ), f"y_transformer should be an instance of YTransformer, but got {type(StandardScaler()).__name__}"])
    with pytest.raises(ValueError):
        _MultipleTransformer([
            TransformerConfig(name="standard_scaler", transformer=StandardScaler()), f"y_transformer should be an instance of YTransformer, but got {type(TransformerConfig()).__name__}"])


def test_multiple_transformer_wrong_input_list():
    """Test MultipleTransformer's response to invalid input -- list of transformers"""
    stand_scaler = StandardScaler()
    func_trans = FunctionTransformer(np.log1p)
    with pytest.raises(ValueError):
        _MultipleTransformer([
            stand_scaler,
            func_trans
        ])


@given(array_1d())
def test_multiple_transformer(X):
    """Test the functionality of the MultipleTransformer class"""
    stand_scaler = StandardScaler()
    func_trans = FunctionTransformer(np.log1p)
    y_transformer = YTransformer(transformers=[
        TransformerConfig(name="standard_scaler", transformer=stand_scaler),
    ])
    multi_trans = _MultipleTransformer(y_transformer=y_transformer)
    # TransformerConfig(name="func_transformer", transformer=func_trans)
    multi_trans.fit(X)
    X_trans = multi_trans.transform(X)
    assert not np.array_equal(X.ravel(), X_trans.ravel())
    X_inv = multi_trans.inverse_transform(X_trans)
    np.testing.assert_almost_equal(X.ravel(), X_inv.ravel())


@given(array_1d_and_2d())
@settings(
    deadline=None, max_examples=10
)
def test_create_estimator(arrays):
    y, X = arrays
    model_config = ModelConfig()
    x_transformer = XTransformer()
    y_transformer = YTransformer()
    ttr = create_estimator(model_config=model_config,
                           x_transformer=x_transformer,
                           y_transformer=y_transformer)
    assert isinstance(ttr, CustomTransformedTargetRegressor)

    ttr.fit(X, y)
    y_pred = ttr.predict(X)
    # Check shape of predicted y
    assert y_pred.shape == y.flatten().shape
    # Testing predict_std method
    y_pred_std = ttr.predict_std(X)
    # Check shape of predicted std
    assert y_pred_std.shape == y.flatten().shape


@given(array_1d_and_2d())
@settings(
    deadline=None, max_examples=10
)
def test_create_estimator_None(arrays):
    """Test create_estimator's response to None input"""
    # This should not raise an error and use default parameters instead
    y, X = arrays
    print(X.shape, y.shape)

    ttr = create_estimator(None, None, None)
    assert isinstance(ttr, CustomTransformedTargetRegressor)

    print(X.shape, y.shape)
    ttr.fit(X, y)
    y_pred = ttr.predict(X)
    # Check shape of predicted y
    assert y_pred.shape == y.flatten().shape
    # Testing predict_std method
    y_pred_std = ttr.predict_std(X)
    # Check shape of predicted std
    assert y_pred_std.shape == y.flatten().shape
