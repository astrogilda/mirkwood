from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from random import shuffle
from utils.custom_transformers_and_estimators import *
from utils.custom_transformers_and_estimators import _MultipleTransformer
from utils.odds_and_ends import *

from ngboost.distns import Normal
from ngboost.scores import LogScore
from sklearn.preprocessing import FunctionTransformer, StandardScaler, RobustScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from pydantic.error_wrappers import ValidationError


@st.composite
def array_1d_and_2d(draw):
    """Strategy to generate a pair of 1D and 2D numpy arrays with shared elements"""
    n_elements = draw(st.integers(100, 1000))
    unique_ratio = draw(st.floats(min_value=0.1, max_value=0.99))
    n_unique = round(n_elements * unique_ratio)
    n_repeat = n_elements - n_unique

    # feel free to adjust the range and size of this pool
    elements_pool = np.random.uniform(-1000, 1000, 50000)
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


def test_customngbregressor_init():
    """Test the initialization of the CustomNGBRegressor class"""
    ngb = CustomNGBRegressor(**ModelConfig().dict())
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
    assert ngb.early_stopping_rounds is None


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

    ngb = CustomNGBRegressor()
    ngb.fit(X_train, y_train, sample_weight=sample_weight, X_val=X_val,
            y_val=y_val, val_sample_weight=val_sample_weight)

    preds = ngb.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]

    preds_std = ngb.predict_std(X)
    assert isinstance(preds_std, np.ndarray)
    assert preds_std.shape[0] == X.shape[0]


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


@settings(deadline=None)
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
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42)

    model_config = ModelConfig()
    X_transformer = XTransformer()
    y_transformer = YTransformer()
    ttr = create_estimator(model_config=model_config,
                           X_transformer=X_transformer,
                           y_transformer=y_transformer)
    assert isinstance(ttr, CustomTransformedTargetRegressor)

    ttr.fit(X_train, y_train, X_val=X_val, y_val=y_val, weight_flag=True)
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
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42)

    ttr = create_estimator(None, None, None)
    assert isinstance(ttr, CustomTransformedTargetRegressor)

    ttr.fit(X_train, y_train, X_val=X_val, y_val=y_val, weight_flag=True)
    y_pred = ttr.predict(X)
    # Check shape of predicted y
    assert y_pred.shape == y.flatten().shape
    # Testing predict_std method
    y_pred_std = ttr.predict_std(X)
    # Check shape of predicted std
    assert y_pred_std.shape == y.flatten().shape


def test_create_estimator_invalid():
    """Test create_estimator's response to invalid input"""
    # with pytest.raises(TypeError):
    #    create_estimator("not a ModelConfig", None, None)
    with pytest.raises(TypeError):
        create_estimator(None, "not a XTransformer", None)
    with pytest.raises(TypeError):
        create_estimator(None, None, "not a YTransformer")


@given(array_2d())
@settings(deadline=None, max_examples=10)
def test_xtransformer_passing_and_inverse_transforming(array):
    # Define the transformer configuration
    x_transformer = XTransformer()

    # Loop over each transformer in the XTransformer
    for transformer_config in x_transformer.transformers:

        # Fit and transform
        transformed_array = transformer_config.transformer.fit_transform(array)

        # Test inverse transform right after the transformation
        inversed_array = transformer_config.transformer.inverse_transform(
            transformed_array)
        assert_almost_equal(inversed_array, array)


@given(array_1d())
@settings(deadline=None, max_examples=10)
def test_ytransformer_passing_and_inverse_transforming(array):
    # Define the transformer configuration
    y_transformer = YTransformer()

    # Loop over each transformer in the XTransformer
    for transformer_config in y_transformer.transformers:

        # Fit and transform
        transformed_array = transformer_config.transformer.fit_transform(
            reshape_to_2d_array(array))

        # Test inverse transform right after the transformation
        inversed_array = transformer_config.transformer.inverse_transform(
            reshape_to_2d_array(transformed_array))
        assert_almost_equal(reshape_to_1d_array(inversed_array), array)


@given(array_1d_and_2d())
@settings(
    deadline=None, max_examples=10
)
def test_custom_transformed_target_regressor(arrays):
    """Test fitting, transforming and predicting of the CustomTransformedTargetRegressor class"""
    y, X = arrays
    X = np.nan_to_num(X)  # replace infinities with large finite numbers
    y = np.nan_to_num(y)  # replace infinities with large finite numbers

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42)

    # Instantiate transformers
    X_transformer = XTransformer()
    y_transformer = YTransformer()

    # Instantiate the regressor
    ngb = CustomNGBRegressor()

    pipeline_X = Pipeline([(transformer.name, transformer.transformer)
                          for transformer in X_transformer.transformers])
    pipeline_y = _MultipleTransformer(y_transformer=y_transformer)
    feature_pipeline = Pipeline([
        ('preprocessor', pipeline_X),
        ('regressor', ngb)
    ])
    ttr = CustomTransformedTargetRegressor(regressor=feature_pipeline,
                                           transformer=pipeline_y)

    # Fit and make predictions
    ttr.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    y_pred = ttr.predict(X)

    # Fit the base regressor for comparison
    ngb.fit(X_train, y_train)
    y_pred_base = ngb.predict(X)

    # Predictions made by CustomTransformedTargetRegressor should be similar to those made by the base estimator
    assert_almost_equal(y_pred, y_pred_base, decimal=5)
