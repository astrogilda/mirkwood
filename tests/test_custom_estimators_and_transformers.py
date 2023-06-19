from src.data_handler import DataHandler, DataHandlerConfig, GalaxyProperty
from sklearn.linear_model import LinearRegression
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from utils.custom_transformers_and_estimators import *
from utils.custom_transformers_and_estimators import _MultipleTransformer
from utils.validate import *
from utils.reshape import *
from utils.resample import *

from ngboost.distns import Normal
from ngboost.scores import LogScore
from sklearn.preprocessing import FunctionTransformer, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from pydantic.error_wrappers import ValidationError


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
    ngb = CustomNGBRegressor(config=ModelConfig())
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

    ngb = CustomNGBRegressor(config=ModelConfig())
    ngb.fit(X_train, y_train, sample_weight=sample_weight, X_val=X_val,
            Y_val=y_val, val_sample_weight=val_sample_weight)

    preds = ngb.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]

    preds_std = ngb.predict_std(X)
    assert isinstance(preds_std, np.ndarray)
    assert preds_std.shape[0] == X.shape[0]


@given(array_2d())
@settings(deadline=None, max_examples=10)
def test_xtransformer_passing_and_inverse_transforming(array):
    # Define the transformer configuration
    x_transformer = XTransformer()

    # Loop over each transformer in the XTransformer
    for transformer_config in x_transformer.transformers:
        # Fit and transform
        transformed_array = transformer_config.transformer.fit_transform(array)
        # Validate if StandardScaler is effectively applied
        if isinstance(transformer_config.transformer, StandardScaler):
            assert_almost_equal(transformed_array.mean(axis=0), 0)
        # Test inverse transform right after the transformation
        inversed_array = transformer_config.transformer.inverse_transform(
            transformed_array)
        assert np.allclose(inversed_array, array, rtol=.05)


def test_xtransformer_failing():
    # Define the transformer configuration
    x_transformer = XTransformer()

    # Test for NotFittedError
    with pytest.raises(NotFittedError):
        x_transformer.transformers[0].transformer.transform(array_2d())


@given(array_1d())
@settings(deadline=None, max_examples=10)
def test_ytransformer_passing_and_inverse_transforming(array):
    # Define the transformer configuration
    y_transformer = YTransformer(transformers=TransformerTuple(
        [
            TransformerConfig(name="robust_scaler",
                              transformer=RobustScaler()),
            TransformerConfig(name="standard_scaler",
                              transformer=StandardScaler())
        ]))

    # Loop over each transformer in the XTransformer
    for transformer_config in y_transformer.transformers:
        # Fit and transform
        transformed_array = transformer_config.transformer.fit_transform(
            reshape_to_2d_array(array))
        # Validate if StandardScaler is effectively applied
        if isinstance(transformer_config.transformer, StandardScaler):
            assert_almost_equal(transformed_array.mean(axis=0), 0)
        # Test inverse transform right after the transformation
        inversed_array = transformer_config.transformer.inverse_transform(
            reshape_to_2d_array(transformed_array))
        assert np.allclose(reshape_to_1d_array(
            inversed_array), array, rtol=.05)


def test_ytransformer_failing():
    # Define the transformer configuration
    y_transformer = YTransformer()

    # Test for NotFittedError
    with pytest.raises(NotFittedError):
        y_transformer.transformers[0].transformer.transform(array_1d())


def test_multiple_transformer_wrong_input():
    """Test MultipleTransformer's response to invalid input -- single transformer"""
    # This should raise a TypeError because MultipleTransformer expects instances of YTransformer
    with pytest.raises(TypeError):
        _MultipleTransformer([StandardScaler(
        ), f"y_transformer should be an instance of YTransformer, but got {type(StandardScaler()).__name__}"])
    with pytest.raises(TypeError):
        transformer = TransformerConfig(
            name="standard_scaler", transformer=StandardScaler())
        _MultipleTransformer(
            [transformer, f"y_transformer should be an instance of YTransformer, but got {type(transformer).__name__}"])


def test_multiple_transformer_wrong_input_list():
    """Test MultipleTransformer's response to invalid input -- list of transformers"""
    stand_scaler = StandardScaler()
    func_trans = FunctionTransformer(np.log1p)
    with pytest.raises(TypeError):
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
    assert np.allclose(X.ravel(), X_inv.ravel(), rtol=.05)


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

    ttr.fit(X_train, y_train, X_val=X_val, y_val=y_val,
            weight_flag=False, sanity_check=True)
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

    ttr.fit(X_train, y_train, X_val=X_val, y_val=y_val,
            weight_flag=False, sanity_check=True)
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
    y_transformer = YTransformer(transformers=[TransformerConfig(
        name="robust_scaler", transformer=RobustScaler())])
    model_config = ModelConfig()

    # Instantiate the regressor
    ngb = CustomNGBRegressor(model_config)

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
    ttr.fit(X_train, y_train, X_val=X_val, y_val=y_val, sanity_check=True)
    y_pred = ttr.predict(X_val)

    # Fit the base regressor for comparison
    ngb.fit(X_train, y_train)
    y_pred_base = ngb.predict(X_val)

    # Predictions made by CustomTransformedTargetRegressor should be similar to those made by the base estimator
    print(f"y: {y}")
    print(f"y_pred: {y_pred}")
    print(f"y_pred_base: {y_pred_base}")
    print("\n")
    assert np.allclose(y_pred, y_pred_base, rtol=.05)


@given(array_2d())
@settings(deadline=None, max_examples=10)
def test_apply_transform_with_checks(X):
    """Test successful transformation with StandardScaler."""
    transformer = StandardScaler()
    transformer, result = apply_transform_with_checks(
        transformer, 'fit_transform', X)
    assert isinstance(result, np.ndarray)
    assert result.shape == X.shape


@given(array_2d())
@settings(deadline=None, max_examples=1)
def test_apply_transform_with_checks_invalid_method(X):
    """Test invalid method exception."""
    transformer = StandardScaler()
    with pytest.raises(ValueError, match=r".*Invalid method name:.*"):
        apply_transform_with_checks(transformer, 'invalid_method', X)


@given(array_2d())
@settings(deadline=None, max_examples=10)
def test_apply_transform_with_checks_sanity_check(X):
    """Test successful inverse transformation with StandardScaler."""
    transformer = StandardScaler()
    transformer, result = apply_transform_with_checks(
        transformer, 'fit_transform', X, sanity_check=True)
    assert isinstance(result, np.ndarray)
    assert result.shape == X.shape


@given(array_2d())
@settings(deadline=None, max_examples=1)
def test_apply_transform_with_checks_fail_transform(X):
    """Test failed transformation exception."""
    transformer = LinearRegression()  # Not a transformer
    with pytest.raises(AttributeError, match=r".*has no attribute 'transform'.*"):
        apply_transform_with_checks(transformer, 'transform', X)


@given(array_1d_and_2d())
@settings(
    deadline=None, max_examples=10
)
def test_apply_transform_with_checks_y_provided(arrays):
    """Test successful transformation when y is provided."""
    y, X = arrays
    transformer = StandardScaler()
    result = apply_transform_with_checks(transformer, 'fit', X, y=y)
    assert isinstance(result, TransformerMixin)


def test_apply_transform_with_checks_edge_case_empty_array():
    """Test edge case with empty array."""
    transformer = StandardScaler()
    X_empty = np.array([])
    with pytest.raises(ValueError, match=r"Failed to transform data with StandardScaler"):
        apply_transform_with_checks(transformer, 'fit_transform', X_empty)


@given(array_2d())
@settings(deadline=None, max_examples=10)
def test_apply_transform_with_checks_edge_case_single_feature(X):
    """Test edge case with single feature array."""
    transformer = StandardScaler()
    X_single_feature = X[:, 0].reshape(-1, 1)
    _, result = apply_transform_with_checks(
        transformer, 'fit_transform', X_single_feature)
    assert isinstance(result, np.ndarray) and result.shape[1] == 1


@given(array_2d())
@settings(deadline=None, max_examples=1)
def test_apply_transform_with_checks_none_transformer(X):
    """Test with None as transformer."""
    with pytest.raises(AttributeError, match=r"'NoneType' object has no attribute 'fit_transform'"):
        apply_transform_with_checks(None, 'fit_transform', X)


@given(array_2d())
@settings(deadline=None, max_examples=1)
def test_apply_transform_with_checks_missing_fit_method(X):
    """Test transformer that doesn't have a fit method."""
    class BadTransformer:
        def transform(self, X):
            return X * 2

    transformer = BadTransformer()
    with pytest.raises(AttributeError, match=r".*has no attribute 'fit_transform'.*"):
        apply_transform_with_checks(transformer, 'fit_transform', X)


@given(array_2d())
@settings(deadline=None, max_examples=1)
def test_apply_transform_with_checks_missing_transform_method(X):
    """Test transformer that doesn't have a transform method."""
    class BadTransformer:
        def fit(self, X):
            return self

    transformer = BadTransformer()
    with pytest.raises(AttributeError, match=r".*has no attribute 'transform'.*"):
        apply_transform_with_checks(transformer, 'transform', X)


def test_apply_transform_with_checks_string_X():
    """Test with string as X."""
    transformer = StandardScaler()
    with pytest.raises(ValueError, match=r".*could not convert string to float.*"):
        apply_transform_with_checks(transformer, 'fit_transform', "invalid_X")


@given(array_2d())
@settings(deadline=None, max_examples=1)
def test_apply_transform_with_checks_string_y(X):
    """Test with string as y."""
    transformer = DecisionTreeRegressor()
    with pytest.raises(ValueError, match=r".*cannot be considered a valid collection*"):
        apply_transform_with_checks(transformer, 'fit', X, y="invalid_y")


@given(array_2d())
@settings(deadline=None, max_examples=1)
def test_apply_transform_with_checks_none_method_name(X):
    """Test with None as method name."""
    transformer = StandardScaler()
    with pytest.raises(ValueError, match=r".*Invalid method name: None. Must be one of .*"):
        apply_transform_with_checks(transformer, None, X)


@given(array_1d())
@settings(deadline=None, max_examples=1)
def test_apply_transform_with_checks_1d_input_for_2d_transformer(X):
    """Test case where a 1D input is passed to a transformer expecting 2D input."""
    transformer = StandardScaler()
    with pytest.raises(ValueError, match=r".*Expected 2D array, got 1D array instead.*"):
        apply_transform_with_checks(transformer, 'fit_transform', X)


def test_postprocessy():
    """
    Test to check if the DataHandler postprocess_y method works correctly.
    """
    dtype = np.dtype([
        ('log_stellar_mass', float),
        ('log_dust_mass', float),
        ('log_metallicity', float),
        ('log_sfr', float),
    ])
    y_array = np.zeros(2, dtype=dtype)
    y_array['log_stellar_mass'] = [1, 2]
    y_array['log_dust_mass'] = [3, 4]
    y_array['log_metallicity'] = [5, 6]
    y_array['log_sfr'] = [7, 8]

    config = DataHandlerConfig(mulfac=1.0)
    handler = DataHandler(config)
    postprocessed_y = PostProcessY(prop=GalaxyProperty.STELLAR_MASS).transform(
        y_array['log_stellar_mass'])
    expected_output = np.zeros(2, dtype=float)
    expected_output = [10, 100]
    np.testing.assert_array_equal(postprocessed_y, expected_output)

    postprocessed_y = PostProcessY(
        prop=GalaxyProperty.DUST_MASS).transform(y_array['log_dust_mass'])
    expected_output = np.zeros(2, dtype=float)
    expected_output = [999, 9999]
    np.testing.assert_array_equal(postprocessed_y, expected_output)

    postprocessed_y = PostProcessY(
        prop=GalaxyProperty.METALLICITY).transform(y_array['log_metallicity'])
    expected_output = np.zeros(2, dtype=float)
    expected_output = [100000, 1000000]
    np.testing.assert_array_equal(postprocessed_y, expected_output)

    postprocessed_y = PostProcessY(
        prop=GalaxyProperty.SFR).transform(y_array['log_sfr'])
    expected_output = np.zeros(2, dtype=float)
    expected_output = [9999999, 99999999]
    np.testing.assert_array_equal(postprocessed_y, expected_output)

    # test if postprocess_y raises an error when ys is a tuple of arrays
    postprocessed_y = PostProcessY(prop=GalaxyProperty.SFR).transform(
        (y_array['log_sfr'], y_array['log_sfr']))
    expected_output = np.zeros(2, dtype=float)
    expected_output = [[9999999, 99999999], [9999999, 99999999]]
    np.testing.assert_array_equal(postprocessed_y, expected_output)
