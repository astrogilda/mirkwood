import numpy as np
import pytest
from hypothesis import given, settings
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import hypothesis.strategies as st
from sklearn.base import BaseEstimator, TransformerMixin

from utils.transform_with_checks import apply_transform_with_checks


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
    with pytest.raises(AttributeError, match=r".*does not have a method called transform.*"):
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
    with pytest.raises(ValueError, match=r"Expected 2D array, got 1D array instead"):
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
    with pytest.raises(AttributeError, match=r"NoneType does not have a method called fit_transform"):
        apply_transform_with_checks(None, 'fit_transform', X)


@given(array_2d())
@settings(deadline=None, max_examples=1)
def test_apply_transform_with_checks_missing_fit_method(X):
    """Test transformer that doesn't have a fit method."""
    class BadTransformer:
        def transform(self, X):
            return X * 2

    transformer = BadTransformer()
    with pytest.raises(AttributeError, match=r".*does not have a method called fit_transform.*"):
        apply_transform_with_checks(transformer, 'fit_transform', X)


@given(array_2d())
@settings(deadline=None, max_examples=1)
def test_apply_transform_with_checks_missing_transform_method(X):
    """Test transformer that doesn't have a transform method."""
    class BadTransformer:
        def fit(self, X):
            return self

    transformer = BadTransformer()
    with pytest.raises(AttributeError, match=r".*does not have a method called transform.*"):
        apply_transform_with_checks(transformer, 'transform', X)


def test_apply_transform_with_checks_string_X():
    """Test with string as X."""
    transformer = StandardScaler()
    with pytest.raises(ValueError, match=r".*Expected 2D array, got scalar array instead*"):
        apply_transform_with_checks(transformer, 'fit_transform', "invalid_X")


@given(array_2d())
@settings(deadline=None, max_examples=1)
def test_apply_transform_with_checks_string_y(X):
    """Test with string as y."""
    transformer = DecisionTreeRegressor()
    with pytest.raises(ValueError, match=r".*dtype='numeric' is not compatible with arrays of bytes/strings*"):
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


@given(array_2d())
@settings(deadline=None, max_examples=10)
def test_apply_transform_with_checks_unfitted_transformer(X):
    """Test case where transformer method is called before fit."""
    transformer = StandardScaler()
    with pytest.raises(ValueError, match=r".*has not been fitted yet.*"):
        apply_transform_with_checks(transformer, 'transform', X)


@given(array_2d(), array_2d())
@settings(deadline=None, max_examples=10)
def test_apply_transform_with_checks_val_data_provided(X, X_val):
    """Test successful transformation with validation data provided."""
    transformer = StandardScaler()
    transformer, result = apply_transform_with_checks(
        transformer, 'fit_transform', X, X_val=X_val)
    assert isinstance(result, np.ndarray)
    assert result.shape == X.shape


@given(array_2d(), array_1d())
@settings(deadline=None, max_examples=10)
def test_apply_transform_with_checks_val_data_mismatch(X, y):
    """Test case where validation data has a different number of features."""
    transformer = StandardScaler()
    if X.shape[0] != y.shape[0]:
        with pytest.raises(ValueError, match=r".*Found input variables with inconsistent numbers of samples.*"):
            apply_transform_with_checks(
                transformer, 'fit_transform', X, y=y)


@given(array_2d(), array_1d())
@settings(deadline=None, max_examples=10)
def test_apply_transform_with_checks_sample_weights_provided(X, sample_weight):
    """Test successful transformation with sample weights provided."""
    transformer = StandardScaler()
    transformer, result = apply_transform_with_checks(
        transformer, 'fit_transform', X, sample_weight=sample_weight)
    assert isinstance(result, np.ndarray)
    assert result.shape == X.shape


@given(array_2d(), array_1d())
@settings(deadline=None, max_examples=10)
def test_apply_transform_with_checks_val_sample_weights_provided(X, val_sample_weight):
    """Test successful transformation with validation sample weights provided."""
    transformer = StandardScaler()
    transformer, result = apply_transform_with_checks(
        transformer, 'fit_transform', X, val_sample_weight=val_sample_weight)
    assert isinstance(result, np.ndarray)
    assert result.shape == X.shape


@given(array_2d())
@settings(deadline=None, max_examples=10)
def test_apply_transform_with_checks_predict_method(X):
    """Test successful prediction with LinearRegression."""
    transformer = LinearRegression().fit(X, np.random.rand(X.shape[0]))
    result = apply_transform_with_checks(
        transformer, 'predict', X)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == X.shape[0]


@given(array_2d())
@settings(deadline=None, max_examples=1)
def test_apply_transform_with_checks_invalid_transformer_method(X):
    """Test transformer that doesn't have a predict_std method."""
    transformer = StandardScaler().fit(X)
    with pytest.raises(AttributeError, match=r".*does not have a method called predict_std.*"):
        apply_transform_with_checks(transformer, 'predict_std', X)


@given(array_1d_and_2d())
@settings(deadline=None, max_examples=10)
def test_apply_transform_with_checks_y_and_sample_weight_provided(arrays):
    """Test successful transformation when y and sample weight are provided."""
    y, X = arrays
    sample_weight = np.random.rand(y.shape[0])
    transformer = LinearRegression()
    transformer = apply_transform_with_checks(
        transformer, 'fit', X, y=y, sample_weight=sample_weight)
    assert isinstance(transformer, BaseEstimator)


@given(array_1d_and_2d(), array_1d_and_2d())
@settings(deadline=None, max_examples=10)
def test_apply_transform_with_checks_val_data_and_y_val_provided(arrays, val_arrays):
    """Test successful transformation when validation data and y_val are provided."""
    y, X = arrays
    y_val, X_val = val_arrays
    transformer = LinearRegression()
    transformer = apply_transform_with_checks(
        transformer, 'fit', X, y, X_val=X_val, y_val=y_val)
    assert isinstance(transformer, BaseEstimator)
