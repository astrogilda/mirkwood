import re
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from sklearn.preprocessing import FunctionTransformer, StandardScaler, RobustScaler, MinMaxScaler
from src.transformers.multiple_transformer import MultipleTransformer
from src.transformers.xandy_transformers import XTransformer, YTransformer, TransformerConfig, TransformerBase
from utils.reshape import reshape_to_2d


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


@settings(deadline=None)
@given(array_2d())
def test_multiple_transformer_wrong_input(X):
    """Test MultipleTransformer's response to invalid input -- single transformer"""
    # This should raise a TypeError because MultipleTransformer expects inputs of type List[TransformerConfig]

    # Input is a list of single TransformMixin object, not a list of TransformerConfig
    mt = MultipleTransformer(
        transformers=[StandardScaler()], sanity_check=False)
    with pytest.raises(ValueError, match="Invalid transformer configuration"):
        mt.fit(X)

    # Input is a single TransformerConfig object, not a list of TransformerConfig
    transformer = TransformerConfig(
        name="standard_scaler", transformer=StandardScaler())
    mt = MultipleTransformer(transformer)
    with pytest.raises(TypeError, match=f"Expected transformers to be a list, but got {type(transformer).__name__}"):
        mt.fit(X)

    # Input is neither a list of TransformerConfig nor does it have a "name" attribute
    mt = MultipleTransformer(
        transformers=[{"transformer": StandardScaler()}], sanity_check=False)
    with pytest.raises(ValueError, match="Invalid transformer configuration"):
        mt.fit(X)


@settings(deadline=None)
@given(array_2d())
def test_multiple_transformer_wrong_input_list(X):
    """Test MultipleTransformer's response to invalid input -- list of transformers"""

    # Input is a list of TransfomerMixin objects, not a list of TransformerConfig
    stand_scaler = StandardScaler()
    func_trans = FunctionTransformer(np.log1p)
    transformers = [
        stand_scaler,
        func_trans
    ]
    mt = MultipleTransformer(transformers=transformers)
    with pytest.raises(ValueError, match="Invalid transformer configuration"):
        mt.fit(X)


@settings(deadline=None)
@given(array_1d())
def test_multiple_transformer(X):
    """Test the functionality of the MultipleTransformer class"""
    X = reshape_to_2d(X)
    stand_scaler = StandardScaler()
    func_trans = FunctionTransformer(np.log1p)
    transformer = TransformerBase(transformers=[
        TransformerConfig(name="standard_scaler", transformer=stand_scaler),
    ])
    multi_trans = MultipleTransformer(**vars(transformer))
    # TransformerConfig(name="func_transformer", transformer=func_trans)
    multi_trans.fit(X)
    X_trans = multi_trans.transform(X)
    assert not np.array_equal(X.ravel(), X_trans.ravel())
    X_inv = multi_trans.inverse_transform(X_trans)
    assert np.allclose(X.ravel(), X_inv.ravel(), rtol=.05)


# Edge cases


def test_multiple_transformer_empty_y_transformer():
    """Test MultipleTransformer's response to an empty YTransformer"""
    y_transformer = YTransformer(transformers=[])
    multi_trans = MultipleTransformer(**vars(y_transformer))
    # Expecting no exceptions here as the transformer list is empty but valid
    multi_trans.fit(np.array([[1, 2], [3, 4]]))
    # Check that the transformation is a passthrough
    X = np.array([5, 6, 7, 8])
    X_trans = multi_trans.transform(X)
    assert np.array_equal(X, X_trans)


@settings(deadline=None)
@given(array_2d())
def test_multiple_transformer_transformers_None(X):
    """Test MultipleTransformer's response to a transformers=None attribute"""
    multi_trans = MultipleTransformer(None)
    multi_trans.fit(X)
    # Check that the transformation is a passthrough
    X_pred = multi_trans.transform(X)
    X_inv = multi_trans.inverse_transform(X_pred)
    np.testing.assert_almost_equal(X, X_pred)
    np.testing.assert_almost_equal(X, X_inv)


def test_multiple_transformer_y_transformer_without_transformer_attribute():
    """Test MultipleTransformer's response to a YTransformer without transformer attribute"""
    y_transformer = YTransformer()
    del y_transformer.transformers
    multi_trans = MultipleTransformer(**vars(y_transformer))
    multi_trans.fit(np.array([[1, 2], [3, 4]]))


def test_multiple_transformer_multiple_transformers():
    """Test the functionality of the MultipleTransformer class with multiple transformers"""
    transformers_list = [
        TransformerConfig(name="standard_scaler",
                          transformer=StandardScaler()),
        TransformerConfig(name="min_max_scaler", transformer=MinMaxScaler()),
        TransformerConfig(name="robust_scaler", transformer=RobustScaler())
    ]
    y_transformer = YTransformer(transformers=transformers_list)
    multi_trans = MultipleTransformer(**vars(y_transformer))
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    multi_trans.fit(X)
    X_trans = multi_trans.transform(X)
    assert not np.array_equal(X, X_trans)
    X_inv = multi_trans.inverse_transform(X_trans)
    assert np.allclose(X, X_inv, rtol=.05)


@settings(deadline=None)
@given(array_2d())
def test_multiple_transformer_with_2d_input(X):
    """Test the functionality of the MultipleTransformer class with 2D input"""
    transformers_list = [
        TransformerConfig(name="standard_scaler",
                          transformer=StandardScaler()),
        TransformerConfig(name="min_max_scaler", transformer=MinMaxScaler()),
    ]
    y_transformer = TransformerBase(transformers=transformers_list)
    multi_trans = MultipleTransformer(**vars(y_transformer))
    multi_trans.fit(X)
    X_trans = multi_trans.transform(X)
    assert not np.array_equal(X, X_trans)
    X_inv = multi_trans.inverse_transform(X_trans)
    assert np.allclose(X, X_inv, rtol=.05)


def test_multiple_transformer_empty_X():
    """Test MultipleTransformer's response to an empty input array X"""
    transformers_list = [
        TransformerConfig(name="standard_scaler",
                          transformer=StandardScaler()),
        TransformerConfig(name="min_max_scaler", transformer=MinMaxScaler()),
        TransformerConfig(name="robust_scaler", transformer=RobustScaler())
    ]
    y_transformer = YTransformer(transformers=transformers_list)
    multi_trans = MultipleTransformer(**vars(y_transformer))
    X = np.array([[]])
    # This should raise a ValueError as the input array X is empty
    with pytest.raises(ValueError):
        multi_trans.fit(X)


def test_multiple_transformer_None_transformer():
    """Test MultipleTransformer's response to a None transformer in the list"""
    transformers_list = [
        TransformerConfig(name="standard_scaler",
                          transformer=StandardScaler()),
        None,
        TransformerConfig(name="min_max_scaler", transformer=MinMaxScaler()),
    ]
    # If we create a y_transformer using transformers_list -- as is the right way to do it -- it will raise a ValidationError, which again is the right error to raise. However, we are trying to catch a ValueError in MultipleTransformer, so we pass in transformers_list manually.
    multi_trans = MultipleTransformer(transformers_list)
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    # This should raise a ValueError as one of the transformer configurations is None
    with pytest.raises(ValueError, match="Invalid transformer configuration."):
        multi_trans.fit(X)
