import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from sklearn.preprocessing import FunctionTransformer, StandardScaler, RobustScaler, MinMaxScaler
from src.multiple_transformer import MultipleTransformer
from src.xandy_transformers import XTransformer, YTransformer, TransformerConfig
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


def test_multiple_transformer_wrong_input():
    """Test MultipleTransformer's response to invalid input -- single transformer"""
    # This should raise a TypeError because MultipleTransformer expects instances of YTransformer
    with pytest.raises(TypeError):
        MultipleTransformer([StandardScaler(
        ), f"y_transformer should be an instance of YTransformer, but got {type([StandardScaler()]).__name__}"])
    with pytest.raises(TypeError):
        transformer = TransformerConfig(
            name="standard_scaler", transformer=StandardScaler())
        MultipleTransformer(
            [transformer, f"y_transformer should be an instance of YTransformer, but got {type(transformer).__name__}"])


def test_multiple_transformer_wrong_input_list():
    """Test MultipleTransformer's response to invalid input -- list of transformers"""
    stand_scaler = StandardScaler()
    func_trans = FunctionTransformer(np.log1p)
    with pytest.raises(TypeError):
        MultipleTransformer([
            stand_scaler,
            func_trans
        ])


@settings(deadline=None)
@given(array_1d())
def test_multiple_transformer(X):
    """Test the functionality of the MultipleTransformer class"""
    X = reshape_to_2d(X)
    stand_scaler = StandardScaler()
    func_trans = FunctionTransformer(np.log1p)
    y_transformer = YTransformer(transformers=[
        TransformerConfig(name="standard_scaler", transformer=stand_scaler),
    ])
    multi_trans = MultipleTransformer(y_transformer=y_transformer)
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
    multi_trans = MultipleTransformer(y_transformer=y_transformer)
    # Expecting no exceptions here as the transformer list is empty but valid
    multi_trans.fit(np.array([[1, 2], [3, 4]]))
    # Check that the transformation is a passthrough
    X = np.array([5, 6, 7, 8])
    X_trans = multi_trans.transform(X)
    assert np.array_equal(X, X_trans)


def test_multiple_transformer_y_transformer_with_invalid_transformer():
    """Test MultipleTransformer's response to a YTransformer with an invalid transformer"""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        y_transformer = YTransformer(transformers=[None])


def test_multiple_transformer_y_transformer_without_transformer_attribute():
    """Test MultipleTransformer's response to a YTransformer without transformer attribute"""
    with pytest.raises(AttributeError):
        y_transformer = YTransformer()
        del y_transformer.transformers
        multi_trans = MultipleTransformer(y_transformer=y_transformer)
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
    multi_trans = MultipleTransformer(y_transformer=y_transformer)
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
    y_transformer = YTransformer(transformers=transformers_list)
    multi_trans = MultipleTransformer(y_transformer=y_transformer)
    multi_trans.fit(X)
    X_trans = multi_trans.transform(X)
    assert not np.array_equal(X, X_trans)
    X_inv = multi_trans.inverse_transform(X_trans)
    assert np.allclose(X, X_inv, rtol=.05)
