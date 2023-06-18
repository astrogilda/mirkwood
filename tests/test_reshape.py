import pytest
from hypothesis import given, strategies as st, settings
import numpy as np
from utils.reshape import *


@settings(deadline=None)
@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=10))
def test_reshape_to_1d_array(numpy_array):
    numpy_array = np.array(numpy_array)
    reshaped = reshape_to_1d_array(numpy_array)
    assert reshaped.ndim == 1
    assert np.array_equal(reshaped, numpy_array.ravel())


@settings(deadline=None)
@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=10))
def test_reshape_to_2d_array(numpy_array):
    numpy_array = np.array(numpy_array)
    reshaped = reshape_to_2d_array(numpy_array)
    assert reshaped.ndim == 2
    assert reshaped.shape[1] == 1
    assert np.array_equal(reshaped, numpy_array.reshape(-1, 1))


# Test empty array input for reshape_to_1d_array
def test_reshape_to_1d_array_empty():
    numpy_array = np.array([])
    with pytest.raises(ValueError, match="Input array must not be empty."):
        reshape_to_1d_array(numpy_array)


# Test empty array input for reshape_to_2d_array
def test_reshape_to_2d_array_empty():
    numpy_array = np.array([])
    with pytest.raises(ValueError, match="Input array must not be empty."):
        reshape_to_2d_array(numpy_array)


# Test non-floating point input for reshape_to_1d_array
@given(st.lists(st.integers(min_value=-10, max_value=10), min_size=1, max_size=10))
def test_reshape_to_1d_array_integers(numpy_array):
    numpy_array = np.array(numpy_array)
    reshaped = reshape_to_1d_array(numpy_array)
    assert reshaped.ndim == 1
    assert np.array_equal(reshaped, numpy_array.ravel())


# Test non-floating point input for reshape_to_2d_array
@given(st.lists(st.integers(min_value=-10, max_value=10), min_size=1, max_size=10))
def test_reshape_to_2d_array_integers(numpy_array):
    numpy_array = np.array(numpy_array)
    reshaped = reshape_to_2d_array(numpy_array)
    assert reshaped.ndim == 2
    assert reshaped.shape[1] == 1
    assert np.array_equal(reshaped, numpy_array.reshape(-1, 1))


# Test invalid shape input for reshape_to_1d_array
def test_reshape_to_1d_array_invalid_shape():
    numpy_array = np.arange(12).reshape(3, 4)
    with pytest.raises(ValueError, match="Input array must have shape \\(n,\\) or \\(n, 1\\)."):
        reshape_to_1d_array(numpy_array)


# Test invalid shape input for reshape_to_2d_array
def test_reshape_to_2d_array_invalid_shape():
    numpy_array = np.arange(12).reshape(3, 4)
    with pytest.raises(ValueError, match="Input array must have shape \\(n,\\) or \\(n, 1\\)."):
        reshape_to_2d_array(numpy_array)
