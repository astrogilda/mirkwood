import pytest
from hypothesis import given, strategies as st, settings
import numpy as np
from utils.odds_and_ends import *


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


@settings(deadline=None)
@given(st.lists(st.tuples(st.floats(allow_nan=False, allow_infinity=False),
                          st.floats(allow_nan=False, allow_infinity=False)),
                min_size=2, max_size=10))
def test_resample_data(paired_list):
    x_list, y_list = zip(*paired_list)
    x = np.array(x_list)
    y = np.array(y_list)

    unique_elements_x = len(set(x))
    unique_elements_y = len(set(y))

    # Failing case: Should raise error when x and y are not of the same length
    with pytest.raises(ValueError):
        resample_data(x[:-1], y)

    # Normal case: Check if resampling works correctly
    (res_x, res_y), (oob_x, oob_y) = resample_data(x, y)
    assert res_x.shape[0] == res_y.shape[0] == len(x)

    resampled_unique_elements_x = len(set(res_x))
    resampled_unique_elements_y = len(set(res_y))
    oob_unique_elements_x = len(set(oob_x))
    oob_unique_elements_y = len(set(oob_y))

    if unique_elements_x == 1:
        assert len(oob_x) == 0
    if unique_elements_y == 1:
        assert len(oob_y) == 0

    assert resampled_unique_elements_x + oob_unique_elements_x >= unique_elements_x
    assert resampled_unique_elements_y + oob_unique_elements_y >= unique_elements_y

    assert set(np.unique(res_y)).issubset(set(np.unique(y)))
    assert set(np.unique(oob_y)).issubset(set(np.unique(y)))


frac_samples_strategy = st.one_of(
    st.just(1.0),
    st.floats(min_value=-1.0, max_value=0.0).map(lambda x: round(x, 2)),
    st.floats(min_value=1.1, max_value=2.0).map(lambda x: round(x, 2))
)


@settings(deadline=None)
@given(numpy_array=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2, max_size=10), frac_samples=frac_samples_strategy, seed=st.integers(0, 2**32 - 1))
def test_resample_data_edge_and_failing_cases(numpy_array, frac_samples, seed):
    x = y = np.array(numpy_array)

    if frac_samples == 1.0:
        # Edge case: Resampling with frac_samples=1 should yield all original data and no out-of-bag data
        (res_x, res_y), (oob_x, oob_y) = resample_data(
            x, y, frac_samples=frac_samples, seed=seed)
        assert isinstance(res_x, np.ndarray)
        assert isinstance(res_y, np.ndarray)
        assert isinstance(oob_x, np.ndarray)
        assert isinstance(oob_y, np.ndarray)
        assert res_x.shape[0] == res_y.shape[0] == len(x)
        unique_elements_x, unique_elements_y = len(set(x)), len(set(y))
        if unique_elements_x == unique_elements_y == 1:
            assert len(oob_x) == len(oob_y) == 0

    else:
        # Failing cases: Should raise error when frac_samples is out of range [0, 1]
        with pytest.raises(ValueError, match='frac_samples must be greater than 0 and less than or equal to 1.'):
            resample_data(x, y, frac_samples=frac_samples)


@settings(deadline=None)
@given(numpy_array=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=20, max_size=100), frac_samples=st.floats(min_value=0.1, max_value=0.9))
def test_resample_data_different_seeds(numpy_array, frac_samples):
    x = y = np.array(numpy_array)

    # Perform resampling with two different seeds
    (res_x1, res_y1), (oob_x1, oob_y1) = resample_data(
        x, y, frac_samples=frac_samples, seed=0)
    (res_x2, res_y2), (oob_x2, oob_y2) = resample_data(
        x, y, frac_samples=frac_samples, seed=1)

    # Ensure the resampled indices are not identical for different seeds
    unique_elements = len(set(numpy_array))
    if unique_elements > 1:
        assert not np.array_equal(res_x1, res_x2)
        assert not np.array_equal(res_y1, res_y2)
        assert not np.array_equal(oob_x1, oob_x2)
        assert not np.array_equal(oob_y1, oob_y2)
