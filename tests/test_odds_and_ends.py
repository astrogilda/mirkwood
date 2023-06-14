import numpy as np
from typing import List
from hypothesis import given, strategies as st, settings
import pytest

from utils.odds_and_ends import *


@given(st.lists(st.floats(allow_nan=False), min_size=1, max_size=10))
def test_reshape_to_1d_array(array: List[float]):
    np_array = np.array(array)
    reshaped = reshape_to_1d_array(np_array)
    assert reshaped.ndim == 1
    assert np.array_equal(reshaped, np_array.ravel())


@given(st.lists(st.floats(allow_nan=False), min_size=1, max_size=10))
def test_reshape_to_2d_array(array: List[List[float]]):
    np_array = np.array(array)
    reshaped = reshape_to_2d_array(np_array)
    assert reshaped.ndim == 2
    assert reshaped.shape[1] == 1
    assert np.array_equal(reshaped, np_array.reshape(-1, 1))


@given(
    st.lists(st.integers(min_value=0, max_value=1000),
             min_size=2, max_size=10),
    st.integers(min_value=1, max_value=10),
)
@settings(deadline=None)
def test_numba_resample(idx: List[int], n_samples: int):
    np_idx = np.array(idx)
    resampled = numba_resample(np_idx, n_samples)
    assert resampled.shape[0] == n_samples
    assert set(resampled).issubset(set(np_idx))


def generate_numpy_arrays(min_rows: int, max_rows: int, min_cols: int, max_cols: int):
    return st.tuples(
        st.integers(min_value=min_rows, max_value=max_rows),
        st.integers(min_value=min_cols, max_value=max_cols),
    ).flatmap(
        lambda shape: st.lists(
            st.lists(
                st.floats(allow_nan=False, allow_infinity=False),
                min_size=shape[1],
                max_size=shape[1],
            ),
            min_size=shape[0],
            max_size=shape[0],
        )
    ).map(np.array)


@given(
    generate_numpy_arrays(1, 10, 2, 5),
    generate_numpy_arrays(1, 10, 1, 1),
    st.floats(min_value=0.1, max_value=1.0),
)
@settings(deadline=None)
def test_resample_data(x: np.ndarray, y: np.ndarray, frac_samples: float):
    if len(x) != len(y):
        with pytest.raises(ValueError):
            resample_data(x, y, frac_samples=frac_samples)
    else:
        res_x, res_y = resample_data(x, y, frac_samples=frac_samples)
        assert res_x.shape[0] == res_y.shape[0] == int(frac_samples * len(x))
        assert set(np.unique(res_y)).issubset(set(np.unique(y)))
