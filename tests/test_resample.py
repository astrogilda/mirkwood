import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np
from pydantic import ValidationError
from utils.resample import ResamplingParams, Resampler

# Test normal resampling cases


@settings(deadline=None)
@given(st.lists(st.tuples(st.floats(allow_nan=False, allow_infinity=False),
                          st.floats(allow_nan=False, allow_infinity=False)),
                min_size=2, max_size=10))
def test_normal_resample_data(paired_list):
    x_list, y_list = zip(*paired_list)
    x, y = np.array(x_list), np.array(y_list)
    resampler = Resampler(ResamplingParams(
        frac_samples=0.5, seed=0, replace=False))

    if len(set(x)) > 2 and len(set(y)) > 2:
        (ib_x, ib_y), (oob_x, oob_y), _, _ = resampler.resample_data([x, y])
        assert ib_x.shape[0] == ib_y.shape[0]
        assert np.array_equal(
            np.sort(np.concatenate((ib_x, oob_x))), np.sort(x))
        assert np.array_equal(
            np.sort(np.concatenate((ib_y, oob_y))), np.sort(y))

# Test cases for failing resampling


@pytest.mark.parametrize("frac_samples,seed,expected_error",
                         [(-1, 0, ValueError),
                          (2, 0, ValueError),
                          # case for too small frac_samples
                          (0.00001, 0, ValueError),
                          (0.5, "seed", ValidationError),
                          (0.5, -1, ValidationError),
                          (0.5, 2**32, ValidationError)])
def test_failing_resample_data(frac_samples, seed, expected_error):
    array = np.random.rand(10)
    with pytest.raises(expected_error) as exc_info:
        resampler = Resampler(ResamplingParams(
            frac_samples=frac_samples, seed=seed))
        resampler.resample_data([array])


# Test different seeds yield different results


@settings(deadline=None)
@given(numpy_array=arrays(dtype=float, shape=st.integers(min_value=100, max_value=1000), elements=st.floats(allow_nan=False, allow_infinity=False)), frac_samples=st.floats(min_value=1/10, max_value=0.9))
def test_different_seeds(numpy_array, frac_samples):
    x = y = np.array(numpy_array)
    resampler1 = Resampler(ResamplingParams(frac_samples=frac_samples, seed=0))
    resampler2 = Resampler(ResamplingParams(frac_samples=frac_samples, seed=1))

    if len(set(numpy_array)) == 1:
        with pytest.raises(ValueError, match="Resampling would be nonsensical when all arrays have only one unique element."):
            resampler1.resample_data([x, y])
    else:
        (ib_x1, ib_y1), _, _, _ = resampler1.resample_data([x, y])
        (ib_x2, ib_y2), _, _, _ = resampler2.resample_data([x, y])

        # Check the fraction of unique samples is larger than a threshold
        if len(set(numpy_array)) > 5 and frac_samples > 0.2:
            assert not np.array_equal(ib_x1, ib_x2)

# Test resampling with different frac_samples


@settings(deadline=None)
@given(numpy_array=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2, max_size=10), frac_samples=st.floats(min_value=0.1, max_value=1))
def test_resample_frac_samples(numpy_array, frac_samples):
    x = y = np.array(numpy_array)
    resampler = Resampler(ResamplingParams(frac_samples=frac_samples, seed=0))

    if len(np.unique(np.array(numpy_array))) == 1:
        with pytest.raises(ValueError, match="Resampling would be nonsensical when all arrays have only one unique element."):
            resampler.resample_data([x, y])

    elif frac_samples < 1/len(numpy_array):
        with pytest.raises(ValueError, match=f"Given 'frac_samples' of {frac_samples} and data size of {len(numpy_array)}, resampling results in fewer than one sample. Increase 'frac_samples'."):
            resampler.resample_data([x, y])
    else:
        (ib_x, ib_y), (oob_x, oob_y), _, _ = resampler.resample_data([x, y])

        if frac_samples == 1:
            assert len(ib_x) == len(ib_y) == len(x)
            print(f"ib_x: {ib_x}")
            print(f"oob_x: {oob_x}")
            print("\n")
            assert oob_x.size == oob_y.size == 0
        else:
            assert ib_x.shape[0] == ib_y.shape[0]
            assert set(ib_x).issubset(set(x))
            assert set(oob_x).issubset(set(x))
            assert set(ib_y).issubset(set(y))
            assert set(oob_x).issubset(set(y))

# Test inconsistent lengths


def test_inconsistent_lengths():
    np.random.seed(0)
    array1 = np.random.rand(10)
    array2 = np.random.rand(9)
    resampler = Resampler(ResamplingParams(frac_samples=0.5, seed=0))

    with pytest.raises(ValueError):
        resampler.resample_data([array1, array2])

# Test resampling with single unique element


def test_resample_single_unique():
    array1 = np.array([1] * 10)
    array2 = np.array([2] * 10)
    resampler = Resampler(ResamplingParams(frac_samples=0.5, seed=0))

    with pytest.raises(ValueError):
        resampler.resample_data([array1, array2])

# Test resampling for an empty list


def test_resample_empty():
    array1 = np.array([])
    array2 = np.array([])
    resampler = Resampler(ResamplingParams(frac_samples=0.5, seed=0))

    with pytest.raises(ValueError):
        resampler.resample_data([array1, array2])


def test_randomness_of_resampling():
    np.random.seed(0)
    array = np.random.rand(10)
    resampler1 = Resampler(ResamplingParams(frac_samples=0.5, seed=0))
    resampler2 = Resampler(ResamplingParams(frac_samples=0.5, seed=1))

    # First run
    ib_array1, _, _, _ = resampler1.resample_data([array])
    # Second run
    ib_array2, _, _, _ = resampler2.resample_data([array])

    assert not np.array_equal(ib_array1, ib_array2)


def test_resampling_with_no_replacement():
    np.random.seed(0)
    array = np.random.rand(10)
    resampler = Resampler(ResamplingParams(
        frac_samples=0.5, seed=0, replace=False))

    # Resample data
    _, _, ib_idx, _ = resampler.resample_data([array])

    # Check if there are any repeated indices
    assert len(ib_idx) == len(set(ib_idx))
