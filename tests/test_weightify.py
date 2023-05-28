from hypothesis.extra.numpy import arrays
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from sklearn.exceptions import NotFittedError
from utils.weightify import Weightify, Style
import math

# Define a Hypothesis strategy for valid weightify initializers
valid_weightify_args = st.fixed_dictionaries({
    'style': st.sampled_from(Style),
    'lds_ks': st.integers(1, 10),
    'n_bins': st.integers(10, 1000),
    'beta': st.floats(0, 1),
    'bw_method': st.one_of(st.floats(0.1, 10), st.text()),
    'lds_sigma': st.floats(0.02, 100),
})

# Define a Hypothesis strategy for numpy arrays
numpy_array_strategy = arrays(dtype=float, shape=st.integers(
    100, 10000), elements=st.floats(.01, 1000))


def test_weightify_bad_init():
    """
    Test Weightify initialization with invalid arguments.

    Raises
    ------
    ValueError
        If Weightify is initialized with invalid arguments.
    """
    # Test a variety of incorrect initializers
    with pytest.raises(ValueError):
        Weightify(lds_ks=-11)
    with pytest.raises(ValueError):
        Weightify(n_bins=9)
    with pytest.raises(ValueError):
        Weightify(beta=-0.1)
    with pytest.raises(ValueError):
        Weightify(bw_method=0)
    with pytest.raises(ValueError):
        Weightify(lds_sigma=0.01)
    with pytest.raises(ValueError):
        Weightify(lds_sigma=101)


@given(valid_weightify_args)
def test_weightify_init(args):
    """
    Test Weightify initialization with valid arguments.

    Parameters
    ----------
    args : dict
        Dictionary containing valid Weightify arguments.

    Raises
    ------
    AssertionError
        If Weightify is not initialized correctly.
    """
    # Test that we can initialize Weightify with a variety of valid arguments
    w = Weightify(**args)
    assert isinstance(w, Weightify)
    for k, v in args.items():
        assert getattr(w, k) == v


@given(st.builds(Weightify), numpy_array_strategy)
def test_weightify_transform_before_fit(w, y):
    """
    Test calling transform before fit raises NotFittedError.

    Parameters
    ----------
    w : Weightify
        Initialized Weightify object.
    X : numpy array
        Input array.

    Raises
    ------
    NotFittedError
        If transform is called before fit.
    """
    # Test that calling transform before fit raises a NotFittedError
    with pytest.raises(NotFittedError):
        w.transform(y)


@given(st.builds(Weightify), numpy_array_strategy)
@settings(deadline=None, max_examples=10)
def test_weightify_fit_transform(w, y):
    """
    Test fitting and transforming input arrays.

    Parameters
    ----------
    w : Weightify
        Initialized Weightify object.
    y : numpy array
        Input array.

    Raises
    ------
    AssertionError
        If the length of transformed weights is not equal to the length of the input array,
        or if any weight is not within the range [0.1, 10].
    """
    # Test that we can fit and transform a variety of input arrays
    w.fit(y)
    weights = w.transform(y)
    assert len(weights) == len(y)
    assert all(0.1 <= w <= 10 for w in weights)


@given(st.builds(Weightify), numpy_array_strategy)
@settings(deadline=None, max_examples=100)
def test_weightify_fit_transform_idempotent(w, y):
    """
    Test calling fit_transform twice gives the same result.

    Parameters
    ----------
    w : Weightify
        Initialized Weightify object.
    y : numpy array
        Input array.

    Raises
    ------
    AssertionError
        If the output weights from two fit_transform calls are not close.
    """
    # Test that calling fit_transform twice gives the same result
    w.fit(y)
    weights1 = w.transform(y)
    w.fit(y)
    weights2 = w.transform(y)
    assert np.allclose(weights1, weights2)


@given(st.builds(Weightify), numpy_array_strategy)
@settings(deadline=None, max_examples=100)
def test_weightify_fit_transform_normalizes(w, y):
    """
    Test that the sum of the output weights is approximately equal to the length of the input.

    Parameters
    ----------
    w : Weightify
        Initialized Weightify object.
    y : numpy array
        Input array.

    Raises
    ------
    AssertionError
        If the sum of the output weights is not close to the length of the input array.
    """
    # Test that the sum of the output weights is approximately equal to the length of the input
    w.fit(y)
    weights = w.transform(y)
    assert math.isclose(np.sum(weights), len(y), rel_tol=.2)
