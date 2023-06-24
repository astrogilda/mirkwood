from sklearn.datasets import make_regression
from utils.weightify import Weightify
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from hypothesis.extra.numpy import arrays
from hypothesis import given, strategies as st, settings
from sklearn.exceptions import NotFittedError
import numpy as np
import pytest
from pydantic import ValidationError

from utils.validate import check_estimator_compliance
from src.regressors.customngb_regressor import CustomNGBRegressor
from src.transformers.multiple_transformer import MultipleTransformer
from utils.weightify import Weightify, WeightifyConfig, Style, inv_weights, sqrt_inv_weights, special_weights, dir_weights

# Define a Hypothesis strategy for valid weightify initializers
valid_weightify_args = st.builds(WeightifyConfig,
                                 n_bins=st.integers(10, 1000),
                                 bw_method=st.one_of(
                                     st.floats(0.1, 10), st.sampled_from(["scott", "silverman"])),
                                 lds_ks=st.integers(1, 10),
                                 lds_sigma=st.floats(0.1, 10),
                                 style=st.sampled_from(Style),
                                 beta=st.floats(0, 1))

# Define a Hypothesis strategy for numpy arrays
numpy_array_strategy = arrays(dtype=float, shape=st.integers(
    100, 10000), elements=st.floats(.01, 1000))


@given(valid_weightify_args)
def test_weightify_init(config):
    """
    Test Weightify initialization with valid arguments.

    Parameters
    ----------
    config : WeightifyConfig
        Valid WeightifyConfig instance.

    Raises
    ------
    AssertionError
        If Weightify is not initialized correctly.
    """
    w = Weightify()
    w.config = config
    assert isinstance(w, Weightify)
    assert w.config == config


@given(numpy_array_strategy)
def test_weightify_transform_before_fit(y):
    """
    Test calling transform before fit raises NotFittedError.

    Parameters
    ----------
    w : Weightify
        Initialized Weightify object.
    y : numpy array
        Input array.

    Raises
    ------
    NotFittedError
        If transform is called before fit.
    """
    w = Weightify()
    with pytest.raises(NotFittedError):
        w.transform(y)


@given(valid_weightify_args, numpy_array_strategy)
@settings(deadline=None, max_examples=10)
def test_weightify_fit_transform(c, y):
    """
    Test fitting and transforming input arrays.

    Parameters
    ----------
    c : WeightifyConfig
        Valid WeightifyConfig instance.
    y : numpy array
        Input array.

    Raises
    ------
    AssertionError
        If the length of transformed weights is not equal to the length of the input array,
        or if any weight is not within the range [0.1, 10].
    """
    w = Weightify(**vars(c))
    w.fit(y)
    weights = w.transform(y)
    assert len(weights) == len(y)
    assert all(0.1 <= w <= 10 for w in weights)


@given(valid_weightify_args, numpy_array_strategy)
@settings(deadline=None, max_examples=100)
def test_weightify_fit_transform_idempotent(c, y):
    """
    Test calling fit_transform twice gives the same result.

    Parameters
    ----------

    c : WeightifyConfig
        Valid WeightifyConfig instance.
    y : numpy array
        Input array.

    Raises
    ------
    AssertionError
        If the output weights from two fit_transform calls are not close.
    """
    w = Weightify(**vars(c))
    w.fit(y)
    weights1 = w.transform(y)
    w.fit(y)
    weights2 = w.transform(y)
    # if not np.isnan(weights1).any():
    assert np.allclose(weights1, weights2)


@given(valid_weightify_args, numpy_array_strategy)
@settings(deadline=None, max_examples=100)
def test_weightify_transform_after_fit_with_diff_data(c, y):
    """
    Test calling transform on different data after fit.

    Parameters
    ----------

    c : WeightifyConfig
        Valid WeightifyConfig instance.
    y : numpy array
        Input array.

    Raises
    ------
    AssertionError
        If the output weights from two different input arrays are close after fitting on one of them.
    """
    z = np.random.rand(len(y))
    z = z.reshape(y.shape)
    z.dtype = y.dtype
    w = Weightify(**vars(c))
    w.fit(y)
    weights_y = w.transform(y)
    weights_z = w.transform(z)
    if not np.array_equal(y, z):
        assert not np.allclose(weights_y, weights_z)


# TODO: below test fails; fix it
'''
# Define a Hypothesis strategy for invalid weightify configurations
invalid_weightify_args = st.one_of(
    st.builds(WeightifyConfig, n_bins=st.integers(1, 9)),  # n_bins too low
    # bw_method out of range
    st.builds(WeightifyConfig, bw_method=st.floats(-1, 0)),
    # lds_ks out of range
    st.builds(WeightifyConfig, lds_ks=st.integers(-10, 0)),
    # lds_sigma out of range
    st.builds(WeightifyConfig, lds_sigma=st.floats(-1, 0)),
    st.builds(WeightifyConfig, beta=st.floats(-1, 0))  # beta out of range
)


@given(invalid_weightify_args, numpy_array_strategy)
def test_weightify_invalid_init(c, y):
    """
    Test Weightify initialization with invalid arguments.

    Parameters
    ----------
    c : WeightifyConfig
        Valid WeightifyConfig instance.
    y : numpy array
        Input array.

    Raises
    ------
    ValueError
        If Weightify is fit with invalid arguments.
    """
    w = Weightify()
    with pytest.raises(ValidationError):
        w.fit(y, c)
'''


# Initialize the Weightify transformer
weightify = Weightify(style="lds", lds_ks=3, lds_sigma=1,
                      n_bins=10, bw_method="scott", beta=0.5)

# Initialize the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Use standard scaler as preprocessor
    ('weightify', weightify)  # Apply Weightify transformer
])

# Initialize the TransformedTargetRegressor
ttregressor = TransformedTargetRegressor(
    regressor=LinearRegression(), transformer=pipeline)


def test_weightify():
    X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
    ttregressor.fit(X, y)
    predictions = ttregressor.predict(X)

    # Test that model can fit and predict without errors
    assert predictions.shape == y.shape

    # Test that Weightify attributes are set after fitting
    assert hasattr(ttregressor.transformer.named_steps['weightify'], 'fitted_')
    assert hasattr(
        ttregressor.transformer.named_steps['weightify'], 'poly_coeffs_')

    # Test that predicted values have expected properties
    assert predictions.min() <= y.max()
    assert predictions.max() >= y.min()


@pytest.mark.parametrize("params", [
    {"style": Style.DIR, "lds_ks": 3, "lds_sigma": 1,
        "n_bins": 10, "bw_method": "scott", "beta": 0.5},
    {"style": Style.INV, "lds_ks": 3, "lds_sigma": 1,
        "n_bins": 10, "bw_method": "scott", "beta": 0.5},
])
def test_weightify_params(params):
    weightify = Weightify(**params)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('weightify', weightify)
    ])
    ttregressor = TransformedTargetRegressor(
        regressor=LinearRegression(), transformer=pipeline)

    X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
    ttregressor.fit(X, y)

    # Test that model can fit and predict with different parameters
    predictions = ttregressor.predict(X)
    assert predictions.shape == y.shape


@given(samples_per_bin=st.lists(st.floats(min_value=0.0001, max_value=1000), min_size=10, max_size=100))
def test_inv_weights(samples_per_bin):
    weights = inv_weights(np.array(samples_per_bin))
    assert len(weights) == len(samples_per_bin)
    assert weights.sum() == len(samples_per_bin)
    assert all(weight > 0 for weight in weights)


@given(samples_per_bin=st.lists(st.floats(min_value=0.0001, max_value=1000), min_size=10, max_size=100))
def test_sqrt_inv_weights(samples_per_bin):
    weights = sqrt_inv_weights(np.array(samples_per_bin))
    assert len(weights) == len(samples_per_bin)
    assert weights.sum() == len(samples_per_bin)
    assert all(weight > 0 for weight in weights)
