
import pytest
import numpy as np
from sklearn.utils.validation import NotFittedError

from transformers.yscaler import YScaler


def test_processy_stellar_mass():
    """
    Test to check if the YScaler works correctly for STELLAR_MASS.
    """
    stellar_mass = np.array([10, 100])
    transformer = YScaler("stellar_mass")
    transformer.fit(stellar_mass)
    log_stellar_mass = transformer.transform(stellar_mass)
    postprocessed_y = transformer.inverse_transform(log_stellar_mass)
    np.testing.assert_array_almost_equal(stellar_mass, postprocessed_y)


def test_processy_dust_mass():
    """
    Test to check if the YScaler works correctly for DUST_MASS.
    """
    dust_mass = np.array([999, 9999])
    transformer = YScaler(prop="dust_mass")
    transformer.fit(dust_mass)
    log_dust_mass = transformer.transform(dust_mass)
    postprocessed_y = transformer.inverse_transform(log_dust_mass)
    np.testing.assert_array_almost_equal(dust_mass, postprocessed_y)


def test_processy_metallicity():
    """
    Test to check if the YScaler works correctly for METALLICITY.
    """
    metallicity = np.array([100000, 1000000])
    transformer = YScaler(prop="metallicity")
    transformer.fit(metallicity)
    log_metallicity = transformer.transform(metallicity)
    postprocessed_y = transformer.inverse_transform(log_metallicity)
    np.testing.assert_array_almost_equal(metallicity, postprocessed_y)


def test_processy_sfr():
    """
    Test to check if the YScaler works correctly for SFR.
    """
    sfr = np.array([9999999, 99999999])
    transformer = YScaler(prop="sfr")
    transformer.fit(sfr)
    log_sfr = transformer.transform(sfr)
    postprocessed_y = transformer.inverse_transform(log_sfr)
    np.testing.assert_array_almost_equal(sfr, postprocessed_y)


def test_processy_invalid_prop():
    """
    Test to check if the YScaler raises an error for an invalid prop.
    """
    data = np.array([7, 8])
    transformer = YScaler(prop='INVALID')
    with pytest.raises(ValueError):
        transformer.fit(data)


def test_processy_complex_data():
    """
    Test to check if the YScaler raises an error for complex data.
    """
    complex_data = np.array([1+1j, 2+2j])
    transformer = YScaler(prop="sfr")
    with pytest.raises(ValueError):
        transformer.fit(complex_data)


def test_processy_non_numeric_data():
    """
    Test to check if the YScaler raises an error for non-numeric data.
    """
    non_numeric_data = np.array(['a', 'b'])
    transformer = YScaler(prop="sfr")
    with pytest.raises(ValueError):
        transformer.fit(non_numeric_data)


def test_processy_none_prop_fit_transform():
    """
    Test to check if the YScaler works correctly when prop is None.
    """
    data = np.array([7, 8])
    transformer = YScaler()
    transformed = transformer.fit_transform(data)
    np.testing.assert_array_equal(transformed, data)


def test_processy_none_prop_fit_then_transform():
    """
    Test to check if the YScaler works correctly when prop is None.
    """
    data = np.array([7, 8])
    transformer = YScaler()
    transformer.fit(data)
    transformed = transformer.transform(data)
    np.testing.assert_array_equal(transformed, data)


def test_processy_negative_input():
    """
    Test to check if the YScaler works correctly for negative input values.
    """
    data = np.array([-1, -2])
    transformer = YScaler(prop="stellar_mass")
    with pytest.raises(ValueError, match="All elements of X must be non-negative when prop is specified."):
        transformer.fit(data)


def test_processy_zero_input():
    """
    Test to check if the YScaler works correctly for zero input values.
    """
    data = np.array([0, 0])
    transformer = YScaler(prop="stellar_mass")
    with pytest.raises(ValueError, match="All elements of X must be non-negative when prop is specified."):
        transformer.fit(data)


def test_processy_nan_input():
    """
    Test to check if the YScaler raises an error for input data with NaN values.
    """
    data = np.array([np.nan, 3])
    transformer = YScaler(prop="stellar_mass")
    with pytest.raises(ValueError, match="Input contains NaN."):
        transformer.fit(data)


def test_processy_transform_without_fit():
    """
    Test to check if the YScaler raises an error when transform is called without first calling fit.
    """
    log_sfr = np.array([7, 8])
    transformer = YScaler(prop="sfr")
    with pytest.raises(NotFittedError):
        transformer.transform(log_sfr)


def test_processy_inverse_transform_output():
    """
    Test to check if the output of inverse_transform is almost equal to the input.
    """
    log_sfr = np.array([7, 8])
    transformer = YScaler(prop="sfr")
    transformer.fit(log_sfr)
    postprocessed_y = transformer.inverse_transform(log_sfr)
    inverse_transformed_y = transformer.transform(
        postprocessed_y)
    np.testing.assert_array_almost_equal(inverse_transformed_y, log_sfr)


def test_inverse_transform_without_fit():
    """
    Test to check if the YScaler raises an error when inverse_transform is called without first calling fit.
    """
    log_sfr = np.array([7, 8])
    transformer = YScaler(prop="sfr")
    with pytest.raises(NotFittedError):
        transformer.inverse_transform(log_sfr)


def test_processy_large_input():
    """
    Test to check if the YScaler works correctly for large input values.
    """
    data = np.array([1e20, 1e21])
    transformer = YScaler(prop="stellar_mass")
    transformer.fit(data)
    transformed = transformer.transform(data)
    expected_output = np.array([20, 21])
    np.testing.assert_array_almost_equal(transformed, expected_output)


def test_processy_small_input():
    """
    Test to check if the YScaler works correctly for small input values.
    """
    data = np.array([1e-6, 1e-7])
    transformer = YScaler(prop="stellar_mass")
    transformer.fit(data)
    transformed = transformer.transform(data)
    expected_output = np.array([-6, -7])
    assert np.allclose(transformed, expected_output, rtol=.1)


def test_processy_2d_array_input():
    """
    Test to check if the YScaler works correctly for 2D array as input.
    """
    data = np.array([[1, 2], [3, 4]])
    transformer = YScaler(prop="stellar_mass")
    transformer.fit(data)
    transformed = transformer.transform(data)
    expected_output = np.array([[0, 0.30103], [0.47712, 0.60206]])
    np.testing.assert_array_almost_equal(
        transformed, expected_output, decimal=5)


def test_processy_len_1_input():
    """
    Test to check if the YScaler works correctly for 1D array of length 1 as input.
    """
    data = np.array([1])
    transformer = YScaler(prop="stellar_mass")
    transformer.fit(data)
    transformed = transformer.transform(data)
    expected_output = np.array([0])
    np.testing.assert_array_almost_equal(transformed, expected_output)


def test_processy_empty_input():
    """
    Test to check if the YScaler works correctly for an empty array as input.
    """
    data = np.array([])
    transformer = YScaler(prop="stellar_mass")
    with pytest.raises(ValueError):
        transformer.fit(data)
