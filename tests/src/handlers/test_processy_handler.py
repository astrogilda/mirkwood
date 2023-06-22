import pytest
import numpy as np

from src.handlers.processy_handler import ProcessYHandler, GalaxyProperty


def test_processy_stellar_mass():
    """
    Test to check if the ProcessYHandler works correctly for STELLAR_MASS.
    """
    log_stellar_mass = np.array([1, 2])
    transformer = ProcessYHandler("stellar_mass")
    transformer.fit(log_stellar_mass)
    postprocessed_y = transformer.inverse_transform(log_stellar_mass)
    expected_output = np.array([10, 100])
    np.testing.assert_array_equal(postprocessed_y, expected_output)


def test_processy_dust_mass():
    """
    Test to check if the ProcessYHandler works correctly for DUST_MASS.
    """
    log_dust_mass = np.array([3, 4])
    transformer = ProcessYHandler(prop="dust_mass")
    transformer.fit(log_dust_mass)
    postprocessed_y = transformer.inverse_transform(log_dust_mass)
    expected_output = np.array([999, 9999])
    np.testing.assert_array_equal(postprocessed_y, expected_output)


def test_processy_metallicity():
    """
    Test to check if the ProcessYHandler works correctly for METALLICITY.
    """
    log_metallicity = np.array([5, 6])
    transformer = ProcessYHandler(prop="metallicity")
    transformer.fit(log_metallicity)
    postprocessed_y = transformer.inverse_transform(log_metallicity)
    expected_output = np.array([100000, 1000000])
    np.testing.assert_array_equal(postprocessed_y, expected_output)


def test_processy_sfr():
    """
    Test to check if the ProcessYHandler works correctly for SFR.
    """
    log_sfr = np.array([7, 8])
    transformer = ProcessYHandler(prop="sfr")
    transformer.fit(log_sfr)
    postprocessed_y = transformer.inverse_transform(log_sfr)
    expected_output = np.array([9999999, 99999999])
    np.testing.assert_array_equal(postprocessed_y, expected_output)


def test_processy_invalid_prop():
    """
    Test to check if the ProcessYHandler raises an error for an invalid prop.
    """
    log_sfr = np.array([7, 8])
    transformer = ProcessYHandler(prop='INVALID')
    with pytest.raises(ValueError):
        transformer.fit(log_sfr)


def test_processy_complex_data():
    """
    Test to check if the ProcessYHandler raises an error for complex data.
    """
    complex_data = np.array([1+1j, 2+2j])
    transformer = ProcessYHandler(prop="sfr")
    with pytest.raises(ValueError):
        transformer.fit(complex_data)


def test_processy_non_numeric_data():
    """
    Test to check if the ProcessYHandler raises an error for non-numeric data.
    """
    non_numeric_data = np.array(['a', 'b'])
    transformer = ProcessYHandler(prop="sfr")
    with pytest.raises(ValueError):
        transformer.fit(non_numeric_data)


def test_processy_transform_without_fit():
    """
    Test to check if the ProcessYHandler raises an error when transform is called without first calling fit.
    """
    log_sfr = np.array([7, 8])
    transformer = ProcessYHandler(prop="sfr")
    with pytest.raises(Exception):
        transformer.transform(log_sfr)


def test_processy_inverse_transform_output():
    """
    Test to check if the output of inverse_transform is almost equal to the input.
    """
    log_sfr = np.array([7, 8])
    transformer = ProcessYHandler(prop="sfr")
    transformer.fit(log_sfr)
    postprocessed_y = transformer.inverse_transform(log_sfr)
    inverse_transformed_y = transformer.transform(
        postprocessed_y)
    np.testing.assert_array_almost_equal(inverse_transformed_y, log_sfr)


def test_inverse_transform_without_fit():
    """
    Test to check if the ProcessYHandler raises an error when inverse_transform is called without first calling fit.
    """
    log_sfr = np.array([7, 8])
    transformer = ProcessYHandler(prop="sfr")
    with pytest.raises(Exception):
        transformer.inverse_transform(log_sfr)


def test_processy_none_prop():
    """
    Test to check if the ProcessYHandler works correctly when prop is None.
    """
    data = np.array([7, 8])
    transformer = ProcessYHandler()
    transformer.fit(data)
    transformed = transformer.transform(data)
    np.testing.assert_array_equal(transformed, data)


def test_processy_negative_input():
    """
    Test to check if the ProcessYHandler works correctly for negative input values.
    """
    data = np.array([-1, -2])
    transformer = ProcessYHandler(prop="stellar_mass")
    with pytest.raises(ValueError, match="All elements of X must be non-negative when prop is specified."):
        transformer.fit(data)


def test_processy_zero_input():
    """
    Test to check if the ProcessYHandler works correctly for zero input values.
    """
    data = np.array([0, 0])
    transformer = ProcessYHandler(prop="stellar_mass")
    with pytest.raises(ValueError, match="All elements of X must be non-negative when prop is specified."):
        transformer.fit(data)


def test_processy_large_input():
    """
    Test to check if the ProcessYHandler works correctly for large input values.
    """
    data = np.array([1e20, 1e21])
    transformer = ProcessYHandler(prop="stellar_mass")
    transformer.fit(data)
    transformed = transformer.transform(data)
    expected_output = np.array([20, 21])
    np.testing.assert_array_almost_equal(transformed, expected_output)


def test_processy_small_input():
    """
    Test to check if the ProcessYHandler works correctly for small input values.
    """
    data = np.array([1e-6, 1e-7])
    transformer = ProcessYHandler(prop="stellar_mass")
    transformer.fit(data)
    transformed = transformer.transform(data)
    expected_output = np.array([-6, -7])
    assert np.allclose(transformed, expected_output, rtol=.1)


def test_processy_2d_array_input():
    """
    Test to check if the ProcessYHandler works correctly for 2D array as input.
    """
    data = np.array([[1, 2], [3, 4]])
    transformer = ProcessYHandler(prop="stellar_mass")
    transformer.fit(data)
    transformed = transformer.transform(data)
    expected_output = np.array([[0, 0.30103], [0.47712, 0.60206]])
    np.testing.assert_array_almost_equal(
        transformed, expected_output, decimal=5)


def test_processy_len_1_input():
    """
    Test to check if the ProcessYHandler works correctly for 1D array of length 1 as input.
    """
    data = np.array([1])
    transformer = ProcessYHandler(prop="stellar_mass")
    transformer.fit(data)
    transformed = transformer.transform(data)
    expected_output = np.array([0])
    np.testing.assert_array_almost_equal(transformed, expected_output)


def test_processy_empty_input():
    """
    Test to check if the ProcessYHandler works correctly for an empty array as input.
    """
    data = np.array([])
    transformer = ProcessYHandler(prop="stellar_mass")
    with pytest.raises(ValueError):
        transformer.fit(data)


def test_processy_nan_input():
    """
    Test to check if the ProcessYHandler raises an error for input data with NaN values.
    """
    data = np.array([np.nan, 3])
    transformer = ProcessYHandler(prop="stellar_mass")
    with pytest.raises(ValueError, match="Input contains NaN."):
        transformer.fit(data)
