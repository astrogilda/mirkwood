import pytest
from handlers.data_handler import GalaxyProperty
from transformers.postprocessy_transformer import PostProcessY
import numpy as np


def test_postprocessy_stellar_mass():
    """
    Test to check if the PostProcessY works correctly for STELLAR_MASS.
    """
    log_stellar_mass = np.array([1, 2])
    transformer = PostProcessY(prop="stellar_mass")
    transformer.fit(log_stellar_mass)
    postprocessed_y = transformer.transform(log_stellar_mass)
    expected_output = np.array([10, 100])
    np.testing.assert_array_equal(postprocessed_y, expected_output)


def test_postprocessy_dust_mass():
    """
    Test to check if the PostProcessY works correctly for DUST_MASS.
    """
    log_dust_mass = np.array([3, 4])
    transformer = PostProcessY(prop="dust_mass")
    transformer.fit(log_dust_mass)
    postprocessed_y = transformer.transform(log_dust_mass)
    expected_output = np.array([999, 9999])
    np.testing.assert_array_equal(postprocessed_y, expected_output)


def test_postprocessy_metallicity():
    """
    Test to check if the PostProcessY works correctly for METALLICITY.
    """
    log_metallicity = np.array([5, 6])
    transformer = PostProcessY(prop="metallicity")
    transformer.fit(log_metallicity)
    postprocessed_y = transformer.transform(log_metallicity)
    expected_output = np.array([100000, 1000000])
    np.testing.assert_array_equal(postprocessed_y, expected_output)


def test_postprocessy_sfr():
    """
    Test to check if the PostProcessY works correctly for SFR.
    """
    log_sfr = np.array([7, 8])
    transformer = PostProcessY(prop="sfr")
    transformer.fit(log_sfr)
    postprocessed_y = transformer.transform(log_sfr)
    expected_output = np.array([9999999, 99999999])
    np.testing.assert_array_equal(postprocessed_y, expected_output)


def test_postprocessy_invalid_prop():
    """
    Test to check if the PostProcessY raises an error for an invalid prop.
    """
    log_sfr = np.array([7, 8])
    transformer = PostProcessY(prop='INVALID')
    with pytest.raises(ValueError):
        transformer.fit(log_sfr)


def test_postprocessy_complex_data():
    """
    Test to check if the PostProcessY raises an error for complex data.
    """
    complex_data = np.array([1+1j, 2+2j])
    transformer = PostProcessY(prop="sfr")
    with pytest.raises(ValueError):
        transformer.fit(complex_data)


def test_postprocessy_non_numeric_data():
    """
    Test to check if the PostProcessY raises an error for non-numeric data.
    """
    non_numeric_data = np.array(['a', 'b'])
    transformer = PostProcessY(prop="sfr")
    with pytest.raises(ValueError):
        transformer.fit(non_numeric_data)


def test_postprocessy_transform_without_fit():
    """
    Test to check if the PostProcessY raises an error when transform is called without first calling fit.
    """
    log_sfr = np.array([7, 8])
    transformer = PostProcessY(prop="sfr")
    with pytest.raises(Exception):
        transformer.transform(log_sfr)


def test_postprocessy_inverse_transform_output():
    """
    Test to check if the output of inverse_transform is almost equal to the input.
    """
    log_sfr = np.array([7, 8])
    transformer = PostProcessY(prop="sfr")
    transformer.fit(log_sfr)
    postprocessed_y = transformer.transform(log_sfr)
    inverse_transformed_y = transformer._apply_inverse_transform(
        postprocessed_y, transformer._get_label_rev_func())
    np.testing.assert_array_almost_equal(inverse_transformed_y, log_sfr)
