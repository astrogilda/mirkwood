from handlers.data_handler import DataHandler, DataHandlerConfig, GalaxyProperty
import numpy as np
from transformers.postprocessy_transformer import PostProcessY


def test_postprocessy():
    """
    Test to check if the DataHandler postprocess_y method works correctly.
    """
    dtype = np.dtype([
        ('log_stellar_mass', float),
        ('log_dust_mass', float),
        ('log_metallicity', float),
        ('log_sfr', float),
    ])
    y_array = np.zeros(2, dtype=dtype)
    y_array['log_stellar_mass'] = [1, 2]
    y_array['log_dust_mass'] = [3, 4]
    y_array['log_metallicity'] = [5, 6]
    y_array['log_sfr'] = [7, 8]

    config = DataHandlerConfig(mulfac=1.0)
    handler = DataHandler(config)
    postprocessed_y = PostProcessY(prop=GalaxyProperty.STELLAR_MASS).transform(
        y_array['log_stellar_mass'])
    expected_output = np.zeros(2, dtype=float)
    expected_output = [10, 100]
    np.testing.assert_array_equal(postprocessed_y, expected_output)

    postprocessed_y = PostProcessY(
        prop=GalaxyProperty.DUST_MASS).transform(y_array['log_dust_mass'])
    expected_output = np.zeros(2, dtype=float)
    expected_output = [999, 9999]
    np.testing.assert_array_equal(postprocessed_y, expected_output)

    postprocessed_y = PostProcessY(
        prop=GalaxyProperty.METALLICITY).transform(y_array['log_metallicity'])
    expected_output = np.zeros(2, dtype=float)
    expected_output = [100000, 1000000]
    np.testing.assert_array_equal(postprocessed_y, expected_output)

    postprocessed_y = PostProcessY(
        prop=GalaxyProperty.SFR).transform(y_array['log_sfr'])
    expected_output = np.zeros(2, dtype=float)
    expected_output = [9999999, 99999999]
    np.testing.assert_array_equal(postprocessed_y, expected_output)

    # test if postprocess_y raises an error when ys is a tuple of arrays
    postprocessed_y = PostProcessY(prop=GalaxyProperty.SFR).transform(
        (y_array['log_sfr'], y_array['log_sfr']))
    expected_output = np.zeros(2, dtype=float)
    expected_output = [[9999999, 99999999], [9999999, 99999999]]
    np.testing.assert_array_equal(postprocessed_y, expected_output)
