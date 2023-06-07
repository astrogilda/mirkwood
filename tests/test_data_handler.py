from src.data_handler import DataHandler, DataHandlerConfig, DataSet, TrainData, GalaxyProperty
from hypothesis.strategies import floats, lists, sampled_from
import pytest
from hypothesis import given, assume, settings
from hypothesis.strategies import floats, lists, text, sampled_from
from pydantic import ValidationError
from src.data_handler import DataHandler, DataHandlerConfig, DataSet, TrainData
import numpy as np
from typing import List
import pandas as pd


@given(mulfac=floats(min_value=0.0, exclude_min=True))
def test_data_handler_config_mulfac(mulfac: float):
    """
    Test to check if the DataHandlerConfig is correctly assigning 'mulfac' when a positive float is provided.

    The function uses the hypothesis library to generate random positive floats as input.
    """
    config = DataHandlerConfig(mulfac=mulfac)
    assert config.mulfac == mulfac


@given(mulfac=floats(max_value=0.0))
def test_data_handler_config_mulfac_negative(mulfac: float):
    """
    Test to check if the DataHandlerConfig raises a ValidationError when 'mulfac' is non-positive.

    The function uses the hypothesis library to generate random non-positive floats as input.
    """
    with pytest.raises(ValidationError):
        config = DataHandlerConfig(mulfac=mulfac)


@given(name=sampled_from(TrainData))
def test_data_set_load_invalid(name: TrainData):
    """
    Test to check if the DataSet raises an IOError when trying to load a dataset with a random name.

    The function uses the hypothesis library to generate random strings as input.
    """
    # we only want to test invalid names here
    assume(name not in [TrainData.SIMBA, TrainData.EAGLE, TrainData.TNG])
    dataset = DataSet(name=name)
    with pytest.raises(IOError):
        dataset.load()


def test_data_set_load_valid():
    """
    Test to check if the DataSet successfully loads a dataset with a known valid name.
    """
    for name in [TrainData.SIMBA, TrainData.EAGLE, TrainData.TNG]:
        dataset = DataSet(name=name)
        try:
            dataset.load()
        except Exception:
            pytest.fail(f"Unexpected error when loading dataset {name}")


@given(train_data=lists(sampled_from([TrainData.SIMBA, TrainData.EAGLE, TrainData.TNG]), min_size=1), mulfac=floats(min_value=0.0, exclude_min=True))
@settings(deadline=None)
def test_data_handler_get_data(train_data: List[TrainData], mulfac: float):
    """
    Test to check if the DataHandler can get data with specific names.
    The function uses the hypothesis library to generate lists of specific strings and positive floats as input.
    """
    config = DataHandlerConfig(mulfac=mulfac)
    handler = DataHandler(config)
    try:
        handler.get_data(train_data)
    except IOError:
        pytest.fail(
            "IOError should not be raised when train_data contains specific values.")


def test_data_handler_postprocess_y():
    """
    Test to check if the DataHandler postprocess_y method works correctly.
    """
    dtype = np.dtype([
        ('log_stellar_mass', float),
        ('log_dust_mass', float),
        ('log_metallicity', float),
        ('log_sfr', float),
    ])
    y_array = np.empty(2, dtype=dtype)
    y_array['log_stellar_mass'] = [1, 2]
    y_array['log_dust_mass'] = [3, 4]
    y_array['log_metallicity'] = [5, 6]
    y_array['log_sfr'] = [7, 8]

    config = DataHandlerConfig(mulfac=1.0)
    handler = DataHandler(config)
    postprocessed_y = handler.postprocess_y(y_array)

    expected_output = pd.DataFrame({
        GalaxyProperty.STELLAR_MASS: [10, 100],
        GalaxyProperty.DUST_MASS: [999, 9999],
        GalaxyProperty.METALLICITY: [100000, 1000000],
        GalaxyProperty.SFR: [9999999, 99999999],
    })

    for col in expected_output.columns:
        expected_output[col] = expected_output[col].astype(
            dtype["log_"+col].type)

    pd.testing.assert_frame_equal(postprocessed_y, expected_output)


def test_data_handler_postprocess_y():
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
    postprocessed_y = handler.postprocess_y(
        y_array, GalaxyProperty.STELLAR_MASS)

    expected_output = np.zeros_like(y_array)
    expected_output['stellar_mass'] = [10, 100]

    np.testing.assert_array_equal(postprocessed_y, expected_output)

    postprocessed_y = handler.postprocess_y(y_array, GalaxyProperty.DUST_MASS)

    expected_output = np.zeros_like(y_array)
    expected_output['dust_mass'] = [999, 9999]

    np.testing.assert_array_equal(postprocessed_y, expected_output)

    postprocessed_y = handler.postprocess_y(
        y_array, GalaxyProperty.METALLICITY)

    expected_output = np.zeros_like(y_array)
    expected_output['metallicity'] = [100000, 1000000]

    np.testing.assert_array_equal(postprocessed_y, expected_output)

    postprocessed_y = handler.postprocess_y(y_array, GalaxyProperty.SFR)

    expected_output = np.zeros_like(y_array)
    expected_output['sfr'] = [9999999, 99999999]

    np.testing.assert_array_equal(postprocessed_y, expected_output)


def test_data_handler_preprocess_y():
    """
    Test to check if the DataHandler preprocess_y method works correctly.
    """
    y = pd.DataFrame({
        GalaxyProperty.STELLAR_MASS: [100, 1000],
        GalaxyProperty.DUST_MASS: [100, 200],
        GalaxyProperty.METALLICITY: [1, 2],
        GalaxyProperty.SFR: [100, 200]
    })

    config = DataHandlerConfig(mulfac=1.0)
    handler = DataHandler(config)
    preprocessed_y = handler.preprocess_y(y)

    assert isinstance(preprocessed_y, np.ndarray)
    assert np.allclose(preprocessed_y['log_stellar_mass'], [2, 3])
    assert np.allclose(preprocessed_y['log_dust_mass'], [
                       np.log10(101), np.log10(201)])
    assert np.allclose(preprocessed_y['log_metallicity'], [0, np.log10(2)])
    assert np.allclose(preprocessed_y['log_sfr'], [
                       np.log10(101), np.log10(201)])


def test_data_set_is_loaded():
    """
    Test to check if the DataSet is_loaded property works correctly.
    """
    dataset = DataSet(name=TrainData.EAGLE)
    assert not dataset.is_loaded

    try:
        dataset.load()
        assert dataset.is_loaded
    except Exception:
        pytest.fail("Unexpected error when loading dataset.")


def test_data_set_load():
    """
    Test to check if the DataSet load method works correctly.
    """
    dataset = DataSet(name=TrainData.EAGLE)

    try:
        dataset.load()
        assert dataset.is_loaded
    except Exception:
        pytest.fail("Unexpected error when loading dataset.")

    assert isinstance(dataset.X, pd.DataFrame)
    assert isinstance(dataset.y, pd.DataFrame)
