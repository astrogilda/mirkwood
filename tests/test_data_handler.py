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


@given(name=text())
def test_data_set_load_invalid(name: str):
    """
    Test to check if the DataSet raises an IOError when trying to load a dataset with a random name.

    The function uses the hypothesis library to generate random strings as input.
    """
    assume(name not in ["simba", "eagle", "tng"]
           )  # we only want to test invalid names here

    dataset = DataSet(name=name)
    with pytest.raises(IOError):
        dataset.load()


def test_data_set_load_valid():
    """
    Test to check if the DataSet successfully loads a dataset with a known valid name.
    """
    for name in ["simba", "eagle", "tng"]:
        dataset = DataSet(name=name)
        try:
            dataset.load()
        except Exception:
            pytest.fail(f"Unexpected error when loading dataset {name}")


# sampled_from takes a finite iterable and returns a strategy which produces any of its elements.
dataset_names = sampled_from(["simba", "eagle", "tng"])

# lists takes a strategy and generates lists whose elements are drawn from that strategy.
# By setting the min_size parameter to 1, we're specifying that the list should never be empty.
non_empty_dataset_name_lists = lists(dataset_names, min_size=1)


@given(train_data=non_empty_dataset_name_lists, mulfac=floats(min_value=0.0, exclude_min=True))
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
    # Create a mock y array
    dtype = np.dtype([
        ('log_stellar_mass', float),
        ('log_dust_mass', float),
        ('log_metallicty', float),
        ('log_sfr', float),
    ])
    y_array = np.empty(2, dtype=dtype)
    y_array['log_stellar_mass'] = [1, 2]
    y_array['log_dust_mass'] = [3, 4]
    y_array['log_metallicty'] = [5, 6]
    y_array['log_sfr'] = [7, 8]

    # Create a DataHandler object
    config = DataHandlerConfig(mulfac=1.0)
    handler = DataHandler(config)

    # Call the postprocess_y method
    postprocessed_y = handler.postprocess_y(y_array)

    # Expected output DataFrame
    expected_output = pd.DataFrame({
        'stellar_mass': [10, 100],
        'dust_mass': [999, 9999],
        'metallicity': [100000, 1000000],
        'sfr': [9999999, 99999999],
    })

    # Change dtype of 'log' columns to match y_array dtype
    for col in expected_output.columns:
        expected_output[col] = expected_output[col].astype(
            dtype["log_"+col].type)

    # Compare the postprocessed y DataFrame with the expected output DataFrame
    pd.testing.assert_frame_equal(postprocessed_y, expected_output)


def test_data_handler_preprocess_y():
    """
    Test to check if the DataHandler preprocess_y method works correctly.
    """
    # Create a mock y DataFrame
    y = pd.DataFrame({
        'stellar_mass': [100, 1000],
        'dust_mass': [100, 200],
        'metallicity': [1, 2],
        'sfr': [100, 200]
    })

    # Create a DataHandler object
    config = DataHandlerConfig(mulfac=1.0)
    handler = DataHandler(config)

    # Call the preprocess_y method
    preprocessed_y = handler.preprocess_y(y)

    # Check the types and values
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
    # Create an unloaded DataSet
    dataset = DataSet(name="eagle")
    assert not dataset.is_loaded

    # Load the DataSet
    try:
        dataset.load()
        assert dataset.is_loaded
    except Exception:
        pytest.fail("Unexpected error when loading dataset.")


def test_data_set_load():
    """
    Test to check if the DataSet load method works correctly.
    """
    # Create an unloaded DataSet
    dataset = DataSet(name="eagle")

    # Try loading the DataSet
    try:
        dataset.load()
        assert dataset.is_loaded
    except Exception:
        pytest.fail("Unexpected error when loading dataset.")

    # Check that the loaded data are pandas DataFrame instances
    assert isinstance(dataset.X, pd.DataFrame)
    assert isinstance(dataset.y, pd.DataFrame)
