from hypothesis.strategies import sampled_from, lists
import pytest
from hypothesis import given, assume, settings
from hypothesis.strategies import floats, lists, text
from pydantic import ValidationError
from src.data_handler import DataHandler, DataHandlerConfig, DataSet, TrainData
import numpy as np
from typing import List


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


def test_data_handler_preprocess_y():
    """
    Test to check if the DataHandler preprocess_y method works correctly.
    """
    pass  # your implementation here


def test_data_set_is_loaded():
    """
    Test to check if the DataSet is_loaded property works correctly.
    """
    pass  # your implementation here


def test_data_set_load():
    """
    Test to check if the DataSet load method works correctly.
    """
    pass  # your implementation here
