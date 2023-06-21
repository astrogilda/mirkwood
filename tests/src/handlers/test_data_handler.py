from hypothesis.strategies import floats, lists, sampled_from, just, one_of
import pytest
from hypothesis import given, assume, settings, HealthCheck
from pydantic import ValidationError
import numpy as np
from typing import List
import pandas as pd
from pathlib import Path

from src.handlers.data_handler import DataHandler, DataHandlerConfig, DataSet, TrainData, DataLoader


@pytest.fixture
def loader():
    base_path = Path.cwd()
    simulation_path = base_path.joinpath('Simulations')
    return DataLoader(simulation_path=simulation_path)


@given(mulfac=floats(min_value=0.0, max_value=10, exclude_min=True))
def test_data_handler_config_mulfac(mulfac: float):
    config = DataHandlerConfig(mulfac=mulfac)
    assert config.mulfac == mulfac


@given(mulfac=floats(max_value=0.0))
def test_data_handler_config_mulfac_negative(mulfac: float):
    with pytest.raises(ValidationError):
        config = DataHandlerConfig(mulfac=mulfac)


@given(name=one_of(just("abc"), just("def"), just("ghi")))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_data_loader_load_invalid(loader, name: str):
    assume(name not in [member.value for member in TrainData])
    dataset = DataSet(name=name)
    with pytest.raises(FileNotFoundError):
        loader.load(dataset)


def test_data_loader_load_valid(loader):
    for name in [TrainData.SIMBA, TrainData.EAGLE, TrainData.TNG]:
        dataset = DataSet(name=name.value)
        try:
            loader.load(dataset)
        except Exception:
            pytest.fail(f"Unexpected error when loading dataset {name}")


@given(train_data=lists(sampled_from([TrainData.SIMBA, TrainData.EAGLE, TrainData.TNG]), min_size=1), mulfac=floats(min_value=0.0, exclude_min=True, max_value=10))
@settings(deadline=None)
def test_data_handler_get_data(train_data: List[TrainData], mulfac: float):
    config = DataHandlerConfig(mulfac=mulfac, datasets={
                               data.name: DataSet(name=data.value) for data in train_data})
    handler = DataHandler(config=config)
    try:
        handler.get_data(train_data)
    except FileNotFoundError:
        pytest.fail(
            "FileNotFoundError should not be raised when train_data contains specific values.")


def test_data_set_is_loaded(loader):
    dataset = DataSet(name=TrainData.EAGLE.value)
    assert not dataset.is_loaded

    try:
        loader.load(dataset)
        assert dataset.is_loaded
    except Exception:
        pytest.fail("Unexpected error when loading dataset.")


def test_data_set_load(loader):
    dataset = DataSet(name=TrainData.EAGLE.value)

    try:
        loader.load(dataset)
        assert dataset.is_loaded
    except Exception:
        pytest.fail("Unexpected error when loading dataset.")

    assert isinstance(dataset.X, pd.DataFrame)
    assert isinstance(dataset.y, pd.DataFrame)


def test_data_handler_get_data_validity():
    config = DataHandlerConfig(mulfac=1.0, datasets={
                               TrainData.EAGLE.name: DataSet(name=TrainData.EAGLE.value)})
    handler = DataHandler(config=config)

    try:
        data = handler.get_data([TrainData.EAGLE])
    except Exception:
        pytest.fail("Unexpected error when getting data.")

    assert isinstance(data, tuple)
    X, y = data
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.DataFrame)
    assert not X.empty
    assert not y.empty
    assert X.shape[0] == y.shape[0]


@given(train_data=sampled_from([TrainData.SIMBA, TrainData.EAGLE, TrainData.TNG]))
def test_data_handler_get_data_correctness(train_data: TrainData):
    # Make sure to use TrainData enum values (SIMBA, EAGLE, TNG) as dataset names
    config = DataHandlerConfig(mulfac=1.0, datasets={
        train_data.name: DataSet(name=train_data.value)
    })
    handler = DataHandler(config=config)

    # Act
    X, y = handler.get_data([train_data])

    # Assert
    assert not X.empty
    assert not y.empty
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.DataFrame)


def test_data_handler_config_default():
    config = DataHandlerConfig()
    assert config.mulfac == 1.0
    assert list(config.datasets.keys()).sort() == [
        'eagle', 'simba', 'tng'].sort()


def test_data_handler_empty_train_data():
    config = DataHandlerConfig(mulfac=1.0, datasets={})
    assert config.mulfac == 1.0
    assert config.datasets == {
        TrainData.EAGLE.name: DataSet(name=TrainData.EAGLE.value),
        TrainData.SIMBA.name: DataSet(name=TrainData.SIMBA.value),
        TrainData.TNG.name: DataSet(name=TrainData.TNG.value)
    }


def test_data_handler_invalid_train_data():
    with pytest.raises(ValidationError):
        DataHandlerConfig(mulfac=1.0, datasets={
            "invalid": DataSet(name="invalid")})


def test_data_handler_config_invalid_mulfac():
    with pytest.raises(ValidationError):
        DataHandlerConfig(mulfac=11)


def test_data_set_load_multiple_times(loader):
    dataset = DataSet(name=TrainData.EAGLE.value)

    try:
        loader.load(dataset)
        loader.load(dataset)
    except Exception:
        pytest.fail("Unexpected error when loading dataset multiple times.")
