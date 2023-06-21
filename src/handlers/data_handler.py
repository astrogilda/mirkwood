from pydantic import root_validator
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Callable
import numpy as np
import pandas as pd

from pydantic import BaseModel, Field, validator, parse_obj_as, confloat

from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)


class TrainData(str, Enum):
    EAGLE = "eagle"
    TNG = "tng"
    SIMBA = "simba"


class DataSet(BaseModel):
    """
    Class to represent a dataset.

    Attributes
    ----------
    name : str
        Name of the dataset.
    X : Optional[pd.DataFrame]
        The X matrix (features) of the dataset.
    y : Optional[pd.DataFrame]
        The y vector (target) of the dataset.
    """

    name: str
    X: Optional[pd.DataFrame] = None
    y: Optional[pd.DataFrame] = None

    class Config:
        arbitrary_types_allowed: bool = True

    @property
    def is_loaded(self) -> bool:
        """
        Check if the data set is loaded.

        Returns
        -------
        bool
            Whether the data set is loaded.
        """
        return self.X is not None and self.y is not None

    def __str__(self):
        return f'DataSet {self.name}: Loaded={self.is_loaded}'


class DataLoader:
    """
    Class to load datasets from disk.
    """

    def __init__(self, simulation_path: str):
        self.simulation_path = simulation_path

    def load(self, dataset: DataSet) -> None:
        """
        Load the dataset from disk.

        Raises
        ------
        FileNotFoundError
            If the files 'X.pkl' or 'y.pkl' cannot be found.
        """
        if not dataset.is_loaded:
            path = Path(self.simulation_path).joinpath(dataset.name).resolve()

            try:
                dataset.X = pd.read_pickle(path.joinpath('X.pkl'))
                dataset.y = pd.read_pickle(path.joinpath('y.pkl'))
                logging.info(f"Successfully loaded data from {path}")

            except FileNotFoundError as e:
                logging.error(f"Error loading data from {path}: {e}")
                raise FileNotFoundError(f"Error loading data from {path}: {e}")


class DataHandlerConfig(BaseModel):
    """
    Configuration class for DataHandler.

    Attributes
    ----------
    mulfac : float
        Multiplicative factor applied to the X matrix.
    datasets : Dict[str, DataSet]
        Available datasets.
    """
    mulfac: float = Field(
        1.0, gt=0.0, le=10.0, description="Multiplicative factor applied to the X matrix.")
    datasets: Dict[str, DataSet] = Field(default_factory=dict)
    simulation_path: str = Field(default=Path.cwd().joinpath('Simulations'))

    @root_validator
    def check_datasets(cls, values):
        datasets = values.get('datasets', {})
        if not datasets:
            # If datasets is empty, populate with default datasets
            datasets = {
                TrainData.EAGLE.name: DataSet(name=TrainData.EAGLE.value),
                TrainData.SIMBA.name: DataSet(name=TrainData.SIMBA.value),
                TrainData.TNG.name: DataSet(name=TrainData.TNG.value)
            }
        else:
            # Check if all keys in user-provided dict are valid
            for name in datasets.keys():
                if name not in TrainData.__members__:
                    raise ValueError(f'invalid dataset name: {name}')
        values['datasets'] = datasets
        return values


class DataHandler:
    """
    Class to handle loading and preprocessing of datasets.

    Attributes
    ----------
    config : DataHandlerConfig
        Configuration for the data handler.
    """

    def __init__(self, config: Optional[DataHandlerConfig] = None):
        if config is None:
            self.config = DataHandlerConfig(datasets={
                TrainData.EAGLE.value: DataSet(name=TrainData.EAGLE.value),
                TrainData.SIMBA.value: DataSet(name=TrainData.SIMBA.value),
                TrainData.TNG.value: DataSet(name=TrainData.TNG.value)
            })
        else:
            self.config = config

        self.datasets = self.config.datasets
        self.loader = DataLoader(
            simulation_path=self.config.simulation_path)
        self.mulfac = self.config.mulfac

    def get_data(self, train_data: List[TrainData]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load and preprocess datasets.

        Parameters
        ----------
        train_data : List[str]
            List of datasets to be used for training.

        Returns
        -------
        Tuple[pd.DataFrame, np.ndarray]
            Preprocessed X matrix and y vector.
        """
        if not train_data:
            raise ValueError("No datasets specified for loading.")

        X_list, y_list = [], []

        # Load and concatenate datasets
        for key in train_data:
            dataset = self.config.datasets[key.name]
            if not dataset.is_loaded:
                self.loader.load(dataset)
            X_list.append(dataset.X)
            y_list.append(dataset.y)

        X = pd.concat(X_list, axis=0).reset_index(drop=True)
        y = pd.concat(y_list, axis=0).reset_index(drop=True)

        return X * self.config.mulfac, y
