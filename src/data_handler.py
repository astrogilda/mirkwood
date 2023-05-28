from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd

from pydantic import BaseModel, Field, validator, parse_obj_as

from enum import Enum

# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")


class TrainData(str, Enum):
    EAGLE = "eagle"
    TNG = "tng"
    SIMBA = "simba"


class DataFrameField(pd.DataFrame):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value):
        if value is not None and not isinstance(value, pd.DataFrame):
            raise ValueError('Value must be a pandas DataFrame')
        return parse_obj_as(pd.DataFrame, value)


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

    arbitrary_types_allowed: bool = True
    name: str
    X: Optional[DataFrameField] = None
    y: Optional[DataFrameField] = None

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

    def load(self) -> None:
        """
        Load the dataset from disk.

        Raises
        ------
        IOError
            If an error occurs while loading the data.
        """
        if not self.is_loaded:
            path = Path('Simulations') / self.name
            try:
                self.X = pd.read_pickle(path / 'X.pkl')
                self.y = pd.read_pickle(path / 'y.pkl')
            except Exception as e:
                raise IOError(f"Error loading data from {path}: {e}")


class DataHandlerConfig(BaseModel):
    """
    Configuration class for DataHandler.

    Attributes
    ----------
    np_seed : int
        Seed for the numpy random generator.
    eps : float
        Small value used to avoid division by zero or log of zero.
    mulfac : float
        Multiplicative factor applied to the X matrix.
    train_data : List[str]
        List of datasets to be used for training.
    datasets : Dict[str, DataSet]
        Available datasets.
    """
    np_seed: int = 1
    eps: float = 1e-6
    mulfac: float = 1.0
    train_data: List[str] = Field(default_factory=list)
    datasets: Dict[str, DataSet] = Field(
        default_factory=lambda: {
            'simba': DataSet(name='simba'),
            'eagle': DataSet(name='eagle'),
            'tng': DataSet(name='tng'),
        }
    )

    @validator("mulfac")
    def _check_mulfac(cls, value: float) -> float:
        """
        Validate the mulfac attribute.

        Parameters
        ----------
        value : float
            Value to be validated.

        Returns
        -------
        float
            Validated value.

        Raises
        ------
        ValueError
            If value is not greater than zero.
        """
        if value <= 0:
            raise ValueError("mulfac should be greater than zero")
        return value


class DataHandler:
    """
    Class to handle loading and preprocessing of datasets.

    Attributes
    ----------
    config : DataHandlerConfig
        Configuration for the data handler.
    """

    def __init__(self, config: DataHandlerConfig):
        self.config = config
        np.random.seed(self.config.np_seed)

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

        X, y = pd.DataFrame(), pd.DataFrame()

        # Load and concatenate datasets
        for key in train_data:
            dataset = self.config.datasets[key]
            if not dataset.is_loaded:
                dataset.load()
            X = pd.concat((X, dataset.X), axis=0).reset_index(drop=True)
            y = pd.concat((y, dataset.y), axis=0).reset_index(drop=True)

        y = self.preprocess_y(y)
        return X * self.config.mulfac, y

    def preprocess_y(self, y: pd.DataFrame) -> np.ndarray:
        """
        Preprocess the y vector.

        Parameters
        ----------
        y : pd.DataFrame
            Raw y vector.

        Returns
        -------
        np.ndarray
            Preprocessed y vector.
        """
        logmass = np.log10(y['stellar_mass'].values)
        logdustmass = np.log10(1 + y['dust_mass']).values
        logmet = np.log10(y['metallicity']).values
        logsfr = np.log10(1 + y['sfr']).values

        # Avoid log of zero or negative values
        logmass[logmass < self.config.eps] = 0
        logsfr[logsfr < self.config.eps] = 0
        logdustmass[logdustmass < self.config.eps] = 0

        dtype = np.dtype([
            ('logmass', float),
            ('logdustmass', float),
            ('logmet', float),
            ('logsfr', float),
        ])

        y_array = np.empty(len(y), dtype=dtype)
        y_array['logmass'] = logmass
        y_array['logdustmass'] = logdustmass
        y_array['logmet'] = logmet
        y_array['logsfr'] = logsfr

        return y_array
