from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Callable
import numpy as np
import pandas as pd

from pydantic import BaseModel, Field, validator, parse_obj_as

from enum import Enum


class TrainData(str, Enum):
    EAGLE = "eagle"
    TNG = "tng"
    SIMBA = "simba"


class GalaxyProperty(str, Enum):
    STELLAR_MASS = "stellar_mass"
    SFR = "sfr"
    METALLICITY = "metallicity"
    DUST_MASS = "dust_mass"


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

    def load(self) -> None:
        """
        Load the dataset from disk.

        Raises
        ------
        FileNotFoundError
            If the files 'X.pkl' or 'y.pkl' cannot be found.
        """
        if not self.is_loaded:

            base_path = Path.cwd()
            simulations_path = base_path.joinpath('Simulations')
            path = simulations_path.joinpath(self.name).resolve()

            try:
                self.X = pd.read_pickle(path.joinpath('X.pkl'))
                self.y = pd.read_pickle(path.joinpath('y.pkl'))

            except FileNotFoundError as e:
                raise FileNotFoundError(f"Error loading data from {path}: {e}")

    def __str__(self):
        return f'DataSet {self.name}: Loaded={self.is_loaded}'


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

    def __init__(self, config: DataHandlerConfig = DataHandlerConfig()):
        self.config = config
        np.random.seed(self.config.np_seed)

    '''
    @property
    def inverse_transforms(self) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
        return {
            'log_stellar_mass': self.label_rev_func()[GalaxyProperty.STELLAR_MASS],
            'log_dust_mass': self.label_rev_func()[GalaxyProperty.DUST_MASS],
            'log_metallicity': self.label_rev_func()[GalaxyProperty.METALLICITY],
            'log_sfr': self.label_rev_func()[GalaxyProperty.SFR],
        }


    @staticmethod
    def label_rev_func():
        return {GalaxyProperty.STELLAR_MASS: lambda x: np.float_power(10, np.clip(x, a_min=0, a_max=20)),
                GalaxyProperty.DUST_MASS: lambda x: np.float_power(10, np.clip(x, a_min=0, a_max=20)) - 1,
                GalaxyProperty.METALLICITY: lambda x: np.float_power(10, np.clip(x, a_min=-1e1, a_max=1e1)),
                GalaxyProperty.SFR: lambda x: np.float_power(10, np.clip(x, a_min=0, a_max=1e2)) - 1,
                }
    '''

    def get_data(self, train_data: Union[List[TrainData], TrainData]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load and preprocess datasets.

        Parameters
        ----------
        train_data : List[str] or str
            List of datasets (or solitary dataset) to be used for training.

        Returns
        -------
        Tuple[pd.DataFrame, np.ndarray]
            Preprocessed X matrix and y vector.
        """
        if not train_data:
            raise ValueError("No datasets specified for loading.")

        if isinstance(train_data, TrainData):
            train_data = [train_data]

        X_list, y_list = [], []

        # Load and concatenate datasets
        for key in train_data:
            dataset = self.config.datasets[key.value]
            if not dataset.is_loaded:
                dataset.load()
            X_list.append(dataset.X)
            y_list.append(dataset.y)

        X = pd.concat(X_list, axis=0).reset_index(drop=True)
        y = pd.concat(y_list, axis=0).reset_index(drop=True)

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
        log_stellar_mass = np.log10(
            y[GalaxyProperty.STELLAR_MASS].values + self.config.eps)  # .reshape(-1, 1)
        log_dust_mass = np.log10(
            1 + y[GalaxyProperty.DUST_MASS].values)  # .reshape(-1, 1)
        log_metallicity = np.log10(
            y[GalaxyProperty.METALLICITY].values)  # .reshape(-1, 1)
        log_sfr = np.log10(1 + y[GalaxyProperty.SFR].values)  # .reshape(-1, 1)

        dtype = np.dtype([
            ('log_stellar_mass', float),
            ('log_dust_mass', float),
            ('log_metallicity', float),
            ('log_sfr', float),
        ])

        y_array = np.empty(len(y), dtype=dtype)
        y_array['log_stellar_mass'] = log_stellar_mass
        y_array['log_dust_mass'] = log_dust_mass
        y_array['log_metallicity'] = log_metallicity
        y_array['log_sfr'] = log_sfr

        return y_array
