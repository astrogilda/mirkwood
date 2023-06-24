import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from typing import Optional, Union, Tuple, Dict, Callable, Type

from enum import Enum

from utils.logger import LoggingUtility

logger = LoggingUtility.get_logger(
    __name__, log_file='logs/process_handler.log')
logger.info('Saving logs from process_handler.py')


EPS = 1e-8


class GalaxyProperty(str, Enum):
    STELLAR_MASS = "stellar_mass"
    SFR = "sfr"
    METALLICITY = "metallicity"
    DUST_MASS = "dust_mass"


def get_label_transform_func() -> Dict[str, Tuple[Callable, Callable]]:
    """
    Static method that returns a dictionary of tuples containing the forward
    and inverse transformation functions for the different galaxy properties.

    Returns
    -------
    Dict[str, Tuple[Callable, Callable]]
        A dictionary with keys as galaxy properties and values as tuples containing
        the forward and inverse transformation functions.
    """
    return {
        GalaxyProperty.STELLAR_MASS: (
            lambda x: np.log10(x + EPS),
            lambda x: np.float_power(10, np.clip(x, a_min=0, a_max=20))
        ),
        GalaxyProperty.DUST_MASS: (
            lambda x: np.log10(x + 1),
            lambda x: np.float_power(10, np.clip(x, a_min=0, a_max=20)) - 1
        ),
        GalaxyProperty.METALLICITY: (
            lambda x: np.log10(x + EPS),
            lambda x: np.float_power(10, np.clip(x, a_min=-1e1, a_max=1e1))
        ),
        GalaxyProperty.SFR: (
            lambda x: np.log10(x + 1),
            lambda x: np.float_power(10, np.clip(x, a_min=0, a_max=1e2)) - 1
        ),
    }


def check_transform_func_consistency() -> None:
    """
    Check the consistency between the GalaxyProperty Enum and the
    get_label_transform_func function. Raises a ValueError if inconsistencies are found.
    """
    transform_func_dict = get_label_transform_func()

    # Check if all Enum members have corresponding transformations
    for member in GalaxyProperty:
        if member not in transform_func_dict:
            raise ValueError(
                f"Enum member '{member}' does not have a corresponding transformation in get_label_transform_func.")

    # Check if all transformations have corresponding Enum members
    for key in transform_func_dict:
        if key not in GalaxyProperty:
            raise ValueError(
                f"Transformation key '{key}' does not have a corresponding Enum member in GalaxyProperty.")


# Call the check_transform_func_consistency function whenever the module is imported or run
check_transform_func_consistency()


class ProcessYHandler:
    """
    Transformer for preprocessing and postprocessing target variable y based on the specified galaxy property.
    """

    def __init__(self, prop: Optional[str] = None):
        """
        Constructs all the necessary attributes for the PreProcessY object.

        Parameters
        ----------
        prop : str, optional
            The name of the galaxy property to be used for preprocessing.
        """
        self.prop = prop

    def fit(self, X: np.ndarray, y=None):
        """
        Checks the input data and sets up the transformer.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        y : None
            Ignored parameter.

        Raises
        ------
        ValueError
            If X is complex or non-numeric data.
        """
        X = check_array(X, accept_sparse=True,
                        force_all_finite=True, ensure_2d=False)

        if np.iscomplexobj(X):
            raise ValueError("Complex data not supported")

        if X.dtype == object:
            raise ValueError("Non-numeric data (dtype=object) not supported")

        member_values = [member.value for member in GalaxyProperty]
        if self.prop is not None and self.prop not in member_values:
            raise ValueError(f"Invalid prop value: '{self.prop}'")

        self.prop = GalaxyProperty(self.prop) if self.prop else None

        if self.prop is not None and (X <= 0).any():
            raise ValueError(
                "All elements of X must be non-negative when prop is specified.")

        self.is_fitted_ = True

    def transform(self, X: np.ndarray):
        """
        Applies the preprocessing steps on the data.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Raises
        ------
        ValueError
            If X is complex or non-numeric data.

        Returns
        -------
        np.ndarray
            Preprocessed data.
        """
        check_is_fitted(self)

        if np.iscomplexobj(X):
            raise ValueError("Complex data not supported")

        if X.dtype == object:
            raise ValueError("Non-numeric data (dtype=object) not supported")

        X = check_array(X, accept_sparse=True,
                        force_all_finite=True, ensure_2d=False)
        transforms = get_label_transform_func()

        if self.prop is None:
            return X
        else:
            if self.prop.value in transforms:
                forward_transform, _ = transforms[self.prop.value]
                func_trans = FunctionTransformer(func=forward_transform)
                return func_trans.transform(X)
            else:
                return X

    def inverse_transform(self, X: np.ndarray):
        """
        Applies the inverse transformation on the data.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Raises
        ------
        ValueError
            If X is complex or non-numeric data.

        Returns
        -------
        np.ndarray
            Inverse transformed data.
        """
        check_is_fitted(self)

        if np.iscomplexobj(X):
            raise ValueError("Complex data not supported")

        if X.dtype == object:
            raise ValueError("Non-numeric data (dtype=object) not supported")

        X = check_array(X, accept_sparse=True,
                        force_all_finite=True, ensure_2d=False)
        transforms = get_label_transform_func()

        if self.prop is None:
            return X
        else:
            if self.prop.value in transforms:
                _, inverse_transform = transforms[self.prop.value]
                func_trans = FunctionTransformer(func=inverse_transform)
                return func_trans.transform(X)
            else:
                return X
