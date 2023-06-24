import logging
from typing import Optional, Callable, Dict, Tuple
from enum import Enum
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.preprocessing import FunctionTransformer

from utils.logger import LoggingUtility

logger = LoggingUtility.get_logger(
    __name__, log_file='logs/process_handler.log')

EPS = 1e-8


class GalaxyProperty(str, Enum):
    STELLAR_MASS = "stellar_mass"
    SFR = "sfr"
    METALLICITY = "metallicity"
    DUST_MASS = "dust_mass"


def get_label_transform_func() -> Dict[str, Tuple[Callable, Callable]]:
    """ Returns a dictionary of tuples containing the forward
    and inverse transformation functions for the different galaxy properties. """
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


class ProcessYHandler(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for preprocessing and postprocessing target variable y based on the specified galaxy property.
    """

    def __init__(self, prop: Optional[str] = None):
        """
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

        logger.debug("Fitting ProcessYHandler.")

        # Check if X contains complex or non-numeric data
        X = check_array(X, accept_sparse=True,
                        force_all_finite=True, ensure_2d=False)

        if np.iscomplexobj(X):
            raise ValueError("Complex data not supported")

        if X.dtype == object:
            raise ValueError("Non-numeric data (dtype=object) not supported")

        # Check if prop is a valid galaxy property
        member_values = [member.value for member in GalaxyProperty]
        if self.prop is not None and self.prop not in member_values:
            raise ValueError(f"Invalid prop value: '{self.prop}'")

        # Check if X contains negative values when prop is specified
        if self.prop is not None and (X <= 0).any():
            raise ValueError(
                "All elements of X must be non-negative when prop is specified.")

        self.is_fitted_ = True
        logger.debug("ProcessYHandler has been fitted.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
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

        logger.debug("Transforming data using ProcessYHandler.")

        check_is_fitted(self, 'is_fitted_')

        # Check if X contains complex or non-numeric data
        X = check_array(X, accept_sparse=True,
                        force_all_finite=True, ensure_2d=False)

        if np.iscomplexobj(X):
            raise ValueError("Complex data not supported")

        if X.dtype == object:
            raise ValueError("Non-numeric data (dtype=object) not supported")

        transforms = get_label_transform_func()

        if self.prop is None:
            return X
        else:
            forward_transform, _ = transforms[self.prop]
            func_trans = FunctionTransformer(func=forward_transform)
            return func_trans.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
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

        logger.debug("Applying inverse transformation using ProcessYHandler.")

        check_is_fitted(self, 'is_fitted_')

        # Check if X contains complex or non-numeric data
        X = check_array(X, accept_sparse=True,
                        force_all_finite=True, ensure_2d=False)

        if np.iscomplexobj(X):
            raise ValueError("Complex data not supported")

        if X.dtype == object:
            raise ValueError("Non-numeric data (dtype=object) not supported")

        transforms = get_label_transform_func()

        if self.prop is None:
            return X
        else:
            _, inverse_transform = transforms[self.prop]
            func_trans = FunctionTransformer(func=inverse_transform)
            return func_trans.transform(X)

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : np.ndarray
            Input data to fit_transform
        y : None
            Ignored parameter

        Returns
        -------
        X_new : np.ndarray
            Transformed array.
        """

        if y is not None:
            raise ValueError("y should be None.")

        # Fit and transform the data.
        self.fit(X, y=None)
        return self.transform(X)
