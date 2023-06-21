from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from handlers.data_handler import GalaxyProperty
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PostProcessY(BaseEstimator, TransformerMixin):
    """
    Transformer for applying postprocessing steps on output data based on specified galaxy properties.

    Attributes
    ----------
    prop : str
        The name of the galaxy property to be used for postprocessing.

    Methods
    -------
    fit(X, y=None)
        Checks the input data and sets up the transformer.
    transform(X)
        Applies the postprocessing steps on the data.
    _apply_inverse_transform(X, inverse_transforms)
        Applies the specific inverse transformation based on the galaxy property.
    """

    def __init__(self, prop: Optional[str] = None):
        """
        Constructs all the necessary attributes for the PostProcessY object.

        Parameters
        ----------
        prop : str
            The name of the galaxy property to be used for postprocessing.
        """
        self.prop = prop

    @staticmethod
    def _get_label_rev_func() -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
        """
        Static method that returns a dictionary of lambda functions for the inverse transformations
        of the different galaxy properties.

        Returns
        -------
        Dict[str, Callable[[np.ndarray], np.ndarray]]
            A dictionary with keys as galaxy properties and values as the inverse transformation functions.
        """
        return {
            GalaxyProperty.STELLAR_MASS: lambda x: np.float_power(10, np.clip(x, a_min=0, a_max=20)),
            GalaxyProperty.DUST_MASS: lambda x: np.float_power(10, np.clip(x, a_min=0, a_max=20)) - 1,
            GalaxyProperty.METALLICITY: lambda x: np.float_power(10, np.clip(x, a_min=-1e1, a_max=1e1)),
            GalaxyProperty.SFR: lambda x: np.float_power(10, np.clip(x, a_min=0, a_max=1e2)) - 1,
        }

    def fit(self, X: np.ndarray, y=None) -> 'PostProcessY':
        """
        Checks the input data and sets up the transformer.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        y : np.ndarray, optional
            Target data. (Default is None)

        Raises
        ------
        ValueError
            If X is complex or non-numeric data.

        Returns
        -------
        self : PostProcessY
            The instance of the transformer.
        """
        X = check_array(X, accept_sparse=True,
                        force_all_finite=False, ensure_2d=False)

        if np.iscomplexobj(X):
            raise ValueError("Complex data not supported")

        if X.dtype == object:
            raise ValueError("Non-numeric data (dtype=object) not supported")

        inverse_transforms = self._get_label_rev_func()
        if self.prop is not None and self.prop not in GalaxyProperty._member_names_:
            raise ValueError(f"Invalid prop value: '{self.prop}'")

        self.prop = GalaxyProperty(self.prop) if self.prop else None
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies the postprocessing steps on the data.

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
            Postprocessed data.
        """
        check_is_fitted(self)

        if np.iscomplexobj(X):
            raise ValueError("Complex data not supported")

        if X.dtype == object:
            raise ValueError("Non-numeric data (dtype=object) not supported")

        X = check_array(X, accept_sparse=True,
                        force_all_finite=False, ensure_2d=False)
        inverse_transforms = self._get_label_rev_func()

        if self.prop is None:
            return X
        else:
            return self._apply_inverse_transform(X, inverse_transforms)

    def _apply_inverse_transform(self, X: np.ndarray, inverse_transforms: Dict[str, Callable]) -> np.ndarray:
        """
        Applies the specific inverse transformation based on the galaxy property.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        inverse_transforms : Dict[str, Callable]
            A dictionary of the inverse transformation functions for each galaxy property.

        Returns
        -------
        np.ndarray
            Postprocessed data.
        """
        return inverse_transforms[self.prop.value](X) if self.prop in inverse_transforms else X
