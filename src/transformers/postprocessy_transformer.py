from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Callable, Dict, Union, Tuple
from sklearn.base import BaseEstimator, RegressorMixin
from handlers.data_handler import GalaxyProperty
from typing import Any, List, Optional, Union, Tuple, Dict, Callable
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.base import BaseEstimator, TransformerMixin, clone
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EPS = 1e-6


class PostProcessY(BaseEstimator, TransformerMixin):
    """
    Custom transformer to postprocess data according to the specified galaxy property.

    Parameters
    ----------
    prop : GalaxyProperty
        The galaxy property to apply the inverse transform.
    """

    def __init__(self, prop: Optional[GalaxyProperty] = None):
        self.prop = prop

    @staticmethod
    def _get_label_rev_func() -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
        """Get inverse transforms for galaxy properties."""
        return {GalaxyProperty.STELLAR_MASS: lambda x: np.float_power(10, np.clip(x, a_min=0, a_max=20)),
                GalaxyProperty.DUST_MASS: lambda x: np.float_power(10, np.clip(x, a_min=0, a_max=20)) - 1,
                GalaxyProperty.METALLICITY: lambda x: np.float_power(10, np.clip(x, a_min=-1e1, a_max=1e1)),
                GalaxyProperty.SFR: lambda x: np.float_power(10, np.clip(x, a_min=0, a_max=1e2)) - 1,
                }

    def fit(self, X: np.ndarray, y=None) -> 'PostProcessY':
        """
        Fit the transformer. 
        Not used in this transformer, hence only checks for complex data and returns self.
        """

        X = check_array(X, accept_sparse=True,
                        force_all_finite=False, ensure_2d=False, dtype=None)

        if np.iscomplexobj(X):
            raise ValueError("Complex data not supported")

        if X.dtype == object:
            raise ValueError("Non-numeric data (dtype=object) not supported")

        # Check the prop value is valid
        inverse_transforms = self._get_label_rev_func()
        if self.prop is not None and self.prop.value not in inverse_transforms.keys():
            raise ValueError("Invalid prop value")

        return self

    def transform(self, *Xs: Union[np.ndarray, Tuple[np.ndarray]]) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """
        Apply inverse transformation to the data according to the specified galaxy property.

        Parameters
        ----------
        *ys : Union[np.ndarray, Tuple[np.ndarray]]
            Preprocessed data.

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray]]
            Postprocessed data.
        """
        # Ensure fit has been called
        check_is_fitted(self)

        if any(np.iscomplexobj(X) for X in Xs):
            raise ValueError("Complex data not supported")

        for X in Xs:
            if X.dtype == object:
                raise ValueError(
                    "Non-numeric data (dtype=object) not supported")
            X = check_array(X, accept_sparse=True,
                            force_all_finite=False, ensure_2d=False, dtype=None)

        inverse_transforms = self._get_label_rev_func()
        if self.prop is None:
            return Xs
        else:
            postprocessed_Xs = [self._apply_inverse_transform(
                X, inverse_transforms) for X in Xs]
            return tuple(postprocessed_Xs) if len(postprocessed_Xs) > 1 else postprocessed_Xs[0]

    def _apply_inverse_transform(self, X: np.ndarray, inverse_transforms: Dict[str, Callable]) -> np.ndarray:
        """Apply the inverse transformation for the specific galaxy property."""
        postprocessed_X = np.zeros_like(X)
        for key, func in inverse_transforms.items():
            if self.prop.value in key:
                postprocessed_X = func(X)
        return postprocessed_X
