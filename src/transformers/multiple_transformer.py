from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
import numpy as np
import logging
from typing import List, Optional

from src.transformers.xandy_transformers import YTransformer, TransformerBase, TransformerConfig
from utils.transform_with_checks import apply_transform_with_checks
from utils.validate import validate_input
from utils.reshape import reshape_to_1d_array

from utils.logger import LoggingUtility

logger = LoggingUtility.get_logger(
    __name__, log_file='logs/multiple_transformer.log')
logger.info('Saving logs from multiple_transformer.py')


EPS = 1e-6


"""
Notes:
- MultipleTransformer is a transformer that applies a list of transformers sequentially.
- It is designed to be used with both the XTransformer and YTransformer as inputs for `transformers`; see the tests for example usage.
- However, it is recommended to use it only with an instance of YTransformer as input.
- Since it has been designed to be generic, please check by hand before passing in an instance of YTransformer that the shape of `X` in `fit`, `transform`, and `inverse_transform` is (n,1). If it is not, please reshape it to (n,1) before passing it in.
- It is designed to be used both directly, and in a TransformedTargetRegressor, and in a Pipeline.

"""


class MultipleTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies a list of transformers sequentially.

    Parameters
    ----------
    transformers : List[TransformerConfig]
    sanity_check : bool, default=True
    """

    def __init__(self, transformers: Optional[List[TransformerConfig]] = [], sanity_check: bool = True):
        self.transformers = transformers if transformers is not None else []
        self.sanity_check = sanity_check

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params) -> "MultipleTransformer":
        validate_input(list, transformers=self.transformers)
        validate_input(bool, sanity_check=self.sanity_check)

        if y is None:
            X = check_array(X, accept_sparse=True,
                            force_all_finite=True, ensure_2d=True)
        else:
            X, y = check_X_y(X, reshape_to_1d_array(y), accept_sparse=True,
                             force_all_finite=True, multi_output=False, ensure_2d=True)

        self.transformers_ = []

        if self.transformers is not None:
            for transformer_config in self.transformers:
                if transformer_config is None or not (hasattr(transformer_config, "transformer") and hasattr(transformer_config, "name")):
                    raise ValueError("Invalid transformer configuration.")
                fitted_transformer = apply_transform_with_checks(
                    transformer=transformer_config.transformer, method_name='fit',
                    X=X, y=y, sanity_check=self.sanity_check, **fit_params)
                self.transformers_.append(fitted_transformer)

        self.fitted_ = True
        return self

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params) -> np.ndarray:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : np.ndarray
            Input data to fit_transform
        y : Optional[np.ndarray]
            Target data to fit_transform
        fit_params : dict
            Optional parameters to use during fitting.

        Returns
        -------
        X_new : np.ndarray
            Transformed array.
        """

        if y is None:
            # Fit and transform the data.
            self.fit(X, **fit_params)
            return self.transform(X)
        else:
            # Fit and transform the data.
            self.fit(X, y, **fit_params)
            return self.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'fitted_')
        X = check_array(X, accept_sparse=True,
                        force_all_finite=True, ensure_2d=True)
        if len(self.transformers_) > 0:
            for transformer in self.transformers_:
                X = apply_transform_with_checks(
                    transformer=transformer, method_name='transform',
                    X=X, sanity_check=self.sanity_check)
            return X
        else:
            logger.info("No transformers found. Acting as a passthrough.")
            return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'fitted_')
        X = check_array(X, accept_sparse=True,
                        force_all_finite='allow-nan', ensure_2d=True)
        if len(self.transformers_) > 0:
            for transformer in reversed(self.transformers_):
                X = apply_transform_with_checks(
                    transformer=transformer, method_name='inverse_transform',
                    X=X, sanity_check=self.sanity_check)
            return X
        else:
            logger.info("No transformers found. Acting as a passthrough.")
            return X

    def get_params(self, deep=True):
        return {"transformers": self.transformers, "sanity_check": self.sanity_check}

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self
