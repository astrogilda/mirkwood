from sklearn.base import BaseEstimator, TransformerMixin
from utils.reshape import reshape_to_1d_array, reshape_to_2d_array
from typing import Any, List, Optional, Union, Tuple, Dict, Callable
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.base import BaseEstimator, TransformerMixin, clone
import numpy as np
import logging
from xandy_transformers import YTransformer, XTransformer
from utils.validate import apply_transform_with_checks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EPS = 1e-6


class MultipleTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies a list of transformers sequentially.
    This class is for internal use and should not be instantiated directly.
    Please use the create_estimator function, or YTransformer, instead.

    Parameters
    ----------
    y_transformer : YTransformer
    """

    def __init__(self, y_transformer: YTransformer, sanity_check: bool = True):
        self.y_transformer = y_transformer
        self.sanity_check = sanity_check

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MultipleTransformer":
        if not isinstance(self.y_transformer, YTransformer):
            raise ValueError(
                "y_transformer should be an instance of YTransformer")

        if y is None:
            X = check_array(X, accept_sparse=True,
                            force_all_finite='allow-nan', ensure_2d=False)
        else:
            X, y = check_X_y(X, y, accept_sparse=True,
                             force_all_finite='allow-nan', multi_output=True)

        self.transformers_ = []
        if self.y_transformer.transformers is not None:
            for transformer in self.y_transformer.transformers:
                # use sanity_check attribute here
                fitted_transformer = apply_transform_with_checks(
                    transformer=clone(transformer.transformer), method_name='fit',
                    X=reshape_to_2d_array(X), y=y, sanity_check=self.sanity_check)
                self.transformers_.append(fitted_transformer)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X, accept_sparse=True,
                        force_all_finite='allow-nan', ensure_2d=False)

        if hasattr(self, "transformers_") and self.transformers_:
            for transformer in self.transformers_:
                # use sanity_check attribute here
                X = apply_transform_with_checks(
                    transformer=transformer, method_name='transform',
                    X=reshape_to_2d_array(X), sanity_check=self.sanity_check)
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X, accept_sparse=True,
                        force_all_finite='allow-nan', ensure_2d=False)

        if hasattr(self, "transformers_") and self.transformers_:
            for transformer in reversed(self.transformers_):
                # use sanity_check attribute here
                X = apply_transform_with_checks(
                    transformer=transformer, method_name='inverse_transform',
                    X=X, sanity_check=self.sanity_check)
        return X

    def get_params(self, deep=True):
        return {"y_transformer": self.y_transformer, "sanity_check": self.sanity_check}

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self
