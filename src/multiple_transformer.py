from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
import numpy as np
import logging
from src.xandy_transformers import YTransformer
from utils.validate import apply_transform_with_checks
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EPS = 1e-6


class MultipleTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies a list of transformers sequentially.

    Parameters
    ----------
    y_transformer : YTransformer
    sanity_check : bool, default=True
    """

    def __init__(self, y_transformer: YTransformer, sanity_check: bool = True):
        if not isinstance(y_transformer, YTransformer):
            raise TypeError(
                f"y_transformer should be an instance of YTransformer, but got {type(y_transformer).__name__} instead")
        if not isinstance(sanity_check, bool):
            raise TypeError("sanity_check should be a boolean")
        self.y_transformer = y_transformer
        self.sanity_check = sanity_check

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MultipleTransformer":
        if y is None:
            X = check_array(X, accept_sparse=True,
                            force_all_finite='allow-nan', ensure_2d=False)
        else:
            X, y = check_X_y(X, y, accept_sparse=True,
                             force_all_finite='allow-nan', multi_output=True)

        if hasattr(self.y_transformer, "transformers"):
            self.transformers_ = []
            if self.y_transformer.transformers is not None:
                for transformer in self.y_transformer.transformers:
                    fitted_transformer = apply_transform_with_checks(
                        transformer=transformer.transformer, method_name='fit',
                        X=X, y=y, sanity_check=self.sanity_check)
                    self.transformers_.append(fitted_transformer)
        else:
            raise AttributeError(
                "The input y_transformer does not have attribute 'transformers'")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X, accept_sparse=True,
                        force_all_finite='allow-nan', ensure_2d=False)
        if hasattr(self, "transformers_"):
            if self.transformers_:
                for transformer in self.transformers_:
                    X = apply_transform_with_checks(
                        transformer=transformer, method_name='transform',
                        X=X, sanity_check=self.sanity_check)
            return X
        else:
            raise AttributeError("Transformer not fitted")

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X, accept_sparse=True,
                        force_all_finite='allow-nan', ensure_2d=False)
        if hasattr(self, "transformers_"):
            if self.transformers_:
                for transformer in reversed(self.transformers_):
                    X = apply_transform_with_checks(
                        transformer=transformer, method_name='inverse_transform',
                        X=X, sanity_check=self.sanity_check)
            return X
        else:
            raise AttributeError("Transformer not fitted")

    def get_params(self, deep=True):
        return {"y_transformer": self.y_transformer, "sanity_check": self.sanity_check}

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self
