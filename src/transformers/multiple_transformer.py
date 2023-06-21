from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
import numpy as np
import logging
from src.transformers.xandy_transformers import YTransformer, TransformerBase, TransformerConfig
from utils.transform_with_checks import apply_transform_with_checks
from utils.validate import validate_input
from utils.reshape import reshape_to_1d_array, reshape_to_2d_array
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MultipleTransformer":
        validate_input(list, transformers=self.transformers)
        validate_input(bool, sanity_check=self.sanity_check)

        if y is None:
            X = check_array(X, accept_sparse=True,
                            force_all_finite=True, ensure_2d=True)
        else:
            X, y = check_X_y(X, y, accept_sparse=True,
                             force_all_finite=True, multi_output=False, ensure_2d=True)

        self.transformers_ = []

        if self.transformers is not None:
            for transformer_config in self.transformers:
                if transformer_config is None or not (hasattr(transformer_config, "transformer") and hasattr(transformer_config, "name")):
                    raise ValueError("Invalid transformer configuration.")
                fitted_transformer = apply_transform_with_checks(
                    transformer=transformer_config.transformer, method_name='fit',
                    X=X, y=y, sanity_check=self.sanity_check)
                self.transformers_.append(fitted_transformer)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'transformers_')
        X = check_array(X, accept_sparse=True,
                        force_all_finite=True, ensure_2d=False)
        if hasattr(self, "transformers_"):
            for transformer in self.transformers_:
                X = apply_transform_with_checks(
                    transformer=transformer, method_name='transform',
                    X=X, sanity_check=self.sanity_check)
            return X
        else:
            raise AttributeError("Transformer not fitted")

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'transformers_')
        X = check_array(X, accept_sparse=True,
                        force_all_finite='allow-nan', ensure_2d=False)
        if hasattr(self, "transformers_"):
            for transformer in reversed(self.transformers_):
                X = apply_transform_with_checks(
                    transformer=transformer, method_name='inverse_transform',
                    X=X, sanity_check=self.sanity_check)
            return X
        else:
            raise AttributeError("Transformer not fitted")

    def get_params(self, deep=True):
        return {"transformers": self.transformers, "sanity_check": self.sanity_check}

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self
