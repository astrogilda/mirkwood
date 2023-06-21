from sklearn.utils.validation import check_array, check_X_y
from typing import Optional, Union
from typing import Any, List, Optional, Union, Tuple, Dict, Callable, Type
from sklearn.base import BaseEstimator, TransformerMixin, clone
import logging
import numpy as np

from utils.validate import is_estimator_fitted

logger = logging.getLogger(__name__)

EPS = 1e-6


def apply_transform_with_checks(transformer: Union[BaseEstimator, TransformerMixin], method_name: str, X: np.ndarray,
                                y: Optional[np.ndarray] = None, sample_weight: Optional[np.ndarray] = None,
                                sanity_check: bool = False, X_val: Optional[np.ndarray] = None,
                                y_val: Optional[np.ndarray] = None,
                                val_sample_weight: Optional[np.ndarray] = None,
                                **kwargs) -> Union[np.ndarray, BaseEstimator, TransformerMixin, Tuple[Union[BaseEstimator, TransformerMixin], np.ndarray]]:

    logger.info(
        f'Applying transformation using {transformer.__class__.__name__}.')

    valid_methods = ['transform', 'fit', 'fit_transform',
                     'inverse_transform', 'predict', 'predict_std']

    if method_name not in valid_methods:
        raise ValueError(
            f"Invalid method name: {method_name}. Must be one of {valid_methods}")

    method = getattr(transformer, method_name)

    # Perform checks
    X = check_array(X, ensure_2d=True, allow_nd=False)
    if X_val is not None:
        X_val = check_array(X_val, ensure_2d=True, allow_nd=False)

    if y is not None:
        y = check_array(y, ensure_2d=False)
        X, y = check_X_y(X, y, accept_sparse=True, force_all_finite=True)
    if y_val is not None:
        y_val = check_array(y_val, ensure_2d=False)
        X_val, y_val = check_X_y(
            X_val, y_val, accept_sparse=True, force_all_finite=True)

    if sample_weight is not None:
        sample_weight = check_array(sample_weight, ensure_2d=False)
    if val_sample_weight is not None:
        val_sample_weight = check_array(val_sample_weight, ensure_2d=False)

    # Add kwargs to pass on to method if needed
    kwargs_to_pass = {}
    if method_name in ['fit', 'fit_transform'] and hasattr(transformer, 'predict'):
        kwargs_to_pass.update({
            'X_val': X_val,
            'y_val': y_val,
            'sample_weight': sample_weight,
            'val_sample_weight': val_sample_weight
        })

    # Check if transformer is fitted if needed
    if method_name in ['transform', 'inverse_transform', 'predict', 'predict_std']:
        if not is_estimator_fitted(transformer):
            raise ValueError(
                f"{transformer.__class__.__name__} has not been fitted yet. You must call 'fit' before calling '{method_name}'.")

    try:
        if y is None:
            transformed_data = method(X, **kwargs_to_pass)
        else:
            transformed_data = method(X, y, **kwargs_to_pass)
    except Exception as e:
        raise ValueError(
            f"Failed to transform data with {transformer.__class__.__name__}. Original error: {e}")

    # Perform sanity checks
    if sanity_check:
        if method_name in ['transform', 'fit_transform']:
            inverse_transformed_data = transformer.inverse_transform(
                transformed_data)
            assert np.allclose(
                a=X, b=inverse_transformed_data, rtol=0.05, atol=1e-10), 'The inverse transformation correctly reverts the data.'
        elif method_name == 'inverse_transform' and transformer.__class__.__name__ == 'StandardScaler':
            transformed_mean = np.mean(transformed_data, axis=0)
            transformer_mean = transformer.mean_
            mean_check = np.allclose(
                transformed_mean, transformer_mean, rtol=EPS, atol=EPS)
            if not np.allclose(X, np.mean(X, axis=0), rtol=1e-1):
                transformed_std = np.std(transformed_data, axis=0)
                transformer_std = np.sqrt(transformer.var_)
                std_check = np.allclose(
                    transformed_std, transformer_std, rtol=EPS, atol=EPS)
            else:
                std_check = True
            assert mean_check and std_check, 'The inverse transformation correctly reverts the data.'

    logger.info(
        f'Transformation using {transformer.__class__.__name__} completed.')

    if method_name not in ['fit_transform', 'fit']:
        return transformed_data
    elif method_name == 'fit_transform':
        return transformer, transformed_data
    else:
        return transformer
