from sklearn.utils.validation import check_array, check_X_y
from typing import Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin
import logging
import numpy as np
import inspect
from utils.validate import is_estimator_fitted
from utils.reshape import reshape_to_1d_array, reshape_to_2d_array

# Create a custom logger
logger = logging.getLogger(__name__)
# This removes all handlers from the logger object, if any exist.
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
# prevent log events from being passed to the root logger
logger.propagate = False
# Set level of logging
logger.setLevel(logging.INFO)
# Create handlers
handler = logging.FileHandler('transform_with_checks.log')
handler.setLevel(logging.INFO)
# Create formatters and add it to handlers
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
# Add handlers to the logger
logger.addHandler(handler)


def apply_transform_with_checks(
    transformer: Union[BaseEstimator, TransformerMixin],
    method_name: str,
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    **kwargs
) -> Union[np.ndarray, BaseEstimator, TransformerMixin]:

    logger.info(
        f'Applying transformation using {transformer.__class__.__name__}.')

    valid_methods = ['transform', 'fit', 'fit_transform',
                     'inverse_transform', 'predict', 'predict_std', 'pred_dist']

    if method_name not in valid_methods:
        raise ValueError(
            f"Invalid method name: {method_name}. Must be one of {valid_methods}")

    if not hasattr(transformer, method_name):
        raise AttributeError(
            f"{transformer.__class__.__name__} does not have a method called {method_name}.")

    method = getattr(transformer, method_name)
    method_signature = inspect.signature(method)
    method_params = method_signature.parameters

    kwargs_to_pass = {param: kwargs[param]
                      for param in kwargs if param in method_params}

    X = check_array(X, ensure_2d=True, allow_nd=False, force_all_finite=True)
    if 'X_val' in kwargs and kwargs['X_val'] is not None:
        kwargs['X_val'] = check_array(
            kwargs['X_val'], ensure_2d=True, allow_nd=False, force_all_finite=True)

    if y is not None:
        y = check_array(y, ensure_2d=False, force_all_finite=True)
        X, y = check_X_y(X, reshape_to_1d_array(
            y), accept_sparse=True, force_all_finite=True)
    if 'y_val' in kwargs and kwargs['y_val'] is not None:
        kwargs['y_val'] = check_array(
            kwargs['y_val'], ensure_2d=False, force_all_finite=True)
        kwargs['X_val'], kwargs['y_val'] = check_X_y(
            kwargs['X_val'], reshape_to_1d_array(kwargs['y_val']), accept_sparse=True, force_all_finite=True)

    if 'sample_weight' in kwargs and kwargs['sample_weight'] is not None:
        kwargs['sample_weight'] = check_array(
            kwargs['sample_weight'], ensure_2d=False, force_all_finite=True)
    if 'val_sample_weight' in kwargs and kwargs['val_sample_weight'] is not None:
        kwargs['val_sample_weight'] = check_array(
            kwargs['val_sample_weight'], ensure_2d=False, force_all_finite=True)

    if method_name in ['transform', 'inverse_transform', 'predict', 'predict_std', 'pred_dist']:
        if not is_estimator_fitted(transformer):
            raise ValueError(
                f"{transformer.__class__.__name__} has not been fitted yet. You must call 'fit' before calling '{method_name}'.")

    try:
        if y is None:
            transformed_data = method(X, **kwargs_to_pass)
        else:
            transformed_data = method(X, y, **kwargs_to_pass)
    except ValueError as e:
        raise ValueError(
            f"Failed to transform data with {transformer.__class__.__name__}. Original error: {e}")

    logger.info(
        f'Transformation using {transformer.__class__.__name__} completed.')

    if method_name == 'fit':
        return transformer
    elif method_name == 'fit_transform':
        return transformer, transformed_data
    else:
        return transformed_data
