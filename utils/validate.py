from typing import Any, List, Optional, Union, Tuple, Dict, Callable, Type
import functools
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.estimator_checks import (
    check_parameters_default_constructible,
    check_estimator,
    _yield_all_checks
)
from sklearn.utils.estimator_checks import check_parameters_default_constructible, check_estimator
from sklearn.utils import all_estimators
import logging
import numpy as np
from pathlib import Path


logger = logging.getLogger(__name__)

EPS = 1e-6


def validate_file_path(file_path: Optional[Path], fitting_mode: bool) -> None:
    """
    Validates a file path based on the fitting mode. 
    If a directory doesn't exist in fitting mode, it will be created.

    Args:
        file_path: File path to validate.
        fitting_mode: If True, checks that the parent directory of the file path exists.
                      If False, checks that the file at the provided file path exists.
    """
    if file_path is None:
        return

    if fitting_mode:
        if not file_path.parent.exists():
            logger.warning(
                f"The directory {file_path.parent} does not exist. Creating it now.")
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(
                    f"Error creating directory {file_path.parent}: {str(e)}")
                raise e
    else:
        if not file_path.exists():
            error_msg = f"No file found at the path: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        elif file_path.is_dir():
            error_msg = f"Expected a file but got a directory: {file_path}"
            logger.error(error_msg)
            raise IsADirectoryError(error_msg)


def validate_input(expected_type: Type, **kwargs: Any) -> None:
    """
    Validate the format and type of inputs.
    Args:
        expected_type: The expected type of the inputs.
        **kwargs: Dictionary of input arguments.
    Raises:
        ValueError: If any of the inputs is not in the expected format or type.
    """
    if not kwargs:
        raise TypeError("No arguments were provided")

    if not isinstance(expected_type, type):
        raise ValueError("Expected type is not a type")

    for arg, value in kwargs.items():
        if not isinstance(value, expected_type):
            error_msg = f"Expected {arg} to be a {expected_type.__name__}, but got {type(value).__name__}"
            logger.error(error_msg)
            raise TypeError(error_msg)


def is_estimator_fitted(estimator: Any) -> bool:
    """
    Generalized function to check if an estimator (or any object with a fit method)
    has been fitted. This is based on the scikit-learn convention that fitted
    estimators have attributes with trailing underscores.

    Parameters
    ----------
    estimator: The estimator to check.

    Returns
    -------
    bool: True if the estimator is fitted, False otherwise.
    """
    # common fitted attributes in scikit-learn
    fitted_attr = [
        "coef_", "intercept_", "classes_", "n_iter_", "n_features_in_",  # general attributes
        "cluster_centers_", "labels_",  # clustering
        "components_", "explained_variance_", "singular_values_",  # decomposition
        "best_score_", "best_params_", "best_estimator_",  # model selection
        "n_clusters_", "children_", "n_components_",  # miscellaneous
        "feature_importances_", "oob_score_", "n_outputs_", "n_classes_",
        "class_count_", "class_prior_", "n_features_",  # naive bayes
        "theta_", "sigma_",  # gaussian naive bayes
        "fitted_", "is_fitted_",  # CustomNGBRegressor
        "transformers_", "n_features_in_", "n_features_out_",  # transformers
        "n_samples_seen_", "scale_", "var_", "mean_",  # scalers
        "n_features_", "n_outputs_", "n_input_features_",  # feature selectors
        "n_neighbors_", "effective_metric_", "effective_metric_params_",  # neighbors
        "n_trees_", "n_features_", "n_outputs_", "n_classes_",  # ensemble
        "n_features_in_", "n_features_out_", "n_components_",  # feature extraction
        "n_features_", "n_outputs_", "n_classes_", "n_layers_",  # neural network

    ]

    return any(hasattr(estimator, attr) for attr in fitted_attr)


def check_estimator_compliance(estimator: BaseEstimator, skips: set = {}) -> None:
    """
    Checks the compliance of a given estimator with scikit-learn's estimator API.

    The function first checks whether the estimator's parameters can be set to their default values.
    It then checks if the estimator adheres to scikit-learn's API.

    If the estimator doesn't pass the checks, an AssertionError is raised, indicating the reason for the failure.

    Parameters
    ----------
    estimator : BaseEstimator
        An instance of the estimator that is to be validated. 

    Raises
    ------
    AssertionError
        If the given estimator doesn't adhere to scikit-learn's estimator API.
    """
    try:
        check_parameters_default_constructible(estimator.__class__, estimator)
    except TypeError as e:
        # Handle estimators without 'get_params' method
        logger.error(
            f"Failed to check parameters for estimator {estimator.__class__}. Error: {str(e)}")

    # Customize which checks to skip
    # add more checks to skip if necessary
    skips = {"check_sample_weights_invariance",
             "check_classifiers_multilabel_output_format_decision_function"}

    def get_check_name(check):
        # The check can either be a functools.partial or a function
        if isinstance(check, functools.partial):
            return check.func.__name__
        else:
            return check.__name__

    checks_generator = (check for check in _yield_all_checks(
        estimator) if get_check_name(check) not in skips)

    estimator_clone = clone(estimator)

    for check in checks_generator:
        try:
            check(estimator_clone, estimator)
        except TypeError as e:  # handle checks that only require one argument
            if "required positional argument" in str(e):
                raise e
            else:
                check(estimator_clone)


def apply_transform_with_checks(transformer: TransformerMixin, method_name: str, X: np.ndarray, y: Optional[np.ndarray] = None, sanity_check: bool = False, **kwargs) -> Union[np.ndarray, TransformerMixin]:
    logger.info(
        f'Applying transformation using {transformer.__class__.__name__}.')

    valid_methods = ['transform', 'fit', 'fit_transform',
                     'inverse_transform', 'predict', 'predict_std']
    if method_name not in valid_methods:
        raise ValueError(
            f"Invalid method name: {method_name}. Must be one of {valid_methods}")

    method = getattr(transformer, method_name)

    try:
        if y is None:
            transformed_data = method(X)
        else:
            transformed_data = method(X, y)
    except Exception as e:
        raise ValueError(
            f"Failed to transform data with {transformer.__class__.__name__}. Original error: {e}")

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


'''
def check_all_estimators():
    estimators = all_estimators()
    for name, Estimator in estimators:
        yield check_estimator_compliance(Estimator)
'''
