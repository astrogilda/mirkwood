import logging
import numpy as np
from numba import jit
from pathlib import Path
from sklearn.utils.validation import check_consistent_length
from typing import Optional, Any, Type


logger = logging.getLogger(__name__)


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
        raise ValueError("No arguments were provided")

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
        # tree based attributes
        "feature_importances_", "oob_score_", "n_outputs_", "n_classes_",
        "class_count_", "class_prior_", "n_features_",  # naive bayes
        "theta_", "sigma_"  # gaussian naive bayes
    ]

    return any(hasattr(estimator, attr) for attr in fitted_attr)
