import logging
import numpy as np
from numba import jit
from pathlib import Path
from sklearn.utils.validation import check_consistent_length
from typing import Optional, Any


logger = logging.getLogger(__name__)


def validate_file_path(file_path: Optional[Path], fitting_mode: bool) -> None:
    """
    Validates a file path based on the fitting mode.

    Args:
        file_path: File path to validate.
        fitting_mode: If True, checks that the parent directory of the file path exists.
                      If False, checks that the file at the provided file path exists.
    """
    if file_path is None:
        return

    if fitting_mode:
        if not file_path.parent.exists():
            error_msg = f"The directory {file_path.parent} does not exist. Provide a valid path."
            logger.error(error_msg)
            raise ValueError(error_msg)
    else:
        if not file_path.exists():
            error_msg = f"No file found at the path: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)


def validate_npndarray_input(**kwargs: Any) -> None:
    """
    Validate the format and type of inputs.
    Args:
        **kwargs: Dictionary of input arguments.
    Raises:
        ValueError: If any of the inputs is not in the expected format or type.
    """
    for arg, value in kwargs.items():
        if not isinstance(value, np.ndarray):
            error_msg = f"Expected {arg} to be a numpy ndarray, but got {type(value)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
