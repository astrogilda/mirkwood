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
