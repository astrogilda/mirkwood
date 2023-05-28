from hypothesis import given, settings
from hypothesis.strategies import integers
from utils.custom_cv import CustomCV
import numpy as np


@given(n_folds=integers(min_value=1, max_value=100))
@settings(
    deadline=None, max_examples=10
)
def test_data_handler_custom_cv(n_folds: int):
    """
    Test to check if the custom_cv function in DataHandler is returning the correct number of folds.

    The function uses the hypothesis library to generate random integers as input.
    """
    y = np.random.rand(100)
    try:
        cv_indices = CustomCV(y, n_folds=n_folds).get_indices()
        assert len(cv_indices) == n_folds
    except ValueError as ve:
        if n_folds < 2:
            assert str(ve) == 'n_folds must be at least 2'
        elif n_folds > len(y):
            assert str(
                ve) == 'n_folds cannot be greater than the number of samples'
        else:
            raise
