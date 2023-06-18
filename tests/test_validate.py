import pytest
from hypothesis import given, settings, strategies as st
from pathlib import Path
from utils.validate import validate_file_path, validate_npndarray_input
import numpy as np


# Test cases for validate_file_path
@pytest.mark.parametrize("fitting_mode", [True, False])
@given(file_path=st.sampled_from([Path("test_file"), None]))
@settings(deadline=None)
def test_validate_file_path(file_path: Path, fitting_mode: bool) -> None:
    if not file_path:
        validate_file_path(file_path, fitting_mode)
    else:
        with open(file_path, 'w') as f:
            f.write("test content")

        if fitting_mode:
            validate_file_path(file_path.parent, fitting_mode)
        else:
            validate_file_path(file_path, fitting_mode)

        file_path.unlink()


# Test cases for validate_npndarray_input
valid_nparray = st.just(np.array([1, 2, 3]))
invalid_nparray = st.sampled_from(["not_nparray", 42, [1, 2, 3]])


@pytest.mark.parametrize("arg_value_strategy", [valid_nparray, invalid_nparray])
@given(arg_value=st.data())
@settings(deadline=None)
def test_validate_npndarray_input(arg_value_strategy, arg_value) -> None:
    arg = arg_value.draw(arg_value_strategy)
    if isinstance(arg, np.ndarray):
        validate_npndarray_input(arg=arg)
    else:
        with pytest.raises(ValueError):
            validate_npndarray_input(arg=arg)
