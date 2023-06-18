import pytest
from hypothesis import given, settings, strategies as st
from pathlib import Path
from utils.validate import validate_file_path, validate_npndarray_input
import numpy as np
import os
import tempfile


@pytest.mark.parametrize("fitting_mode", [True, False])
@given(file_path=st.sampled_from([Path("test_file"), None]))
@settings(deadline=None)
def test_validate_file_path(file_path: Path, fitting_mode: bool) -> None:
    if not file_path:
        validate_file_path(file_path, fitting_mode)
    else:
        # Create a temporary directory as the parent directory
        with tempfile.TemporaryDirectory() as temp_dir:
            full_path = Path(temp_dir) / file_path
            if not fitting_mode:
                with open(full_path, 'w') as f:
                    f.write("test content")

            validate_file_path(full_path, fitting_mode)

            if fitting_mode:
                assert full_path.parent.exists(
                ), f"{full_path.parent} was not created."
            else:
                assert full_path.exists(), f"{full_path} does not exist."

        if not fitting_mode:
            assert not full_path.exists(
            ), f"{full_path} still exists after the temporary directory was removed."


# Test case for when an invalid path is given, that is neither a string nor a Path object
@given(file_path=st.sampled_from([123, True, False, 1.23, [1, 2, 3]]))
@settings(deadline=None)
def test_validate_file_path_edge_cases(file_path: Path) -> None:
    fitting_mode = True
    with pytest.raises(Exception):
        validate_file_path(file_path, fitting_mode)


# Test case for when a valid path is given but the file/directory doesn't exist
@given(file_path=st.sampled_from([Path("test_file")]))
@settings(deadline=None)
def test_validate_file_path_nonexistent(file_path: Path) -> None:
    fitting_mode = False
    if file_path.exists():
        file_path.unlink()
    with pytest.raises(FileNotFoundError):
        validate_file_path(file_path, fitting_mode)


# Test case for when the directory already exists
def test_validate_file_path_existing_directory() -> None:
    fitting_mode = True
    with tempfile.TemporaryDirectory() as temp_dir:
        validate_file_path(Path(temp_dir), fitting_mode)


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
