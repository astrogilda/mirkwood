from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
import pytest
from hypothesis import given, settings, strategies as st
from pathlib import Path
from utils.validate import *
import numpy as np
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


# Test cases for validate_input
valid_nparray = st.just(np.array([1, 2, 3]))
invalid_nparray = st.sampled_from(["not_nparray", 42, [1, 2, 3]])
valid_list = st.lists(st.integers())
invalid_list = st.sampled_from([42, "not_a_list", np.array([1, 2, 3])])


@pytest.mark.parametrize(
    "expected_type,arg_value_strategy",
    [
        (np.ndarray, valid_nparray),
        (np.ndarray, invalid_nparray),
        (list, valid_list),
        (list, invalid_list)
    ]
)
@given(arg_value=st.data())
@settings(deadline=None)
def test_validate_input(expected_type, arg_value_strategy, arg_value) -> None:
    arg = arg_value.draw(arg_value_strategy)
    if isinstance(arg, expected_type):
        validate_input(expected_type, arg=arg)
    else:
        with pytest.raises(TypeError):
            validate_input(expected_type, arg=arg)


# Edge case when no argument is given
def test_validate_input_no_arg() -> None:
    with pytest.raises(TypeError, match="No arguments were provided"):
        validate_input(np.ndarray)


# Edge case when None is passed
def test_validate_input_none_arg() -> None:
    with pytest.raises(TypeError):
        validate_input(np.ndarray, arg=None)


# Edge case when an invalid type is provided as expected_type
def test_validate_input_invalid_expected_type() -> None:
    with pytest.raises(ValueError):
        validate_input("not_a_type", arg=np.array([1, 2, 3]))


# Edge case when multiple arguments of different types are provided
def test_validate_input_multiple_args() -> None:
    arg1 = np.array([1, 2, 3])
    arg2 = np.array([4, 5, 6])
    validate_input(np.ndarray, arg1=arg1, arg2=arg2)

    arg1 = [1, 2, 3]
    arg2 = [4, 5, 6]
    validate_input(list, arg1=arg1, arg2=arg2)

    arg1 = np.array([1, 2, 3])
    arg2 = [4, 5, 6]
    with pytest.raises(TypeError):
        validate_input(np.ndarray, arg1=arg1, arg2=arg2)

    arg1 = [1, 2, 3]
    arg2 = np.array([4, 5, 6])
    with pytest.raises(TypeError):
        validate_input(list, arg1=arg1, arg2=arg2)


# Hypothesis strategy to generate both fitted and unfitted models
@given(st.integers(min_value=0, max_value=2**32-1), st.booleans())
def test_is_estimator_fitted(random_state, is_fitted):
    # Generate data for fitting
    X, y = make_classification(n_samples=100, n_features=20)

    # Create models
    models = [LinearRegression(), SVC(), RandomForestClassifier(
        random_state=random_state)]
    model = models[random_state % len(models)]

    # Train or create unfitted model based on the test case
    if is_fitted:
        model.fit(X, y)

    # Use your utility function
    result = is_estimator_fitted(model)

    # Assert result matches expectation
    assert result == is_fitted, f"Expected {is_fitted}, but got {result}"


# Unfitted models
@given(st.randoms())
def test_unfitted_models(random_generator):
    models = [LinearRegression(), SVC(), RandomForestClassifier(
        random_state=random_generator)]
    model = models[random_generator.randint(0, len(models)-1)]

    assert not is_estimator_fitted(
        model), f"{model} should be considered unfitted, but isn't."


# Test model not following sklearn API
class NotSklearnAPIModel:
    def fit(self, X, y):
        self.coef_ = 42
        return self


def test_not_sklearn_model():
    model = NotSklearnAPIModel().fit([1, 2, 3], [1, 2, 3])
    assert is_estimator_fitted(
        model), "Model not following sklearn API should still be considered as fitted."


# Test various types of fitted models
def test_various_fitted_models():
    X, y = make_classification(n_samples=100, n_features=20)
    models = [LinearRegression(), SVC(), RandomForestClassifier(), PCA(), KMeans(), GridSearchCV(
        estimator=SVC(), param_grid={'C': [1, 10]}), GaussianNB(), GradientBoostingRegressor(), DecisionTreeClassifier()]
    for model in models:
        model.fit(X, y)
        assert is_estimator_fitted(
            model), f"Fitted {model} should be considered fitted, but isn't."


# Test various types of unfitted models
def test_various_unfitted_models():
    models = [LinearRegression(), SVC(), RandomForestClassifier(), PCA(), KMeans(), GridSearchCV(
        estimator=SVC(), param_grid={'C': [1, 10]}), GaussianNB(), GradientBoostingRegressor(), DecisionTreeClassifier()]
    for model in models:
        assert not is_estimator_fitted(
            model), f"Unfitted {model} should be considered unfitted, but isn't."


# Test validate_file_path with non-existent parent directory and fitting_mode=True
def test_validate_file_path_nonexistent_parent():
    fitting_mode = True
    with tempfile.TemporaryDirectory() as temp_dir:
        parent = Path(temp_dir) / "nonexistent"
        file_path = parent / "test_file"
        validate_file_path(file_path, fitting_mode)
        assert parent.exists(), f"Parent directory {parent} was not created."


# Test validate_file_path with a file when a directory is expected and vice versa
def test_validate_file_path_type_mismatch():
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test_file"
        with pytest.raises(FileNotFoundError):
            validate_file_path(file_path, fitting_mode=False)

        dir_path = Path(temp_dir) / "test_dir"
        dir_path.mkdir()
        with pytest.raises(IsADirectoryError):
            validate_file_path(dir_path, fitting_mode=False)


# Test validate_input with additional types
@pytest.mark.parametrize("expected_type,invalid_values", [
    (int, ["not_int", 1.23, [1, 2, 3], {"a": 1}]),
    (str, [42, 1.23, [1, 2, 3], {"a": 1}]),
    (tuple, [42, 1.23, [1, 2, 3], {"a": 1}]),
    (dict, [42, 1.23, [1, 2, 3], "not_dict"]),
])
def test_validate_input_additional_types(expected_type, invalid_values):
    for invalid_value in invalid_values:
        with pytest.raises(TypeError):
            validate_input(expected_type, arg=invalid_value)


# Test is_estimator_fitted with sklearn objects other than estimators
def test_is_estimator_fitted_scaler():
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit([[0, 0], [0, 0], [1, 1], [1, 1]])
    assert is_estimator_fitted(scaler)


#  Test is_estimator_fitted with custom estimator which doesn't follow underscore convention
class BadEstimator:
    def fit(self, X, y):
        self.coef = [1, 2, 3]  # No trailing underscore
        return self


def test_is_estimator_fitted_bad_estimator():
    estimator = BadEstimator().fit([1, 2, 3], [1, 2, 3])
    assert not is_estimator_fitted(
        estimator), "Estimator was incorrectly identified as fitted."


#  Test check_estimator_compliance with non-sklearn estimator
def test_check_estimator_compliance_bad_estimator():
    estimator = BadEstimator().fit([1, 2, 3], [1, 2, 3])
    with pytest.raises((ValueError, TypeError)):
        check_estimator_compliance(estimator)


#  Test check_estimator_compliance with sklearn estimator
def test_check_estimator_compliance_bad_estimator():
    estimator = RandomForestClassifier()
    check_estimator_compliance(estimator)
