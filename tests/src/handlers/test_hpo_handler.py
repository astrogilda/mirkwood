import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from src.handlers.hpo_handler import HPOHandler, HPOHandlerConfig, ParamGridConfig, crps_scorer
from utils.custom_cv import CustomCV
import optuna
import numpy as np
from pydantic import ValidationError


# Fixture for common setup
@pytest.fixture
def setup_data():
    X, y = load_diabetes(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)
    cv = CustomCV(y_train, n_folds=5).get_indices()
    param_grid = ParamGridConfig(
        learning_rate=optuna.distributions.FloatDistribution(0.01, 0.3),
        n_estimators=optuna.distributions.IntDistribution(100, 1000),
        minibatch_frac=optuna.distributions.FloatDistribution(0.1, 1.0),
    )
    return X_train, X_val, y_train, y_val, param_grid, cv


def test_cv_is_empty(setup_data):
    X_train, _, y_train, _, param_grid, _ = setup_data
    with pytest.raises(ValidationError):
        HPOHandlerConfig(param_grid=param_grid,
                         n_trials=10, loss=crps_scorer, cv=[])


def test_cv_has_non_tuple_elements(setup_data):
    X_train, _, y_train, _, param_grid, _ = setup_data
    with pytest.raises(ValidationError):
        HPOHandlerConfig(
            param_grid=param_grid, n_trials=10, loss=crps_scorer, cv=[np.array([1, 2, 3])])


def test_cv_has_tuple_elements_of_wrong_size(setup_data):
    X_train, _, y_train, _, param_grid, _ = setup_data
    with pytest.raises(ValidationError):
        HPOHandlerConfig(param_grid=param_grid, n_trials=10, loss=crps_scorer, cv=[
            (np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3]))])


def test_cv_has_tuple_elements_of_wrong_type(setup_data):
    X_train, _, y_train, _, param_grid, _ = setup_data
    with pytest.raises(ValidationError):
        HPOHandlerConfig(
            param_grid=param_grid, n_trials=10, loss=crps_scorer, cv=[('wrong', 'type')])


def test_cv_has_2d_arrays(setup_data):
    X_train, _, y_train, _, param_grid, _ = setup_data
    with pytest.raises(ValidationError):
        HPOHandlerConfig(param_grid=param_grid, n_trials=10, loss=crps_scorer, cv=[
            (np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9], [10, 11, 12]]))])


def test_hpo_handler_fit_with_invalid_n_trials(setup_data):
    X_train, _, y_train, _, param_grid, cv = setup_data
    with pytest.raises(ValidationError):
        HPOHandlerConfig(
            param_grid=param_grid, n_trials=-10, loss=crps_scorer, cv=cv, timeout=5*60)


def test_hpo_handler_fit_with_invalid_timeout(setup_data):
    X_train, _, y_train, _, param_grid, cv = setup_data
    with pytest.raises(ValidationError):
        HPOHandlerConfig(
            param_grid=param_grid, n_trials=10, loss=crps_scorer, cv=cv, timeout=-5*60)


def test_hpo_handler_fit_with_invalid_loss_function(setup_data):
    X_train, _, y_train, _, param_grid, cv = setup_data
    with pytest.raises(ValidationError):
        HPOHandlerConfig(
            param_grid=param_grid, n_trials=10, loss="not_a_function", cv=cv, timeout=5*60)


def test_hpo_handler_no_data_fit(setup_data):
    _, _, _, _, param_grid, cv = setup_data

    config = HPOHandlerConfig(
        param_grid=param_grid, n_trials=10, loss=crps_scorer, cv=cv, timeout=5*60)

    hpo = HPOHandler(config=config)

    with pytest.raises(ValueError):
        hpo.fit(None, None)


def test_hpo_handler_predict_without_fit(setup_data):
    _, _, _, _, param_grid, cv = setup_data

    config = HPOHandlerConfig(
        param_grid=param_grid, n_trials=10, loss=crps_scorer, cv=cv, timeout=5*60)

    hpo = HPOHandler(config=config)

    with pytest.raises(ValueError):
        hpo.predict(np.array([[1, 2, 3, 4, 5]]))


def test_hpo_handler_predict_std_without_fit(setup_data):
    _, _, _, _, param_grid, cv = setup_data

    config = HPOHandlerConfig(
        param_grid=param_grid, n_trials=10, loss=crps_scorer, cv=cv, timeout=5*60)

    hpo = HPOHandler(config=config)

    with pytest.raises(ValueError):
        hpo.predict_std(np.array([[1, 2, 3, 4, 5]]))


def test_hpo_handler_fit_with_empty_param_grid(setup_data):
    """Passing an empty list as param_grid should raise a ValidationError, PROVIDED cv is not provided. If cv is provided, the param_grid is set to its default value"""
    X_train, _, y_train, _, _, cv = setup_data
    param_grid = []
    HPOHandlerConfig(param_grid=param_grid, n_trials=10,
                     loss=crps_scorer, cv=cv, timeout=5*60)
    with pytest.raises(ValidationError):
        HPOHandlerConfig(param_grid=param_grid)


def test_hpo_handler_fit_with_no_loss_function(setup_data):
    """This should not raise a ValidationError, since a loss of None results in the score function being used in CustomTransformedTargetRegressor"""
    X_train, _, y_train, _, param_grid, cv = setup_data
    config = HPOHandlerConfig(
        param_grid=param_grid, n_trials=10, loss=None, cv=cv, timeout=5*60)


@settings(max_examples=2, deadline=None, suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(st.integers(min_value=11, max_value=100))
def test_hpo_handler_fit_and_predict_and_predict_std(n_trials, setup_data):
    X_train, X_val, y_train, y_val, param_grid, cv = setup_data

    config = HPOHandlerConfig(
        param_grid=param_grid, n_trials=n_trials, loss=crps_scorer, cv=cv, timeout=5*60)

    hpo = HPOHandler(config=config)
    hpo.fit(X_train, y_train)
    assert isinstance(hpo.best_trial.params, dict)

    y_pred = hpo.predict(X_val)
    assert len(y_pred) == len(X_val)

    y_pred_std = hpo.predict_std(X_val)
    assert len(y_pred_std) == len(X_val)


'''
# TODO: enforce validations on param_grid via ParamGridConfig. As it stands the below test does not fail, as indicated


@settings(max_examples=2, deadline=None, suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(params=st.tuples(
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=100),
    st.floats(min_value=0.1, max_value=1.0)
))
def test_hpo_handler_with_invalid_param(params, setup_data):
    learning_rate, n_estimators, minibatch_frac = params
    X_train, _, y_train, _, _, cv = setup_data

    param_grid = ParamGridConfig(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        minibatch_frac=minibatch_frac
    )

    config = HPOHandlerConfig(
        param_grid=param_grid, n_trials=10, loss=crps_scorer, cv=cv, timeout=5*60)

    hpo = HPOHandler(config=config)

    with pytest.raises(ValueError):
        hpo.fit(X_train, y_train)
'''
