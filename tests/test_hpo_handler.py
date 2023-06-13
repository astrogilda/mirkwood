import numpy as np
import pytest
from hypothesis import given, strategies as st
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from src.hpo_handler import HPOHandler, HPOHandlerParams, ParamGridConfig
from sklearn.metrics import mean_squared_error, make_scorer
import optuna


@given(st.integers(min_value=1, max_value=100))
def test_hpo_handler(n_trials: int):
    """
    Test HPOHandler with a range of n_trials values.
    """
    X, y = load_diabetes(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)

    param_grid = ParamGridConfig(
        learning_rate=optuna.distributions.FloatDistribution(0.01, 0.3),
        n_estimators=optuna.distributions.IntDistribution(100, 1000),
        minibatch_frac=optuna.distributions.FloatDistribution(0.1, 1.0),
    )

    params = HPOHandlerParams(
        param_grid=param_grid, n_trials=n_trials, loss=make_scorer(mean_squared_error))

    hpo = HPOHandler(params)
    hpo.fit(X_train, y_train)

    assert isinstance(hpo.best_params_, dict)


def test_hpo_handler_predict():
    """
    Test the predict method of HPOHandler.
    """
    X, y = load_diabetes(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)

    param_grid = ParamGridConfig(
        learning_rate=optuna.distributions.FloatDistribution(0.01, 0.3),
        n_estimators=optuna.distributions.IntDistribution(100, 1000),
        minibatch_frac=optuna.distributions.FloatDistribution(0.1, 1.0),
    )

    params = HPOHandlerParams(
        param_grid=param_grid, n_trials=10, loss=make_scorer(mean_squared_error))

    hpo = HPOHandler(params)
    hpo.fit(X_train, y_train)

    y_pred = hpo.predict(X_val)

    assert y_pred.shape == y_val.shape


@given(st.integers(min_value=1, max_value=100), st.integers(min_value=1, max_value=100), st.floats(min_value=0.1, max_value=1.0))
def test_hpo_handler_with_invalid_param(learning_rate: int, n_estimators: int, minibatch_frac: float):
    """
    Test HPOHandler with an invalid param_grid.
    """
    X, y = load_diabetes(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)

    param_grid = ParamGridConfig(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        minibatch_frac=minibatch_frac
    )

    params = HPOHandlerParams(
        param_grid=param_grid, n_trials=10, loss=make_scorer(mean_squared_error))

    hpo = HPOHandler(params)

    with pytest.raises(ValueError):
        hpo.fit(X_train, y_train)
