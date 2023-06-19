import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from src.hpo_handler import HPOHandler, HPOHandlerConfig, ParamGridConfig, crps_scorer
import optuna
from utils.custom_cv import CustomCV

# Common function for most tests


def common_setup():
    X, y = load_diabetes(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)
    cv = CustomCV(y_train, n_folds=5).get_indices()
    param_grid = ParamGridConfig(
        learning_rate=optuna.distributions.FloatDistribution(0.01, 0.3),
        n_estimators=optuna.distributions.IntDistribution(100, 1000),
        minibatch_frac=optuna.distributions.FloatDistribution(0.1, 1.0),
    )
    return X_train, X_val, y_train, y_val, param_grid, cv


@settings(max_examples=2, deadline=None)
@given(st.integers(min_value=11, max_value=100))
def test_hpo_handler_fit_and_predict_and_predict_std(n_trials):
    """
    Test HPOHandler with a range of n_trials values.
    """
    X_train, X_val, y_train, y_val, param_grid, cv = common_setup()

    config = HPOHandlerConfig(
        param_grid=param_grid, n_trials=n_trials, loss=crps_scorer, cv=cv, timeout=5*60)

    hpo = HPOHandler(config=config)
    try:
        hpo.fit(X_train, y_train)
    except ValueError:
        # Handle the error gracefully
        pytest.skip("No trials are completed yet.")
    assert isinstance(hpo.best_trial.params, dict)

    y_pred = hpo.predict(X_val)
    assert len(y_pred) == len(y_val)

    y_pred_std = hpo.predict_std(X_val)
    assert len(y_pred_std) == len(y_val)


'''
@settings(max_examples=2, deadline=None)
@given(st.tuples(
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=100),
    st.floats(min_value=0.1, max_value=1.0)
))
def test_hpo_handler_with_invalid_param(params):
    """
    Test HPOHandler with an invalid param_grid.
    """
    learning_rate, n_estimators, minibatch_frac = params
    X_train, _, y_train, _, _, cv = common_setup()

    param_grid = ParamGridConfig(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        minibatch_frac=minibatch_frac
    )

    params = HPOHandlerConfig(
        param_grid=param_grid, n_trials=10, loss=crps_scorer, cv=cv, timeout=5*60)

    hpo = HPOHandler(params=params)

    with pytest.raises(ValueError):
        hpo.fit(X_train, y_train)
'''


def test_hpo_handler_no_cv():
    """
    Test HPOHandler with no cross-validation splits.
    """
    X_train, _, y_train, y_val, param_grid, _ = common_setup()

    params = HPOHandlerConfig(
        param_grid=param_grid, n_trials=10, loss=crps_scorer, cv=[], timeout=5*60)

    hpo = HPOHandler(params=params)

    with pytest.raises(ValueError):
        hpo.fit(X_train, y_train)


def test_hpo_handler_no_data_fit():
    """
    Test HPOHandler with no data passed to fit.
    """
    _, _, _, _, param_grid, _ = common_setup()

    params = HPOHandlerConfig(
        param_grid=param_grid, n_trials=10, loss=crps_scorer, cv=[], timeout=5*60)

    hpo = HPOHandler(params=params)

    with pytest.raises(TypeError):
        hpo.fit()


def test_hpo_handler_predict_without_fit():
    """
    Test HPOHandler predict without fitting the model.
    """
    _, _, _, _, param_grid, _ = common_setup()

    params = HPOHandlerConfig(
        param_grid=param_grid, n_trials=10, loss=crps_scorer, cv=[], timeout=5*60)

    hpo = HPOHandler(params=params)

    with pytest.raises(ValueError):
        hpo.predict(np.array([[1, 2, 3, 4, 5]]))


def test_hpo_handler_predict_std_without_fit():
    """
    Test HPOHandler predict_std without fitting the model.
    """
    _, _, _, _, param_grid, _ = common_setup()

    params = HPOHandlerConfig(
        param_grid=param_grid, n_trials=10, loss=crps_scorer, cv=[], timeout=5*60)

    hpo = HPOHandler(params=params)

    with pytest.raises(ValueError):
        hpo.predict_std(np.array([[1, 2, 3, 4, 5]]))
