# Suppress all warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from src.trainpredict_handler import TrainPredictHandler
from hypothesis.extra.numpy import arrays
import pytest
from hypothesis import given, strategies as st, settings
import numpy as np
from src.model_handler import ModelHandler
from pydantic import ValidationError
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Define the strategies for generating valid input data
x_strategy = arrays(dtype=float, shape=st.integers(
    min_value=100, max_value=1000))
y_strategy = arrays(dtype=float, shape=st.integers(
    min_value=100, max_value=1000))
n_folds_strategy = st.integers(min_value=2, max_value=10)
bool_strategy = st.booleans()
float_strategy = st.floats(
    min_value=0.0001, max_value=1, allow_nan=False, allow_infinity=False)
int_strategy = st.integers(min_value=1, max_value=10)

train_predict_handler_strategy = st.builds(
    TrainPredictHandler,
    x=x_strategy,
    y=y_strategy,
    n_folds=n_folds_strategy,
    x_noise=st.none() | x_strategy,
    x_transformer=st.none(),
    y_transformer=st.none(),
    frac_samples_best=float_strategy,
    weight_bins=st.integers(min_value=1, max_value=100),
    reversifyfn=st.none(),
    testfoldnum=int_strategy,
    fitting_mode=st.just(True),
    num_bs=st.integers(min_value=1, max_value=100),
    weight_flag=bool_strategy,
    n_workers=int_strategy,
)

train_predict_handler_strategy_invalid = st.builds(
    TrainPredictHandler,
    x=x_strategy,
    y=y_strategy,
    n_folds=n_folds_strategy,
    x_noise=st.none() | x_strategy,
    x_transformer=st.none(),
    y_transformer=st.none(),
    frac_samples_best=float_strategy,
    weight_bins=st.integers(min_value=1, max_value=100),
    reversifyfn=st.none(),
    testfoldnum=int_strategy,
    fitting_mode=st.just(False),
    num_bs=st.integers(min_value=1, max_value=100),
    weight_flag=bool_strategy,
    n_workers=int_strategy,
)


@pytest.fixture
def train_predict_handler():
    # Create an instance of TrainPredictHandler using the strategy
    return train_predict_handler_strategy.example()


@given(train_predict_handler=train_predict_handler_strategy_invalid)
@settings(deadline=None, max_examples=10)
def test_train_predict_invalid_combination(train_predict_handler):
    with pytest.raises(FileNotFoundError) as exc_info:
        train_predict_handler.train_predict()


@given(train_predict_handler=train_predict_handler_strategy)
def test_train_predict(train_predict_handler):
    # Check if the model_handler property works
    assert isinstance(train_predict_handler.model_handler, ModelHandler)

    # Check if the fitting_ attribute is set correctly
    assert train_predict_handler.fitting_ is True

    # Perform the train_predict operation
    result = train_predict_handler.train_predict()

    # Check the result type and shape
    assert isinstance(result, tuple)
    assert len(result) == 7
    assert all(isinstance(arr, np.ndarray) for arr in result)

    # Check the length of the prediction arrays
    pred_mean, pred_std, pred_lower, pred_upper, pred_std_epis, actuals, shap_mean = result
    assert len(pred_mean) == len(pred_std) == len(pred_lower) == len(
        pred_upper) == len(pred_std_epis) == len(actuals) == len(shap_mean)

    # Check the shape of the prediction arrays
    assert pred_mean.shape == pred_std.shape == pred_lower.shape == pred_upper.shape == pred_std_epis.shape == actuals.shape == shap_mean.shape


def test_train_predict_invalid_input():
    # Create an instance of TrainPredictHandler with invalid input
    with pytest.raises(ValidationError) as exc_info:
        invalid_handler = TrainPredictHandler(
            x=np.array([1, 2, 3]), y=np.array([4, 5, 6]), n_folds=2)

    # Assert that the validation error contains the expected error message
    assert "ensure this value has at least 100 elements" in str(exc_info.value)


def test_train_predict_empty_cv():
    # Create an instance of TrainPredictHandler with empty CV
    with pytest.raises(ValidationError) as exc_info:
        empty_cv_handler = TrainPredictHandler(
            x=np.array(list(range(100))), y=np.array(list(range(100))), n_folds=0)

    # Assert that the validation error contains the expected error message
    # assert exc_info.value.errors()[0]["msg"] == "CV is empty"


def test_train_predict_minimum_folds():
    # Create an instance of TrainPredictHandler with the minimum valid number of folds
    min_fold_handler = TrainPredictHandler(
        x=np.array(list(range(100))), y=np.array(list(range(100))), n_folds=2)

    # Perform the train_predict operation
    result = min_fold_handler.train_predict()

    # Check the result type and shape
    assert isinstance(result, tuple)
    assert len(result) == 7
    assert all(isinstance(arr, np.ndarray) for arr in result)

    # Check the length of the prediction arrays
    pred_mean, pred_std, pred_lower, pred_upper, pred_std_epis, actuals, shap_mean = result
    assert len(pred_mean) == len(pred_std) == len(pred_lower) == len(
        pred_upper) == len(pred_std_epis) == len(actuals) == len(shap_mean) == 100

    # Check the shape of the prediction arrays
    assert pred_mean.shape == pred_std.shape == pred_lower.shape == pred_upper.shape == pred_std_epis.shape == actuals.shape == shap_mean.shape == (
        100,)


def test_train_predict_with_reversifyfn():
    # Create an instance of TrainPredictHandler with reversifyfn
    def reversifyfn(arr):
        return np.flip(arr)

    handler = TrainPredictHandler(
        x=np.array(list(range(100))), y=np.array(list(range(100))), n_folds=2, reversifyfn=reversifyfn)

    # Perform the train_predict operation
    result = handler.train_predict()

    # Perform assertions on the reversed predictions
    pred_mean, pred_std, pred_lower, pred_upper, pred_std_epis, actuals, shap_mean = result
    assert np.array_equal(pred_mean, np.flip(pred_mean))
    assert np.array_equal(pred_std, np.flip(pred_std))
    assert np.array_equal(pred_lower, np.flip(pred_lower))
    assert np.array_equal(pred_upper, np.flip(pred_upper))
    assert np.array_equal(pred_std_epis, np.flip(pred_std_epis))
    assert np.array_equal(actuals, np.flip(actuals))
    assert np.array_equal(shap_mean, np.flip(shap_mean))


def test_train_predict_invalid_weight_bins():
    # Create an instance of TrainPredictHandler with invalid weight_bins
    with pytest.raises(ValidationError) as exc_info:
        invalid_handler = TrainPredictHandler(
            x=np.array(list(range(100))), y=np.array(list(range(100))), n_folds=2, weight_bins=0
        )

    # Assert that the validation error contains the expected error message
    assert "ensure this value is greater than 0" in str(exc_info.value)


def test_train_predict_with_x_noise():
    # Create an instance of TrainPredictHandler with x_noise
    handler = TrainPredictHandler(
        x=np.array(list(range(100))), y=np.array(list(range(100))), n_folds=2, x_noise=np.array([0.1]*100)
    )

    # Perform the train_predict operation
    result = handler.train_predict()

    # Perform assertions on the result
    assert isinstance(result, tuple)
    assert len(result) == 7

    pred_mean, pred_std, pred_lower, pred_upper, pred_std_epis, actuals, shap_mean = result
    assert len(pred_mean) == len(pred_std) == len(pred_lower) == len(
        pred_upper) == len(pred_std_epis) == len(actuals) == len(shap_mean) == 1

    assert np.allclose(pred_mean, np.array([4.0]))
    assert np.allclose(pred_std, np.array([0.0]))
    assert np.allclose(pred_lower, np.array([4.0]))
    assert np.allclose(pred_upper, np.array([4.0]))
    assert np.allclose(pred_std_epis, np.array([0.0]))
    assert np.allclose(actuals, np.array([5.0]))
    assert np.allclose(shap_mean, np.array([0.0]))


def test_train_predict_with_transformers():
    # Create an instance of TrainPredictHandler with transformers
    transformer = StandardScaler()  # TransformerMixin()
    handler = TrainPredictHandler(
        x=np.array(list(range(100))).reshape(-1, 1), y=np.array(list(range(100))), n_folds=2, x_transformer=transformer, y_transformer=transformer
    )

    # Perform the train_predict operation
    result = handler.train_predict()

    # Perform assertions on the result
    assert isinstance(result, tuple)
    assert len(result) == 7

    pred_mean, pred_std, pred_lower, pred_upper, pred_std_epis, actuals, shap_mean = result
    assert len(pred_mean) == len(pred_std) == len(pred_lower) == len(
        pred_upper) == len(pred_std_epis) == len(actuals) == len(shap_mean) == 1

    assert np.allclose(pred_mean, np.array([4.0]))
    assert np.allclose(pred_std, np.array([0.0]))
    assert np.allclose(pred_lower, np.array([4.0]))
    assert np.allclose(pred_upper, np.array([4.0]))
    assert np.allclose(pred_std_epis, np.array([0.0]))
    assert np.allclose(actuals, np.array([5.0]))
    assert np.allclose(shap_mean, np.array([0.0]))


def test_train_predict_with_chained_transformers():
    # Create pipeline of two transformers for x and y
    x_transformer = Pipeline(
        [('scaler', StandardScaler()), ('minmax', MinMaxScaler())])
    y_transformer = Pipeline(
        [('scaler', StandardScaler()), ('minmax', MinMaxScaler())])

    # Create an instance of TrainPredictHandler with the chained transformers
    handler = TrainPredictHandler(
        x=np.array(list(range(100))).reshape(-1, 1),
        y=np.array(list(range(100))),
        n_folds=2,
        x_transformer=x_transformer,
        y_transformer=y_transformer
    )

    # Perform the train_predict operation
    result = handler.train_predict()

    # Perform assertions on the result
    assert isinstance(result, tuple)
    assert len(result) == 7

    pred_mean, pred_std, pred_lower, pred_upper, pred_std_epis, actuals, shap_mean = result
    assert len(pred_mean) == len(pred_std) == len(pred_lower) == len(
        pred_upper) == len(pred_std_epis) == len(actuals) == len(shap_mean) == 100
