import pytest
import numpy as np
from src.model_handler import ModelHandler, ModelConfig
from src.bootstrap_handler import BootstrapHandler
from pydantic import ValidationError
from numpy.testing import assert_array_equal
from multiprocessing import Pool
import random
import string
from utils.metrics import ProbabilisticErrorMetrics, DeterministicErrorMetrics, calculate_z_score


def generate_data(n_samples: int, n_features: int):
    X = np.random.rand(n_samples, n_features).astype(np.float64)
    y = np.random.rand(n_samples).astype(np.float64)
    feature_names = [''.join(random.choices(string.ascii_letters + string.digits, k=10))
                     for _ in range(n_features)]
    return X, y, feature_names


# ModelHandler for tests
X_train, y_train, feature_names = generate_data(50, 20)
X_val, y_val, _ = generate_data(30, 20)
model_handler = ModelHandler(X_train=X_train, y_train=y_train,
                             X_val=X_val, y_val=y_val, feature_names=feature_names)


@pytest.mark.parametrize("frac_samples_best,expect_pass", [
    (0.8, True),
    (0.0, False),
    (1.0, True),
    (1.5, False),
    (-0.2, False)
])
def test_BootstrapHandler_init(frac_samples_best, expect_pass):
    if expect_pass:
        BootstrapHandler(model_handler=model_handler,
                         frac_samples_best=frac_samples_best)
    else:
        with pytest.raises(ValidationError):
            BootstrapHandler(model_handler=model_handler,
                             frac_samples_best=frac_samples_best)


@pytest.mark.parametrize("frac_samples_best,X,y", [
    (0.8, np.random.rand(50, 20), np.random.rand(50)),
    (0.9, np.random.rand(100, 20), np.random.rand(100)),
    (1.0, np.random.rand(200, 20), np.random.rand(200)),
    (0.5, np.random.rand(400, 20), np.random.rand(400)),
])
def test_bootstrap_func_mp_with_different_samples(frac_samples_best, X, y):
    feature_names = [''.join(random.choices(string.ascii_letters + string.digits, k=10))
                     for _ in range(20)]
    model_handler = ModelHandler(
        X_train=X, y_train=y, X_val=X, y_val=y, feature_names=feature_names)
    bootstrap_handler = BootstrapHandler(
        model_handler=model_handler, frac_samples_best=frac_samples_best)

    y_test, y_pred_mean, y_pred_std, shap_values_mean = bootstrap_handler.bootstrap_func_mp(
        iteration_num=0)

    # Assert the shapes are consistent
    assert_array_equal(y_test.ravel(), bootstrap_handler.model_handler.y_val)
    assert y_pred_mean.shape == y_test.shape
    assert y_pred_std.shape == y_test.shape
    assert shap_values_mean.shape == y_test.shape

    # Assert that the standard deviations are positive
    assert np.all(y_pred_std >= 0)

    # Assert that SHAP values are finite
    assert np.all(np.isfinite(shap_values_mean))


@pytest.mark.parametrize("iteration_num,expect_pass", [
    (0, True),
    (-1, False),
    (100, True),
    ('a', False),
    (None, False)
])
def test_bootstrap_func_mp_with_invalid_iteration_num(iteration_num, expect_pass):
    bootstrap_handler = BootstrapHandler(
        model_handler=model_handler, frac_samples_best=0.8)

    if expect_pass:
        bootstrap_handler.bootstrap_func_mp(iteration_num=iteration_num)
    elif isinstance(iteration_num, int):
        with pytest.raises(ValueError):
            bootstrap_handler.bootstrap_func_mp(iteration_num=iteration_num)
    else:
        with pytest.raises(TypeError):
            bootstrap_handler.bootstrap_func_mp(iteration_num=iteration_num)


def test_bootstrap_func_mp_metrics():
    # Test that the metrics are calculated correctly
    bootstrap_handler = BootstrapHandler(model_handler=model_handler)
    y_val, y_pred_mean, y_pred_std, shap_values_mean = bootstrap_handler.bootstrap_func_mp(
        iteration_num=0)

    confidence_level = 0.95
    z_score = calculate_z_score(confidence_level=confidence_level)

    deterministic_error_metrics_handler = DeterministicErrorMetrics(
        yp=y_pred_mean, yt=y_val)
    assert isinstance(deterministic_error_metrics_handler.nrmse(), float)
    assert isinstance(deterministic_error_metrics_handler.nmae(), float)
    assert isinstance(deterministic_error_metrics_handler.medianae(), float)
    assert isinstance(deterministic_error_metrics_handler.mape(), float)
    assert isinstance(deterministic_error_metrics_handler.bias(), float)
    assert isinstance(deterministic_error_metrics_handler.nbe(), float)

    probabilistic_error_metrics_handler = ProbabilisticErrorMetrics(
        yp=y_pred_mean, yt=y_val, yp_lower=y_pred_mean - z_score * y_pred_std, yp_upper=y_pred_mean + z_score * y_pred_std)
    assert isinstance(
        probabilistic_error_metrics_handler.gaussian_crps(), float)
    assert isinstance(probabilistic_error_metrics_handler.ace(), float)
    assert isinstance(probabilistic_error_metrics_handler.pinaw(), float)
    assert isinstance(
        probabilistic_error_metrics_handler.interval_sharpness(), float)


'''

# Test Data
X_train = np.random.rand(50, 3).astype(np.float64)
y_train = np.random.rand(50).astype(np.float64)
X_val = np.random.rand(30, 3).astype(np.float64)
y_val = np.random.rand(30).astype(np.float64)

feature_names = []
for _ in range(X_train.shape[1]):
    random_string = ''.join(random.choices(
        string.ascii_letters + string.digits, k=10))
    feature_names.append(random_string)

from src.data_handler import DataHandler, DataHandlerConfig, TrainData
from sklearn.model_selection import train_test_split

data_handler = DataHandler(DataHandlerConfig())
X, y = data_handler.get_data(TrainData.SIMBA)
y = y['log_stellar_mass']
feature_names = X.columns.tolist()
X = np.log10(X.values+1)
X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42)


# create_estimator for tests
ce = create_estimator(None, None, None)
ce.fit(X=X_train, y=y_train, X_val=X_val, y_val=y_val, sanity_check=True)
y_val_pred_mean_ce, y_val_pred_std_ce = ce.predict(
    X_val), ce.predict_std(X_val)
y_train_pred_mean_ce, y_train_pred_std_ce = ce.predict(
    X_train), ce.predict_std(X_train)

# ngboost model for tests
cngb = CustomNGBRegressor(config=ModelConfig())
cngb.fit(X_train, y_train, X_val=X_val, Y_val=y_val)
y_val_pred_mean_cngb, y_val_pred_std_cngb = cngb.predict(
    X_val), cngb.predict_std(X_val)
y_train_pred_mean_cngb, y_train_pred_std_cngb = cngb.predict(
    X_train), cngb.predict_std(X_train)


# modelhandler for tests
model_handler = ModelHandler(
    X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, feature_names=feature_names, estimator=ce)
model_handler.fit()
y_val_pred_mean_mh, y_val_pred_std_mh = model_handler.predict(X_test=X_val, return_std=True)

np.allclose(y_val_pred_mean_mh, y_val_pred_mean_ce)
np.allclose(y_val_pred_std_mh, y_val_pred_std_ce)

# BootstrapHandler for tests
bootstrap_handler = BootstrapHandler(model_handler=model_handler, galaxy_property=None)

#TODO: move galaxy_property to model_handler

n_jobs_bs = 10
num_bs_inner = 5
with Pool(num_bs_inner) as p:
    args = ((bootstrap_handler, j)
            for j in range(num_bs_inner))
    concat_output = p.starmap(
        BootstrapHandler.bootstrap_func_mp, args)

        
# first three elements are of shape (n, 1) and the fourth element is of shape (n, m)

# Split var into four lists, each containing all the versions of a specific element
list1, list2, list3, list4 = zip(*concat_output)

# Convert the lists to arrays and stack along a new dimension
array1 = np.stack(list1, axis=0)  # shape is (num_bs_inner, n, 1)
array2 = np.stack(list2, axis=0)  # shape is (num_bs_inner, n, 1)
array3 = np.stack(list3, axis=0)  # shape is (num_bs_inner, n, 1)
array4 = np.stack(list4, axis=0)  # shape is (num_bs_inner, n, m)

# Now calculate the mean across the k versions for each array
mean_array1 = np.mean(array1, axis=0)  # shape will be (n, 1)
mean_array2 = np.mean(array2, axis=0)  # shape will be (n, 1)
mean_array3 = np.mean(array3, axis=0)  # shape will be (n, 1)
mean_array4 = np.mean(array4, axis=0)  # shape will be (n, m)



orig_array, mu_array, std_array, shap_mu_array = np.array(
    concat_output).T



'''
