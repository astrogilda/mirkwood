import pytest
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.bootstrap_handler import BootstrapHandler, numba_resample
from src.model_handler import ModelHandler, ModelConfig
from pydantic import ValidationError


def test_numba_resample():
    idx = np.array([1, 2, 3, 4, 5])
    n_samples = 3
    resampled_idx = numba_resample(idx, n_samples)
    assert len(resampled_idx) == n_samples


def test_BootstrapHandler_init():
    x = np.random.rand(5).astype(np.float32)
    y = np.random.rand(5).astype(np.float32)

    with pytest.raises(ValidationError):
        # This should fail as frac_samples_best should be in (0, 1]
        BootstrapHandler(x=x, y=y, frac_samples_best=1.5)

    with pytest.raises(ValidationError):
        # This should fail as x and y must be 1-dimensional
        BootstrapHandler(x=np.random.rand(5, 1).astype(np.float32), y=y)

    # This should pass
    BootstrapHandler(x=x, y=y)


def test_resample_data():
    x = np.random.rand(5).astype(np.float32)
    y = np.random.rand(5).astype(np.float32)
    handler = BootstrapHandler(x=x, y=y, frac_samples_best=0.5)
    resampled = handler.resample_data(x, y)

    assert len(resampled[0]) == len(resampled[1]) == int(0.5*len(x))
    assert len(resampled[2]) == int(0.5*len(x))


def test_apply_inverse_transform():
    x = np.random.rand(5).astype(np.float32)
    y = np.random.rand(5).astype(np.float32)
    handler = BootstrapHandler(x=x, y=y)

    # Using StandardScaler for demonstration
    scaler = StandardScaler()
    scaled_y = scaler.fit_transform(y.reshape(-1, 1))

    inverse_transformed = handler.apply_inverse_transform(
        scaled_y, scaled_y, scaled_y, [scaler])

    assert np.allclose(inverse_transformed[0].reshape(-1, ), y, atol=1e-5)
    assert np.allclose(inverse_transformed[1].reshape(-1, ), y, atol=1e-5)
    assert np.allclose(inverse_transformed[2].reshape(-1, ), y, atol=1e-5)


def test_apply_reversify():
    x = np.random.rand(5).astype(np.float32)
    y = np.random.rand(5).astype(np.float32)
    handler = BootstrapHandler(x=x, y=y)

    # Simple reversify function
    def reversify_fn(arr):
        return arr[::-1]

    reversed = handler.apply_reversify(x, x, x, reversify_fn)

    assert np.array_equal(reversed[0], x[::-1])
    assert np.array_equal(reversed[1], x[::-1])
    assert np.array_equal(reversed[2], x[::-1])


def test_bootstrap_func_mp():

    x = np.random.rand(5).astype(np.float32)
    y = np.random.rand(5).astype(np.float32)
    handler = BootstrapHandler(x=x, y=y)

    model_handler = ModelHandler(x=x, y=y, config=ModelConfig())

    bootstrap_output = handler.bootstrap_func_mp(
        model_handler=model_handler, iteration_num=0)

    assert np.array_equal(bootstrap_output[0], x)
    assert np.array_equal(bootstrap_output[1], x)
    assert np.array_equal(bootstrap_output[2], x)
    assert np.array_equal(bootstrap_output[3], x)
    assert np.array_equal(bootstrap_output[4], x)
