from hypothesis.strategies import lists, floats, tuples
from hypothesis import given
import pytest
import numpy as np
from utils.metrics import DeterministicErrorMetrics, ProbabilisticErrorMetrics, EPS


def test_deterministic_metrics():
    # Test deterministic error metrics calculations
    yt = np.array([1, 2, 3, 4, 5])
    yp = np.array([1.1, 1.9, 3.2, 4.1, 4.8])
    metrics = DeterministicErrorMetrics(yt=yt, yp=yp)

    assert np.isclose(metrics.get_iqr(), 3.8 + EPS, atol=1e-6)
    assert np.isclose(metrics.nrmse(), 0.0657894737, atol=1e-6)
    assert np.isclose(metrics.nmae(), 0.0526315789, atol=1e-6)
    assert np.isclose(metrics.medianae(), 0.2, atol=1e-6)
    assert np.isclose(metrics.mape(), 0.0466666667, atol=1e-6)
    assert np.isclose(metrics.bias(), -0.2, atol=1e-6)
    assert np.isclose(metrics.nbe(), -0.0131578947, atol=1e-6)


def test_deterministic_metrics_failure():
    # Test failure case for deterministic error metrics
    with pytest.raises(ValueError):
        DeterministicErrorMetrics(yt=np.array([1, 2, 3]), yp=np.array([1, 2]))


def test_probabilistic_metrics():
    # Test probabilistic error metrics calculations
    yt = np.array([1, 2, 3, 4, 5])
    yp = np.array([1.1, 1.9, 3.2, 4.1, 4.8])
    yp_lower = np.array([0.9, 1.8, 3.1, 3.9, 4.6])
    yp_upper = np.array([1.2, 2.1, 3.3, 4.3, 5.0])
    metrics = ProbabilisticErrorMetrics(
        yt=yt, yp=yp, yp_lower=yp_lower, yp_upper=yp_upper)

    assert np.isclose(metrics.ace(), 0.0173, atol=1e-4)
    assert np.isclose(metrics.pinaw(), 0.0526315789, atol=1e-6)


def test_probabilistic_metrics_failure():
    # Test failure case for probabilistic error metrics
    with pytest.raises(ValueError):
        ProbabilisticErrorMetrics(yt=np.array([1, 2, 3]), yp=np.array(
            [1, 2]), yp_lower=np.array([1, 2]), yp_upper=np.array([1, 2]))


@given(lists(floats(allow_nan=False, allow_infinity=False)),
       lists(floats(allow_nan=False, allow_infinity=False)))
def test_deterministic_metrics_hypothesis(yt, yp):
    # Test deterministic error metrics using Hypothesis
    if len(yt) != len(yp) or len(yt) == 0:
        with pytest.raises(ValueError):
            metrics = DeterministicErrorMetrics(
                yt=np.array(yt), yp=np.array(yp))
    else:
        metrics = DeterministicErrorMetrics(yt=np.array(yt), yp=np.array(yp))
        assert isinstance(metrics.get_iqr(), float)
        assert isinstance(metrics.nrmse(), float)
        assert isinstance(metrics.nmae(), float)
        assert isinstance(metrics.medianae(), float)
        assert isinstance(metrics.mape(), float)
        assert isinstance(metrics.bias(), float)
        assert isinstance(metrics.nbe(), float)


@given(lists(floats(allow_nan=False, allow_infinity=False)),
       tuples(lists(floats(allow_nan=False, allow_infinity=False)),
              lists(floats(allow_nan=False, allow_infinity=False)),
              lists(floats(allow_nan=False, allow_infinity=False))))
def test_probabilistic_metrics_hypothesis(yt, yp):
    # Test probabilistic error metrics using Hypothesis
    if len(yt) != len(yp[0]) or len(yt) != len(yp[1]) or len(yt) != len(yp[2]) or len(yt) == 0:
        with pytest.raises(ValueError):
            metrics = ProbabilisticErrorMetrics(yt=np.array(yt), yp=yp)
    else:
        metrics = ProbabilisticErrorMetrics(yt=np.array(yt), yp=(
            np.array(yp[0]), np.array(yp[1]), np.array(yp[2])))
        assert isinstance(metrics.ace(), float)
        assert isinstance(metrics.pinaw(), float)
        assert isinstance(metrics.interval_sharpness(), float)
